# noise_profile_scanner_core.py
# v1 — Task 6 of the noise profile scanner build.
#
# Pure logic for the talent-environment noise floor gate. Importable by the GUI
# wrapper (Task 7) and by REAPER's Lua-Python bridge. No CLI / __main__, matching
# intake_scanner_core_v7's "core is pure logic" pattern.
#
# Implements the locked Task 3 output contract
# (noise_profile_scanner_output_contract.md):
#   - one source-of-truth dict per scan; both artifacts serialize from it
#   - JSON safety: null (None) never NaN/Infinity; paths stored as str()
#   - output WAV suffix `_np` (trimmed profile, distinct from the raw `_noise`
#     input so re-runs never overwrite the take); sidecar shares the WAV stem
#   - central JSONL log captures every scan, both modalities
#   - log-only-on-failure: a REJECT writes no WAV and no sidecar
#
# Calibration is the Default profile in noise_profile_scanner_default.yaml.
# Threshold model and gate behaviour match that yaml; the embedded DEFAULT_CONFIG
# below mirrors it so the core runs without an external file (intake's pattern).
#
# Measurement convention matches intake_scanner_core_v7: A-weighted (IEC 61672)
# RMS, computed identically so the two tools' numbers are comparable.

import os
import sys
import json
import hashlib
import math
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime

# ------------------------------------------------------------
# CRITICAL: Make ffmpeg/ffprobe work in frozen app (PyInstaller)
# Mirrors intake_scanner_core_v7 so the GUI .app behaves the same.
# ------------------------------------------------------------
if getattr(sys, 'frozen', False):
    bundle_dir = sys._MEIPASS
    os.environ["PATH"] = bundle_dir + os.pathsep + os.environ.get("PATH", "")
    try:
        from pydub import AudioSegment
        AudioSegment.converter = os.path.join(bundle_dir, "ffmpeg")
        AudioSegment.ffprobe = os.path.join(bundle_dir, "ffprobe")
    except Exception:
        pass

# ------------------------------------------------------------
# Imports (safe)
# ------------------------------------------------------------
try:
    import soundfile as sf
    SF_AVAILABLE = True
except ImportError:
    SF_AVAILABLE = False

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

# ------------------------------------------------------------
# Identity constants (stamped into every record)
# ------------------------------------------------------------
SCANNER_VERSION = "noise_profile_scanner_core_v1"  # code version → record.scanner_version
# 1.1: talent_id changed from null stub to reserved empty string, and language_code
# added — both per the cross-pipeline identity convention (level_check_output_contract
# §2.1). Measurement logic is unchanged, so SCANNER_VERSION holds; the bump is on the
# record SHAPE, which is what a downstream JSONL consumer keys off.
SCHEMA_VERSION = "1.1"                              # output-contract version → record.schema_version

# Central log default. macOS-sensible, configurable per the contract.
DEFAULT_LOG_PATH = Path.home() / "Library" / "Application Support" / "MAD" / "noise_profile_scans.jsonl"

# Output WAV naming. The trimmed profile carries a DISTINCT `_np` suffix so it
# never collides with the raw `_noise`/`_blank` dead-air input — re-runs overwrite
# the prior profile, not the take. RT60 Pass-1 auto-detect knows `_np` and prefers
# it over a raw recording (rt60_analyzer_gui.py).
_OUTPUT_SUFFIX = "_np"
# Raw dead-air recording suffixes (input side). Used only by _strip_noise_suffix to
# trim a raw suffix off the stem before appending _OUTPUT_SUFFIX; the output `_np` is
# deliberately OUTSIDE this set so input and output never share a suffix.
_NOISE_PROFILE_SUFFIXES = ("-blank", "_blank", "-noise", "_noise")

# Level-check phrase file naming. The level-check test runs before the noise scan in
# the 3-step suite and drops its take in the same folder; the scanner reads its
# A-weighted RMS as the speech level for SNR. These are placeholders for this build —
# confirm/replace once the level-check tool defines its own output suffix. Must NOT
# overlap _NOISE_PROFILE_SUFFIXES (contract hard rule: only the noise WAV is suffixed
# `_noise`, so RT60's Pass-1 never grabs the phrase file).
_TEST_PHRASE_SUFFIXES = ("-level", "_level", "-phrase", "_phrase")

_VALID_MODALITIES = ("audition", "session")

# Production WAV spec (matches intake's format validation).
_SPEC_SAMPLE_RATE = 48000
_SPEC_BIT_DEPTH = "24"
_SPEC_CHANNELS = 1

# ------------------------------------------------------------
# Config — embedded Default mirrors noise_profile_scanner_default.yaml.
# load_config() merges an external yaml over this (intake's merge pattern).
# ------------------------------------------------------------
DEFAULT_CONFIG = {
    "profile_name": "Default",
    "config_version": "1.1-provisional",
    # A-weighted SNR bands (dB). SNR = speech_level_dba - noise_floor_dba.
    "snr": {
        "reject_below": 38.0,
        "warn_below": 48.0,
        "caution_below": 55.0,
        "pass_above": 55.0,
    },
    # Binary gate: only REJECT fails. WARN/CAUTION/PASS all pass with a quality label.
    # Every token here resolves to disposition REJECT (verdict FAIL).
    "gate": {
        "fail_on": [
            "snr_reject",
            "format_fail",
            "short_recording",
            "insufficient_clean_noise",
        ],
    },
    # Noise recording validity. Target 5 s; 3 s hard minimum. min_valid_noise_sec is
    # CONTINUOUS (unbroken run), not aggregate; the floor is measured on that run.
    "recording": {
        "min_duration_sec": 3.0,
        "target_duration_sec": 5.0,
        "min_valid_noise_sec": 3.0,
    },
    # Impulse detection: A-weighted short-hop transient excess vs a local median
    # baseline. An event whose peak excess exceeds reject_excess_db is disqualifying
    # and excised; minor events ride along. reject_excess_db PROVISIONAL (calibrated
    # by ear against a held-out listening set).
    "impulse": {
        "detect_hop_ms": 5,
        "baseline_sec": 0.5,
        "merge_ms": 10,
        "reject_excess_db": 11.5,
        "guard_ms": 30,
    },
    "display": {
        "primary_metric": "noise_floor_dba",
        "emphasize_on_fail": "noise_floor_dba",
        "noise_floor_guidance_target_dba": -72.0,
        "speech_level_reference_dba": -34.0,
    },
}


def load_config(config_path: Path = None) -> dict:
    """Load a YAML config and merge over DEFAULT_CONFIG. Missing keys fall back
    to defaults. Returns a complete, resolved config dict. (intake pattern.)"""
    config = _deep_copy_dict(DEFAULT_CONFIG)
    if config_path is None:
        return config

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        user_config = yaml.safe_load(f)

    if not isinstance(user_config, dict):
        raise ValueError(f"Config file must contain a YAML mapping, got {type(user_config).__name__}")

    _merge_config(config, user_config)
    return config


def _deep_copy_dict(d: dict) -> dict:
    """Simple deep copy for nested dicts/lists of primitives."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _deep_copy_dict(v)
        elif isinstance(v, list):
            out[k] = v[:]
        else:
            out[k] = v
    return out


def _merge_config(base: dict, override: dict):
    """Recursively merge override into base, modifying base in place."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _merge_config(base[k], v)
        else:
            base[k] = v


def config_sha256(config: dict) -> str:
    """SHA-256 over the resolved config dict. Computed on a JSON-safe, key-sorted
    serialization so an unversioned edit is still detectable (contract decision 2)."""
    serialized = json.dumps(_json_safe(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# ------------------------------------------------------------
# A-weighting (IEC 61672) — copied verbatim from intake_scanner_core_v7
# so the two tools produce identical A-weighted numbers.
# ------------------------------------------------------------
def a_weight(data: np.ndarray, sr: int) -> np.ndarray:
    """Apply IEC 61672 A-weighting. Analog prototype → digital SOS via bilinear
    transform at the given sample rate."""
    from scipy.signal import zpk2sos, bilinear_zpk, sosfilt

    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12194.217

    z_analog = np.array([0, 0, 0, 0])
    p_analog = np.array([
        -2 * np.pi * f1,
        -2 * np.pi * f1,
        -2 * np.pi * f2,
        -2 * np.pi * f3,
        -2 * np.pi * f4,
        -2 * np.pi * f4,
    ])

    # Normalize so 1 kHz = 0 dB.
    num_1k = (2 * np.pi * 1000) ** 4
    denom_1k = 1.0
    for p in p_analog:
        denom_1k *= abs(1j * 2 * np.pi * 1000 - p)
    k_analog = denom_1k / num_1k

    z_dig, p_dig, k_dig = bilinear_zpk(z_analog, p_analog, k_analog, fs=sr)
    sos = zpk2sos(z_dig, p_dig, k_dig)
    return sosfilt(sos, data).astype(np.float32)


# ------------------------------------------------------------
# Audio I/O — copied from intake_scanner_core_v7
# ------------------------------------------------------------
def get_audio_info(file_path: Path) -> dict:
    if SF_AVAILABLE:
        try:
            info = sf.info(str(file_path))
            bit_depth = info.subtype.split("_")[-1] if "_" in info.subtype else "N/A"
            return {
                "channels": info.channels,
                "bit_depth": bit_depth,
                "sample_rate": info.samplerate,
                "duration": info.duration or 0.0,
            }
        except Exception:
            pass

    if PYDUB_AVAILABLE:
        try:
            audio = AudioSegment.from_file(str(file_path))
            return {
                "channels": audio.channels,
                "bit_depth": audio.sample_width * 8,
                "sample_rate": audio.frame_rate,
                "duration": len(audio) / 1000.0,
            }
        except Exception:
            pass

    return {"channels": "?", "bit_depth": "?", "sample_rate": "?", "duration": 0}


def load_audio(file_path: Path) -> tuple[np.ndarray, int]:
    errors = []

    if SF_AVAILABLE:
        try:
            data, sr = sf.read(str(file_path), dtype="float32")
            if data.ndim > 1:
                data = data[:, 0]
            return data, sr
        except Exception as e:
            errors.append(f"soundfile: {e}")

    if PYDUB_AVAILABLE:
        try:
            audio = AudioSegment.from_file(str(file_path)).set_channels(1)
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            if audio.sample_width == 2:
                samples /= 32768.0
            elif audio.sample_width == 3:
                samples /= 8388608.0
            elif audio.sample_width == 4:
                samples /= 2147483648.0
            return samples, audio.frame_rate
        except Exception as e:
            errors.append(f"pydub: {e}")

    if errors:
        raise RuntimeError(f"All audio backends failed: {'; '.join(errors)}")
    raise RuntimeError("No audio backend available (soundfile or pydub)")


# ------------------------------------------------------------
# Measurement
# ------------------------------------------------------------
def measure_aweighted_rms_dba(data: np.ndarray, sr: int) -> float | None:
    """Full-file A-weighted RMS in dBFS(A). The whole file is treated as the
    measurement (no silence gating) — matches intake's reference-profile measure.
    Returns None if unmeasurable (empty), -120.0 for digital silence."""
    if data is None or len(data) == 0:
        return None
    weighted = a_weight(data, sr)
    rms = np.sqrt(np.mean(weighted ** 2))
    if rms < 1e-10:
        return -120.0
    return float(20 * np.log10(rms))


def measure_broadband_rms_dbfs(data: np.ndarray) -> float | None:
    """Full-file unweighted RMS in dBFS. Spec bridge only (ACX / voiceover); does
    not gate. None if unmeasurable, -120.0 for digital silence."""
    if data is None or len(data) == 0:
        return None
    rms = np.sqrt(np.mean(data ** 2))
    if rms < 1e-10:
        return -120.0
    return float(20 * np.log10(rms))


# ------------------------------------------------------------
# Impulse detection + valid-noise measurement
#
# A single calibrated transient detector flags disqualifying impulses (sharp clicks
# AND sustained events both show up — both spike the short-window envelope):
#   - A-weight the signal (discounts LF rumble the way the floor metric does, so
#     HVAC/rumble is not mistaken for a transient).
#   - Short-hop (impulse.detect_hop_ms, default 5 ms) A-weighted RMS envelope in dB.
#   - Local baseline = running median over impulse.baseline_sec (~0.5 s).
#     EXCESS = envelope - baseline, so a loud-but-STEADY room reads ~0 (validated:
#     Treated Noisy, a -59.8 dBA room, produced 0 events).
#   - An event whose PEAK excess exceeds impulse.reject_excess_db is disqualifying.
#     Threshold calibrated by ear against a held-out listening set (pass events topped
#     ~+10.8 dB, reject events started ~+13.8 dB; line set at +11.5 dB, PROVISIONAL).
# Minor events below the threshold are left in place — they do not measurably affect
# a denoiser's noise profile. Disqualifying impulses are excised; the take is salvaged
# if a CONTINUOUS clean run of at least recording.min_valid_noise_sec survives, and
# the deliverable is trimmed to that run.
#
# (RT60's measure_noise_profile does whole-file band power with no exclusion, and its
# validate_clap tests the inverse — that a clap IS impulsive — so neither was reusable;
# only the crest-factor concept and corpus numbers informed the calibration.)
# ------------------------------------------------------------
def _transient_excess(data: np.ndarray, sr: int, hop_ms: float, baseline_sec: float):
    """A-weighted short-hop RMS envelope (dB) minus its local running-median baseline.
    Returns (excess_per_hop, hop_samples). Excess isolates transients from the local
    floor, so a steady (even loud) room reads near zero."""
    from scipy.signal import medfilt
    dw = a_weight(data, sr)
    hop = max(int(hop_ms / 1000 * sr), 1)
    n = (len(dw) - hop) // hop + 1 if len(dw) >= hop else 0
    if n <= 0:
        return np.zeros(0), hop
    env = np.empty(n)
    for i in range(n):
        seg = dw[i * hop:i * hop + hop]
        r = np.sqrt(np.mean(seg ** 2))
        env[i] = 20 * np.log10(r) if r > 1e-10 else -120.0
    k = int(baseline_sec / (hop_ms / 1000))
    k += (k + 1) % 2                       # force odd kernel
    k = min(k, n - (n + 1) % 2)            # not larger than the series (odd)
    base = medfilt(env, kernel_size=max(k, 1))
    return env - base, hop


def detect_impulses(data: np.ndarray, sr: int, config: dict) -> list[tuple[int, int]]:
    """Return [(start_sample, end_sample), ...] for transient events whose peak excess
    exceeds impulse.reject_excess_db. Each event is padded by impulse.guard_ms and
    overlapping events are merged. end is exclusive."""
    imp = config.get("impulse", DEFAULT_CONFIG["impulse"])
    hop_ms = imp.get("detect_hop_ms", 5)
    baseline_sec = imp.get("baseline_sec", 0.5)
    thresh = imp.get("reject_excess_db", 11.5)
    merge_hops = max(int(imp.get("merge_ms", 10) / hop_ms), 0)
    guard = int(imp.get("guard_ms", 30) / 1000 * sr)

    excess, hop = _transient_excess(data, sr, hop_ms, baseline_sec)
    if len(excess) == 0:
        return []

    flagged = excess > thresh
    events = []
    i = 0
    while i < len(flagged):
        if not flagged[i]:
            i += 1
            continue
        start = i
        last = i
        j = i
        while j < len(flagged) and (j - last) <= merge_hops:
            if flagged[j]:
                last = j
            j += 1
        s0 = max(0, start * hop - guard)
        e0 = min(len(data), (last + 1) * hop + guard)
        events.append((s0, e0))
        i = last + 1

    # Merge any intervals the guard padding made overlap.
    merged = []
    for s, e in events:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def analyze_noise_windows(data: np.ndarray, sr: int, config: dict) -> dict:
    """Detect disqualifying impulses, then measure the floor on the longest CONTINUOUS
    run of clean (impulse-free) noise.

    "Continuous" is the gate, not aggregate: a take with impulses scattered through it
    can hold plenty of clean noise overall while never having an unbroken stretch, so
    we require an uninterrupted run >= recording.min_valid_noise_sec. The floor is
    measured on that same run, and the deliverable WAV is trimmed to it, so RT60 gets
    one contiguous clean span with the disqualifying impulses excised.

    Returns:
      noise_floor_dba            A-weighted RMS over the longest clean run, or None
      noise_floor_dbfs_broadband unweighted RMS over the longest clean run, or None
      valid_noise_sec            duration of the longest CONTINUOUS clean run
      clean_run_bounds           (start, end) sample indices of that run, or None
      impulse_count              number of disqualifying impulses excised
    """
    out = {
        "noise_floor_dba": None,
        "noise_floor_dbfs_broadband": None,
        "valid_noise_sec": 0.0,
        "clean_run_bounds": None,
        "impulse_count": 0,
    }
    if data is None or len(data) == 0:
        return out

    data_w = a_weight(data, sr)
    impulses = detect_impulses(data, sr, config)
    out["impulse_count"] = len(impulses)

    # Clean runs are the gaps between (merged, sorted) impulse intervals across the take.
    clean_runs = []
    cursor = 0
    for s, e in impulses:
        if s > cursor:
            clean_runs.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < len(data):
        clean_runs.append((cursor, len(data)))
    if not clean_runs:
        return out

    # Longest unbroken clean span. For a clean take this is the whole file, so the
    # floor matches a whole-file (intake) measurement.
    s0, e0 = max(clean_runs, key=lambda r: r[1] - r[0])
    out["valid_noise_sec"] = (e0 - s0) / sr
    out["clean_run_bounds"] = (s0, e0)

    seg_w = data_w[s0:e0]
    rms_w = np.sqrt(np.mean(seg_w ** 2))
    out["noise_floor_dba"] = -120.0 if rms_w < 1e-10 else float(20 * np.log10(rms_w))
    seg_bb = data[s0:e0]
    rms_bb = np.sqrt(np.mean(seg_bb ** 2))
    out["noise_floor_dbfs_broadband"] = -120.0 if rms_bb < 1e-10 else float(20 * np.log10(rms_bb))
    return out


def find_test_phrase(noise_wav: Path, suffixes: tuple = _TEST_PHRASE_SUFFIXES) -> Path | None:
    """Find the level-check phrase WAV in the noise file's folder by suffix.
    Returns the first match (sorted), excluding the noise file itself, or None."""
    folder = noise_wav.parent
    for f in sorted(folder.glob("*.wav")):
        if f == noise_wav:
            continue
        stem = f.stem.lower()
        if any(stem.endswith(s) for s in suffixes):
            return f
    return None


# ------------------------------------------------------------
# Format validation (intake convention) and output stem derivation
# ------------------------------------------------------------
def validate_format(info: dict, file_path: Path) -> list[str]:
    """Return a list of human-readable format issues vs the production spec.
    Empty list means the file conforms. Same checks as intake_scanner_core_v7."""
    issues = []
    if info["channels"] != _SPEC_CHANNELS:
        issues.append(f"{info['channels']}ch — expected Mono")
    if info["sample_rate"] != _SPEC_SAMPLE_RATE:
        issues.append(f"{info['sample_rate']} Hz — expected {_SPEC_SAMPLE_RATE} Hz")
    if str(info["bit_depth"]) != _SPEC_BIT_DEPTH:
        issues.append(f"{info['bit_depth']}-bit — expected {_SPEC_BIT_DEPTH}-bit")
    # Accept lossless WAV or FLAC. FLAC (PCM_24 subtype) decodes bit-identically via
    # soundfile and carries the same sample_rate/bit_depth/channels, so it is a valid
    # profile input; the platform's working copies are FLAC.
    if file_path.suffix.lower() not in (".wav", ".flac"):
        issues.append(f"File is {file_path.suffix} — expected .wav or .flac")
    return issues


def _strip_noise_suffix(stem: str) -> str:
    """Remove a trailing raw noise/blank suffix (case-insensitive) so a raw
    'room_noise.wav' yields 'room_np.wav', not 'room_noise_np.wav'."""
    low = stem.lower()
    for suf in _NOISE_PROFILE_SUFFIXES:
        if low.endswith(suf):
            return stem[: -len(suf)]
    return stem


def derive_output_wav_path(source_wav: Path, output_dir: Path = None) -> Path:
    """Canonical, session-derived output path: <base>_np.wav in output_dir
    (default: the source's folder). The `_np` suffix is distinct from the raw
    `_noise`/`_blank` input, so the raw take ALWAYS survives. Stable across
    rechecks: a re-run overwrites the prior *_np.wav profile (the folder holds at
    most one), never the raw recording (contract: Session rechecks)."""
    out_dir = Path(output_dir) if output_dir else source_wav.parent
    base = _strip_noise_suffix(source_wav.stem)
    return out_dir / f"{base}{_OUTPUT_SUFFIX}.wav"


# ------------------------------------------------------------
# Disposition + gate
# ------------------------------------------------------------
def classify_disposition(snr_db: float, config: dict) -> str:
    """Four-way label from A-weighted SNR. Caller handles the unmeasurable case
    (snr_db None) — this expects a real number."""
    snr = config["snr"]
    if snr_db < snr["reject_below"]:
        return "REJECT"
    if snr_db < snr["warn_below"]:
        return "WARN"
    if snr_db < snr.get("caution_below", snr["warn_below"]):
        return "CAUTION"
    return "PASS"


def _evaluate_gate(snr_db: float | None, format_issues: list[str],
                   raw_duration: float | None, valid_noise_sec: float | None,
                   config: dict) -> tuple[str, str, str | None]:
    """Resolve (disposition, verdict, gate_reason).

    Contract invariant: verdict == "FAIL" iff disposition == "REJECT". Every reason
    here is a yaml gate.fail_on token. Precedence, worst-first: format → too short →
    too few clean-noise frames → SNR. (Missing speech level never reaches this — it
    raises upstream as a suite setup error, not a room REJECT.)"""
    rec = config.get("recording", DEFAULT_CONFIG["recording"])

    if format_issues:
        return "REJECT", "FAIL", "format_fail"
    if raw_duration is None or raw_duration < rec["min_duration_sec"]:
        return "REJECT", "FAIL", "short_recording"
    # valid_noise_sec is the longest CONTINUOUS clean run; too short (or no clean run
    # at all, so snr_db is None) means the take is too impulsive to certify.
    if valid_noise_sec is None or valid_noise_sec < rec["min_valid_noise_sec"] or snr_db is None:
        return "REJECT", "FAIL", "insufficient_clean_noise"

    disposition = classify_disposition(snr_db, config)
    if disposition == "REJECT":
        return "REJECT", "FAIL", "snr_reject"
    return disposition, "PASS", None


# ------------------------------------------------------------
# JSON safety (contract): no NaN/Infinity, paths as strings
# ------------------------------------------------------------
def _json_safe(value):
    """Recursively coerce a value into strict-JSON-safe form:
      - NaN / +Inf / -Inf floats -> None
      - Path -> str
      - numpy scalars -> python scalars
      - dict / list recursed
    Everything else passes through unchanged."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):  # numpy float64/int64/etc.
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def _coerce_bit_depth(bit_depth):
    """Store bit_depth numerically when it is a clean integer (keeps the corpus
    numeric, matching the channels-as-int deviation); otherwise keep the string."""
    try:
        return int(bit_depth)
    except (ValueError, TypeError):
        return bit_depth


# ------------------------------------------------------------
# Record assembly — THE single source-of-truth dict (contract)
# Both artifacts serialize from this; there is no second assembly path.
# ------------------------------------------------------------
def build_record(
    *,
    source_wav: Path,
    output_wav: Path | None,
    modality: str,
    info: dict,
    duration: float | None,
    noise_floor_dba: float | None,
    noise_floor_dbfs_broadband: float | None,
    speech_level_dba: float | None,
    snr_db: float | None,
    disposition: str,
    verdict: str,
    gate_reason: str | None,
    config: dict,
    talent_id: str | None = None,
    language_code: str | None = None,
) -> dict:
    snr_cfg = config["snr"]
    record = {
        "schema_version": SCHEMA_VERSION,
        "scanner_version": SCANNER_VERSION,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "modality": modality,
        "source_wav": str(source_wav),
        "output_wav": str(output_wav) if output_wav is not None else None,
        "sample_rate": info["sample_rate"],
        "bit_depth": _coerce_bit_depth(info["bit_depth"]),
        "channels": info["channels"],
        "duration": float(duration) if duration is not None else None,
        "snr_db": snr_db,
        "noise_floor_dba": noise_floor_dba,
        "noise_floor_dbfs_broadband": noise_floor_dbfs_broadband,
        "speech_level_dba": speech_level_dba,
        "profile": config.get("profile_name", "Default"),
        "thresholds": {
            "name": config.get("profile_name", "Default"),
            "config_version": config.get("config_version"),
            "config_sha256": config_sha256(config),
            "reject_below": snr_cfg["reject_below"],
            "warn_below": snr_cfg["warn_below"],
            "caution_below": snr_cfg["caution_below"],
            "pass_above": snr_cfg["pass_above"],
        },
        "disposition": disposition,
        "verdict": verdict,
        "gate_reason": gate_reason,
        # Identity (cross-pipeline convention, contract §2.1). Both reserved as ""
        # at v1 — stored as a stable str, never None, so consumers never face a
        # None/str union. A caller (the screening pipeline / a future platform)
        # passes the real values; None coerces to "" so unsupplied stays typed.
        "talent_id": talent_id or "",
        "language_code": language_code or "",
        "qa_tags": [],       # reserved; reconciliation tool appends here later
    }
    # Final safety pass: guarantees strict-JSON serialization for both artifacts.
    return _json_safe(record)


# ------------------------------------------------------------
# Artifact writers
# ------------------------------------------------------------
def write_sidecar(record: dict, output_wav: Path) -> Path:
    """Derived artifact 1: pretty-printed sidecar sharing the WAV stem."""
    sidecar_path = output_wav.with_suffix(".json")
    sidecar_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return sidecar_path


def append_log(record: dict, log_path: Path = None) -> Path:
    """Derived artifact 2: append one compact, independently-parseable JSON line.
    Open append, write line + '\\n', flush on close. (Single-operator sequential
    use is safe; concurrent writers would need a lock — flagged in the contract.)"""
    log_path = Path(log_path) if log_path else DEFAULT_LOG_PATH
    log_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, separators=(",", ":"))
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    return log_path


def _write_output_wav(samples: np.ndarray, sr: int, output_wav: Path):
    """Persist the noise-profile deliverable, trimmed to the longest continuous clean
    run, as 48 kHz / 24-bit mono (the spec format, already validated). soundfile is
    required for 24-bit output. The output carries the `_np` suffix, distinct from the
    raw `_noise`/`_blank` input, so it never overwrites the take; a re-run overwrites
    only the prior *_np.wav (the folder keeps exactly one — contract: Session rechecks).

    Trimming is effectively lossless: the source is 24-bit PCM, read as float32 (24-bit
    mantissa) and written back as PCM_24, so in-range samples round-trip exactly."""
    if not SF_AVAILABLE:
        raise RuntimeError("soundfile is required to write the 24-bit output WAV")
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_wav), samples, sr, subtype="PCM_24")


# ------------------------------------------------------------
# Top-level scan
# ------------------------------------------------------------
def scan_noise_profile(
    noise_wav: Path,
    *,
    modality: str = "session",
    speech_level_dba: float = None,
    test_phrase_wav: Path = None,
    config: dict = None,
    output_dir: Path = None,
    log_path: Path = None,
    write_outputs: bool = True,
    talent_id: str = None,
    language_code: str = None,
) -> dict:
    """Scan a dead-air noise WAV (5 s target, 3 s minimum), gate it, and emit the
    contract artifacts.

    The gate metric is A-weighted SNR = speech_level_dba - noise_floor_dba, where the
    noise floor is measured only on the non-impulsive portion of the take. The speech
    level comes from the suite's level-check phrase step, resolved in this order:
      1. speech_level_dba — a float already measured by the level-check step, or
      2. test_phrase_wav — an explicit path; its A-weighted RMS is measured, or
      3. a *-level / *-phrase WAV auto-detected in the noise file's folder
         (the level-check take; a placeholder for this build).
    If none resolves, the level-check step did not run — that is a suite setup error,
    so this RAISES rather than logging a misleading room REJECT.

    NOTE: the yaml's speech_level_reference_dba (-34) is display-only and is
    deliberately NOT used as a gate fallback — the gate requires measured speech.

    Gate (all fail closed to REJECT, worst-first): off-spec format, take shorter than
    recording.min_duration_sec, no continuous clean run >= recording.min_valid_noise_sec,
    then SNR below reject_below.

    modality:
      "session"  -> on PASS: write <stem>_np.wav + sidecar, and append the log.
                    The WAV is TRIMMED to the longest continuous clean run, so RT60
                    only ever sees certified-clean noise (as long as is usable, not a
                    flat 3 s). record["duration"] stays the recorded take length (what
                    the 5 s / 3 s spec validates); the deliverable may be shorter.
      "audition" -> append the log only; no WAV, no sidecar (contract decision 7).
    On FAIL (REJECT) nothing persists but the log line (log-only-on-failure).

    write_outputs=False is a dry-run escape hatch (tests / GUI preview): it returns
    the record with output_wav=null and writes no files at all.

    talent_id / language_code: optional identity (cross-pipeline convention, §2.1).
    Persisted into the record (sidecar + JSONL); unsupplied stays "" so the field
    type is always str. Existing callers that omit them are unaffected.

    Returns the single source-of-truth dict (strict-JSON-safe).
    """
    noise_wav = Path(noise_wav)
    if modality not in _VALID_MODALITIES:
        raise ValueError(f"modality must be one of {_VALID_MODALITIES}, got {modality!r}")
    if config is None:
        config = _deep_copy_dict(DEFAULT_CONFIG)

    info = get_audio_info(noise_wav)
    format_issues = validate_format(info, noise_wav)

    # Measure only if the format is trustworthy. Off-spec input (e.g. one channel of
    # a stereo file, wrong rate) yields misleading numbers, so we skip measurement
    # and let the gate fail closed on format.
    noise_floor_dba = None
    noise_floor_dbfs_broadband = None
    valid_noise_sec = None
    clean_run_bounds = None
    data = sr = None
    duration = info.get("duration")
    if not format_issues:
        data, sr = load_audio(noise_wav)
        duration = len(data) / sr if sr else info.get("duration")
        windows = analyze_noise_windows(data, sr, config)
        noise_floor_dba = windows["noise_floor_dba"]
        noise_floor_dbfs_broadband = windows["noise_floor_dbfs_broadband"]
        valid_noise_sec = windows["valid_noise_sec"]
        clean_run_bounds = windows["clean_run_bounds"]

    # Resolve speech level: explicit float > explicit path > folder auto-detect.
    if speech_level_dba is None:
        if test_phrase_wav is None:
            test_phrase_wav = find_test_phrase(noise_wav)
        if test_phrase_wav is not None:
            tp_data, tp_sr = load_audio(Path(test_phrase_wav))
            speech_level_dba = measure_aweighted_rms_dba(tp_data, tp_sr)
    if speech_level_dba is None:
        raise ValueError(
            "No speech level available: the level-check phrase step must run before the "
            "noise scan. Pass speech_level_dba, pass test_phrase_wav, or place a "
            "*-level / *-phrase WAV in the noise file's folder."
        )

    # SNR is the gate metric. None if the floor was unmeasurable (off-spec or no
    # clean frames) — the gate then fails on format / insufficient_clean_noise.
    if noise_floor_dba is not None:
        snr_db = float(speech_level_dba - noise_floor_dba)
    else:
        snr_db = None

    disposition, verdict, gate_reason = _evaluate_gate(
        snr_db, format_issues, duration, valid_noise_sec, config
    )

    # Decide persistence before assembling the record so output_wav reflects reality.
    output_wav_path = derive_output_wav_path(noise_wav, output_dir)
    should_persist = write_outputs and verdict == "PASS" and modality == "session"

    persisted_wav = None
    if should_persist:
        # A PASS guarantees a qualifying clean run, so clean_run_bounds is set.
        s0, e0 = clean_run_bounds
        _write_output_wav(data[s0:e0], sr, output_wav_path)
        persisted_wav = output_wav_path

    record = build_record(
        source_wav=noise_wav,
        output_wav=persisted_wav,
        modality=modality,
        info=info,
        duration=duration,
        noise_floor_dba=noise_floor_dba,
        noise_floor_dbfs_broadband=noise_floor_dbfs_broadband,
        speech_level_dba=speech_level_dba,
        snr_db=snr_db,
        disposition=disposition,
        verdict=verdict,
        gate_reason=gate_reason,
        config=config,
        talent_id=talent_id,
        language_code=language_code,
    )

    if write_outputs:
        # Central log captures every scan, both modalities, pass or fail.
        append_log(record, log_path)
        # Sidecar travels with the WAV — session modality, passing scans only.
        if should_persist:
            write_sidecar(record, output_wav_path)

    return record
