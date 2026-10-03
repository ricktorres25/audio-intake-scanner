# level_check_phrase_core.py
#
# Core measurement + evaluation logic for the level check phrase test (third gate
# in the talent environment screening suite). Pure logic, no __main__ — importable
# by a GUI wrapper and by REAPER's Lua-Python bridge, matching the noise scanner /
# intake scanner pattern.

import json
import math
import hashlib
import numpy as np
import pyloudnorm as pyln
from pathlib import Path
from datetime import datetime, timezone
from scipy.signal import resample_poly

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
SCANNER_VERSION = "level_check_phrase_core_v1"
SCHEMA_VERSION = "1.0"

DEFAULT_LOG_PATH = Path.home() / "Library" / "Application Support" / "MAD" / "level_check_scans.jsonl"

# Embedded default config. Mirrors the noise scanner's DEFAULT_CONFIG /
# load_config pattern (a YAML profile would merge over this); v1 ships with
# the plan's locked threshold numbers only — no external profile loading yet.
DEFAULT_CONFIG = {
    "profile_name": "default",
    "config_version": "1.0",
    "peak": {
        "clip_at_dbtp": 0.0,
        "hot_above_dbfs": -3.0,
    },
    "lufs": {
        "too_quiet_below": -42.0,
        "quiet_below": -36.0,
    },
}


# ------------------------------------------------------------
# A-weighting (IEC 61672) — copied verbatim from noise_profile_scanner_core /
# intake_scanner_core_v7 so all three cores produce identical A-weighted numbers.
# Required here because speech_level_dba must be the SAME metric as the noise
# scanner's (locked handoff contract — see level_check_output_contract.md §3):
# full-file A-weighted RMS, no silence gating, so the two are directly comparable
# for SNR without the noise scanner re-deriving it from a different filter.
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


def measure_aweighted_rms_dba(data: np.ndarray, sr: int) -> float | None:
    """Full-file A-weighted RMS in dBFS(A), no silence gating — identical metric
    to the noise scanner's speech_level_dba so the two are directly comparable
    for SNR (locked handoff contract). None if unmeasurable (empty), -120.0 for
    digital silence. (Copied verbatim from noise_profile_scanner_core.)"""
    if data is None or len(data) == 0:
        return None
    weighted = a_weight(data, sr)
    rms = np.sqrt(np.mean(weighted ** 2))
    if rms < 1e-10:
        return -120.0
    return float(20 * np.log10(rms))


# ------------------------------------------------------------
# Measurement
# ------------------------------------------------------------
def measure_sample_peak(data: np.ndarray) -> float | None:
    """Max absolute sample value in dBFS. None if unmeasurable (empty),
    -120.0 for digital silence."""
    if data is None or len(data) == 0:
        return None
    peak = np.max(np.abs(data))
    if peak < 1e-10:
        return -120.0
    return float(20 * np.log10(peak))


def measure_true_peak(data: np.ndarray, sr: int) -> float | None:
    """4x-oversampled true peak in dBTP (ITU-R BS.1770-4), via scipy's
    resample_poly — same oversampling call as intake_scanner_core_v7.py's
    find_peaks_with_timecodes (line 452), applied here across the whole file
    rather than windowed segments around candidate transients (this core has
    no peak-detection step; the phrase WAV is short enough to oversample whole).
    None if unmeasurable (empty), -120.0 for digital silence."""
    if data is None or len(data) == 0:
        return None
    peak = np.max(np.abs(data))
    if peak < 1e-10:
        return -120.0
    upsampled = resample_poly(data, 4, 1)
    true_peak = np.max(np.abs(upsampled))
    if true_peak < 1e-10:
        return -120.0
    return float(20 * np.log10(true_peak))


def measure_integrated_lufs(data: np.ndarray, sr: int) -> float | None:
    """Integrated loudness (LUFS-I, ITU-R BS.1770-4) via pyloudnorm's Meter.
    None if unmeasurable (empty); pyloudnorm returns -inf for digital silence,
    which is normalized to -120.0 to match the suite's silence convention."""
    if data is None or len(data) == 0:
        return None
    meter = pyln.Meter(sr)
    lufs = meter.integrated_loudness(data)
    if not np.isfinite(lufs):
        return -120.0
    return float(lufs)


def measure_crest_factor(data: np.ndarray) -> float | None:
    """Peak/RMS in dB (20 * log10(peak / rms)). Informational at v1, not a gate
    criterion. None if unmeasurable (empty), 0.0 for digital silence (peak and
    rms both collapse to the floor — matches intake's compute_crest_factor)."""
    if data is None or len(data) == 0:
        return None
    rms = np.sqrt(np.mean(data ** 2))
    if rms < 1e-10:
        return 0.0
    peak = np.max(np.abs(data))
    return float(20 * np.log10(peak / rms))


# ------------------------------------------------------------
# Evaluation — locked threshold model (PROJECT_PLAN.md)
#
# Peak dimension: dBTP gates clip first (it dominates regardless of dBFS), then
# dBFS splits Hot vs Pass. LUFS dimension: three bands, half-open like the noise
# scanner's _lufs_flag / _snr_flag (the boundary value belongs to the stricter
# adjacent tier — e.g. exactly -36 LUFS is Quiet, not Pass; exactly -42 is Quiet,
# not Too quiet). Defaults below are the plan's locked v1 numbers; evaluate_*
# read from config so a profile can override them later.
# ------------------------------------------------------------
_DEFAULT_THRESHOLDS = {
    "peak": {
        "clip_at_dbtp": 0.0,
        "hot_above_dbfs": -3.0,
    },
    "lufs": {
        "too_quiet_below": -42.0,
        "quiet_below": -36.0,
    },
}


def evaluate_peak(sample_peak_dbfs: float, true_peak_dbtp: float, config: dict | None = None) -> str:
    """Peak dimension disposition: 'CLIP', 'HOT', or 'PASS'.

    Clip is checked first — dBTP >= 0.0 fails regardless of dBFS. Otherwise Hot
    if dBFS exceeds the hot threshold, else Pass. Boundary values fall to the
    stricter neighbor: exactly -3 dBFS is Pass, not Hot; exactly 0.0 dBTP is Clip."""
    cfg = (config or {}).get("peak", {})
    clip_at = cfg.get("clip_at_dbtp", _DEFAULT_THRESHOLDS["peak"]["clip_at_dbtp"])
    hot_above = cfg.get("hot_above_dbfs", _DEFAULT_THRESHOLDS["peak"]["hot_above_dbfs"])

    if true_peak_dbtp >= clip_at:
        return "CLIP"
    if sample_peak_dbfs > hot_above:
        return "HOT"
    return "PASS"


def evaluate_lufs(integrated_lufs: float, config: dict | None = None) -> str:
    """LUFS dimension disposition: 'TOO_QUIET', 'QUIET', or 'PASS'.

    Boundary values fall to the stricter neighbor: exactly -42 LUFS is Quiet
    (not Too quiet), exactly -36 LUFS is Quiet (not Pass) — matches the plan's
    '-42 to -36 = Quiet' band reading as closed, with strict '<' / '>' guarding
    the Too-quiet and Pass edges."""
    cfg = (config or {}).get("lufs", {})
    too_quiet_below = cfg.get("too_quiet_below", _DEFAULT_THRESHOLDS["lufs"]["too_quiet_below"])
    quiet_below = cfg.get("quiet_below", _DEFAULT_THRESHOLDS["lufs"]["quiet_below"])

    if integrated_lufs < too_quiet_below:
        return "TOO_QUIET"
    if integrated_lufs <= quiet_below:
        return "QUIET"
    return "PASS"


def combine_verdict(peak_disposition: str, lufs_disposition: str) -> tuple[str, list[str]]:
    """Combine the independent peak and LUFS dispositions into (verdict, fail_reasons).

    PASS only when both dimensions are clean (PASS/PASS, with HOT or QUIET flags
    riding along as Pass-with-flag per the plan's combined table). Any FAIL on
    either axis is a hard fail; HOT + QUIET is also a hard fail in its own right
    (impulsive peak content masking a too-quiet speech level — named explicitly
    per the plan's requirement, not folded into the generic per-axis reasons)."""
    fail_reasons = []

    if peak_disposition == "CLIP":
        fail_reasons.append("Peak: clipping detected (true peak at or above 0.0 dBTP)")
    if lufs_disposition == "TOO_QUIET":
        fail_reasons.append("Loudness: integrated level too quiet (below -42 LUFS)")

    if peak_disposition == "HOT" and lufs_disposition == "QUIET":
        fail_reasons.append(
            "Hot peak with quiet overall level — likely extreme plosive or intermittent "
            "noise driving the peak while speech level remains too low to use"
        )

    if fail_reasons:
        return "FAIL", fail_reasons
    return "PASS", []


# ------------------------------------------------------------
# Audio I/O — copied from noise_profile_scanner_core / intake_scanner_core_v7
# (each core in the suite is self-contained; no shared infra module exists).
# ------------------------------------------------------------
def get_audio_info(file_path: Path) -> dict:
    """Format facts (channels, bit depth, sample rate, duration) via soundfile,
    falling back to pydub. '?' placeholders if neither backend can read the file."""
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
    """Load audio as float32 mono via soundfile, falling back to pydub.
    Multi-channel input is collapsed to its first channel."""
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
# JSON safety + record assembly helpers — mirror noise_profile_scanner_core
# verbatim so the suite's two JSONL logs stay byte-for-byte consistent in style.
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
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def _coerce_bit_depth(bit_depth):
    """Store bit_depth numerically when it is a clean integer; otherwise keep
    the string (matches noise scanner's _coerce_bit_depth)."""
    try:
        return int(bit_depth)
    except (ValueError, TypeError):
        return bit_depth


def config_sha256(config: dict) -> str:
    """SHA-256 over the resolved config dict, JSON-safe + key-sorted, so an
    unversioned edit is still detectable (matches noise scanner's config_sha256)."""
    serialized = json.dumps(_json_safe(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# ------------------------------------------------------------
# Record assembly — THE single source-of-truth dict (output contract v1.0).
# Both artifacts (sidecar + JSONL line) serialize from this; no second path.
# ------------------------------------------------------------
def build_record(
    *,
    source_wav: Path,
    info: dict,
    duration: float | None,
    sample_peak_dbfs: float | None,
    true_peak_dbtp: float | None,
    integrated_lufs: float | None,
    crest_factor_db: float | None,
    speech_level_dba: float | None,
    peak_disposition: str,
    lufs_disposition: str,
    verdict: str,
    fail_reasons: list[str],
    config: dict,
    talent_id: str | None = None,
    language_code: str | None = None,
) -> dict:
    peak_cfg = config["peak"]
    lufs_cfg = config["lufs"]
    record = {
        "schema_version": SCHEMA_VERSION,
        "scanner_version": SCANNER_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_wav": str(source_wav),
        "sample_rate": info["sample_rate"],
        "bit_depth": _coerce_bit_depth(info["bit_depth"]),
        "channels": info["channels"],
        "duration": float(duration) if duration is not None else None,
        "sample_peak_dbfs": sample_peak_dbfs,
        "true_peak_dbtp": true_peak_dbtp,
        "integrated_lufs": integrated_lufs,
        "crest_factor_db": crest_factor_db,
        "speech_level_dba": speech_level_dba,
        "peak_disposition": peak_disposition,
        "lufs_disposition": lufs_disposition,
        "verdict": verdict,
        "fail_reasons": fail_reasons,
        "profile": config.get("profile_name", "default"),
        "thresholds": {
            "name": config.get("profile_name", "default"),
            "config_version": config.get("config_version"),
            "config_sha256": config_sha256(config),
            "clip_at_dbtp": peak_cfg["clip_at_dbtp"],
            "hot_above_dbfs": peak_cfg["hot_above_dbfs"],
            "too_quiet_below_lufs": lufs_cfg["too_quiet_below"],
            "quiet_below_lufs": lufs_cfg["quiet_below"],
        },
        # Identity (cross-pipeline convention, §2.1). Reserved as "" when a caller
        # omits them; a caller (screening pipeline / future platform) passes the
        # real values. None coerces to "" so the field type is always str.
        "talent_id": talent_id or "",
        "language_code": language_code or "",
    }
    # Final safety pass: guarantees strict-JSON serialization for both artifacts.
    return _json_safe(record)


# ------------------------------------------------------------
# Artifact writers
# ------------------------------------------------------------
def write_sidecar(record: dict, phrase_wav: Path) -> Path:
    """Derived artifact 1: pretty-printed sidecar sharing the phrase WAV's stem."""
    sidecar_path = Path(phrase_wav).with_suffix(".json")
    sidecar_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return sidecar_path


def append_log(record: dict, log_path: Path = None) -> Path:
    """Derived artifact 2: append one compact, independently-parseable JSON line.
    Open append, write line + '\\n', flush on close. (Single-operator sequential
    use is safe; concurrent writers would need a lock.)"""
    log_path = Path(log_path) if log_path else DEFAULT_LOG_PATH
    log_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, separators=(",", ":"))
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    return log_path


# ------------------------------------------------------------
# Top-level scan
# ------------------------------------------------------------
def scan_phrase(
    phrase_wav: Path,
    *,
    config: dict = None,
    log_path: Path = None,
    write_outputs: bool = True,
    talent_id: str = None,
    language_code: str = None,
) -> dict:
    """Analyze a level-check phrase WAV and emit the contract artifacts.

    Measures sample peak, true peak (4x oversampled), integrated LUFS, crest
    factor, and speech_level_dba (A-weighted full-file RMS — handed off to the
    noise scanner's SNR step per the locked speech_level_dba contract). Evaluates
    peak and LUFS independently against the locked threshold model and combines
    them into a single verdict.

    Always writes the sidecar JSON (next to the WAV, same stem) and appends to
    the central JSONL log — pass or fail. This differs from the noise scanner's
    log-only-on-failure behavior: the level check is a measurement record for
    every take, not a deliverable gate that only persists on success.

    write_outputs=False is a dry-run escape hatch (tests / GUI preview): returns
    the record without writing any files.

    Does not rename, move, or trim the input WAV — suffix/placement enforcement
    is the caller's (GUI/CLI) responsibility per the plan.

    talent_id / language_code: optional identity (cross-pipeline convention, §2.1),
    persisted into the record. Unsupplied stays "" so the field type is always str;
    existing callers that omit them are unaffected.

    Returns the single source-of-truth dict (strict-JSON-safe).
    """
    phrase_wav = Path(phrase_wav)
    if config is None:
        config = DEFAULT_CONFIG

    info = get_audio_info(phrase_wav)
    data, sr = load_audio(phrase_wav)
    duration = len(data) / sr if sr else info.get("duration")

    sample_peak_dbfs = measure_sample_peak(data)
    true_peak_dbtp = measure_true_peak(data, sr)
    integrated_lufs = measure_integrated_lufs(data, sr)
    crest_factor_db = measure_crest_factor(data)
    speech_level_dba = measure_aweighted_rms_dba(data, sr)

    peak_disposition = evaluate_peak(sample_peak_dbfs, true_peak_dbtp, config)
    lufs_disposition = evaluate_lufs(integrated_lufs, config)
    verdict, fail_reasons = combine_verdict(peak_disposition, lufs_disposition)

    record = build_record(
        source_wav=phrase_wav,
        info=info,
        duration=duration,
        sample_peak_dbfs=sample_peak_dbfs,
        true_peak_dbtp=true_peak_dbtp,
        integrated_lufs=integrated_lufs,
        crest_factor_db=crest_factor_db,
        speech_level_dba=speech_level_dba,
        peak_disposition=peak_disposition,
        lufs_disposition=lufs_disposition,
        verdict=verdict,
        fail_reasons=fail_reasons,
        config=config,
        talent_id=talent_id,
        language_code=language_code,
    )

    if write_outputs:
        append_log(record, log_path)
        write_sidecar(record, phrase_wav)

    return record
