# intake_scanner_core_v7.py
# v7.0 changes:
# - Content-based noise profile fallback in find_noise_profiles():
#   when no named -blank/-noise file exists in a folder, scan remaining
#   WAVs by crest factor and pick the lowest-CF candidate at or below
#   14.65 dB as the noise profile (corpus-calibrated threshold: 1 dB above
#   highest observed noise CF of 13.59 dB, 1.36 dB below lowest speech CF
#   of 16.01 dB). Files at or below -90 dBFS RMS are skipped (silent/empty).
#   find_noise_profiles() now returns a dict of
#   {folder: (path, method)} where method is 'name' or 'content'.
#   Callers that only need the path can call find_noise_profile_path().
#
# v6.1 changes (preserved):
# - Format-mismatch files are skipped up front (no metrics computed)
# - Skipped files listed once in report header, not in per-file body
# - Report header trimmed: Sample threshold line and Noise floor detection
#   line removed (still applied internally, just not displayed)
# - Default profile name cleaned: "Strict (Research-Grade)" -> "Strict"
#
# v6.0 changes (preserved):
# - A-weighted noise floor and SNR (IEC 61672 via scipy.signal SOS filter)
# - Integrated LUFS per file (BS.1770-4 via pyloudnorm)
# - New config section "loudness" with reject/warn thresholds
# - Noise floor report label: dBFS -> dBA
# - Noise profile reference: detect -blank/-noise/_blank/_noise files,
#   measure full-file A-weighted RMS as folder baseline for SNR
# - Fallback to per-file silence detection when no reference profile exists
#
# v5.1 changes (preserved):
# - Relative silence threshold mode for noise floor detection
# - Format validation: flags non-mono, non-48kHz, non-24-bit, non-WAV files
#
# v5.0 changes (preserved):
# - Thresholds loaded from YAML config (strict/forgiving profiles)
# - Three-way sort: pass / salvageable / reject per Phase 0 interface contract
# - Config object threaded through flagging, sorting, and report generation

import os
import sys
import re
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime
from scipy.signal import resample_poly, sosfilt

# ------------------------------------------------------------
# CRITICAL: Make ffmpeg/ffprobe work in frozen app (PyInstaller)
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

try:
    import pyloudnorm as pyln
    PYLN_AVAILABLE = True
except ImportError:
    PYLN_AVAILABLE = False

# ------------------------------------------------------------
# Config – true-peak detection (hardcoded, not profile-dependent)
# ------------------------------------------------------------
SAMPLE_THRESHOLD_DB = -0.2
TRUE_NEAR_DBTP      = -0.5
TRUE_HARD_DBTP      = 0.0
DEBOUNCE_MS         = 30
CONTEXT_MS          = 5
MAX_BURST_MS        = 500
SAFE_SAMPLE_THRESHOLD = 0.89125

# Measurement bias compensation — see v4 comments for rationale.
TP_CLIP_BIAS_DB = -0.10

_TRUE_NEAR_LINEAR = 10 ** (TRUE_NEAR_DBTP / 20)
_TRUE_HARD_LINEAR = 10 ** ((TRUE_HARD_DBTP + TP_CLIP_BIAS_DB) / 20)

# Sentinel for unavailable noise floor
NOISE_FLOOR_UNAVAILABLE = float('nan')

# ------------------------------------------------------------
# YAML config loading
# ------------------------------------------------------------

# Default thresholds (strict profile)
# SNR thresholds calibrated against known-quality test corpus (April 2026)
DEFAULT_CONFIG = {
    "profile_name": "Strict",
    "snr": {
        "reject_below": 45.0,
        "warn_below": 53.0,
        "caution_below": 60.0,
        "pass_above": 60.0,
    },
    "crest_factor": {
        "warn_low": 10.0,
        "caution_low": 13.0,
        "caution_high": 999.0,
        "warn_high": 999.0,
    },
    "noise_floor": {
        "silence_threshold_mode": "relative",
        "silence_threshold_db": -40.0,
        "silence_threshold_offset": -20.0,
        "min_segment_ms": 50,
    },
    "loudness": {
        "reject_below": -42.0,
        "caution_below": -36.0,
    },
    "sort_rules": {
        "reject_on": ["snr_reject", "tp_clip", "format_fail", "lufs_reject"],
        "salvageable_on": ["snr_warn", "snr_caution", "cf_warn", "lufs_caution"],
    },
}


def load_config(config_path: Path = None) -> dict:
    """Load a YAML config file and merge with defaults.
    Missing keys fall back to DEFAULT_CONFIG values.
    Returns a complete config dict."""
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

    # Merge top-level scalars and nested dicts
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


def get_builtin_config_dir() -> Path:
    """Return the path to the bundled configs/ directory.
    Works both in development and in a PyInstaller frozen app."""
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS) / "configs"
    return Path(__file__).parent / "configs"


def list_builtin_profiles() -> list[Path]:
    """Return paths to all .yaml files in the builtin configs directory."""
    config_dir = get_builtin_config_dir()
    if not config_dir.exists():
        return []
    return sorted(config_dir.glob("*.yaml"))


# ------------------------------------------------------------
# A-weighting (IEC 61672)
# ------------------------------------------------------------
def a_weight(data: np.ndarray, sr: int) -> np.ndarray:
    """Apply IEC 61672 A-weighting filter to audio data.
    Uses a pre-computed analog prototype converted to digital SOS form
    via bilinear transform at the given sample rate."""
    from scipy.signal import zpk2sos, bilinear_zpk

    # IEC 61672 A-weighting analog prototype poles and zeros
    # Zeros: 4 at s=0 (two double zeros)
    # Poles from the standard filter frequencies
    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12194.217

    # Analog zeros and poles (rad/s)
    z_analog = np.array([0, 0, 0, 0])
    p_analog = np.array([
        -2 * np.pi * f1,
        -2 * np.pi * f1,
        -2 * np.pi * f2,
        -2 * np.pi * f3,
        -2 * np.pi * f4,
        -2 * np.pi * f4,
    ])

    # Gain: normalize so that 1 kHz = 0 dB
    # The A-weighting transfer function magnitude at 1 kHz
    # |H(j*2*pi*1000)| should equal 1 (0 dB)
    # Compute the analog gain constant
    num_1k = (2 * np.pi * 1000) ** 4  # from 4 zeros at origin
    denom_1k = 1.0
    for p in p_analog:
        denom_1k *= abs(1j * 2 * np.pi * 1000 - p)
    k_analog = denom_1k / num_1k

    # Convert to digital via bilinear transform
    z_dig, p_dig, k_dig = bilinear_zpk(z_analog, p_analog, k_analog, fs=sr)

    # Convert to second-order sections for numerical stability
    sos = zpk2sos(z_dig, p_dig, k_dig)

    return sosfilt(sos, data).astype(np.float32)


# ------------------------------------------------------------
# Noise profile detection and reference measurement
# ------------------------------------------------------------
_NOISE_PROFILE_SUFFIXES = ("-blank", "_blank", "-noise", "_noise")

def is_noise_profile(file_path: Path) -> bool:
    """Check if a file is a noise profile based on naming convention.
    Matches filenames ending in -blank, _blank, -noise, or _noise (before extension)."""
    stem = file_path.stem.lower()
    return any(stem.endswith(s) for s in _NOISE_PROFILE_SUFFIXES)

# Crest factor ceiling for content-based noise profile detection.
# Calibrated against the intake + RT60 test corpus (May 2026):
#   highest noise profile CF observed: 13.59 dB
#   lowest speech file CF observed:    16.01 dB
# 14.65 dB sits 1.06 dB above the noise ceiling and 1.36 dB below the speech floor.
_NOISE_CONTENT_CF_MAX_DB = 14.65

# Files at or below this RMS are silent/empty and skipped during content detection.
_NOISE_CONTENT_RMS_MIN_DBFS = -90.0


def find_noise_profiles(folder: Path, extensions: set = None) -> dict[Path, tuple[Path, str]]:
    """Find noise profile files grouped by parent folder, with content fallback.

    Pass 1 (name): any file whose stem ends with -blank/_blank/-noise/_noise.
    Pass 2 (content): if Pass 1 finds nothing in a folder, scan remaining WAVs
        by crest factor and pick the lowest-CF candidate at or below
        _NOISE_CONTENT_CF_MAX_DB (14.65 dB). Files at or below
        _NOISE_CONTENT_RMS_MIN_DBFS (-90 dBFS) are skipped as silent/empty.

    Scans recursively. Returns {folder_path: (noise_profile_path, method)}
    where method is 'name' or 'content'.
    If multiple named profiles exist in one folder, uses the first match.
    """
    if extensions is None:
        extensions = {".wav", ".m4a", ".mp3", ".aiff", ".flac", ".aac"}

    # Collect all candidate files grouped by folder
    by_folder: dict[Path, list[Path]] = {}
    for f in sorted(folder.rglob("*")):
        if f.is_file() and f.suffix.lower() in extensions:
            by_folder.setdefault(f.parent, []).append(f)

    profiles: dict[Path, tuple[Path, str]] = {}

    for parent, candidates in by_folder.items():
        # Pass 1: naming convention
        for f in candidates:
            if is_noise_profile(f):
                profiles[parent] = (f, "name")
                break

        if parent in profiles:
            continue

        # Pass 2: content scan — lowest CF at or below threshold
        # Only WAV files are considered (non-WAV blanks are unusual and
        # crest factor is less reliable across codecs at low levels).
        best_path: Path | None = None
        best_cf = float("inf")
        for f in candidates:
            if f.suffix.lower() != ".wav":
                continue
            try:
                data, _ = load_audio(f)
                cf_db, rms_db = compute_crest_factor(data)
                if rms_db <= _NOISE_CONTENT_RMS_MIN_DBFS:
                    continue  # silent / empty file
                if cf_db <= _NOISE_CONTENT_CF_MAX_DB and cf_db < best_cf:
                    best_cf = cf_db
                    best_path = f
            except Exception:
                continue

        if best_path is not None:
            profiles[parent] = (best_path, "content")

    return profiles


def find_noise_profile_path(folder: Path, extensions: set = None) -> dict[Path, Path]:
    """Convenience wrapper — returns {folder_path: noise_profile_path} without method info.
    Equivalent to the v6 find_noise_profiles() signature for callers that don't need method."""
    return {k: v for k, (v, _) in find_noise_profiles(folder, extensions).items()}

def measure_reference_noise_floor(file_path: Path) -> float:
    """Measure A-weighted RMS of an entire noise profile file.
    The whole file is treated as noise — no silence gating needed.
    Returns noise floor in dBA."""
    data, sr = load_audio(file_path)
    if len(data) == 0:
        return NOISE_FLOOR_UNAVAILABLE
    data_weighted = a_weight(data, sr)
    rms = np.sqrt(np.mean(data_weighted ** 2))
    if rms < 1e-10:
        return -120.0
    return 20 * np.log10(rms)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def sec_to_timecode(seconds: float) -> str:
    """Convert an input of seconds (float) and return a timecode string 
    with zero padding that follows timecode convention (MM:SS.sss).

    Examples: 10.7 s renders 00:10.700, and 8.95 s renders 00:08.950
    """
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins:02d}:{secs:06.3f}"

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
# Peak detection (unchanged from v3/v4)
# ------------------------------------------------------------
def find_peaks_with_timecodes(data: np.ndarray, sr: int,
                              bias_db: float = TP_CLIP_BIAS_DB) -> list[str]:
    sample_threshold = 10 ** (SAMPLE_THRESHOLD_DB / 20)
    debounce_samples = int(DEBOUNCE_MS / 1000 * sr)
    context_samples  = max(int(CONTEXT_MS / 1000 * sr), 32)
    max_burst_samples = int(MAX_BURST_MS / 1000 * sr)

    true_near_linear = _TRUE_NEAR_LINEAR
    true_hard_linear = 10 ** ((TRUE_HARD_DBTP + bias_db) / 20)

    results = []
    candidate_idx = np.where(np.abs(data) >= sample_threshold)[0]

    i = 0
    while i < len(candidate_idx):
        burst_start = candidate_idx[i]
        j = i + 1
        while j < len(candidate_idx) and candidate_idx[j] - candidate_idx[j-1] <= debounce_samples:
            j += 1
        burst_end = candidate_idx[j-1]

        segment = data[burst_start:burst_end+1]
        max_pos_rel = np.argmax(np.abs(segment))
        max_pos = burst_start + max_pos_rel
        raw_peak = np.abs(segment[max_pos_rel])

        if raw_peak >= SAFE_SAMPLE_THRESHOLD:
            burst_len = burst_end - burst_start + 1
            if burst_len <= max_burst_samples:
                s = max(0, burst_start - context_samples)
                e = min(len(data), burst_end + 1 + context_samples)
            else:
                half = max_burst_samples // 2
                s = max(0, max_pos - half - context_samples)
                e = min(len(data), max_pos + half + context_samples)

            upsampled = resample_poly(data[s:e], 4, 1)

            trim = context_samples * 4
            center = upsampled[trim : len(upsampled) - trim]
            if len(center) == 0:
                center = upsampled

            local_peak = np.max(np.abs(center))

            peak_idx_us = int(np.argmax(np.abs(center)))
            t = (s + context_samples + peak_idx_us / 4) / sr

            dbtp = 20 * np.log10(local_peak + 1e-10)
            if local_peak >= true_hard_linear:
                results.append(f"{sec_to_timecode(t)} → TP CLIP ({dbtp:+.2f} dBTP)")
            elif local_peak >= true_near_linear:
                results.append(f"{sec_to_timecode(t)} → near-clip ({dbtp:+.2f} dBTP)")

        elif raw_peak >= true_near_linear:
            dbtp = 20 * np.log10(raw_peak + 1e-10)
            t = max_pos / sr
            results.append(f"{sec_to_timecode(t)} → near-clip ({dbtp:+.2f} dBTP)")

        i = j

    return results if results else ["CLEAN"]

# ------------------------------------------------------------
# Noise floor estimation
# v6: accepts optional data_weighted for A-weighted RMS measurement
# Gating (silence detection) always uses unweighted data.
# RMS measurement uses data_weighted when provided.
# ------------------------------------------------------------
def estimate_noise_floor(data: np.ndarray, sr: int, config: dict = None,
                         data_weighted: np.ndarray = None) -> float:
    """Returns noise floor in dBA (when A-weighted data provided) or dBFS, or NaN if no silent segments found."""
    if config is None:
        config = DEFAULT_CONFIG

    nf_cfg = config["noise_floor"]
    mode = nf_cfg.get("silence_threshold_mode", "absolute")

    # Gating always uses unweighted data
    if mode == "relative":
        signal_rms = np.sqrt(np.mean(data ** 2))
        if signal_rms < 1e-10:
            return NOISE_FLOOR_UNAVAILABLE
        rms_db = 20 * np.log10(signal_rms)
        offset = nf_cfg.get("silence_threshold_offset", -20.0)
        thresh_db = rms_db + offset
        thresh_linear = 10 ** (thresh_db / 20)
    else:
        thresh_linear = 10 ** (nf_cfg["silence_threshold_db"] / 20)

    min_samples   = int(nf_cfg["min_segment_ms"] / 1000 * sr)

    abs_data  = np.abs(data)
    is_silent = abs_data < thresh_linear

    padded  = np.concatenate(([False], is_silent, [False]))
    changes = np.diff(padded.astype(np.int8))
    starts  = np.where(changes == 1)[0]
    ends    = np.where(changes == -1)[0]

    noise_segments = []
    # Pick which data to measure RMS on: A-weighted if provided, else unweighted
    measure_data = data_weighted if data_weighted is not None else data
    for s, e in zip(starts, ends):
        if (e - s) >= min_samples:
            noise_segments.append(measure_data[s:e])

    if not noise_segments:
        return NOISE_FLOOR_UNAVAILABLE

    noise_samples = np.concatenate(noise_segments)
    rms = np.sqrt(np.mean(noise_samples ** 2))

    if rms < 1e-10:
        return -120.0

    return 20 * np.log10(rms)

# ------------------------------------------------------------
# Crest factor and signal RMS (unchanged — stays unweighted)
# ------------------------------------------------------------
def compute_crest_factor(data: np.ndarray) -> tuple[float, float]:
    """Returns (crest_factor_db, rms_dbfs)."""
    rms = np.sqrt(np.mean(data ** 2))

    if rms < 1e-10:
        return 0.0, -120.0

    rms_db  = 20 * np.log10(rms)
    peak    = np.max(np.abs(data))
    peak_db = 20 * np.log10(peak + 1e-10)

    return peak_db - rms_db, rms_db

# ------------------------------------------------------------
# Config-driven flagging
# ------------------------------------------------------------
def _cf_flag(cf_db: float, config: dict) -> str:
    """Return flag label for crest factor using config thresholds."""
    if np.isnan(cf_db):
        return ""
    cf = config["crest_factor"]
    if cf_db <= cf["warn_low"] or cf_db >= cf["warn_high"]:
        return "[WARN]"
    if cf_db <= cf["caution_low"] or cf_db >= cf["caution_high"]:
        return "[CAUTION]"
    return "[OK]"

# SNR below this threshold is treated as a noise profile file (no flags)
NOISE_PROFILE_SNR_THRESHOLD = 1.0

def _snr_flag(snr_db: float, config: dict) -> str:
    """Return flag label for SNR using config thresholds.
    Four tiers: REJECT / WARN / CAUTION / OK.
    SNR near zero is a noise profile — skip flags.
    If caution_below equals warn_below, CAUTION tier is effectively skipped."""
    if np.isnan(snr_db):
        return "[N/A]"
    if snr_db < NOISE_PROFILE_SNR_THRESHOLD:
        return "[NOISE PROFILE]"
    snr = config["snr"]
    if snr_db < snr["reject_below"]:
        return "[REJECT]"
    if snr_db < snr["warn_below"]:
        return "[WARN]"
    if snr_db < snr.get("caution_below", snr["warn_below"]):
        return "[CAUTION]"
    return "[OK]"

def _lufs_flag(lufs: float, config: dict) -> str:
    """Return flag label for integrated LUFS using config thresholds.
    Two tiers only: reject and caution. No warn tier.
    Noise profile files should skip this — caller handles that."""
    if np.isnan(lufs) or lufs == float('-inf'):
        return "[N/A]"
    loud = config.get("loudness", DEFAULT_CONFIG["loudness"])
    if lufs < loud["reject_below"]:
        return "[REJECT]"
    if lufs < loud.get("caution_below", loud.get("warn_below", -36.0)):
        return "[CAUTION]"
    return "[OK]"

def _cf_note(cf_db: float, config: dict) -> str:
    """Brief description for out-of-spec CF. Empty if OK."""
    if np.isnan(cf_db):
        return ""
    cf = config["crest_factor"]
    if cf_db <= cf["warn_low"]:
        return "severely over-compressed — re-record"
    if cf_db <= cf["caution_low"]:
        return "over-compressed — peaks will land below spec after normalization"
    if cf_db >= cf["warn_high"]:
        return "severely dynamic — normalization will clip"
    if cf_db >= cf["caution_high"]:
        return "too dynamic — peaks will exceed spec after normalization"
    return ""

def _snr_note(snr_db: float, config: dict) -> str:
    """Brief description for out-of-spec SNR. Empty if OK."""
    if np.isnan(snr_db):
        return "no silent segments detected"
    if snr_db < NOISE_PROFILE_SNR_THRESHOLD:
        return "noise profile file — no flags applied"
    snr = config["snr"]
    if snr_db < snr["reject_below"]:
        return "noise floor too high — unsalvageable"
    if snr_db < snr["warn_below"]:
        return "noise likely audible after normalization — possibly salvageable with manual editing"
    if snr_db < snr.get("caution_below", snr["warn_below"]):
        return "noise may be faintly audible — candidate for automated de-noise"
    return ""

def _lufs_note(lufs: float, config: dict) -> str:
    """Brief description for out-of-spec LUFS. Empty if OK."""
    if np.isnan(lufs) or lufs == float('-inf'):
        return "measurement unavailable"
    loud = config.get("loudness", DEFAULT_CONFIG["loudness"])
    if lufs < loud["reject_below"]:
        return "speech level too low — unsalvageable without heavy amplification"
    if lufs < loud.get("caution_below", loud.get("warn_below", -36.0)):
        return "speech level low — may need gain adjustment"
    return ""

# ------------------------------------------------------------
# Three-way sort (Phase 0 interface contract)
# Disposition is worst-flag-wins across all metrics.
# ------------------------------------------------------------
def classify_disposition(result: dict, config: dict) -> str:
    """Classify a scan result as 'reject', 'salvageable', 'pass', or 'reference'.
    Returns the disposition string."""
    if result.get("error"):
        return "reject"

    # Noise profile files get special disposition — not subject to normal flagging
    if result.get("is_noise_profile"):
        return "reference"

    sort_rules = config["sort_rules"]
    flags_present = set()

    # Check format issues
    if result.get("format_issues"):
        flags_present.add("format_fail")

    # Check TP clip
    if result.get("has_clip"):
        flags_present.add("tp_clip")

    # Check SNR flags (four tiers: reject / warn / caution / ok)
    # SNR only drives disposition when measured against a reference noise profile.
    # Per-file silence detection is unreliable for disposition (signal-dependent drift)
    # so it is reported as informational only.
    snr_db = result.get("snr_db", float('nan'))
    nf_source = result.get("noise_floor_source", "unavailable")
    if not np.isnan(snr_db) and nf_source == "reference":
        snr_cfg = config["snr"]
        if snr_db < snr_cfg["reject_below"]:
            flags_present.add("snr_reject")
        elif snr_db < snr_cfg["warn_below"]:
            flags_present.add("snr_warn")
        elif snr_db < snr_cfg.get("caution_below", snr_cfg["warn_below"]):
            flags_present.add("snr_caution")

    # Check CF flags
    cf_db = result.get("crest_factor_db", float('nan'))
    if not np.isnan(cf_db):
        cf_cfg = config["crest_factor"]
        if cf_db <= cf_cfg["warn_low"] or cf_db >= cf_cfg["warn_high"]:
            flags_present.add("cf_warn")

    # Check LUFS flags (two tiers: reject and caution)
    lufs = result.get("lufs", float('nan'))
    if not np.isnan(lufs) and lufs != float('-inf'):
        loud_cfg = config.get("loudness", DEFAULT_CONFIG["loudness"])
        if lufs < loud_cfg["reject_below"]:
            flags_present.add("lufs_reject")
        elif lufs < loud_cfg.get("caution_below", loud_cfg.get("warn_below", -36.0)):
            flags_present.add("lufs_caution")

    # Apply sort rules: reject takes priority, then salvageable
    for trigger in sort_rules.get("reject_on", []):
        if trigger in flags_present:
            return "reject"

    for trigger in sort_rules.get("salvageable_on", []):
        if trigger in flags_present:
            return "salvageable"

    return "pass"

# ------------------------------------------------------------
# Peak outlier detection
# Files whose peak is >5 dB below the folder's max peak
# ------------------------------------------------------------
PEAK_OUTLIER_THRESHOLD_DB = 5.0

def flag_peak_outliers(results: list[dict]):
    """Flag files whose highest peak is more than 5 dB below
    the max peak among files in the same parent folder.
    Mutates results in place, adding 'peak_outlier' and 'peak_delta_db' keys."""
    from collections import defaultdict

    # Group by parent folder
    by_folder = defaultdict(list)
    for r in results:
        if not r.get("error"):
            parent = Path(r["path"]).parent
            by_folder[parent].append(r)

    for _, folder_results in by_folder.items():
        peaks = [r["global_dbtp"] for r in folder_results if r["global_dbtp"] != float('-inf')]
        if not peaks:
            continue
        max_peak = max(peaks)

        for r in folder_results:
            if r["global_dbtp"] == float('-inf'):
                continue
            # Skip noise profile files — they're expected to be quiet
            if r.get("is_noise_profile"):
                continue
            delta = max_peak - r["global_dbtp"]
            if delta > PEAK_OUTLIER_THRESHOLD_DB:
                r["peak_outlier"] = True
                r["peak_delta_db"] = delta

# ------------------------------------------------------------
# Severity tier for report ordering
# ------------------------------------------------------------
def _severity_tier(r: dict, config: dict) -> int:
    """Sort tier: 0=reject, 1=salvageable, 2=pass, 3=reference, 4=error (bottom)."""
    disp = r.get("disposition", "pass")
    if disp == "reject":
        return 0
    if disp == "salvageable":
        return 1
    if disp == "pass":
        return 2
    if disp == "reference":
        return 3
    return 4

# ------------------------------------------------------------
# Main scan
# ------------------------------------------------------------
def scan_file(file_path: Path, root_dir: Path = None,
              bias_db: float = TP_CLIP_BIAS_DB,
              config: dict = None,
              reference_noise_floor_db: float = None) -> dict:
    """Scan a single audio file.
    reference_noise_floor_db: when provided, used as the noise floor for SNR
        instead of per-file silence detection. Typically from a folder's
        noise profile (-blank/-noise) file."""
    if config is None:
        config = DEFAULT_CONFIG

    info = get_audio_info(file_path)
    rel_path = file_path.relative_to(root_dir) if root_dir else file_path

    # Format validation up front — mismatched files are skipped entirely.
    # Metrics on a file that needs re-export are wasted work and misleading
    # (e.g. SNR on just the left channel of a stereo file).
    format_issues = []
    if info["channels"] != 1:
        format_issues.append(f"Stereo ({info['channels']}ch) — expected Mono")
    if info["sample_rate"] != 48000:
        format_issues.append(f"{info['sample_rate']} Hz — expected 48000 Hz")
    if str(info["bit_depth"]) != "24":
        format_issues.append(f"{info['bit_depth']}-bit — expected 24-bit")
    if file_path.suffix.lower() != ".wav":
        format_issues.append(f"File is {file_path.suffix} — expected .wav")

    if format_issues:
        ch = info["channels"]
        if ch == 1:
            channels_str = "Mono"
        elif isinstance(ch, int):
            channels_str = f"Stereo ({ch}ch)" if ch == 2 else f"{ch}-channel"
        else:
            channels_str = str(ch)
        return {
            "path": file_path,
            "rel_path": rel_path,
            "duration": info.get("duration", 0) or 0,
            "sample_rate": info["sample_rate"],
            "global_dbtp": float('-inf'),
            "events": ["SKIPPED"],
            "has_clip": False,
            "worst_dbtp": -99.0,
            "channels": channels_str,
            "bit_depth": info["bit_depth"],
            "noise_floor_db": float('nan'),
            "noise_floor_source": "unavailable",
            "crest_factor_db": float('nan'),
            "rms_db": float('nan'),
            "snr_db": float('nan'),
            "lufs": float('nan'),
            "is_noise_profile": False,
            "format_issues": format_issues,
            "skipped": True,
            "disposition": "reject",
        }

    data, sr = load_audio(file_path)

    if len(data) == 0:
        raise RuntimeError("Audio file is empty")

    _is_noise_profile = is_noise_profile(file_path)

    duration = len(data) / sr
    global_peak = np.max(np.abs(data))
    global_dbtp = 20 * np.log10(global_peak + 1e-10) if global_peak > 0 else float('-inf')
    events = find_peaks_with_timecodes(data, sr, bias_db=bias_db)
    has_clip = any("TP CLIP" in e for e in events)

    worst_dbtp = -99.0
    for event in events:
        m = re.search(r'([-+]\d+\.\d+)\s+dBTP', event)
        if m:
            worst_dbtp = max(worst_dbtp, float(m.group(1)))

    # A-weighted data for noise floor and SNR
    data_weighted = a_weight(data, sr)

    # Noise floor determination
    if _is_noise_profile:
        # Noise profile: full-file A-weighted RMS (entire file is noise)
        nf_rms = np.sqrt(np.mean(data_weighted ** 2))
        noise_floor_db = 20 * np.log10(nf_rms) if nf_rms > 1e-10 else -120.0
        noise_floor_source = "full-file"
    elif reference_noise_floor_db is not None:
        # Use the folder's reference noise profile as baseline
        noise_floor_db = reference_noise_floor_db
        noise_floor_source = "reference"
    else:
        # Fallback: per-file silence detection
        noise_floor_db = estimate_noise_floor(data, sr, config=config, data_weighted=data_weighted)
        noise_floor_source = "per-file" if not np.isnan(noise_floor_db) else "unavailable"

    # Crest factor stays unweighted (standard practice)
    crest_factor_db, rms_db = compute_crest_factor(data)

    # SNR computation
    if _is_noise_profile:
        # Noise profile: SNR is meaningless (all noise)
        snr_db = 0.0
    else:
        # A-weighted signal RMS vs noise floor
        signal_rms_weighted = np.sqrt(np.mean(data_weighted ** 2))
        if signal_rms_weighted < 1e-10:
            rms_weighted_db = -120.0
        else:
            rms_weighted_db = 20 * np.log10(signal_rms_weighted)

        if np.isnan(noise_floor_db):
            snr_db = NOISE_FLOOR_UNAVAILABLE
        else:
            snr_db = rms_weighted_db - noise_floor_db

    # Integrated LUFS — skip for noise profiles (not speech)
    lufs = float('nan')
    if not _is_noise_profile and PYLN_AVAILABLE:
        try:
            meter = pyln.Meter(sr)
            # pyloudnorm expects (samples, channels) for mono — reshape to 2-D
            lufs = meter.integrated_loudness(data.reshape(-1, 1))
        except Exception:
            lufs = float('nan')

    channels_str = "Mono" if info["channels"] == 1 else f"Stereo ({info['channels']}ch)"

    result = {
        "path": file_path,
        "rel_path": rel_path,
        "duration": duration,
        "sample_rate": sr,
        "global_dbtp": global_dbtp,
        "events": events,
        "has_clip": has_clip,
        "worst_dbtp": worst_dbtp,
        "channels": channels_str,
        "bit_depth": info["bit_depth"],
        "noise_floor_db": noise_floor_db,
        "noise_floor_source": noise_floor_source,
        "crest_factor_db": crest_factor_db,
        "rms_db": rms_db,
        "snr_db": snr_db,
        "lufs": lufs,
        "is_noise_profile": _is_noise_profile,
        "format_issues": [],
    }

    # Classification (noise profiles get "reference" disposition)
    result["disposition"] = classify_disposition(result, config)

    return result

# ------------------------------------------------------------
# Formatting helpers
# ------------------------------------------------------------
def _fmt_db(value: float, suffix: str = "dB") -> str:
    if np.isnan(value):
        return "N/A"
    return f"{value:+.1f} {suffix}"

_DISPOSITION_LABELS = {
    "reject": "REJECT",
    "salvageable": "SALVAGEABLE",
    "pass": "PASS",
    "reference": "NOISE PROFILE",
}

def _build_snr_threshold_line(config: dict) -> str:
    """Build the SNR thresholds line for the report header.
    Shows caution tier only when it differs from warn (i.e., four-tier system)."""
    snr = config["snr"]
    caution = snr.get("caution_below", snr["warn_below"])
    parts = [f"SNR thresholds: reject < {snr['reject_below']} dB"]
    parts.append(f"warn < {snr['warn_below']} dB")
    if caution != snr["warn_below"]:
        parts.append(f"caution < {caution} dB")
    parts.append(f"pass ≥ {snr['pass_above']} dB")
    return " | ".join(parts)

def _build_lufs_threshold_line(config: dict) -> str:
    """Build the LUFS thresholds line for the report header."""
    loud = config.get("loudness", DEFAULT_CONFIG["loudness"])
    caution = loud.get("caution_below", loud.get("warn_below", -36.0))
    return (f"LUFS thresholds: reject < {loud['reject_below']} LUFS | "
            f"caution < {caution} LUFS")

# ------------------------------------------------------------
# Report generation
# ------------------------------------------------------------
def build_report_lines(results: list[dict],
                       bias_db: float = TP_CLIP_BIAS_DB,
                       config: dict = None) -> list[str]:
    """Generate report lines. Config drives threshold display and flagging."""
    if config is None:
        config = DEFAULT_CONFIG

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = len(results)
    clips = sum(1 for r in results if r["has_clip"])
    profile = config.get("profile_name", "Custom")

    lines = [
        f"INTAKE SCAN v7.0 — {total} files",
        f"Profile: {profile}",
        f"Generated: {timestamp}",
        f"Noise floor / SNR: A-weighted (IEC 61672)",
        _build_snr_threshold_line(config),
        _build_lufs_threshold_line(config),
        # Only the lower bounds (over-compression) are displayed. Upper-bound
        # CF thresholds default to 999 (disabled) so they aren't flagged.
        f"CF thresholds: warn ≤ {config['crest_factor']['warn_low']} dB | "
        f"caution ≤ {config['crest_factor']['caution_low']} dB",
    ]

    # Sort summary
    valid = [r for r in results if not r.get("error")]
    content = [r for r in valid if not r.get("is_noise_profile")]  # exclude noise profiles from counts
    n_pass = sum(1 for r in content if r.get("disposition") == "pass")
    n_salv = sum(1 for r in content if r.get("disposition") == "salvageable")
    n_skip = sum(1 for r in content if r.get("skipped"))
    # Reject count excludes skipped — skipped has its own bucket
    n_rej  = sum(1 for r in content if r.get("disposition") == "reject" and not r.get("skipped"))
    n_ref  = sum(1 for r in valid if r.get("is_noise_profile"))
    n_err  = sum(1 for r in results if r.get("error"))

    disp_parts = f"Disposition: {n_pass} pass / {n_salv} salvageable / {n_rej} reject"
    if n_skip:
        disp_parts += f" / {n_skip} skipped (format)"
    if n_ref:
        disp_parts += f" / {n_ref} noise profile"
    if n_err:
        disp_parts += f" / {n_err} error"
    lines.append(disp_parts)
    lines.append(f"{clips} files contain TP CLIPs")

    # Skipped files (format mismatch) — quick reference list at the top
    # so these files are visible without hunting through the body.
    skipped_files = [r for r in results if r.get("skipped")]
    if skipped_files:
        lines.append(
            f"Skipped (format mismatch) — re-export required: {len(skipped_files)} file(s)"
        )
        for r in skipped_files:
            lines.append(f"  {r['rel_path']} — {'; '.join(r.get('format_issues', []))}")

    # Aggregate stats (exclude noise profiles from content metrics)
    nf_values = [r["noise_floor_db"] for r in content if not np.isnan(r.get("noise_floor_db", float('nan')))]
    cf_values = [r["crest_factor_db"] for r in content if "crest_factor_db" in r]
    snr_values = [r["snr_db"] for r in content if not np.isnan(r.get("snr_db", float('nan')))]
    lufs_values = [r["lufs"] for r in content
                   if not np.isnan(r.get("lufs", float('nan')))
                   and r.get("lufs", float('-inf')) != float('-inf')]

    if nf_values:
        lines.append(
            f"Noise floor across files: "
            f"worst {max(nf_values):+.1f} dBA / "
            f"best {min(nf_values):+.1f} dBA / "
            f"median {np.median(nf_values):+.1f} dBA"
        )
    if snr_values:
        lines.append(
            f"SNR across files: "
            f"worst {min(snr_values):+.1f} dB / "
            f"best {max(snr_values):+.1f} dB / "
            f"median {np.median(snr_values):+.1f} dB"
        )
    if lufs_values:
        lines.append(
            f"LUFS across files: "
            f"lowest {min(lufs_values):+.1f} LUFS / "
            f"highest {max(lufs_values):+.1f} LUFS / "
            f"median {np.median(lufs_values):+.1f} LUFS"
        )
    if cf_values:
        lines.append(
            f"Crest factor across files: "
            f"min {min(cf_values):.1f} dB / "
            f"max {max(cf_values):.1f} dB / "
            f"median {np.median(cf_values):.1f} dB"
        )

    # Flag counts (content files only)
    cf_warns    = sum(1 for r in content if _cf_flag(r.get("crest_factor_db", float('nan')), config) == "[WARN]")
    cf_cautions = sum(1 for r in content if _cf_flag(r.get("crest_factor_db", float('nan')), config) == "[CAUTION]")
    # SNR flags only count reference-based measurements (per-file is informational only)
    ref_content = [r for r in content if r.get("noise_floor_source") == "reference"]
    snr_rejects  = sum(1 for r in ref_content if _snr_flag(r.get("snr_db", float('nan')), config) == "[REJECT]")
    snr_warns    = sum(1 for r in ref_content if _snr_flag(r.get("snr_db", float('nan')), config) == "[WARN]")
    snr_cautions = sum(1 for r in ref_content if _snr_flag(r.get("snr_db", float('nan')), config) == "[CAUTION]")

    lufs_rejects  = sum(1 for r in content if _lufs_flag(r.get("lufs", float('nan')), config) == "[REJECT]")
    lufs_cautions = sum(1 for r in content if _lufs_flag(r.get("lufs", float('nan')), config) == "[CAUTION]")

    if cf_warns or cf_cautions:
        lines.append(f"CF flags: {cf_warns} WARN / {cf_cautions} CAUTION")
    if snr_rejects or snr_warns or snr_cautions:
        lines.append(f"SNR flags: {snr_rejects} REJECT / {snr_warns} WARN / {snr_cautions} CAUTION")
    if lufs_rejects or lufs_cautions:
        lines.append(f"LUFS flags: {lufs_rejects} REJECT / {lufs_cautions} CAUTION")

    # Count files without a reference profile (per-file or unavailable noise floor)
    n_no_ref = sum(1 for r in content if r.get("noise_floor_source") in ("per-file", "unavailable"))
    if n_no_ref and n_ref == 0:
        lines.append(f"WARNING: No noise profiles found — SNR is informational only, not used for disposition")
        lines.append(f"  Add a -blank or -noise file per folder to enable SNR-based sorting")
    elif n_no_ref:
        lines.append(f"{n_no_ref} file(s) without folder noise profile — SNR informational only for those files")

    if n_ref:
        # Show the reference noise floor value for each folder
        ref_results = [r for r in valid if r.get("is_noise_profile")]
        for rr in ref_results:
            nf = rr.get("noise_floor_db", float('nan'))
            nf_str = f"{nf:+.1f} dBA" if not np.isnan(nf) else "N/A"
            lines.append(f"Noise profile: {rr['rel_path']} — baseline {nf_str}")

    # Peak divergence summary — show highest/lowest files and outlier count
    peak_content = [r for r in content
                    if r.get("global_dbtp", float('-inf')) != float('-inf')]
    n_peak_outliers = sum(1 for r in content if r.get("peak_outlier"))
    if peak_content:
        highest = max(peak_content, key=lambda r: r["global_dbtp"])
        lowest = min(peak_content, key=lambda r: r["global_dbtp"])
        spread = highest["global_dbtp"] - lowest["global_dbtp"]
        lines.append(
            f"Peak range: {highest['global_dbtp']:+.1f} dBTP ({Path(highest['rel_path']).name}) "
            f"to {lowest['global_dbtp']:+.1f} dBTP ({Path(lowest['rel_path']).name}) "
            f"— spread {spread:.1f} dB"
        )
        if n_peak_outliers:
            lines.append(
                f"Peak outliers (>{PEAK_OUTLIER_THRESHOLD_DB} dB below folder max): "
                f"{n_peak_outliers}"
            )

    lines.append("=" * 80)
    lines.append("")

    # Sort by disposition (reject first), then by worst peak within tier
    sorted_results = sorted(
        results,
        key=lambda x: (_severity_tier(x, config), -x.get("worst_dbtp", -99.0))
    )

    for r in sorted_results:
        # Skipped files are listed once in the header summary — no body entry.
        if r.get("skipped"):
            continue

        peak_str = f"{r['global_dbtp']:+.2f} dBFS" if r['global_dbtp'] != float('-inf') else "N/A"
        disp = _DISPOSITION_LABELS.get(r.get("disposition", "pass"), "PASS")
        _is_np = r.get("is_noise_profile", False)

        lines.append(f"FILE: {r['path']}")
        lines.append(f"Disposition: {disp}")
        fmt_issues = r.get("format_issues", [])
        if fmt_issues:
            lines.append(f"Format: {r['bit_depth']}-bit | {r['channels']} | {r['sample_rate']} Hz  [REJECT] — {'; '.join(fmt_issues)}")
        else:
            lines.append(f"Format: {r['bit_depth']}-bit | {r['channels']} | {r['sample_rate']} Hz")
        lines.append(f"Duration: {sec_to_timecode(r['duration'])}")
        lines.append(f"Highest measured peak: {peak_str}")

        nf  = r.get("noise_floor_db", float('nan'))
        cf  = r.get("crest_factor_db", float('nan'))
        rms = r.get("rms_db", float('nan'))
        snr = r.get("snr_db", float('nan'))
        lufs = r.get("lufs", float('nan'))
        nf_source = r.get("noise_floor_source", "unavailable")

        if _is_np:
            # Noise profile file — compact display
            nf_str = f"{nf:+.1f} dBA" if not np.isnan(nf) else "N/A"
            lines.append(f"RMS:          {_fmt_db(rms, 'dBFS')}")
            lines.append(f"Noise floor:  {nf_str} (full-file reference measurement)")
            if r["events"] != ["CLEAN"]:
                lines.append("TRUE PEAK ✗")
                lines.extend(f"   {e}" for e in r["events"])
            else:
                lines.append("TRUE PEAK ✓ CLEAN")
            lines.append(f">>> {disp}")
        else:
            # Content file — full metrics
            cf_flag  = _cf_flag(cf, config)
            cf_note  = _cf_note(cf, config)
            cf_suffix  = f"  {cf_flag} — {cf_note}"  if cf_note  else ""

            # SNR flag/note: only show severity flags when reference-based.
            # Per-file SNR is informational only (unreliable for disposition).
            if nf_source == "reference":
                snr_flag = _snr_flag(snr, config)
                snr_note = _snr_note(snr, config)
                snr_suffix = f"  {snr_flag} — {snr_note}" if snr_note else ""
            elif not np.isnan(snr):
                snr_suffix = "  (informational — no noise profile)"
            else:
                snr_suffix = ""

            lufs_f = _lufs_flag(lufs, config)
            lufs_n = _lufs_note(lufs, config)
            lufs_suffix = f"  {lufs_f} — {lufs_n}" if lufs_n else ""

            # Noise floor source annotation
            if nf_source == "reference":
                nf_label = "dBA (ref)"
            elif nf_source == "per-file":
                nf_label = "dBA (per-file)"
            else:
                nf_label = "dBA"

            lines.append(f"RMS:          {_fmt_db(rms, 'dBFS')}")
            lines.append(f"Crest factor: {_fmt_db(cf)}{cf_suffix}")
            lines.append(f"Noise floor:  {_fmt_db(nf, nf_label)}")
            lines.append(f"SNR:          {_fmt_db(snr)}{snr_suffix}")

            # LUFS line
            if np.isnan(lufs) or lufs == float('-inf'):
                lufs_str = "N/A"
            else:
                lufs_str = f"{lufs:+.1f} LUFS"
            lines.append(f"LUFS:         {lufs_str}{lufs_suffix}")

            if r.get("peak_outlier"):
                delta = r.get("peak_delta_db", 0)
                lines.append(f"PEAK:         [CAUTION] — {delta:.1f} dB below folder max peak")

            if r["events"] != ["CLEAN"]:
                lines.append("TRUE PEAK ✗")
                lines.extend(f"   {e}" for e in r["events"])
            else:
                lines.append("TRUE PEAK ✓ CLEAN")
            lines.append(f">>> {disp}" + (f" — TP CLIP" if r['has_clip'] else ""))

        lines.append("-" * 80)
        lines.append("")

    return lines

def write_text_report(results: list[dict], output_path: Path,
                      bias_db: float = TP_CLIP_BIAS_DB,
                      config: dict = None):
    if config is None:
        config = DEFAULT_CONFIG
    lines = build_report_lines(results, bias_db=bias_db, config=config)
    output_path.write_text("\n".join(lines), encoding="utf-8")
