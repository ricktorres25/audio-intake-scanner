# intake_scanner_core.py
# v8.0 changes:
# - Uses vendored copies of the collection platform's two Rec QC cores, under
#   cores/. Importing them runs a start check against cores/VENDORED.json.
#   A-weighting and a reference's whole-file A-weighted RMS now come from the
#   noise core (bit-identical to v7's own copy), and the whole-file true peak
#   from the level core.
# - Clip fix: the clip decision uses the whole-file 4x true peak against
#   0.0 dBTP, or -0.10 with the bias toggle. The windowed finder still supplies
#   the timecodes; a clip it has no event for is listed without a timecode.
# - A noise take found by content scan is treated like a named one. v7 scanned
#   it as content against its own floor and rejected it.
# - The noise core's trimmed output name, _np, counts as a noise-take name.
# - scan_folder(): the folder scan v7 ran inside the GUI's run_scan, moved here
#   so the report rows and the tests run the same path. Display and Finder
#   labels stay in the GUI.
# - INTAKE_REPORT.jsonl beside INTAKE_REPORT.txt: a run record, then one row
#   per file with relative paths, noise-floor method and confidence, policy
#   hashes, and RT60 and drift as n/a with the reason.
# - The text report's FILE: line is relative to the scanned folder, an error
#   line no longer carries the folder's absolute path, and the header states
#   RT60 and drift once as n/a. No other line changes.
# - Platform Band, a third profile: configs/platform/platform_band.yaml points
#   at the cores' own policies and holds no thresholds. scan_folder_platform()
#   runs the platform's two calls per file (write_outputs off), rows carry the
#   records' readout and hashes, and the text report uses platform blocks.
# - FLAC is measured like WAV: 24-bit, mono, 48 kHz, .wav or .flac. FLAC is
#   lossless, the platform stores its takes as FLAC, and the noise core accepts
#   it. The content scan for a noise take considers FLAC too.
# - With no noise take, the per-file floor is the quietest steady stretch of
#   the file (estimate_quiet_floor), not v7's silence gate, whose error grew
#   with how quiet the room was. The number still has low confidence, gives no
#   band and never sorts. Rows name the method per_file_quiet_window
#   (schema 1.1).
# - A noise take of digital silence (whole-file floor at or below -110 dBA) is
#   never a folder's reference, under any profile. It stays a noise take, its
#   row reads confidence none with reason digital_silence, and the report says
#   it is not used. Under Platform Band this differs from the platform's noise
#   core on purpose: an intake folder has no live capture or clap test to catch
#   such a take.
#
# v7.0 changes (preserved):
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
import json
import math
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime
from scipy.signal import resample_poly

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
# Vendored Rec QC cores (v8). Importing the cores package runs the start
# check first: a copy that differs from cores/VENDORED.json raises
# VendoredCoreError here, naming the file, before any vendored code runs.
# ------------------------------------------------------------
from cores import level_check_phrase_core as level_core
from cores import noise_profile_scanner_core as noise_core
from cores import CORES_DIR, load_manifest

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

# TODO(deprecate): remove the operator bias toggle in a future rev — the bias_db
# param (threaded through find_peaks_with_timecodes, scan_file, and the report
# builders) and the GUI "Bias compensation" checkbox / --bias flag are unused in
# practice with no edge-case benefit. TP_CLIP_BIAS_DB stays as the fixed
# compensation; only the operator-facing override goes.
TP_CLIP_BIAS_DB = -0.10

_TRUE_NEAR_LINEAR = 10 ** (TRUE_NEAR_DBTP / 20)
_TRUE_HARD_LINEAR = 10 ** ((TRUE_HARD_DBTP + TP_CLIP_BIAS_DB) / 20)

# Sentinel for unavailable noise floor
NOISE_FLOOR_UNAVAILABLE = float('nan')

# v8: the containers that are measured (with 24-bit, mono, 48 kHz). FLAC is
# lossless, so it carries the same samples a 24-bit WAV would.
MEASURED_EXTENSIONS = (".wav", ".flac")

# ------------------------------------------------------------
# v8: versions, report file names, and the fixed not-applicable fields
#
# ------------------------------------------------------------
SCANNER_VERSION = "intake_scanner_core_v8"
REPORT_VERSION = "8.0"
ROWS_SCHEMA_VERSION = "1.1"   # 1.1: noise_floor_method per_file_quiet_window replaces per_file_silence
REPORT_NAME = "INTAKE_REPORT.txt"
ROWS_NAME = "INTAKE_REPORT.jsonl"

RT60_REASON = ("RT60 needs a clap recording and runs as Rec QC stage 3. "
               "The intake scanner does not measure it.")
DRIFT_REASON = ("Drift compares a re-record against a kept baseline. "
                "An intake batch has neither.")
# The text report's one header line for both (worded so the sibling harnesses'
# "baseline <number> dBA" pattern cannot match it).
RT60_DRIFT_HEADER_LINE = ("RT60 and drift: n/a — not measured here (RT60 needs a clap "
                          "recording; drift needs a re-record and a kept take)")

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

    # v8: a pointer profile carries no thresholds and is not merged over these.
    if user_config.get("readout") == "platform":
        return load_platform_profile(config_path, user_config)

    # Merge top-level scalars and nested dicts
    _merge_config(config, user_config)
    return config


def load_platform_profile(config_path: Path, pointer: dict) -> dict:
    """Resolve a pointer profile. The noise core loads its own
    policy file through its own load_config, the call the platform makes at
    boot; the level core keeps its built-in policy, as the platform passes no
    config. A missing policy file raises: the scan stops, never falls back.

    Returns the profile the Platform Band path reads. It holds no scanner
    thresholds, only the cores' resolved policies. (The per-file estimate takes
    no settings from the profile; see estimate_quiet_floor.)"""
    config_path = Path(config_path)
    cores_section = pointer.get("cores") or {}
    noise_rel = cores_section.get("noise_profile_config")
    if not noise_rel:
        raise ValueError(f"{config_path.name}: cores.noise_profile_config is missing")
    if cores_section.get("level_check_config") is not None:
        raise ValueError(f"{config_path.name}: the level core takes no policy file yet, "
                         "so level_check_config must be null")
    noise_path = CORES_DIR.parent / noise_rel     # relative to the scanner folder
    if not noise_path.is_file():
        raise FileNotFoundError(
            f"Platform policy file not found: {noise_rel} (named by {config_path.name}). "
            "The scan stops here; it does not fall back to another policy.")
    return {
        "profile_name": pointer.get("profile_name", "Platform Band"),
        "readout": "platform",
        "noise_profile_config": noise_rel,
        "noise_policy": noise_core.load_config(noise_path),
        "level_policy": level_core.DEFAULT_CONFIG,
    }


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
    """Return paths to all .yaml files in the builtin configs directory.
    v8: also configs/platform/, where pointer profiles sit out of v7's sight."""
    config_dir = get_builtin_config_dir()
    if not config_dir.exists():
        return []
    return sorted(config_dir.glob("*.yaml")) + sorted((config_dir / "platform").glob("*.yaml"))


# ------------------------------------------------------------
# A-weighting (IEC 61672) — v8 uses the vendored noise core's filter.
# The scanner's own copy was bit-identical to it and is
# removed, so the scanner and Rec QC share one A-weighting.
# ------------------------------------------------------------
a_weight = noise_core.a_weight


# ------------------------------------------------------------
# Noise profile detection and reference measurement
# ------------------------------------------------------------
# v8: "_np" is the noise core's name for a noise take trimmed to its clean run,
# which the platform stores.
_NOISE_PROFILE_SUFFIXES = ("-blank", "_blank", "-noise", "_noise", "_np")

def is_noise_profile(file_path: Path) -> bool:
    """Check if a file is a noise profile based on naming convention.
    Matches filenames ending in -blank, _blank, -noise, _noise or _np (before extension)."""
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

# v8: a noise take this quiet (whole-file A-weighted floor, dBA) is digital silence,
# not a room, and is never a folder's reference, under any profile. The same
# line estimate_quiet_floor draws for frames. Real rooms measured so far sit
# between -80 and -95 dBA.
SILENT_TAKE_DBA = -110.0


def is_digital_silence_take(file_path: Path) -> bool:
    """True when a noise take is digital silence: its whole-file A-weighted floor
    is at or below SILENT_TAKE_DBA. An unreadable file is not called silent here;
    the scan reports it as it always has."""
    try:
        floor = measure_reference_noise_floor(Path(file_path))
    except Exception:
        return False
    return not np.isnan(floor) and floor <= SILENT_TAKE_DBA


def find_noise_profiles(folder: Path, extensions: set = None) -> dict[Path, tuple[Path, str]]:
    """Find noise profile files grouped by parent folder, with content fallback.

    Pass 1 (name): any file whose stem ends with -blank/_blank/-noise/_noise.
    Pass 2 (content): if Pass 1 finds nothing in a folder, scan remaining WAV
        and FLAC files (v8) by crest factor and pick the lowest-CF candidate at or below
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
        # Pass 1: naming convention. v8: a take of digital silence is passed over,
        # so another noise-named file, or the content scan, can serve instead.
        for f in candidates:
            if is_noise_profile(f) and not is_digital_silence_take(f):
                profiles[parent] = (f, "name")
                break

        if parent in profiles:
            continue

        # Pass 2: content scan — lowest CF at or below threshold
        # Only WAV and FLAC files are considered (other blanks are unusual and
        # crest factor is less reliable across lossy codecs at low levels).
        best_path: Path | None = None
        best_cf = float("inf")
        for f in candidates:
            if f.suffix.lower() not in MEASURED_EXTENSIONS:
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
    Returns noise floor in dBA.

    v8: the measurement is the noise core's measure_aweighted_rms_dba, the same
    formula v7 carried (-120.0 for digital silence). Its None for an empty
    file maps to NaN, as before."""
    data, sr = load_audio(file_path)
    floor = noise_core.measure_aweighted_rms_dba(data, sr)
    return NOISE_FLOOR_UNAVAILABLE if floor is None else floor


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
    """Identify true peak clips and near clips in a mono signal, return timecodes with labels.

    Args:
        data: mono audio signal as a 1D numpy array of floats in the range [-1.0, +1.0].
        sr: The sample rate of the audio signal, used to convert sample positions to timecodes.
        bias_db: The bias in decibels to adjust the threshold for true peak clip detection.
            Operator toggled, off by default.

    Returns:
        A list of strings representing the timecodes "MM:SS.mmm" and labels of identified
            peaks, their shape "TP CLIP (±0.60 dBTP)", as well as a "CLEAN" label if no peaks are found.

    Notes:
        True peak detection selected because it oversamples, so it'll catch overs even if sample
            doesn't hit full scale.
        Output is display-format only, not intended for programmatic parsing.
            Designed for human-readable reports, machine parsing would require a structured output format change.
    """
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
# ------------------------------------------------------------
def estimate_noise_floor(data: np.ndarray, sr: int, config: dict = None,
                         data_weighted: np.ndarray = None) -> float:
    """Detect silent stretches in the signal, measure RMS across just those stretches to estimate the
        noise floor. Fallback, best effort mechanism for noise floor estimation, not reliable enough to consider
        in disposition.

    v8: scan_file no longer calls this; it uses estimate_quiet_floor below. Kept as v7 had it, for
        comparison.

    Args:

        data: 1D float mono signal, used for silence gating. Also used for RMS measurement only when data_weighted is absent.
        sr: sample rate.  Convert min_segment_ms to samples.
        config: threshold dict, defaults to DEFAULT_CONFIG.  Only uses noise_floor section of config.
        data_weighted: Optional copy of data, A-weighted.  If present, switch return unit to dBA (only unit, not scope, gating
            still happens).

    Returns:

        dBFS float, unless data_weighted, in which case dBA float.
        NaN when there's no qualifying silent segment, or signal is silent when in relative mode.
        -120.0 when silent segments are detected but their RMS is extremely low (below 1e-10 linear).
            This is a floor value to avoid returning -inf dB.

    Notes:

        Gating takes place on the unweighted data, if data_weighted is provided, RMS measurement will be computed from that
            instead. Unweighted gate determines silence by true amplitude; weighted data is used to correlate measurement with
            perception.
        Two modes: relative uses file's own RMS so content dependent, while absolute uses a fixed dB threshold.
        "Best effort" fallback that overstates noise floor.  Intra speech silence measurement is imperfect because of the residual
            speech/room artifacts in such a short window, so generally skews higher than when using noise profile, with a per
            recording variance that can't be normalized for.
        Caution to not compare dBFS results to dBA-calibrated thresholds.
    """
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
# v8: the per-file floor when a folder has no noise take. scan_file uses this
# instead of estimate_noise_floor above, which stays for comparison. The number
# is shown with low confidence, gives no band and never sorts.
# ------------------------------------------------------------
QUIET_FRAME_MS = 50          # A-weighted frame length
QUIET_HOP_MS = 25            # frame hop
QUIET_SILENT_DBA = -110.0    # a frame at or below this is digital silence, not the room
QUIET_GUARD_MS = 50          # frames this close to digital silence are ignored too (gate edges)
QUIET_WINDOW_S = 0.3         # the length of the quiet stretch measured
QUIET_STEADY_DB = 2.0        # its frames may vary by this much and no more


def estimate_quiet_floor(data_weighted: np.ndarray, sr: int,
                         window_s: float = QUIET_WINDOW_S,
                         steady_db: float = QUIET_STEADY_DB,
                         guard_ms: float = QUIET_GUARD_MS,
                         silent_dba: float = QUIET_SILENT_DBA) -> float:
    """The floor of a file with no noise take: the quietest steady stretch.

    The A-weighted signal is measured in 50 ms frames on a 25 ms hop. Frames of
    digital silence, and the frames within guard_ms of them, are ignored. The
    quietest run of frames spanning window_s is the floor (its mean power, in
    dBA), but only if its frames vary by steady_db or less: room noise is
    steady, a breath or a decay tail is not.

    Returns NaN (no estimate) when no such run exists, or when the quietest one
    is not steady. The keyword arguments exist for the tests' near misses; the
    scanner always uses the defaults.
    """
    n = int(sr * QUIET_FRAME_MS / 1000)
    hop = int(sr * QUIET_HOP_MS / 1000)
    if len(data_weighted) < n:
        return NOISE_FLOOR_UNAVAILABLE
    c = np.concatenate(([0.0], np.cumsum(data_weighted.astype(np.float64) ** 2)))
    starts = np.arange(0, len(data_weighted) - n + 1, hop)
    power = (c[starts + n] - c[starts]) / n
    frame_db = 10 * np.log10(np.maximum(power, 1e-20))

    k = max(int(round(window_s * 1000 / QUIET_HOP_MS)) - 1, 1)   # k frames span window_s
    if len(frame_db) < k:
        return NOISE_FLOOR_UNAVAILABLE

    silent = frame_db <= silent_dba
    ignored = silent.copy()
    for d in range(1, int(guard_ms / QUIET_HOP_MS) + 1):         # widen each silent frame both ways
        ignored[d:] |= silent[:-d]
        ignored[:-d] |= silent[d:]

    # The mean power of every run of k frames with no ignored frame in it.
    usable = np.concatenate(([0], np.cumsum(~ignored)))
    cp = np.concatenate(([0.0], np.cumsum(power)))
    s = np.arange(0, len(frame_db) - k + 1)
    clean = (usable[s + k] - usable[s]) == k
    if not clean.any():
        return NOISE_FLOOR_UNAVAILABLE
    means = np.where(clean, (cp[s + k] - cp[s]) / k, np.inf)
    best = int(np.argmin(means))
    if frame_db[best:best + k].std() > steady_db:
        return NOISE_FLOOR_UNAVAILABLE
    return float(10 * np.log10(means[best]))

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
    flags_present = disposition_flags(result, config)

    # Apply sort rules: reject takes priority, then salvageable
    for trigger in sort_rules.get("reject_on", []):
        if trigger in flags_present:
            return "reject"

    for trigger in sort_rules.get("salvageable_on", []):
        if trigger in flags_present:
            return "salvageable"

    return "pass"


def disposition_flags(result: dict, config: dict) -> set:
    """The flags classify_disposition sorts on (v8: split out of it unchanged,
    so each JSONL row can carry them)."""
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

    return flags_present

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
              reference_noise_floor_db: float = None,
              noise_take: bool = None) -> dict:
    """Scan a single audio file.
    reference_noise_floor_db: when provided, used as the noise floor for SNR
        instead of per-file silence detection. Typically from a folder's
        noise profile (-blank/-noise) file.
    noise_take: v8. True when the caller knows this file is the folder's noise
        take, which is how scan_folder marks a take found by content scan.
        None keeps v7's rule: the file name alone decides."""
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
    if file_path.suffix.lower() not in MEASURED_EXTENSIONS:
        format_issues.append(f"File is {file_path.suffix} — expected .wav or .flac")

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
            # v8 row fields (not measured on a skipped file)
            "true_peak_dbtp": float('nan'),
            "speech_level_dba": float('nan'),
            "channel_count": ch if isinstance(ch, int) else None,
        }

    data, sr = load_audio(file_path)

    if len(data) == 0:
        raise RuntimeError("Audio file is empty")

    _is_noise_profile = is_noise_profile(file_path) if noise_take is None else noise_take

    duration = len(data) / sr
    global_peak = np.max(np.abs(data))
    global_dbtp = 20 * np.log10(global_peak + 1e-10) if global_peak > 0 else float('-inf')
    events = find_peaks_with_timecodes(data, sr, bias_db=bias_db)

    # v8 clip fix. The clip decision is the level core's whole-file 4x true
    # peak against the finder's own line (0.0 dBTP, or -0.10 with bias). The
    # finder only oversamples around samples at or above -0.2 dBFS, so it misses
    # an over whose samples all sit lower; it still supplies the timecodes.
    true_peak_dbtp = level_core.measure_true_peak(data, sr)
    has_clip = true_peak_dbtp is not None and true_peak_dbtp >= TRUE_HARD_DBTP + bias_db
    if has_clip and not any("TP CLIP" in e for e in events):
        missed = (f"--:--.--- → TP CLIP ({true_peak_dbtp:+.2f} dBTP) — "
                  f"whole-file true peak, no timecode")
        events = [missed] if events == ["CLEAN"] else events + [missed]

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
        # Fallback, v8: the quietest steady stretch of the file. Informational only.
        noise_floor_db = estimate_quiet_floor(data_weighted, sr)
        noise_floor_source = "per-file" if not np.isnan(noise_floor_db) else "unavailable"

    # Crest factor stays unweighted (standard practice)
    crest_factor_db, rms_db = compute_crest_factor(data)

    # SNR computation
    speech_level_dba = float('nan')   # v8 row field: content files only
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
        speech_level_dba = rms_weighted_db

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
        # v8 row fields
        "true_peak_dbtp": true_peak_dbtp,
        "speech_level_dba": speech_level_dba,
        "channel_count": info["channels"] if isinstance(info["channels"], int) else None,
    }

    # Classification (noise profiles get "reference" disposition)
    result["disposition"] = classify_disposition(result, config)

    return result

# ------------------------------------------------------------
# Formatting helpers
# ------------------------------------------------------------
def _fmt_db(value: float, suffix: str = "dB") -> str:
    """Format a value (float), return string with sign, value
    at one decimal precision and suffix ("dB" as default).
    Returns "N/A" when value is NaN.

    Examples: -3.456 → "-3.5 dB", 0.0 → "+0.0 dB", NaN → "N/A"
    """
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
    if config.get("readout") == "platform":
        return build_platform_report_lines(results, config)   # v8: platform blocks

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = len(results)
    clips = sum(1 for r in results if r["has_clip"])
    profile = config.get("profile_name", "Custom")

    lines = [
        f"INTAKE SCAN v{REPORT_VERSION} — {total} files",
        f"Profile: {profile}",
        f"Generated: {timestamp}",
        f"Noise floor / SNR: A-weighted (IEC 61672)",
        # v8: stated once here; each JSONL row carries the reasons.
        RT60_DRIFT_HEADER_LINE,
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
        # v8: a take of digital silence keeps v7's line (its floor is what was
        # measured) and gains this one, so nobody reads that baseline as a room.
        for rr in ref_results:
            if (rr.get("noise_reference") or {}).get("gate_reason") == "digital_silence":
                lines.append(f"Noise take not used: {Path(rr['rel_path']).as_posix()} — digital silence, "
                             f"not a room; its folder reads as having no noise take")

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

        # v8: relative to the scanned folder. An absolute path can carry
        # names that do not belong in a report.
        lines.append(f"FILE: {Path(r['rel_path']).as_posix()}")
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


# ------------------------------------------------------------
# v8: the folder scan
# v7 ran these steps inside the GUI's run_scan. They moved here so the JSONL
# rows and the tests run the same path as the app. The GUI keeps the display
# and the Finder labels.
# ------------------------------------------------------------
AUDIO_EXTENSIONS = {".wav", ".m4a", ".mp3", ".aiff", ".flac", ".aac"}

# The noise core's gate reasons that mean the noise take itself is unusable.
# Anything else (no reason, or an SNR band) means the core accepted the take.
_NOISE_TAKE_REJECT_REASONS = ("format_fail", "short_recording", "insufficient_clean_noise")


def check_noise_take(file_path: Path, root_dir: Path, found_by: str, policy: dict = None) -> dict:
    """The noise core's own verdict on a noise take. Under Default
    and Strict it only labels the rows: they keep v7's whole-file floor whatever
    it says. policy: the noise core policy to judge under (None = its default;
    Platform Band passes its own).

    The core's gate checks the format, a take of at least 3 s and a continuous
    clean run of at least 3 s, then an SNR band. The band needs a speech level
    and plays no part here, so any level will do and 0.0 is passed. Only the
    three reasons above count as the core rejecting the take. write_outputs is
    False, so the core writes no file next to the audio and no log line.
    """
    check = {
        "rel_path": Path(file_path).relative_to(root_dir).as_posix(),
        "found_by": found_by,          # "name" or "content_scan"
        "accepted": False,
        "gate_reason": None,           # set only when the core rejects the take
        "valid_noise_sec": None,
        "impulse_count": None,
        "clean_run_floor_dba": None,   # the floor on the longest clean run
    }
    try:
        record = noise_core.scan_noise_profile(Path(file_path), modality="audition",
                                               speech_level_dba=0.0, config=policy,
                                               write_outputs=False)
        reason = record.get("gate_reason")
        check["accepted"] = reason not in _NOISE_TAKE_REJECT_REASONS
        if not check["accepted"]:
            check["gate_reason"] = reason
        if reason != "format_fail":
            # The record leaves these two out, so ask the function the gate used.
            data, sr = noise_core.load_audio(Path(file_path))
            windows = noise_core.analyze_noise_windows(data, sr, policy or noise_core.load_config())
            check["valid_noise_sec"] = windows["valid_noise_sec"]
            check["impulse_count"] = windows["impulse_count"]
            check["clean_run_floor_dba"] = windows["noise_floor_dba"]
    except Exception:
        check["accepted"] = False
        check["gate_reason"] = "unreadable"   # the noise core could not read the file
    # v8: the scanner's own rule on top of the core's gate. The core accepts digital
    # silence as the quietest room there is; the scanner never uses it.
    if check["accepted"] and is_digital_silence_take(file_path):
        check["accepted"] = False
        check["gate_reason"] = "digital_silence"
    return check


def _error_result(file_path: Path, root_dir: Path, error: Exception) -> dict:
    """The result v7's run_scan recorded for a file that failed to scan.
    v8: the message drops the scanned folder's absolute path."""
    message = str(error).replace(str(root_dir) + os.sep, "")
    return {
        "path": file_path,
        "rel_path": file_path.relative_to(root_dir),
        "duration": 0.0,
        "sample_rate": "?",
        "global_dbtp": float('-inf'),
        "events": [f"ERROR → {message}"],
        "has_clip": False,
        "worst_dbtp": -99.0,
        "channels": "?",
        "bit_depth": "?",
        "noise_floor_db": float('nan'),
        "noise_floor_source": "unavailable",
        "crest_factor_db": float('nan'),
        "rms_db": float('nan'),
        "snr_db": float('nan'),
        "lufs": float('nan'),
        "is_noise_profile": False,
        "format_issues": [],
        "error": True,
        "disposition": "reject",
        "true_peak_dbtp": float('nan'),
        "speech_level_dba": float('nan'),
        "channel_count": None,
    }


def _noise_reference_for(result: dict, file_path: Path, root_dir: Path,
                         folder_takes: dict) -> dict | None:
    """The noise core's check of the take this row's floor came from, or None."""
    if result.get("is_noise_profile"):
        take = folder_takes.get(file_path.parent)
        if take is not None and take["rel_path"] == file_path.relative_to(root_dir).as_posix():
            return take
        # A second noise-named file in the same folder: check it on its own.
        found_by = "name" if is_noise_profile(file_path) else "content_scan"
        return check_noise_take(file_path, root_dir, found_by)
    if result.get("noise_floor_source") == "reference":
        return folder_takes.get(file_path.parent)
    return None


def scan_folder(folder_path: Path, config: dict = None, bias_db: float = 0.0,
                progress=None) -> list[dict]:
    """Scan every audio file under folder_path and return one result per file.

    The steps v7's GUI ran in run_scan: find each folder's noise take, measure
    its floor, scan every file against it, then flag peak outliers. Writing the
    reports and setting Finder labels stay with the caller.

    v8 adds two things. A noise take found by content scan is scanned as a noise
    take, not as a content file against its own floor. And
    each noise take is checked by the noise core, so every row can say how far
    to trust its floor (result["noise_reference"]).

    bias_db: 0.0 unless the operator ticks the bias toggle (-0.10), as in v7.
    progress: optional callable(text, tag) that receives v7's live log lines.
    """
    def say(text, tag=None):
        if progress is not None:
            progress(text, tag)

    folder_path = Path(folder_path)
    if config is None:
        config = DEFAULT_CONFIG
    if config.get("readout") == "platform":
        return scan_folder_platform(folder_path, config, progress=progress)

    files = sorted(
        f for f in folder_path.rglob("*")
        if f.suffix.lower() in AUDIO_EXTENSIONS and f.is_file()
    )
    if not files:
        say("No supported audio files found.\n", "yellow")
        return []
    say(f"Found {len(files)} audio files — analyzing...\n")

    # Pre-pass: find and measure each folder's noise take. find_noise_profiles()
    # tries the name first, then the lowest-crest-factor WAV or FLAC at or below 14.65 dB.
    folder_noise_floors = {}   # folder -> reference floor (dBA)
    folder_takes = {}          # folder -> the noise core's check of that take
    detected = find_noise_profiles(folder_path)
    for parent, (profile_path, method) in detected.items():
        try:
            nf_db = measure_reference_noise_floor(profile_path)
            folder_noise_floors[parent] = nf_db
            rel = profile_path.relative_to(folder_path)
            method_label = "name match" if method == "name" else "content scan"
            say(f"Noise profile ({method_label}): {rel} — baseline {nf_db:+.1f} dBA\n", "blue")
        except Exception as e:
            say(f"Failed to measure noise profile {profile_path.name}: {e}\n", "yellow")
            continue
        found_by = "name" if method == "name" else "content_scan"
        folder_takes[parent] = check_noise_take(profile_path, folder_path, found_by)
    if not folder_noise_floors:
        say("No noise profile files found — using per-file noise estimation\n", "yellow")

    results = []
    for i, file_path in enumerate(files, 1):
        say(f"[{i:3}/{len(files)}] {file_path.relative_to(folder_path)}\n")
        # v8: the folder's chosen noise take is a noise take however it was found.
        chosen = detected.get(file_path.parent)
        noise_take = is_noise_profile(file_path) or (chosen is not None and chosen[0] == file_path)
        # A noise take is measured whole; every other file uses its folder's floor.
        ref_nf = None if noise_take else folder_noise_floors.get(file_path.parent)
        try:
            result = scan_file(file_path, root_dir=folder_path,
                               bias_db=bias_db, config=config,
                               reference_noise_floor_db=ref_nf,
                               noise_take=noise_take)
        except Exception as e:
            result = _error_result(file_path, folder_path, e)
            say(f"   {result['events'][0]}\n", "red")
        result["noise_reference"] = _noise_reference_for(result, file_path, folder_path, folder_takes)
        if (result["noise_reference"] or {}).get("gate_reason") == "digital_silence":
            say(f"   noise take of digital silence — not used\n", "yellow")
        results.append(result)

    # Flag peak outliers before report generation
    flag_peak_outliers(results)
    return results


def scan_folder_platform(folder_path: Path, config: dict, progress=None) -> list[dict]:
    """Platform Band: what Rec QC stages 1 and 2 would say about each
    file. For every content file, the platform's two public calls with the same
    argument shapes:

        level = level_core.scan_phrase(file, write_outputs=False)
        noise = noise_core.scan_noise_profile(take, modality="audition",
                    speech_level_dba=level["speech_level_dba"],
                    config=<the pointed policy>, write_outputs=False)

    modality "audition" and write_outputs False are the two deliberate
    differences from the platform's call: nothing is written next to client
    audio and the cores' central logs stay untouched. The records
    are kept as they are (result["platform"]); the scanner re-implements no rule.

    The scanner's own pass still runs for what the cores lack (format, clip
    timecodes, crest factor, peak outliers) and is informational here: a format
    issue is shown, but the file is still read out. The bias toggle is inert,
    because the platform has no bias. A folder with no noise take gets no band;
    its per-file estimate is shown with low confidence.
    """
    def say(text, tag=None):
        if progress is not None:
            progress(text, tag)

    folder_path = Path(folder_path)
    noise_policy = config["noise_policy"]
    # The scanner's own pass needs v7's config shape. Its editing sort is
    # discarded below.
    own_config = _deep_copy_dict(DEFAULT_CONFIG)

    files = sorted(
        f for f in folder_path.rglob("*")
        if f.suffix.lower() in AUDIO_EXTENSIONS and f.is_file()
    )
    if not files:
        say("No supported audio files found.\n", "yellow")
        return []
    say(f"Found {len(files)} audio files — analyzing...\n")

    # Each folder's noise take, found as v7 finds a reference (name, then content),
    # and the noise core's own check of it under the Platform Band policy.
    takes = {}   # folder -> (take path, check)
    for parent, (take_path, method) in find_noise_profiles(folder_path).items():
        found_by = "name" if method == "name" else "content_scan"
        check = check_noise_take(take_path, folder_path, found_by, policy=noise_policy)
        takes[parent] = (take_path, check)
        floor = check["clean_run_floor_dba"]
        floor_text = f"clean run {floor:+.1f} dBA" if floor is not None else "no clean run"
        method_label = "name match" if method == "name" else "content scan"
        say(f"Noise take ({method_label}): {take_path.relative_to(folder_path)} — {floor_text}\n", "blue")
    if not takes:
        say("No noise take found — no band; per-file estimates shown with low confidence\n", "yellow")

    results = []
    for i, file_path in enumerate(files, 1):
        say(f"[{i:3}/{len(files)}] {file_path.relative_to(folder_path)}\n")
        take = takes.get(file_path.parent)
        is_take = is_noise_profile(file_path) or (take is not None and take[0] == file_path)
        platform = {"is_take": is_take, "take": take[1] if take else None,
                    "level": None, "noise": None}
        try:
            if is_take:
                if take is None or take[0] != file_path:
                    # A second noise-named file in the folder: check it on its own.
                    platform["take"] = check_noise_take(file_path, folder_path, "name",
                                                        policy=noise_policy)
                result = scan_file(file_path, root_dir=folder_path, bias_db=0.0,
                                   config=own_config, noise_take=True)
            else:
                platform["level"] = level_core.scan_phrase(file_path, write_outputs=False)
                if take is not None:
                    platform["noise"] = noise_core.scan_noise_profile(
                        take[0], modality="audition",
                        speech_level_dba=platform["level"]["speech_level_dba"],
                        config=noise_policy, write_outputs=False)
                # The scanner's own pass, measured against the platform's floor if any.
                floor = platform["noise"]["noise_floor_dba"] if platform["noise"] else None
                result = scan_file(file_path, root_dir=folder_path, bias_db=0.0,
                                   config=own_config, reference_noise_floor_db=floor,
                                   noise_take=False)
        except Exception as e:
            result = _error_result(file_path, folder_path, e)
            say(f"   {result['events'][0]}\n", "red")
            platform = {"is_take": False, "take": None, "level": None, "noise": None}
        result["platform"] = platform
        result["disposition"] = None        # the editing sort does not apply here
        result["noise_reference"] = None
        results.append(result)

    flag_peak_outliers(results)
    return results


# ------------------------------------------------------------
# v8: rows and the JSONL report
# ------------------------------------------------------------
def _json_safe(value):
    """Strict-JSON form: NaN and infinities become null, paths become POSIX
    strings, numpy scalars become plain Python values."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _int_or_none(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def noise_floor_method(result: dict) -> tuple[str, str]:
    """(noise_floor_method, noise_floor_confidence) for one result.

    high:   the floor is a noise take found by name, and the noise core accepts it
    medium: the same, for a take found by content scan
    low:    the per-file quiet-window estimate, or a take the core rejects (Default and
            Strict still sort on it, as v7 did)
    none:   no floor at all
    """
    if result.get("error") or result.get("skipped"):
        return "unavailable", "none"
    source = result.get("noise_floor_source", "unavailable")
    if source in ("reference", "full-file"):
        take = result.get("noise_reference")
        if take is not None and take.get("gate_reason") == "digital_silence":
            return "noise_take_full_file", "none"      # measured, but not a room
        if take is None or not take["accepted"]:
            return "noise_take_full_file", "low"
        return "noise_take_full_file", ("high" if take["found_by"] == "name" else "medium")
    if source == "per-file":
        return "per_file_quiet_window", "low"
    return "unavailable", "none"


def _policy(config: dict) -> dict:
    """The policy object on every row and on the run record."""
    return {
        "profile": config.get("profile_name", "Custom"),
        "scanner_config_sha256": noise_core.config_sha256(config),
        "level_config_sha256": None,     # Platform Band only
        "band_config_sha256": None,      # Platform Band only
    }


def build_rows(results: list[dict], config: dict = None) -> list[dict]:
    """One JSONL row per result: relative paths only, the noise-floor
    method and confidence, the policy hashes, and RT60 and drift as n/a."""
    if config is None:
        config = DEFAULT_CONFIG
    if config.get("readout") == "platform":
        return build_platform_rows(results, config)
    policy = _policy(config)
    rows = []
    for r in results:
        method, confidence = noise_floor_method(r)
        take = r.get("noise_reference")
        uses_take = take is not None and method == "noise_take_full_file"
        is_take = bool(r.get("is_noise_profile"))
        if r.get("error"):
            role = "error"
        elif r.get("skipped"):
            role = "skipped"
        else:
            role = "noise_take" if is_take else "content"
        events = [e for e in r.get("events", [])
                  if e not in ("CLEAN", "SKIPPED") and not e.startswith("ERROR")]
        rows.append(_json_safe({
            "record": "file",
            # Identity
            "rel_path": Path(r["rel_path"]).as_posix(),
            "role": role,
            # Format
            "bit_depth": _int_or_none(r.get("bit_depth")),
            "channels": r.get("channel_count"),
            "sample_rate": _int_or_none(r.get("sample_rate")),
            "duration_sec": r.get("duration"),
            "format_issues": list(r.get("format_issues", [])),
            # Measures
            "sample_peak_dbfs": r.get("global_dbtp"),
            "true_peak_dbtp": r.get("true_peak_dbtp"),
            "true_peak_events": events,
            "integrated_lufs": r.get("lufs"),
            "crest_factor_db": r.get("crest_factor_db"),
            "rms_dbfs": r.get("rms_db"),
            "speech_level_dba": r.get("speech_level_dba"),
            "noise_floor_dba": r.get("noise_floor_db"),
            "snr_db": None if is_take else r.get("snr_db"),   # meaningless on the take itself
            # Noise method
            "noise_floor_method": method,
            "noise_floor_confidence": confidence,
            "noise_reference_rel_path": take["rel_path"] if uses_take else None,
            "noise_reference_found_by": take["found_by"] if uses_take else None,
            "noise_reference_gate_reason": take["gate_reason"] if uses_take else None,
            "valid_noise_sec": take["valid_noise_sec"] if uses_take else None,
            "impulse_count": take["impulse_count"] if uses_take else None,
            # Editing sort (Default, Strict)
            "disposition": r.get("disposition"),
            "flags": [] if is_take else sorted(disposition_flags(r, config)),
            "peak_outlier": bool(r.get("peak_outlier", False)),
            # Platform readout (Platform Band only)
            "level_verdict": None,
            "peak_disposition": None,
            "lufs_disposition": None,
            "level_fail_reasons": None,
            "noise_band": None,
            "noise_verdict": None,
            "noise_gate_reason": None,
            # Policy
            "policy": policy,
            # Not applicable: always present, never left out
            "rt60_status": "n/a",
            "rt60_reason": RT60_REASON,
            "drift_status": "n/a",
            "drift_reason": DRIFT_REASON,
        }))
    return rows


def _platform_policy(config: dict, level_rec: dict = None, noise_rec: dict = None) -> dict:
    """The policy object under Platform Band. The hashes are copied
    from the cores' records; a row without a record gets the same core function
    over the same policy, so the value is the one the record would carry."""
    noise_policy = config["noise_policy"]
    return {
        "profile": noise_policy.get("profile_name", config.get("profile_name")),
        "scanner_config_sha256": None,
        "level_config_sha256": (level_rec["thresholds"]["config_sha256"] if level_rec
                                else level_core.config_sha256(config["level_policy"])),
        "band_config_sha256": (noise_rec["thresholds"]["config_sha256"] if noise_rec
                               else noise_core.config_sha256(noise_policy)),
    }


def _take_confidence(take: dict) -> str:
    """high or medium for a take the noise core accepts; none when it rejects it."""
    if take is None or not take["accepted"]:
        return "none"
    return "high" if take["found_by"] == "name" else "medium"


def platform_noise_method(result: dict) -> tuple[str, str]:
    """(noise_floor_method, noise_floor_confidence) under Platform Band.
    The floor is the noise core's clean run; a take the core rejects gives no
    floor (none); a folder with no take falls back to the per-file estimate
    (low, no band)."""
    p = result.get("platform") or {}
    if result.get("error"):
        return "unavailable", "none"
    if p.get("is_take"):
        return "noise_take_clean_run", _take_confidence(p.get("take"))
    if p.get("take") is not None:
        noise = p.get("noise")
        if noise is None or noise.get("gate_reason") in _NOISE_TAKE_REJECT_REASONS:
            return "noise_take_clean_run", "none"
        return "noise_take_clean_run", ("high" if p["take"]["found_by"] == "name" else "medium")
    if result.get("noise_floor_source") == "per-file":
        return "per_file_quiet_window", "low"
    return "unavailable", "none"


def build_platform_rows(results: list[dict], config: dict) -> list[dict]:
    """Rows under Platform Band: the same fields as build_rows, with the
    platform readout copied from the two cores' records. The editing sort
    does not apply, so disposition is null and flags are empty."""
    rows = []
    for r in results:
        p = r.get("platform") or {}
        level, noise, take = p.get("level"), p.get("noise"), p.get("take")
        method, confidence = platform_noise_method(r)
        if r.get("error"):
            role = "error"
        elif p.get("is_take"):
            role = "noise_take"
        elif level is not None:
            role = "content"     # read out even with a format issue (informational)
        else:
            role = "skipped"
        if p.get("is_take"):
            floor, snr = (take or {}).get("clean_run_floor_dba"), None
        elif noise is not None:
            floor, snr = noise.get("noise_floor_dba"), noise.get("snr_db")
        else:
            floor, snr = r.get("noise_floor_db"), r.get("snr_db")    # the per-file estimate
        uses_take = take is not None and method == "noise_take_clean_run"
        events = [e for e in r.get("events", [])
                  if e not in ("CLEAN", "SKIPPED") and not e.startswith("ERROR")]
        rows.append(_json_safe({
            "record": "file",
            # Identity
            "rel_path": Path(r["rel_path"]).as_posix(),
            "role": role,
            # Format
            "bit_depth": _int_or_none(r.get("bit_depth")),
            "channels": r.get("channel_count"),
            "sample_rate": _int_or_none(r.get("sample_rate")),
            "duration_sec": r.get("duration"),
            "format_issues": list(r.get("format_issues", [])),
            # Measures: the level record's where there is one
            "sample_peak_dbfs": level["sample_peak_dbfs"] if level else r.get("global_dbtp"),
            "true_peak_dbtp": level["true_peak_dbtp"] if level else r.get("true_peak_dbtp"),
            "true_peak_events": events,
            "integrated_lufs": level["integrated_lufs"] if level else None,
            "crest_factor_db": level["crest_factor_db"] if level else r.get("crest_factor_db"),
            "rms_dbfs": r.get("rms_db"),
            "speech_level_dba": level["speech_level_dba"] if level else None,
            "noise_floor_dba": floor,
            "snr_db": snr,
            # Noise method
            "noise_floor_method": method,
            "noise_floor_confidence": confidence,
            "noise_reference_rel_path": take["rel_path"] if uses_take else None,
            "noise_reference_found_by": take["found_by"] if uses_take else None,
            "noise_reference_gate_reason": take["gate_reason"] if uses_take else None,
            "valid_noise_sec": take["valid_noise_sec"] if uses_take else None,
            "impulse_count": take["impulse_count"] if uses_take else None,
            # Editing sort: not applicable under Platform Band
            "disposition": None,
            "flags": [],
            "peak_outlier": bool(r.get("peak_outlier", False)),
            # Platform readout, copied from the records
            "level_verdict": level["verdict"] if level else None,
            "peak_disposition": level["peak_disposition"] if level else None,
            "lufs_disposition": level["lufs_disposition"] if level else None,
            "level_fail_reasons": level["fail_reasons"] if level else None,
            "noise_band": noise["disposition"] if noise else None,
            "noise_verdict": noise["verdict"] if noise else None,
            "noise_gate_reason": noise["gate_reason"] if noise else None,
            # Policy
            "policy": _platform_policy(config, level, noise),
            # Not applicable: always present, never left out
            "rt60_status": "n/a",
            "rt60_reason": RT60_REASON,
            "drift_status": "n/a",
            "drift_reason": DRIFT_REASON,
        }))
    return rows


def build_run_record(results: list[dict], config: dict = None, bias_db: float = 0.0) -> dict:
    """The first JSONL line: when and how the scan ran, the policy
    hashes, and the vendored core files it ran on."""
    if config is None:
        config = DEFAULT_CONFIG
    platform = config.get("readout") == "platform"
    versions = {"level_check_phrase_core.py": level_core.SCANNER_VERSION,
                "noise_profile_scanner_core.py": noise_core.SCANNER_VERSION}
    vendored = [{"path": e["path"], "version": versions.get(e["path"]),
                 "repo": e["repo"], "commit": e["commit"], "sha256": e["sha256"]}
                for e in load_manifest(CORES_DIR)["files"]]
    return _json_safe({
        "record": "run",
        "schema_version": ROWS_SCHEMA_VERSION,
        "scanner_version": SCANNER_VERSION,
        "scanned_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "profile": config.get("profile_name", "Custom"),
        "readout": "platform" if platform else "editing",
        "policy": _platform_policy(config) if platform else _policy(config),
        # The bias toggle is inert under Platform Band: the platform has no bias.
        "bias_db": 0.0 if platform else bias_db,
        "file_count": len(results),
        "vendored": vendored,
        "rt60_status": "n/a",
        "rt60_reason": RT60_REASON,
        "drift_status": "n/a",
        "drift_reason": DRIFT_REASON,
    })


def _platform_tier(r: dict) -> int:
    """Report order under Platform Band: a file that fails either stage, then
    WARN, CAUTION, PASS, files with no band, noise takes, errors."""
    p = r.get("platform") or {}
    if r.get("error"):
        return 6
    if p.get("is_take"):
        return 5
    level, noise = p.get("level"), p.get("noise")
    if (level and level["verdict"] == "FAIL") or (noise and noise["disposition"] == "REJECT"):
        return 0
    if noise is None:
        return 4
    return {"WARN": 1, "CAUTION": 2, "PASS": 3}.get(noise["disposition"], 4)


def _across(label: str, values: list, unit: str) -> str:
    """One 'across files' header line in v7's shape."""
    return (f"{label} across files: worst {max(values):+.1f} {unit} / best {min(values):+.1f} {unit} / "
            f"median {np.median(values):+.1f} {unit}")


def build_platform_report_lines(results: list[dict], config: dict) -> list[str]:
    """The text report under Platform Band: a header
    with the bands and level-check lines read from the cores, their policy
    hashes and the counts, then one block per file, failing files first. The
    lines the sibling harnesses parse keep v7's shapes (FILE:, RMS:, SNR:,
    Noise floor:, Highest measured peak:, LUFS:, TRUE PEAK, baseline ... dBA)."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    noise_policy, level_policy = config["noise_policy"], config["level_policy"]
    snr, peak, loud = noise_policy["snr"], level_policy["peak"], level_policy["lufs"]
    band_hash = noise_core.config_sha256(noise_policy)[:8]
    level_hash = level_core.config_sha256(level_policy)[:8]

    read_out = [r for r in results if (r.get("platform") or {}).get("level")]
    bands = {}
    for r in read_out:
        noise = r["platform"]["noise"]
        key = noise["disposition"] if noise else "no band"
        bands[key] = bands.get(key, 0) + 1
    verdicts = {"PASS": 0, "FAIL": 0}
    for r in read_out:
        verdicts[r["platform"]["level"]["verdict"]] += 1
    n_hot = sum(1 for r in read_out if r["platform"]["level"]["peak_disposition"] == "HOT")
    n_quiet = sum(1 for r in read_out if r["platform"]["level"]["lufs_disposition"] == "QUIET")

    lines = [
        f"INTAKE SCAN v{REPORT_VERSION} — {len(results)} files",
        f"Profile: {config.get('profile_name', 'Platform Band')}",
        f"Generated: {timestamp}",
        "Noise floor / SNR: A-weighted (IEC 61672)",
        RT60_DRIFT_HEADER_LINE,
        "Platform readout: Rec QC stages 1 and 2, by the platform's own cores — bands and verdicts",
        f"Noise bands: reject < {snr['reject_below']} dB | warn < {snr['warn_below']} dB | "
        f"caution < {snr.get('caution_below', snr['warn_below'])} dB | pass ≥ {snr['pass_above']} dB "
        f"(noise core policy \"{noise_policy.get('profile_name')}\", {band_hash})",
        f"Level check: clip at ≥ {peak['clip_at_dbtp']} dBTP | hot above {peak['hot_above_dbfs']} dBFS | "
        f"too quiet below {loud['too_quiet_below']} LUFS | quiet at or below {loud['quiet_below']} LUFS "
        f"(level core policy \"{level_policy.get('profile_name')}\", {level_hash})",
        f"Noise band: {bands.get('PASS', 0)} PASS / {bands.get('CAUTION', 0)} CAUTION / "
        f"{bands.get('WARN', 0)} WARN / {bands.get('REJECT', 0)} REJECT / {bands.get('no band', 0)} no band",
        f"Level verdict: {verdicts['PASS']} PASS / {verdicts['FAIL']} FAIL — flags {n_hot} HOT, {n_quiet} QUIET",
    ]

    n_err = sum(1 for r in results if r.get("error"))
    if n_err:
        lines.append(f"Errors: {n_err} file(s) could not be read")
    with_issues = [r for r in read_out if r.get("format_issues")]
    if with_issues:
        lines.append(f"Format issues (informational under Platform Band): {len(with_issues)} file(s)")
        for r in with_issues:
            lines.append(f"  {Path(r['rel_path']).as_posix()} — {'; '.join(r['format_issues'])}")
    if bands.get("no band"):
        lines.append(f"{bands['no band']} file(s) without a folder noise take — no band; "
                     f"per-file estimate shown with low confidence")

    floors = [r["platform"]["noise"]["noise_floor_dba"] for r in read_out
              if r["platform"]["noise"] and r["platform"]["noise"]["noise_floor_dba"] is not None]
    snrs = [r["platform"]["noise"]["snr_db"] for r in read_out
            if r["platform"]["noise"] and r["platform"]["noise"]["snr_db"] is not None]
    lufs_values = [r["platform"]["level"]["integrated_lufs"] for r in read_out
                   if r["platform"]["level"]["integrated_lufs"] is not None]
    if floors:
        lines.append(_across("Noise floor (clean run)", floors, "dBA"))
    if snrs:
        lines.append(f"SNR across files: worst {min(snrs):+.1f} dB / best {max(snrs):+.1f} dB / "
                     f"median {np.median(snrs):+.1f} dB")
    if lufs_values:
        lines.append(f"LUFS across files: lowest {min(lufs_values):+.1f} LUFS / highest "
                     f"{max(lufs_values):+.1f} LUFS / median {np.median(lufs_values):+.1f} LUFS")

    for r in results:
        p = r.get("platform") or {}
        take = p.get("take")
        if not p.get("is_take") or take is None:
            continue
        floor = take["clean_run_floor_dba"]
        floor_text = f"baseline {floor:+.1f} dBA" if floor is not None else "no clean run"
        clean = f"{take['valid_noise_sec']:.1f} s" if take["valid_noise_sec"] is not None else "n/a"
        silent = take["gate_reason"] == "digital_silence"
        verdict = ("accepted by the noise core" if take["accepted"]
                   else "digital silence, not used" if silent
                   else f"rejected by the noise core ({take['gate_reason']})")
        lines.append(f"Noise profile: {Path(r['rel_path']).as_posix()} — {floor_text} "
                     f"(clean run {clean}, {take['impulse_count']} impulse(s), "
                     f"found by {take['found_by'].replace('_', ' ')}, {verdict})")
        if silent:   # a deliberate difference from the platform's core
            lines.append(f"Noise take not used: {Path(r['rel_path']).as_posix()} — digital silence, "
                         f"not a room; its folder reads as having no noise take")

    lines.append("=" * 80)
    lines.append("")

    for r in sorted(results, key=lambda x: (_platform_tier(x), -x.get("worst_dbtp", -99.0))):
        p = r.get("platform") or {}
        level, noise, take = p.get("level"), p.get("noise"), p.get("take")
        lines.append(f"FILE: {Path(r['rel_path']).as_posix()}")

        if r.get("error"):
            lines.append("Platform: not read out (error)")
            lines.extend(f"   {e}" for e in r["events"])
            lines.append(">>> ERROR")
            lines.append("-" * 80)
            lines.append("")
            continue

        if p.get("is_take"):
            state = ("accepted by the noise core" if take and take["accepted"]
                     else "digital silence, not used by the scanner"
                     if take and take["gate_reason"] == "digital_silence"
                     else f"rejected ({take['gate_reason']}) by the noise core" if take
                     else "not checked by the noise core")
            lines.append(f"Platform: noise take — {state}")
        else:
            band = noise["disposition"] if noise else "no band"
            flags = [d for d in (level["peak_disposition"], level["lufs_disposition"]) if d != "PASS"]
            lines.append(f"Platform: noise {band} | level {level['verdict']}"
                         + (f" ({', '.join(flags)})" if flags else ""))

        issues = r.get("format_issues", [])
        lines.append(f"Format: {r['bit_depth']}-bit | {r['channels']} | {r['sample_rate']} Hz"
                     + (f"  (informational) — {'; '.join(issues)}" if issues else ""))
        lines.append(f"Duration: {sec_to_timecode(r['duration'])}")
        sample_peak = level["sample_peak_dbfs"] if level else r.get("global_dbtp", float('-inf'))
        lines.append("Highest measured peak: "
                     + (f"{sample_peak:+.2f} dBFS" if sample_peak not in (None, float('-inf')) else "N/A"))
        true_peak = level["true_peak_dbtp"] if level else r.get("true_peak_dbtp")
        lines.append(f"True peak:    {_fmt_db(true_peak if true_peak is not None else float('nan'), 'dBTP')}"
                     " (whole file)")
        lines.append(f"RMS:          {_fmt_db(r.get('rms_db', float('nan')), 'dBFS')}")

        if p.get("is_take"):
            floor = take["clean_run_floor_dba"] if take else None
            lines.append("Noise floor:  "
                         + (f"{floor:+.1f} dBA (clean run)" if floor is not None else "N/A (no clean run)"))
        else:
            crest = level["crest_factor_db"]
            lines.append(f"Crest factor: {_fmt_db(crest if crest is not None else float('nan'))} (informational)")
            if noise is not None:
                floor, snr_db = noise["noise_floor_dba"], noise["snr_db"]
                lines.append("Noise floor:  "
                             + (f"{floor:+.1f} dBA (clean run)" if floor is not None
                                else f"N/A ({noise['gate_reason']})"))
                lines.append("SNR:          "
                             + (f"{snr_db:+.1f} dB  [{noise['disposition']}]" if snr_db is not None
                                else f"N/A  [{noise['disposition']}]"))
            else:
                floor, snr_db = r.get("noise_floor_db", float('nan')), r.get("snr_db", float('nan'))
                lines.append(f"Noise floor:  {_fmt_db(floor, 'dBA (per-file estimate, low confidence)')}")
                lines.append(f"SNR:          {_fmt_db(snr_db)}  (informational — no noise take, no band)")
            lufs = level["integrated_lufs"]
            lines.append("LUFS:         "
                         + (f"{lufs:+.1f} LUFS  [{level['lufs_disposition']}]" if lufs is not None else "N/A"))
            if r.get("peak_outlier"):
                lines.append(f"PEAK:         [CAUTION] — {r.get('peak_delta_db', 0):.1f} dB below "
                             f"folder max peak (informational)")

        if r.get("skipped"):
            lines.append("TRUE PEAK: timecodes not read (format issue)")
        elif r["events"] != ["CLEAN"]:
            lines.append("TRUE PEAK ✗")
            lines.extend(f"   {e}" for e in r["events"])
        else:
            lines.append("TRUE PEAK ✓ CLEAN")

        if p.get("is_take"):
            lines.append(">>> NOISE TAKE")
        else:
            band = noise["disposition"] if noise else "no band"
            lines.append(f">>> noise {band} | level {level['verdict']}")
        lines.append("-" * 80)
        lines.append("")

    return lines


def write_jsonl_report(results: list[dict], output_path: Path,
                       bias_db: float = 0.0, config: dict = None):
    """INTAKE_REPORT.jsonl: the run record, then one row per file. Strict JSON,
    one object per line."""
    if config is None:
        config = DEFAULT_CONFIG
    records = [build_run_record(results, config, bias_db)] + build_rows(results, config)
    text = "".join(json.dumps(rec, ensure_ascii=False, allow_nan=False) + "\n" for rec in records)
    Path(output_path).write_text(text, encoding="utf-8")


# ------------------------------------------------------------
# v8: Finder label colours. The GUI applies them; the mapping lives here so it
# can be checked without opening a window.
# Label indices: 0 none, 1 orange, 2 red, 3 yellow, 4 blue, 5 purple, 6 green, 7 gray
# ------------------------------------------------------------
def finder_label_index(r: dict, config: dict) -> int:
    """The Finder label for one result. Default and Strict keep v7's mapping;
    Platform Band has its own (see _platform_label_index)."""
    if config.get("readout") == "platform":
        return _platform_label_index(r)

    disp = r.get("disposition", "pass")
    if disp == "reject":
        return 2  # Red
    if disp == "reference":
        return 7  # Gray — noise profile reference file
    if disp == "pass":
        if r.get("peak_outlier"):
            return 4  # Blue
        return 6  # Green

    # Salvageable — check which flags are present to pick orange vs yellow
    snr_db = r.get("snr_db", float('nan'))
    cf_db = r.get("crest_factor_db", float('nan'))
    lufs = r.get("lufs", float('nan'))
    snr_cfg = config["snr"]
    cf_cfg = config["crest_factor"]
    loud_cfg = config.get("loudness", DEFAULT_CONFIG["loudness"])

    has_critical = False
    nf_source = r.get("noise_floor_source", "unavailable")
    if not np.isnan(snr_db) and nf_source == "reference" and snr_db < snr_cfg["warn_below"]:
        has_critical = True  # snr_warn level (reference-based only)
    if not np.isnan(cf_db):
        if cf_db <= cf_cfg["warn_low"] or cf_db >= cf_cfg["warn_high"]:
            has_critical = True  # cf_warn level
    if not np.isnan(lufs) and lufs != float('-inf'):
        if lufs < loud_cfg.get("caution_below", loud_cfg.get("warn_below", -36.0)):
            has_critical = True  # lufs_caution level

    return 1 if has_critical else 3  # Orange if critical, Yellow if minor


def _platform_label_index(r: dict) -> int:
    """The default mapping:
    red for a level FAIL or a noise REJECT, orange for WARN, yellow for CAUTION
    or a level flag (HOT, QUIET), green for PASS on both, gray for the noise
    take, and none when the folder has no noise take: an unanswered row never
    shows green."""
    if r.get("error"):
        return 2
    p = r.get("platform") or {}
    if p.get("is_take"):
        return 7
    level, noise = p.get("level"), p.get("noise")
    if level is None:
        return 0
    band = noise["disposition"] if noise else None
    flagged = level["peak_disposition"] != "PASS" or level["lufs_disposition"] != "PASS"
    if level["verdict"] == "FAIL" or band == "REJECT":
        return 2
    if band == "WARN":
        return 1
    if band == "CAUTION" or flagged:
        return 3
    if band == "PASS":
        return 6
    return 0


def write_reports(results: list[dict], folder_path: Path,
                  bias_db: float = 0.0, config: dict = None) -> tuple[Path, Path]:
    """Write INTAKE_REPORT.txt and INTAKE_REPORT.jsonl into the scanned folder.
    Returns both paths."""
    folder_path = Path(folder_path)
    report_path = folder_path / REPORT_NAME
    rows_path = folder_path / ROWS_NAME
    write_text_report(results, report_path, bias_db=bias_db, config=config)
    write_jsonl_report(results, rows_path, bias_db=bias_db, config=config)
    return report_path, rows_path
