# intake_scanner_gui_v7.py
# v7.0 changes:
# - Imports from intake_scanner_core_v7
# - Pre-pass noise profile detection uses new find_noise_profiles() which
#   returns {folder: (path, method)}; method logged as 'name match' or
#   'content scan' so the operator knows how the profile was found
# - Version labels bumped to 7.0
#
# v6.1 changes (preserved):
# - Report header trimmed (Sample threshold / Noise floor detection lines removed)
# - Header lines 1-9 rendered white (no more teal titles in report body)
# - Disposition line: per-bucket colored counts (zeros stay white)
# - "N files contain TP CLIPs" line: count is red when > 0, white when 0
# - "Skipped (format mismatch)" header and its file list rendered red
#
# v6.0 changes (preserved):
# - LUFS line colorization in get_tag_for_line
# - Header line styling for LUFS aggregate/threshold lines
#
# v5.0 changes (preserved):
# - Profile selector dropdown (strict/default/custom YAML)
# - Disposition coloring in report output
# - Headless mode: --headless <folder> [--profile <name_or_path>] [--bias]

import re
import sys
from pathlib import Path

from intake_scanner_core import (
    scan_file, write_text_report, build_report_lines,
    load_config, list_builtin_profiles, get_builtin_config_dir,
    flag_peak_outliers, DEFAULT_CONFIG,
    is_noise_profile, find_noise_profiles, measure_reference_noise_floor,
    load_audio, a_weight,
)


def _resolve_config(profile_arg: str = None) -> dict:
    """Resolve a profile name or YAML path to a config dict.
    Defaults to default profile when no profile is specified."""
    if profile_arg is None:
        profile_arg = "default"

    # If it looks like a file path, load it directly
    p = Path(profile_arg)
    if p.suffix in (".yaml", ".yml") and p.exists():
        return load_config(p)

    # Otherwise match against builtin profile names (case-insensitive)
    for builtin in list_builtin_profiles():
        if profile_arg.lower() in builtin.stem.lower():
            return load_config(builtin)

    raise ValueError(f"Profile not found: {profile_arg}")


def _parse_headless_args() -> tuple:
    """Parse --headless CLI args. Returns (folder_path, bias_db, config) or None."""
    if "--headless" not in sys.argv:
        return None

    args = [a for a in sys.argv[1:] if a != "--headless"]
    bias_db = 0.0
    profile_arg = None
    folder_path = None

    i = 0
    while i < len(args):
        if args[i] == "--bias":
            bias_db = -0.10
        elif args[i] == "--profile" and i + 1 < len(args):
            profile_arg = args[i + 1]
            i += 1
        elif folder_path is None:
            folder_path = Path(args[i])
        i += 1

    if folder_path is None or not folder_path.is_dir():
        return None

    config = _resolve_config(profile_arg)
    return folder_path, bias_db, config


# -- GUI mode ----------------------------------------------------------------
import tkinter as tk
from tkinter import filedialog, scrolledtext
from tkinterdnd2 import DND_FILES, TkinterDnD
import queue
import threading


# -- App window ---------------------------------------------------------------
root = TkinterDnD.Tk()
root.title("Intake Scanner 7.0")
root.configure(bg="#1e1e1e")
root.geometry("1200x900")
root.minsize(1000, 700)

# -- State --------------------------------------------------------------------
_scan_in_progress = False
_log_queue: queue.Queue = queue.Queue()
bias_var = tk.BooleanVar(value=False)

# Profile management
_profile_map: dict[str, Path | None] = {}  # display name -> config path (None = default)
_custom_config_path: Path | None = None


def _build_profile_map() -> dict[str, Path | None]:
    """Build mapping of display names to config file paths."""
    profiles = {"Default": None}  # None = fallback to DEFAULT_CONFIG if YAML missing
    for p in list_builtin_profiles():
        try:
            cfg = load_config(p)
            name = cfg.get("profile_name", p.stem)
            if "default" in p.stem.lower():
                profiles["Default"] = p
            elif "strict" in p.stem.lower():
                profiles[name] = p
            else:
                profiles[name] = p
        except Exception:
            profiles[p.stem] = p
    profiles["Custom YAML..."] = "custom"
    return profiles


_profile_map = _build_profile_map()
profile_var = tk.StringVar(value="Default")


def _get_active_config() -> dict:
    """Load the currently selected profile config."""
    selected = profile_var.get()

    if selected == "Custom YAML...":
        if _custom_config_path and _custom_config_path.exists():
            return load_config(_custom_config_path)
        log("Default profile not found — using built-in strict thresholds\n", "yellow")
        return DEFAULT_CONFIG

    config_path = _profile_map.get(selected)
    if config_path is None or config_path == "custom":
        log("Default profile not found — using built-in strict thresholds\n", "yellow")
        return DEFAULT_CONFIG

    try:
        return load_config(config_path)
    except Exception as e:
        log(f"Failed to load profile '{selected}': {e} — using built-in strict thresholds\n", "yellow")
        return DEFAULT_CONFIG


def _on_profile_change(*args):
    """Handle profile dropdown change. If 'Custom YAML...' is selected, open file picker."""
    global _custom_config_path
    if profile_var.get() == "Custom YAML...":
        path = filedialog.askopenfilename(
            title="Select YAML Config",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
        )
        if path:
            _custom_config_path = Path(path)
            # Validate it loads
            try:
                cfg = load_config(_custom_config_path)
                log(f"Loaded custom config: {cfg.get('profile_name', _custom_config_path.name)}\n", "green")
            except Exception as e:
                log(f"Failed to load config: {e}\n", "red")
                profile_var.set("Default")
        else:
            # User cancelled — revert to default
            profile_var.set("Default")


profile_var.trace_add("write", _on_profile_change)


# -- Tag colorization ---------------------------------------------------------
_active_config: dict = DEFAULT_CONFIG  # set at scan start for color decisions

def _warn_color() -> str:
    """Return the color tag for [WARN] flags — red for strict, orange for default."""
    name = _active_config.get("profile_name", "").lower()
    return "orange" if "default" in name else "red"

def get_tag_for_line(line: str) -> str:
    """Pick a color tag for a report line.
    v6.1 convention: non-error lines are white; error/flag lines use their
    severity color. The Disposition and 'N files contain TP CLIPs' lines are
    handled separately with per-segment coloring, not here.
    """
    line_upper = line.upper()
    stripped = line.strip()

    # Per-file body — these lines carry flag markers and need severity colors
    if stripped.startswith("Format:"):
        if "[REJECT]" in line:
            return "red"
        if "[SKIPPED]" in line:
            return "red"
        return "yellow"
    if stripped.startswith("Crest factor:"):
        return _warn_color() if "[WARN]" in line else ("yellow" if "[CAUTION]" in line else "white")
    if stripped.startswith("SNR:"):
        if "[REJECT]" in line:
            return "red"
        if "[NOISE PROFILE]" in line:
            return "blue"
        return _warn_color() if "[WARN]" in line else ("yellow" if "[CAUTION]" in line else "white")
    if stripped.startswith("LUFS:"):
        if "[REJECT]" in line:
            return "red"
        if "[NOISE PROFILE]" in line:
            return "blue"
        return _warn_color() if "[WARN]" in line else ("yellow" if "[CAUTION]" in line else "white")
    if stripped.startswith("PEAK:"):
        return "blue"

    # Per-file disposition line (one per file in the body)
    if stripped.startswith("Disposition:"):
        if "REJECT" in line_upper:
            return "red"
        if "SALVAGEABLE" in line_upper:
            return "yellow"
        if "NOISE PROFILE" in line_upper:
            return "blue"
        if "SKIPPED" in line_upper:
            return "red"
        if "PASS" in line_upper:
            return "green"
        return "white"

    # Per-file footer: ">>> PASS/REJECT/..." and TRUE PEAK lines
    if "CLEAN" in line_upper and "CLIP" not in line_upper:
        return "green"
    if any(w in line_upper for w in ["TP CLIP", "ERROR", "FAIL", "REJECT"]):
        return "red"
    if any(w in line_upper for w in ["NEAR-CLIP", "SALVAGEABLE"]):
        return "yellow"

    # Flag count lines — severity by content
    if line.startswith("CF flags:") or line.startswith("SNR flags:") or line.startswith("LUFS flags:"):
        wc = _warn_color()
        return wc if "WARN" in line or "REJECT" in line else "yellow"

    # Noise profile baseline line — keep blue as its established convention
    if line.startswith("Noise profile:"):
        return "blue"

    # Missing noise profile warnings stay yellow (cautionary, not error)
    if line.startswith("WARNING:") or "without folder noise profile" in line:
        return "yellow"

    # Peak outlier callouts — blue per existing convention
    if line.startswith("Peak outliers"):
        return "blue"

    # Everything else (header, thresholds, aggregates, separators, per-file body) → white
    return "white"


# -- Thread-safe logging -------------------------------------------------------
def log(text: str, tag: str = None):
    _log_queue.put(("text", text, tag or get_tag_for_line(text)))

def log_segments(segments: list):
    """Log a line composed of multiple (text, tag) segments.
    Used for lines that need per-word coloring (e.g. Disposition counts)."""
    _log_queue.put(("segments", segments))

def log_clear():
    _log_queue.put(("clear",))

def log_scroll_top():
    _log_queue.put(("scroll_top",))

def _process_log_queue():
    try:
        while True:
            item = _log_queue.get_nowait()
            if item[0] == "text":
                _, text, tag = item
                output.config(state=tk.NORMAL)
                output.insert(tk.END, text, tag)
                output.see(tk.END)
                output.config(state=tk.DISABLED)
            elif item[0] == "segments":
                _, segments = item
                output.config(state=tk.NORMAL)
                for text, tag in segments:
                    output.insert(tk.END, text, tag)
                output.see(tk.END)
                output.config(state=tk.DISABLED)
            elif item[0] == "clear":
                output.config(state=tk.NORMAL)
                output.delete(1.0, tk.END)
                output.config(state=tk.DISABLED)
            elif item[0] == "scroll_top":
                output.see("1.0")
            elif item[0] == "scan_done":
                _on_scan_done()
    except queue.Empty:
        pass
    root.after(50, _process_log_queue)


# -- Segment builders for multi-color report lines ----------------------------
# The Disposition and "N files contain TP CLIPs" lines have per-bucket colors.

_DISPOSITION_COLORS = {
    "pass":          "green",
    "salvageable":   "yellow",
    "reject":        "red",
    "skipped":       "gray",     # matches the "N skipped (format)" label
    "noise profile": "blue",
    "error":         "red",
}

def _disposition_color_for(label: str, count: int) -> str:
    """Pick color for a disposition count. Zero counts stay white."""
    if count == 0:
        return "white"
    key = label.strip().lower()
    # "skipped (format)" starts with "skipped" — strip any trailing qualifier
    for prefix, color in _DISPOSITION_COLORS.items():
        if key.startswith(prefix):
            return color
    return "white"

def _log_disposition_line(line: str):
    """Split the Disposition line into colored segments and log it.
    Format: 'Disposition: N pass / N salvageable / N reject [/ ...]'.
    Words and slashes stay white; counts colored per bucket (0 → white)."""
    prefix = "Disposition: "
    if not line.startswith(prefix):
        log(line + "\n")
        return
    rest = line[len(prefix):]
    segments = [(prefix, "white")]
    for i, part in enumerate(rest.split(" / ")):
        if i > 0:
            segments.append((" / ", "white"))
        m = re.match(r"(\d+)\s+(.+)", part)
        if m:
            count = int(m.group(1))
            label = m.group(2)
            segments.append((m.group(1), _disposition_color_for(label, count)))
            segments.append((" " + label, "white"))
        else:
            segments.append((part, "white"))
    segments.append(("\n", "white"))
    log_segments(segments)

def _log_tpclips_line(line: str):
    """Color the count in 'N files contain TP CLIPs'. Count red when > 0."""
    m = re.match(r"(\d+)( files contain TP CLIPs.*)", line)
    if not m:
        log(line + "\n")
        return
    count = int(m.group(1))
    number_color = "red" if count > 0 else "white"
    log_segments([
        (m.group(1), number_color),
        (m.group(2), "white"),
        ("\n", "white"),
    ])

def _on_scan_done():
    global _scan_in_progress
    _scan_in_progress = False
    scan_button.config(
        state=tk.NORMAL,
        text="Select Folder → Scan"
    )


# -- Finder color labels -------------------------------------------------------
# Label indices: 0=None, 1=Orange, 2=Red, 3=Yellow, 4=Blue, 5=Purple, 6=Green, 7=Gray

def _set_finder_label(file_path: Path, label_index: int):
    """Set the Finder color label on a file via AppleScript."""
    import subprocess
    posix = str(file_path)
    subprocess.run(
        ['osascript', '-e',
         f'tell application "Finder" to set label index of '
         f'(POSIX file "{posix}" as alias) to {label_index}'],
        capture_output=True, timeout=5,
    )

def _finder_label_for_result(r: dict, config: dict) -> int:
    """Determine Finder label index from scan result and config.
    Reject (including TP clips) -> Red (2)
    Salvageable with snr_warn or cf_warn -> Orange (1)
    Salvageable with only snr_caution -> Yellow (3)
    Peak outlier (pass but quiet) -> Blue (4)
    Pass -> Green (6)"""
    import numpy as np

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

def _apply_finder_labels(results: list[dict], config: dict) -> int:
    """Apply Finder color labels based on disposition. Returns count of labeled files."""
    count = 0
    for r in results:
        label = _finder_label_for_result(r, config)
        try:
            _set_finder_label(r["path"], label)
            count += 1
        except Exception:
            pass
    return count


# -- Scan logic ----------------------------------------------------------------
def run_scan(folder_path: Path, bias_db: float = -0.10, config: dict = None):
    global _active_config
    if config is None:
        config = DEFAULT_CONFIG
    _active_config = config

    try:
        log_clear()
        profile = config.get("profile_name", "Custom")
        log(f"Intake Scanner v7.0 — {profile}\n", "title")
        log("=" * 70 + "\n", "title")
        log(f"Scanning folder:\n{folder_path}\n")
        if bias_db != 0.0:
            log(f"Bias compensation: {bias_db:+.2f} dB (effective TP CLIP threshold: {bias_db:+.2f} dBTP)\n", "yellow")

        exts = {".wav", ".m4a", ".mp3", ".aiff", ".flac", ".aac"}
        files = sorted(
            f for f in folder_path.rglob("*")
            if f.suffix.lower() in exts and f.is_file()
        )

        if not files:
            log("No supported audio files found.\n", "yellow")
            return

        log(f"Found {len(files)} audio files — analyzing...\n")

        # Pre-pass: find and measure noise profile files per folder.
        # find_noise_profiles() runs naming convention first, then content
        # fallback (lowest-CF WAV at or below 14.65 dB) for folders with no
        # named profile. Method is logged so the operator can verify.
        folder_noise_floors = {}  # parent_folder -> noise_floor_db
        detected_profiles = find_noise_profiles(folder_path)
        for parent, (profile_path, method) in detected_profiles.items():
            try:
                nf_db = measure_reference_noise_floor(profile_path)
                folder_noise_floors[parent] = nf_db
                rel = profile_path.relative_to(folder_path)
                method_label = "name match" if method == "name" else "content scan"
                log(f"Noise profile ({method_label}): {rel} — baseline {nf_db:+.1f} dBA\n", "blue")
            except Exception as e:
                log(f"Failed to measure noise profile {profile_path.name}: {e}\n", "yellow")
        if not folder_noise_floors:
            log("No noise profile files found — using per-file noise estimation\n", "yellow")

        results = []
        for i, file_path in enumerate(files, 1):
            log(f"[{i:3}/{len(files)}] {file_path.relative_to(folder_path)}\n")
            try:
                # Pass folder's reference noise floor if available (not for the profile file itself)
                ref_nf = None
                if not is_noise_profile(file_path):
                    ref_nf = folder_noise_floors.get(file_path.parent)
                result = scan_file(file_path, root_dir=folder_path,
                                   bias_db=bias_db, config=config,
                                   reference_noise_floor_db=ref_nf)
                results.append(result)
            except Exception as e:
                log(f"   ERROR → {e}\n", "red")
                results.append({
                    "path": file_path,
                    "rel_path": file_path.relative_to(folder_path),
                    "duration": 0.0,
                    "sample_rate": "?",
                    "global_dbtp": float('-inf'),
                    "events": [f"ERROR → {e}"],
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
                })

        # Flag peak outliers before report generation
        flag_peak_outliers(results)

        report_path = folder_path / "INTAKE_REPORT.txt"
        try:
            write_text_report(results, report_path, bias_db=bias_db, config=config)
        except Exception as e:
            log(f"\nFAILED to save report: {e}\n", "red")
            return

        log_clear()
        # Route special lines through their own colorizers; everything else
        # falls through to get_tag_for_line.
        in_skipped_block = False
        for line in build_report_lines(results, bias_db=bias_db, config=config):
            if line.startswith("Disposition:"):
                _log_disposition_line(line)
                in_skipped_block = False
                continue
            if re.match(r"\d+ files contain TP CLIPs", line):
                _log_tpclips_line(line)
                in_skipped_block = False
                continue
            if line.startswith("Skipped (format mismatch)"):
                log(line + "\n", "red")
                in_skipped_block = True
                continue
            # Detail lines of the skipped block are 2-space-indented right
            # after the header; end the block on the first non-indented line.
            if in_skipped_block and line.startswith("  "):
                log(line + "\n", "red")
                continue
            in_skipped_block = False
            log(line + "\n")

        # Summary with disposition counts
        content = [r for r in results if not r.get("is_noise_profile")]
        n_pass = sum(1 for r in content if r.get("disposition") == "pass")
        n_salv = sum(1 for r in content if r.get("disposition") == "salvageable")
        n_rej  = sum(1 for r in content if r.get("disposition") == "reject")
        clips  = sum(1 for r in results if r.get("has_clip", False))

        log(f"\nSCAN COMPLETE — {n_pass} pass / {n_salv} salvageable / {n_rej} reject\n", "title")
        if clips:
            log(f"{clips} file(s) with true-peak clipping\n", "red")

        # Apply Finder color labels
        labeled = _apply_finder_labels(results, config)
        if labeled:
            log(f"Finder labels applied: {labeled} files\n", "green")

        log(f"Report saved: {report_path}\n", "title")
        log_scroll_top()

    except Exception as e:
        import traceback
        log(f"\nFATAL ERROR: {e}\n", "red")
        log(traceback.format_exc(), "red")
    finally:
        _log_queue.put(("scan_done",))


# -- Scan launchers ------------------------------------------------------------
def _start_scan(folder_path: Path, bias_db: float = None, config: dict = None):
    global _scan_in_progress
    if _scan_in_progress:
        return
    _scan_in_progress = True
    if bias_db is None:
        bias_db = -0.10 if bias_var.get() else 0.0
    if config is None:
        config = _get_active_config()
    scan_button.config(state=tk.DISABLED, text="Scanning...")
    threading.Thread(target=run_scan, args=(folder_path, bias_db, config), daemon=False).start()

def select_folder():
    folder = filedialog.askdirectory(title="Select Folder to Scan")
    if folder:
        _start_scan(Path(folder))

def on_drop(event):
    raw = event.data.strip()
    if raw.startswith("{"):
        path_str = raw[1:raw.index("}")] if "}" in raw else raw[1:]
    else:
        path_str = raw.split()[0]

    folder_path = Path(path_str)
    if folder_path.is_dir():
        _start_scan(folder_path)
    else:
        log(f"Drop target is not a folder: {path_str}\n", "red")


# -- Drag & drop ---------------------------------------------------------------
root.drop_target_register(DND_FILES)
root.dnd_bind('<<Drop>>', on_drop)


# -- UI layout -----------------------------------------------------------------
tk.Label(root, text="Intake Scanner", font=("Helvetica", 36, "bold"),
         fg="#00ffaa", bg="#1e1e1e", pady=50).pack()

frame = tk.Frame(root, bg="#0a0a0a", bd=3, relief="sunken")
frame.pack(padx=40, pady=(0, 40), fill="both", expand=True)

output = scrolledtext.ScrolledText(
    frame, font=("Menlo", 13), bg="#0d0d0d", fg="#d4d4d4",
    insertbackground="#d4d4d4", state=tk.DISABLED, relief="flat", wrap="word"
)
output.tag_config("green",  foreground="#00ff88")
output.tag_config("red",    foreground="#ff6666")
output.tag_config("orange", foreground="#ff9933")
output.tag_config("yellow", foreground="#ffdd44")
output.tag_config("blue",   foreground="#66aaff")
output.tag_config("gray",   foreground="#888888")
output.tag_config("white",  foreground="#ffffff")
output.tag_config("title",  foreground="#00ffaa", font=("Helvetica", 16, "bold"))
# Default (no-error) lines now render pure white per v6.1 spec
output.tag_config("mono",   font=("Menlo", 12), foreground="#ffffff")
output.pack(fill="both", expand=True, padx=18, pady=18)

# -- Controls row ---------------------------------------------------------------
controls_frame = tk.Frame(root, bg="#1e1e1e")
controls_frame.pack(pady=(0, 5))

# Profile selector
tk.Label(controls_frame, text="Profile:", font=("Helvetica", 12),
         fg="#aaaaaa", bg="#1e1e1e").pack(side="left", padx=(0, 8))

profile_menu = tk.OptionMenu(controls_frame, profile_var, *_profile_map.keys())
profile_menu.config(
    font=("Helvetica", 12), bg="#2a2a2a", fg="#ffffff",
    activebackground="#3a3a3a", activeforeground="#ffffff",
    highlightthickness=0, relief="flat", cursor="hand2",
    width=28,
)
profile_menu["menu"].config(
    font=("Helvetica", 12), bg="#2a2a2a", fg="#ffffff",
    activebackground="#00ccff", activeforeground="black",
)
profile_menu.pack(side="left", padx=(0, 20))

# Bias checkbox
bias_check = tk.Checkbutton(
    controls_frame, text="Bias compensation  (-0.10 dB)",
    variable=bias_var, font=("Helvetica", 12),
    fg="#aaaaaa", bg="#1e1e1e", activebackground="#1e1e1e",
    activeforeground="#ffffff", selectcolor="#1e1e1e",
    cursor="hand2"
)
bias_check.pack(side="left")

scan_button = tk.Button(
    root, text="Select Folder → Scan",
    command=select_folder, font=("Helvetica", 19, "bold"),
    bg="#00ccff", fg="black", activebackground="#00eeff",
    pady=28, relief="flat", cursor="hand2"
)
scan_button.pack(pady=35)

tk.Label(root, text="MAD Audio Tools • v7.0 • May 2026", font=("Helvetica", 10),
         fg="#666666", bg="#1e1e1e").pack(side="bottom", pady=20)


# -- Start queue processor & run -----------------------------------------------
root.after(50, _process_log_queue)

# Auto-start scan if launched with --headless
_headless_args = _parse_headless_args()
if _headless_args:
    _hl_folder, _hl_bias, _hl_config = _headless_args
    root.after(100, lambda: _start_scan(_hl_folder, bias_db=_hl_bias, config=_hl_config))

if __name__ == "__main__":
    root.mainloop()
