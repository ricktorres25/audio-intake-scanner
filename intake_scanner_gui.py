# intake_scanner_gui.py
# v8.0 changes:
# - Imports from intake_scanner_core. If the vendored cores fail their start
#   check (or a package is missing), a dialog names the problem and the app quits.
# - run_scan calls the core's scan_folder() and write_reports(): the scan steps
#   moved into the core, and INTAKE_REPORT.jsonl is written beside the text report.
# - The app opens on Platform Band, and --headless without --profile runs it.
#   v7 (configs/default.yaml, the previous version's Default) and Strict stay in
#   the dropdown.
# - Platform Band in the dropdown (from configs/platform/). Under it the bias
#   checkbox is disabled, the summary reads as noise band and level verdict
#   counts, and a profile that fails to load stops the scan instead of falling
#   back to Strict.
# - Finder labels come from the core's finder_label_index(): v7's mapping for
#   v7 and Strict, a band-based mapping for Platform Band.
# - The platform skin: the platform's design tokens, read from
#   cores/platform_tokens/tokens.css in the appearance macOS shows at launch;
#   a plain dark look if they cannot be read.
# - Version labels bumped to 8.0.
#
# v7.0 changes (preserved):
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


def _refuse_to_start(error: Exception):
    """The scanner cannot run: say why in a dialog (the app has no console), then quit."""
    message = f"Intake Scanner cannot start.\n\n{error}"
    print(message, file=sys.stderr)
    try:
        import tkinter as _tk
        from tkinter import messagebox as _messagebox
        _root = _tk.Tk()
        _root.withdraw()
        _messagebox.showerror("Intake Scanner", message)
        _root.destroy()
    except Exception:
        pass
    sys.exit(1)


try:
    # Importing the core imports the vendored cores, which runs their start check.
    from intake_scanner_core import (
        scan_folder, write_reports, build_report_lines,
        load_config, list_builtin_profiles, get_builtin_config_dir,
        DEFAULT_CONFIG, REPORT_VERSION, finder_label_index, CORES_DIR,
    )
except Exception as _start_error:      # VendoredCoreError, or a missing package
    _refuse_to_start(_start_error)


# v8: the app opens on Platform Band, and a headless run without --profile uses
# it. v7 (configs/default.yaml) and Strict stay in the list.
_STARTUP_PROFILE_STEM = "platform_band"


def _resolve_config(profile_arg: str = None) -> dict:
    """Resolve a profile name or YAML path to a config dict.
    Defaults to Platform Band when no profile is specified (v8)."""
    if profile_arg is None:
        profile_arg = _STARTUP_PROFILE_STEM

    # If it looks like a file path, load it directly
    p = Path(profile_arg)
    if p.suffix in (".yaml", ".yml") and p.exists():
        return load_config(p)

    # Otherwise match against builtin profile names (case-insensitive)
    for builtin in list_builtin_profiles():
        if profile_arg.lower() in builtin.stem.lower():
            return load_config(builtin)

    # v8: or a profile's own name, so --profile v7 finds configs/default.yaml
    for builtin in list_builtin_profiles():
        try:
            config = load_config(builtin)
        except Exception:
            continue
        if config.get("profile_name", "").lower() == profile_arg.lower():
            return config

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


# -- Skin ---------------------------------------------------------------------
# The platform's design tokens, read from the vendored copy of its tokens.css
# (cores/platform_tokens/tokens.css, listed in cores/VENDORED.json), in the
# appearance macOS shows at launch. Read once; there is no toggle. If anything is
# missing, the window keeps a plain dark look and the log says why.

# The five status hues. Dark keeps v7's values. Light takes darker versions that
# hold hue and saturation, at the lightest value clearing 4.5:1 contrast on the
# light --card. They are the scanner's own hues, not platform tokens.
_STATUS_HUES = {
    "dark":  {"green": "#00ff88", "red": "#ff6666", "orange": "#ff9933",
              "yellow": "#ffdd44", "blue": "#66aaff"},
    "light": {"green": "#008346", "red": "#e50000", "orange": "#b25900",
              "yellow": "#876f00", "blue": "#006bf0"},
}
_SKIN_TOKENS = ("--paper", "--card", "--line", "--ink", "--ink-soft", "--accent", "--accent-fill")
_ROUNDED_BUTTON = True    # rounded corners on the scan button; set False if they misbehave


def _system_appearance() -> str:
    """'dark' or 'light': the macOS appearance at launch."""
    import subprocess
    try:
        style = subprocess.run(["defaults", "read", "-g", "AppleInterfaceStyle"],
                               capture_output=True, text=True, timeout=2).stdout.strip()
    except Exception:
        return "light"
    return "dark" if style.lower() == "dark" else "light"


def _load_skin():
    """(skin, problem). skin maps paper, card, line, ink, ink_soft, accent,
    accent_fill and the five hues to this launch's colours; None, with the
    reason, when the tokens cannot be read."""
    try:
        css = (CORES_DIR / "platform_tokens" / "tokens.css").read_text(encoding="utf-8")
    except OSError:
        return None, "cores/platform_tokens/tokens.css is missing"
    # Strip comments first: the file's header comment names the dark selector too.
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)

    def rule(selector):
        start = css.find(selector + " {")
        if start < 0:
            return {}
        body = css[css.index("{", start) + 1:css.index("}", start)]
        return {k: v.strip() for k, v in re.findall(r"(--[\w-]+)\s*:\s*([^;]+);", body)}

    theme = _system_appearance()
    tokens = rule(":root")
    if theme == "dark":
        tokens.update(rule('html[data-theme="dark"]'))
    bad = [t for t in _SKIN_TOKENS if not re.fullmatch(r"#[0-9a-fA-F]{6}", tokens.get(t, ""))]
    if bad:
        return None, f"tokens.css has no usable {', '.join(bad)} for the {theme} appearance"
    skin = {t[2:].replace("-", "_"): tokens[t] for t in _SKIN_TOKENS}
    skin.update(_STATUS_HUES[theme])
    skin["theme"] = theme
    return skin, None


SKIN, SKIN_PROBLEM = _load_skin()

# The colours the window uses: a plain dark look, replaced by the skin when it loads.
LOOK = {
    "window": "#1e1e1e", "title": "#00ffaa", "frame": "#0a0a0a", "panel": "#0d0d0d",
    "panel_text": "#d4d4d4", "text": "#ffffff", "muted": "#888888", "label": "#aaaaaa",
    "footer": "#666666", "log_title": "#00ffaa",
    "green": "#00ff88", "red": "#ff6666", "orange": "#ff9933", "yellow": "#ffdd44", "blue": "#66aaff",
}
if SKIN:
    LOOK.update({
        "window": SKIN["paper"], "title": SKIN["ink"], "frame": SKIN["card"], "panel": SKIN["card"],
        "panel_text": SKIN["ink"], "text": SKIN["ink"], "muted": SKIN["ink_soft"],
        "label": SKIN["ink_soft"], "footer": SKIN["ink_soft"], "log_title": SKIN["accent"],
        **{hue: SKIN[hue] for hue in ("green", "red", "orange", "yellow", "blue")},
    })


# -- App window ---------------------------------------------------------------
root = TkinterDnD.Tk()
root.title(f"Intake Scanner {REPORT_VERSION}")
root.configure(bg=LOOK["window"])
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
    """Build mapping of display names to config file paths.
    v8: each profile is listed under its own profile_name, so configs/default.yaml
    reads "v7"; v7 forced that entry's name to "Default"."""
    profiles = {}
    for p in list_builtin_profiles():
        try:
            cfg = load_config(p)
            profiles[cfg.get("profile_name", p.stem)] = p
        except Exception:
            profiles[p.stem] = p
    profiles["Custom YAML..."] = "custom"
    return profiles


_profile_map = _build_profile_map()


def _startup_profile_name() -> str:
    """The dropdown entry the app opens on: Platform Band, under whatever name it
    is listed. If its policy fails to load it is listed by file name and still
    selected, so the scan stops with the error rather than the app opening
    on another profile. Only if the pointer file itself is absent does it fall
    back to the first profile listed."""
    for name, path in _profile_map.items():
        if path not in (None, "custom") and Path(path).stem == _STARTUP_PROFILE_STEM:
            return name
    return next(iter(_profile_map))


profile_var = tk.StringVar(value=_startup_profile_name())


def _is_platform_profile(path) -> bool:
    """True when a profile file is a pointer profile (readout: platform)."""
    try:
        import yaml
        with open(path, "r") as f:
            return (yaml.safe_load(f) or {}).get("readout") == "platform"
    except Exception:
        return False


def _get_active_config() -> dict | None:
    """Load the currently selected profile config.
    v8: a Platform Band profile that fails to load returns None and the scan
    stops; it never falls back to another policy. The editing profiles
    keep v7's fallback to the built-in strict thresholds."""
    selected = profile_var.get()

    if selected == "Custom YAML...":
        config_path = _custom_config_path if _custom_config_path and _custom_config_path.exists() else None
    else:
        config_path = _profile_map.get(selected)
        if config_path == "custom":
            config_path = None
    if config_path is None:
        log("Profile not found — using built-in strict thresholds\n", "yellow")
        return DEFAULT_CONFIG

    try:
        return load_config(config_path)
    except Exception as e:
        if _is_platform_profile(config_path):
            log(f"Failed to load profile '{selected}': {e}\nThe scan did not run.\n", "red")
            return None
        log(f"Failed to load profile '{selected}': {e} — using built-in strict thresholds\n", "yellow")
        return DEFAULT_CONFIG


def _sync_bias_toggle():
    """The bias toggle does not apply under Platform Band: the platform has no bias."""
    selected = profile_var.get()
    path = _custom_config_path if selected == "Custom YAML..." else _profile_map.get(selected)
    platform = path not in (None, "custom") and _is_platform_profile(path)
    bias_check.config(state=tk.DISABLED if platform else tk.NORMAL,
                      text="Bias compensation  (not used by Platform Band)" if platform
                      else "Bias compensation  (-0.10 dB)")


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
                profile_var.set(_startup_profile_name())
        else:
            # User cancelled — revert to the profile the app opens on
            profile_var.set(_startup_profile_name())
    _sync_bias_toggle()


profile_var.trace_add("write", _on_profile_change)


# -- Tag colorization ---------------------------------------------------------
_active_config: dict = DEFAULT_CONFIG  # set at scan start for color decisions

def _warn_color() -> str:
    """Return the color tag for [WARN] flags — red for strict, orange for default.
    v8: orange under Platform Band too, matching its WARN label, and for v7
    (configs/default.yaml's profile, renamed from Default)."""
    if _active_config.get("readout") == "platform":
        return "orange"
    name = _active_config.get("profile_name", "").lower()
    return "orange" if ("default" in name or name == "v7") else "red"


def _platform_color(line_upper: str) -> str:
    """v8, Platform Band: the colour of the worse readout on a Platform: or >>> line."""
    if "NOISE TAKE" in line_upper:
        return "blue"
    if "REJECT" in line_upper or "LEVEL FAIL" in line_upper or "ERROR" in line_upper:
        return "red"
    if "WARN" in line_upper:
        return "orange"
    if "CAUTION" in line_upper or "HOT" in line_upper or "QUIET" in line_upper:
        return "yellow"
    if "NO BAND" in line_upper:
        return "white"
    return "green"

def get_tag_for_line(line: str) -> str:
    """Pick a color tag for a report line.
    v6.1 convention: non-error lines are white; error/flag lines use their
    severity color. The Disposition and 'N files contain TP CLIPs' lines are
    handled separately with per-segment coloring, not here.
    """
    line_upper = line.upper()
    stripped = line.strip()

    # v8 Platform Band: the per-file readout line and its footer
    if stripped.startswith("Platform:") or stripped.startswith(">>> noise") \
            or stripped.startswith(">>> NOISE TAKE"):
        return _platform_color(line_upper)
    if line.startswith("Format issues (informational"):
        return "yellow"

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
        if "[REJECT]" in line or "[TOO_QUIET]" in line:
            return "red"
        if "[NOISE PROFILE]" in line:
            return "blue"
        if "[QUIET]" in line:
            return "yellow"
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

    # v8: a noise take of digital silence that is not used — a warning, not a baseline
    if line.startswith("Noise take not used"):
        return "yellow"

    # Noise profile baseline line — keep blue as its established convention
    if line.startswith("Noise profile:"):
        return "blue"

    # Missing noise profile warnings stay yellow (cautionary, not error)
    if line.startswith("WARNING:") or "without folder noise profile" in line \
            or "without a folder noise take" in line:
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
    v8: the mapping lives in the core (finder_label_index) so it can be checked
    without a window. v7 and Strict keep v7's colours; under Platform Band
    a file with no answer gets no label."""
    return finder_label_index(r, config)

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
def run_scan(folder_path: Path, bias_db: float = 0.0, config: dict = None):
    global _active_config
    if config is None:
        config = DEFAULT_CONFIG
    _active_config = config
    platform = config.get("readout") == "platform"

    try:
        log_clear()
        profile = config.get("profile_name", "Custom")
        log(f"Intake Scanner v{REPORT_VERSION} — {profile}\n", "title")
        log("=" * 70 + "\n", "title")
        log(f"Scanning folder:\n{folder_path}\n")
        if platform:
            log("Platform Band: the bias toggle does not apply (the platform has no bias)\n", "gray")
        elif bias_db != 0.0:
            log(f"Bias compensation: {bias_db:+.2f} dB (effective TP CLIP threshold: {bias_db:+.2f} dBTP)\n", "yellow")

        # v8: the scan steps (noise takes, per-file scans, peak outliers) live in
        # the core's scan_folder(). Its progress lines are v7's live log lines.
        results = scan_folder(folder_path, config=config, bias_db=bias_db, progress=log)
        if not results:
            return

        try:
            report_path, rows_path = write_reports(results, folder_path, bias_db=bias_db, config=config)
        except Exception as e:
            log(f"\nFAILED to save report: {e}\n", "red")
            return

        log_clear()
        # Route special lines through their own colorizers; everything else
        # falls through to get_tag_for_line.
        in_skipped_block = False
        block_color = "red"
        for line in build_report_lines(results, bias_db=bias_db, config=config):
            if line.startswith("Disposition:"):
                _log_disposition_line(line)
                in_skipped_block = False
                continue
            if re.match(r"\d+ files contain TP CLIPs", line):
                _log_tpclips_line(line)
                in_skipped_block = False
                continue
            if line.startswith("Skipped (format mismatch)") or line.startswith("Format issues (informational"):
                # v8: under Platform Band a format issue is informational (yellow).
                block_color = "red" if line.startswith("Skipped") else "yellow"
                log(line + "\n", block_color)
                in_skipped_block = True
                continue
            # Detail lines of the skipped block are 2-space-indented right
            # after the header; end the block on the first non-indented line.
            if in_skipped_block and line.startswith("  "):
                log(line + "\n", block_color)
                continue
            in_skipped_block = False
            log(line + "\n")

        if platform:
            # v8: Platform Band reads as two counts, noise band and level verdict.
            read_out = [r for r in results if (r.get("platform") or {}).get("level")]
            bands = {}
            for r in read_out:
                noise = r["platform"]["noise"]
                key = noise["disposition"] if noise else "no band"
                bands[key] = bands.get(key, 0) + 1
            n_fail = sum(1 for r in read_out if r["platform"]["level"]["verdict"] == "FAIL")
            log(f"\nSCAN COMPLETE — noise band {bands.get('PASS', 0)} PASS / "
                f"{bands.get('CAUTION', 0)} CAUTION / {bands.get('WARN', 0)} WARN / "
                f"{bands.get('REJECT', 0)} REJECT / {bands.get('no band', 0)} no band"
                f" — level {len(read_out) - n_fail} PASS / {n_fail} FAIL\n", "title")
        else:
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
        log(f"Rows saved:   {rows_path}\n", "title")
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
    if bias_db is None:
        bias_db = -0.10 if bias_var.get() else 0.0
    if config is None:
        config = _get_active_config()
        if config is None:          # v8: a Platform Band profile that failed to load
            return
    _scan_in_progress = True
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
import tkinter.font as tkfont


def _sys_font(size: int, weight: str = "normal"):
    """The macOS system font at a size (the skin's labels and controls use it)."""
    return (tkfont.nametofont("TkDefaultFont").actual("family"), size, weight)


class AccentButton(tk.Canvas):
    """The scan button under the skin, drawn by hand on --accent-fill with a white
    label. config(state=..., text=...) works as on a tk.Button, so the scan
    code calls it the same way. Disabled: --line fill, --ink-soft label. No hover
    colour, because dark --accent under white measures 3.59:1 and fails."""

    def __init__(self, parent, text, command, font, width, height=72, radius=14):
        super().__init__(parent, width=width, height=height, bg=SKIN["paper"],
                         highlightthickness=0, bd=0)
        self._text, self._command, self._font = text, command, font
        self._state = tk.NORMAL
        self._radius = radius if _ROUNDED_BUTTON else 0
        self.bind("<Button-1>", self._click)
        self._draw()

    def _draw(self):
        self.delete("all")
        w, h, r = int(self["width"]) - 1, int(self["height"]) - 1, self._radius
        enabled = self._state != tk.DISABLED
        fill = SKIN["accent_fill"] if enabled else SKIN["line"]
        if r:
            # A smoothed polygon with doubled corner points draws a rounded rectangle.
            points = [r, 0, r, 0, w - r, 0, w - r, 0, w, 0, w, r, w, r, w, h - r, w, h - r,
                      w, h, w - r, h, w - r, h, r, h, r, h, 0, h, 0, h - r, 0, h - r,
                      0, r, 0, r, 0, 0]
            self.create_polygon(points, smooth=True, fill=fill, outline=fill)
        else:
            self.create_rectangle(0, 0, w, h, fill=fill, outline=fill)
        self.create_text(w // 2, h // 2, text=self._text, font=self._font,
                         fill="#ffffff" if enabled else SKIN["ink_soft"])
        self.configure(cursor="hand2" if enabled else "arrow")

    def _click(self, _event):
        if self._state != tk.DISABLED:
            self._command()

    def config(self, cnf=None, **kw):
        """Take a tk.Button's state= and text=; pass anything else to the canvas."""
        redraw = "state" in kw or "text" in kw
        self._state = kw.pop("state", self._state)
        self._text = kw.pop("text", self._text)
        if cnf or kw:
            super().configure(cnf, **kw)
        if redraw:
            self._draw()

    configure = config


title_label = tk.Label(root, text="Intake Scanner",
                       font=("Georgia", 36) if SKIN else ("Helvetica", 36, "bold"),
                       fg=LOOK["title"], bg=LOOK["window"], pady=50)
title_label.pack()

if SKIN:
    # The report panel: a --card ground with a --line edge.
    frame = tk.Frame(root, bg=LOOK["frame"], bd=0, highlightthickness=1,
                     highlightbackground=SKIN["line"], highlightcolor=SKIN["line"])
else:
    frame = tk.Frame(root, bg=LOOK["frame"], bd=3, relief="sunken")
frame.pack(padx=40, pady=(0, 40), fill="both", expand=True)

output = scrolledtext.ScrolledText(
    frame, font=("Menlo", 13), bg=LOOK["panel"], fg=LOOK["panel_text"],
    insertbackground=LOOK["panel_text"], state=tk.DISABLED, relief="flat", wrap="word",
    **({"highlightthickness": 0} if SKIN else {})
)
for _hue in ("green", "red", "orange", "yellow", "blue"):
    output.tag_config(_hue, foreground=LOOK[_hue])
output.tag_config("gray",   foreground=LOOK["muted"])
output.tag_config("white",  foreground=LOOK["text"])
# Under the skin the report stays a monospace log, its title lines included.
output.tag_config("title",  foreground=LOOK["log_title"],
                  font=("Menlo", 13, "bold") if SKIN else ("Helvetica", 16, "bold"))
# Default (no-error) lines now render pure white per v6.1 spec
output.tag_config("mono",   font=("Menlo", 12), foreground=LOOK["text"])
output.pack(fill="both", expand=True, padx=18, pady=18)

# -- Controls row ---------------------------------------------------------------
controls_frame = tk.Frame(root, bg=LOOK["window"])
controls_frame.pack(pady=(0, 5))

_label_font = _sys_font(13) if SKIN else ("Helvetica", 12)

# Profile selector
tk.Label(controls_frame, text="Profile:", font=_label_font,
         fg=LOOK["label"], bg=LOOK["window"]).pack(side="left", padx=(0, 8))

profile_menu = tk.OptionMenu(controls_frame, profile_var, *_profile_map.keys())
if SKIN:
    # The dropdown stays native; only its surround takes the window ground.
    profile_menu.config(width=28, cursor="hand2", highlightthickness=0, bg=LOOK["window"])
else:
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
if SKIN:
    # The checkbox stays native; its label takes the ink on the window ground.
    bias_check = tk.Checkbutton(
        controls_frame, text="Bias compensation  (-0.10 dB)",
        variable=bias_var, font=_label_font,
        fg=SKIN["ink"], bg=LOOK["window"], activebackground=LOOK["window"],
        highlightthickness=0, cursor="hand2"
    )
else:
    bias_check = tk.Checkbutton(
        controls_frame, text="Bias compensation  (-0.10 dB)",
        variable=bias_var, font=("Helvetica", 12),
        fg="#aaaaaa", bg="#1e1e1e", activebackground="#1e1e1e",
        activeforeground="#ffffff", selectcolor="#1e1e1e",
        cursor="hand2"
    )
bias_check.pack(side="left")
_sync_bias_toggle()

if SKIN:
    _button_font = _sys_font(19, "bold")
    scan_button = AccentButton(
        root, text="Select Folder → Scan", command=select_folder, font=_button_font,
        width=tkfont.Font(font=_button_font).measure("Select Folder → Scan") + 120,
    )
else:
    scan_button = tk.Button(
        root, text="Select Folder → Scan",
        command=select_folder, font=("Helvetica", 19, "bold"),
        bg="#00ccff", fg="black", activebackground="#00eeff",
        pady=28, relief="flat", cursor="hand2"
    )
scan_button.pack(pady=35)

tk.Label(root, text=f"MAD Audio Tools • v{REPORT_VERSION} • October 2026",
         font=_sys_font(11) if SKIN else ("Helvetica", 10),
         fg=LOOK["footer"], bg=LOOK["window"]).pack(side="bottom", pady=20)

if SKIN_PROBLEM:
    log(f"Skin not applied: {SKIN_PROBLEM}. The window keeps a plain dark look.\n", "yellow")


# -- Start queue processor & run -----------------------------------------------
root.after(50, _process_log_queue)

# Auto-start scan if launched with --headless
try:
    _headless_args = _parse_headless_args()
except Exception as _headless_error:     # v8: say why in the window, not a silent crash
    _headless_args = None
    log(f"Headless scan not started: {_headless_error}\n", "red")
if _headless_args:
    _hl_folder, _hl_bias, _hl_config = _headless_args
    root.after(100, lambda: _start_scan(_hl_folder, bias_db=_hl_bias, config=_hl_config))

if __name__ == "__main__":
    root.mainloop()
