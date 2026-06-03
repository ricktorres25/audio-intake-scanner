# audio-intake-scanner

Audio batch QC tool for production pipeline intake. Built to validate RAW audio files before they enter an audio data prep workflow.

The scanner walks a folder of audio files and reports format compliance (mono, 48 kHz, 24-bit WAV), true-peak level with 4x oversampling and optional clip-bias compensation, A-weighted noise floor and SNR per IEC 61672, crest factor, and integrated LUFS per BS.1770-4. Each file is sorted into pass, salvageable, or reject driven by YAML threshold profiles. The scanner detects per-folder noise profile files (named `-blank` or `-noise`, underscore variants also recognized) and uses them as the SNR reference; when no named profile exists, it falls back to a content-based scan that picks the lowest crest-factor WAV in the folder as the noise reference. Output is a plain-text report plus, on macOS, Finder color labels per file so bad files are visible in the Finder before you open the report.

## Why use it

If you receive batches of audio from external recording or production sources and need to triage them before they reach your pipeline, this saves the per-file manual check. The three-way pass / salvageable / reject sort tells you which files are good, which are worth fixing, and which to send back for re-record.

## How it works

The tool is split into two files. `intake_scanner_core.py` holds the measurement and sorting logic with no UI in it, so the same code can be called from the GUI, from a batch run, or from another script. `intake_scanner_gui.py` is a thin Tkinter wrapper around it.

Every file goes through the same measurement pass: read the audio, check the format against spec, measure true peak on a 4x upsampled signal, run the A-weighting filter to get a noise floor and SNR, compute crest factor, and measure integrated loudness. The per-file numbers are then compared against a threshold profile, and each file lands in one of three buckets. Reject takes priority, then salvageable, then pass. Noise profile files are scored separately and used as the SNR reference rather than judged as program material.

Thresholds are not hardcoded. They live in YAML profiles in `configs/` (Default for relaxed thresholds, Strict for research-grade), so the same scanner can be pointed at a different spec without touching the code.

## The stack

The library choices are deliberate, not defaults.

`scipy.signal.resample_poly` does the 4x upsampling for true-peak detection, so inter-sample peaks that sit between two samples are caught instead of missed. The A-weighting filter is built with `bilinear_zpk` and `zpk2sos` and applied with `sosfilt`, which is the numerically stable way to run an IIR filter. `pyloudnorm` provides the BS.1770-4 loudness measurement. `soundfile` handles WAV I/O, with `pydub` (and `ffmpeg`) as the fallback for non-WAV inputs. `numpy` underlies all of the signal math, and `pyyaml` loads the threshold profiles.

## Requirements

Python 3.10 or newer, plus the packages in `requirements.txt` (`numpy`, `scipy`, `soundfile`, `pydub`, `pyloudnorm`, `tkinterdnd2`). `pydub` requires `ffmpeg` available on `PATH` for non-WAV inputs.

The Finder color label step uses AppleScript and is macOS-only. The scan and report writing should run on Linux and Windows, but those platforms have not been tested.

## How to run

```
pip install -r requirements.txt
python intake_scanner_gui.py
```

The window opens; drag and drop a folder onto it, or use the Select Folder button. Pick a profile from the dropdown, toggle clip-bias compensation if you want the -0.10 dB allowance applied to the true-peak ceiling, and the scan starts on a background thread. Output appears in the window and a text report is written to `INTAKE_REPORT.txt` at the scanned folder.

For batch use:

```
python intake_scanner_gui.py --headless /path/to/folder --profile strict
```

This auto-runs the scan and writes the report, but still requires a display because the scan logic depends on the Tk event loop. See Known limitations below.

## Example output

The report opens with a summary block (thresholds in effect, the disposition counts, and the spread of each measurement across the folder), then one block per file. The run below was a deliberately low-level test batch, so nothing cleared the pass threshold; it shows the salvageable and reject buckets and the per-file reasoning. File paths are generic placeholders here.

```
INTAKE SCAN v7.0 — 16 files
Profile: Default
Generated: 2026-05-24 18:19:27
Noise floor / SNR: A-weighted (IEC 61672)
SNR thresholds: reject < 38.0 dB | warn < 48.0 dB | caution < 55.0 dB | pass ≥ 55.0 dB
LUFS thresholds: reject < -42.0 LUFS | caution < -36.0 LUFS
CF thresholds: warn ≤ 10.0 dB | caution ≤ 13.0 dB
Disposition: 0 pass / 12 salvageable / 4 reject
0 files contain TP CLIPs
Noise floor across files: worst -95.0 dBA / best -95.0 dBA / median -95.0 dBA
SNR across files: worst +0.0 dB / best +51.2 dB / median +49.4 dB
LUFS across files: lowest -42.6 LUFS / highest -38.4 LUFS / median -40.3 LUFS
Crest factor across files: min 12.0 dB / max 23.2 dB / median 18.8 dB
CF flags: 0 WARN / 1 CAUTION
SNR flags: 0 REJECT / 3 WARN / 12 CAUTION
LUFS flags: 3 REJECT / 12 CAUTION
Peak range: -17.1 dBTP (intake_demo/take-10.wav) to -72.7 dBTP (intake_demo/room-tone.wav) — spread 55.6 dB
Peak outliers (>5.0 dB below folder max): 5
================================================================================

FILE: intake_demo/take-04.wav
Disposition: REJECT
Format: 24-bit | Mono | 48000 Hz
Duration: 00:27.360
Highest measured peak: -21.20 dBFS
RMS:          -42.4 dBFS
Crest factor: +21.2 dB
Noise floor:  -95.0 dBA (ref)
SNR:          +47.4 dB  [WARN] — noise likely audible after normalization — possibly salvageable with manual editing
LUFS:         -42.5 LUFS  [REJECT] — speech level too low — unsalvageable without heavy amplification
TRUE PEAK ✓ CLEAN
>>> REJECT
--------------------------------------------------------------------------------

FILE: intake_demo/take-01.wav
Disposition: SALVAGEABLE
Format: 24-bit | Mono | 48000 Hz
Duration: 00:39.120
Highest measured peak: -19.44 dBFS
RMS:          -39.6 dBFS
Crest factor: +20.2 dB
Noise floor:  -95.0 dBA (ref)
SNR:          +49.6 dB  [CAUTION] — noise may be faintly audible — candidate for automated de-noise
LUFS:         -39.4 LUFS  [CAUTION] — speech level low — may need gain adjustment
TRUE PEAK ✓ CLEAN
>>> SALVAGEABLE
--------------------------------------------------------------------------------

(… 14 more file blocks omitted …)
```

On macOS the same dispositions are written as Finder color labels, so the reject and salvageable files stand out in the folder before you open the report.

## Known limitations

The `--headless` flag is misnamed. It auto-runs the scan without manual interaction, but it still spins up a Tkinter window and requires a display server. A true headless mode is planned.

The threshold profiles ship calibrated against a single known-quality test corpus from April 2026. The SNR, LUFS, and content-based noise-profile thresholds all reflect that corpus, so a different recording domain will likely need its own profile. The thresholds live in YAML for exactly this reason, but the shipped values are a starting point, not a universal standard.

There are no automated tests yet. Behavior has been checked by running against the test corpus and comparing the output to the spec, but there is no `tests/` folder in the repo.
