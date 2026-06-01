# audio-intake-scanner

Audio batch QC tool for production pipeline intake. Built for validation of RAW audio files in an audio data prep workflow.

The scanner walks a folder of audio files and reports format compliance (mono, 48 kHz, 24-bit WAV), true-peak level with 4x oversampling and optional clip-bias compensation, A-weighted noise floor and SNR per IEC 61672, crest factor, and integrated LUFS per BS.1770-4. Each file is sorted into pass, salvageable, or reject buckets driven by YAML threshold profiles. The scanner detects per-folder noise profile files (named `-blank` or `-noise`, underscore variants also recognized) and uses them as the SNR reference; when no named profile exists, it falls back to a content-based scan that picks the lowest crest-factor WAV in the folder as the noise reference. Output is a plain-text report plus, on macOS, Finder color labels per file so bad files are visible in the Finder before opening the report.

## Why use it

If you receive batches of audio from external recording or production sources and need to triage them before they reach your pipeline, this saves the per-file manual check. The three-way pass / salvageable / reject sort tells you which files are good, which are worth fixing, and which to send back for re-record.

## Requirements

Python 3.10 or newer, plus the packages in `requirements.txt` (`numpy`, `scipy`, `soundfile`, `pydub`, `pyloudnorm`, `tkinterdnd2`). `pydub` requires `ffmpeg` available on `PATH` for non-WAV inputs.

The Finder color label step uses AppleScript and is macOS-only. The scan and report writing should run on Linux and Windows, but those platforms have not been tested.

## How to run

```
pip install -r requirements.txt
python intake_scanner_gui.py
```

The window opens; drag and drop a folder onto it, or use the Select Folder button. Pick a profile from the dropdown (Default for relaxed thresholds, Strict for research-grade), toggle clip-bias compensation if you want the -0.10 dB allowance applied to the true-peak ceiling, and the scan starts on a background thread. Output appears in the window and a text report is written to `INTAKE_REPORT.txt` at the scanned folder.

For batch use:

```
python intake_scanner_gui.py --headless /path/to/folder --profile strict
```

This auto-runs the scan and writes the report, but still requires a display because the scan logic depends on the Tk event loop. True command-line headless is on the list for a later version.

## Known limitations

The `--headless` flag is misnamed. It auto-runs the scan without manual interaction, but it still spins up a Tkinter window and requires a display server. A true headless mode is planned.

The threshold profiles ship calibrated against a single known-quality test corpus from April 2026. SNR, LUFS, and the content-based noise profile crest-factor ceiling all reflect that corpus. Different domains will likely need different thresholds.

No automated tests yet. Behavior has been validated by running against the test corpus and checking output against the spec, but there is no `tests/` folder in the repo.

## Status

Work in progress. A v2 README with sample input and output and a tighter limitations write-up is planned.
