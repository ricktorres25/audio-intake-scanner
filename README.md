# audio-intake-scanner

Audio batch QC for production pipeline intake. Give it a folder and it measures every audio file inside, writes two reports into that folder, and sets a Finder label on each file.

Version 8 has three profiles:
- **Platform Band**, the default, reads each file the way a voice-data collection platform's Rec QC stages 1 and 2 would, using copies of that platform's own two cores.
- **v7** and **Strict** answer the editing question, how much work a file needs, with version 7's numbers.

## Why use it

If you receive batches of audio from external recording or production sources and need to triage them before they reach your pipeline, this saves the per-file manual check.
- The editing sort says which files are ready, which are worth fixing, and which to send back for a re-record.
- Platform Band says how each file would fare at the platform's own recording-quality gate.

Every number it uses is a standard measurement, and every decision comes from a stated threshold:
- A-weighted noise floor and SNR (IEC 61672);
- integrated loudness (ITU-R BS.1770-4);
- 4x-oversampled true peak.

Nothing is a learned score.

## Quick start

```
pip install -r requirements.txt
python intake_scanner_gui.py
```

Drop a folder on the window, or click **Select Folder → Scan**. The window opens on Platform Band.

For batch use: `python intake_scanner_gui.py --headless /path/to/folder [--profile v7|strict|<file.yaml>] [--bias]`. See [Headless](#headless).

The output is `INTAKE_REPORT.txt` and `INTAKE_REPORT.jsonl` in the scanned folder, plus a Finder label on every scanned file.

## What changed in v8

- **Platform Band**, a third profile that runs the platform's two Rec QC cores, vendored under `cores/`. The app opens on it, and a headless run without `--profile` uses it.
- **Default is now called v7.** `configs/default.yaml` keeps its file name, so `--profile default` still works, as does `--profile v7`.
- **The clip fix.** A clip is decided on the whole-file true peak. v7 missed an over whose samples all sit below −0.2 dBFS. See [Clipping](#clipping).
- **Noise takes.** A take found by content scan is treated as a noise take; v7 scanned it as content against its own floor and rejected it. `_np`, the name the noise core gives a trimmed take, counts as a noise-take name. A take of digital silence is never used. See [Noise takes](#noise-takes).
- **FLAC is measured like WAV.** A 24-bit, mono, 48 kHz `.flac` is no longer skipped.
- **The estimate for a folder with no noise take** is the quietest steady stretch of each file, not v7's silence gate. It is still a number only. See [No noise take](#no-noise-take-the-per-file-estimate).
- **`INTAKE_REPORT.jsonl`**: one row per file, with relative paths, the noise-floor method and confidence, and the policy hashes. See [Reports](#reports).
- **The text report.** `FILE:` paths are relative to the scanned folder, error lines no longer carry the folder's absolute path, and the header states once that RT60 and drift are not measured. Under v7 and Strict every other line keeps its v7 shape.
- **The platform skin.** The window takes the platform's design tokens, light or dark.
- **PyYAML** is in `requirements.txt`.

## How it is verified

- **Against the platform.** The Platform Band path was run on every stored stage result in the platform's development data: 51 readings from 17 auditions, each under the policy that scored it. Every number matched bit for bit, apart from SNR, which differed by under 1e-13 dB. Every verdict, band and policy hash agreed.
- **Against v7.** On a 113-file test corpus, v8 under v7 and Strict sorts every file as v7 did, apart from the noise takes v7 rejected. The numbers differ only on those takes and in one folder whose noise take is digital silence.
- **The test suite.** The suite behind these claims is in a private repository; every check in it is first shown to fail on a near miss.

## Profiles

| Profile | File | Answers | Per file |
|---|---|---|---|
| **Platform Band** | `configs/platform/platform_band.yaml` | What Rec QC stages 1 and 2 would say | A noise band and a level verdict |
| **v7** | `configs/default.yaml` | How much editing work the file needs | pass, salvageable or reject |
| **Strict** | `configs/strict.yaml` | The same, with higher SNR lines | pass, salvageable or reject |

Choose in the dropdown, or with `--profile` headless. **Custom YAML...** loads any file in the shape of `strict.yaml`; keys it leaves out take Strict's values.

### v7 and Strict: the editing sort

The numbers are v7's, calibrated in April 2026 against a known-quality test corpus. The sort aims at a common delivery target, −6 to −3 dBFS sample peak and −23 to −20 LUFS integrated: a flagged problem predicts a failure against it after normalization.

| Metric | v7 | Strict |
|---|---|---|
| SNR reject | below 38 dB | below 45 dB |
| SNR warn | 38 to 48 dB | 45 to 53 dB |
| SNR caution | 48 to 55 dB | 53 to 60 dB |
| SNR pass | 55 dB and up | 60 dB and up |
| LUFS reject | below −42 | below −42 |
| LUFS caution | −42 up to −36 | −42 up to −36 |
| Crest factor warn (over-compressed) | 10 dB or lower | 10 dB or lower |
| Crest factor caution | 13 dB or lower | 13 dB or lower |
| True-peak clip | 0.0 dBTP or higher (−0.10 with bias) | the same |
| Near-clip note | −0.5 dBTP or higher | the same |

Only a low crest factor (over-compression) is flagged; a high one is left to the operator.

The worst flag decides:

- **reject**: an SNR reject, a true-peak clip, a format failure or a LUFS reject.
- **salvageable**: an SNR warn or caution, a crest-factor warn, or a LUFS caution.
- **pass**: anything else. A crest-factor caution, a near-clip note and a peak outlier do not move a file out of pass.

A file that cannot be read is reported as an error and labelled red.

**SNR sorts only against a noise take.** Without one, SNR is shown as informational and never sorts. See [No noise take](#no-noise-take-the-per-file-estimate).

**Format.** The scanner finds `.wav`, `.m4a`, `.mp3`, `.aiff`, `.flac` and `.aac` files in the folder and all its subfolders. Only 24-bit, mono, 48 kHz `.wav` or `.flac` is measured. Anything else is skipped, listed once at the top of the report with the reason, and labelled red.

**Peak outlier.** A content file whose sample peak is more than 5 dB below the loudest file in its folder gets a `PEAK: [CAUTION]` line, and a blue label if it passes. Noise takes are left out.

### Platform Band: the platform readout

For each content file the scanner makes the platform's two public calls, on its own vendored copies of the cores:

```python
level = level_core.scan_phrase(file, write_outputs=False)
noise = noise_core.scan_noise_profile(noise_take, modality="audition",
            speech_level_dba=level["speech_level_dba"],
            config=<the Platform Band policy>, write_outputs=False)
```

`write_outputs=False` means nothing is written next to the audio and the cores keep no log. Neither argument changes the scoring.

The scanner copies the cores' records as they are and applies no rule of its own.

- **Noise band** (noise core).
  - SNR is the content file's A-weighted speech level minus the noise take's A-weighted floor, measured on the take's longest continuous run free of impulses.
  - The gate runs in order: wrong format, a take shorter than 3 s, a clean run shorter than 3 s, then the SNR band. The first three reject whatever the SNR is.
  - The bands are REJECT (below 20 dB), WARN (20 to 48), CAUTION (48 to 55) and PASS (55 and up). Only REJECT fails the stage.
- **Level verdict** (level core).
  - `CLIP` when the whole-file true peak is at or above 0.0 dBTP.
  - `HOT` when the sample peak is above −3 dBFS.
  - `TOO_QUIET` below −42 LUFS, and `QUIET` at or below −36 LUFS.
  - FAIL on CLIP, on TOO_QUIET, or on HOT together with QUIET. HOT alone or QUIET alone passes with a flag.

**The numbers are not in the scanner.** `configs/platform/platform_band.yaml` is a pointer: it names the noise core's own policy file and leaves the level core on its built-in policy. The report header prints the lines in force, read from the cores, with each policy's name and hash. A missing policy file stops the scan; it never falls back to another policy.

**What a row means.** The platform measures speech level on a dedicated level-check phrase; the scanner measures it on each content file with the same formula. A row answers: which band would this file land in if it were the level-check take in this room.

**The scanner's own pass still runs** for what the cores do not cover: the format check, clip timecodes, crest factor and peak outliers. Under Platform Band these are informational and never change a band or a verdict. The bias toggle does nothing, because the platform has no bias.

One boundary differs from v7: at exactly −36.0 LUFS the v7 profile passes a file and the level core calls it QUIET.

An excerpt from a test folder, with its file names replaced:

```
INTAKE SCAN v8.0 — 16 files
Profile: Platform Band
Generated: 2026-10-02 10:43:36
Noise floor / SNR: A-weighted (IEC 61672)
RT60 and drift: n/a — not measured here (RT60 needs a clap recording; drift needs a re-record and a kept take)
Platform readout: Rec QC stages 1 and 2, by the platform's own cores — bands and verdicts
Noise bands: reject < 20.0 dB | warn < 48.0 dB | caution < 55.0 dB | pass ≥ 55.0 dB (noise core policy "Platform Band", 20e56786)
Level check: clip at ≥ 0.0 dBTP | hot above -3.0 dBFS | too quiet below -42.0 LUFS | quiet at or below -36.0 LUFS (level core policy "default", 881b9757)
Noise band: 0 PASS / 12 CAUTION / 3 WARN / 0 REJECT / 0 no band
Level verdict: 12 PASS / 3 FAIL — flags 0 HOT, 12 QUIET
Noise floor (clean run) across files: worst -95.0 dBA / best -95.0 dBA / median -95.0 dBA
SNR across files: worst +47.0 dB / best +51.2 dB / median +49.5 dB
LUFS across files: lowest -42.6 LUFS / highest -38.4 LUFS / median -40.3 LUFS
Noise profile: room-tone.wav — baseline -95.0 dBA (clean run 5.2 s, 0 impulse(s), found by content scan, accepted by the noise core)
================================================================================

FILE: take-10.wav
Platform: noise WARN | level FAIL (TOO_QUIET)
Format: 24-bit | Mono | 48000 Hz
Duration: 00:27.360
Highest measured peak: -21.20 dBFS
True peak:    -21.2 dBTP (whole file)
RMS:          -42.4 dBFS
Crest factor: +21.2 dB (informational)
Noise floor:  -95.0 dBA (clean run)
SNR:          +47.4 dB  [WARN]
LUFS:         -42.5 LUFS  [TOO_QUIET]
TRUE PEAK ✓ CLEAN
>>> noise WARN | level FAIL
--------------------------------------------------------------------------------

(… 14 more file blocks …)

FILE: room-tone.wav
Platform: noise take — accepted by the noise core
Format: 24-bit | Mono | 48000 Hz
Duration: 00:05.206
Highest measured peak: -72.66 dBFS
True peak:    -72.4 dBTP (whole file)
RMS:          -84.6 dBFS
Noise floor:  -95.0 dBA (clean run)
TRUE PEAK ✓ CLEAN
>>> NOISE TAKE
--------------------------------------------------------------------------------
```

## Noise takes

Each folder that holds audio is searched for its own noise take, in two passes.

1. **By name.** A file whose name, before the extension, ends in `-blank`, `_blank`, `-noise`, `_noise` or `_np`. The first in sorted order wins. `_np` is the name the noise core gives a take trimmed to its clean run. A raw take and its trimmed copy (`room_noise.wav` and `room_np.wav`) sort with the raw take first, so the raw take stays the reference.
2. **By content**, when no name matches: the WAV or FLAC with the lowest crest factor at or below 14.65 dB. Files at or below −90 dBFS RMS are skipped as silent.

A noise take is never sorted as content. It shows as `NOISE PROFILE` (v7, Strict) or `NOISE TAKE` (Platform Band), gets a gray label and stays out of the counts.

**A take of digital silence is never the reference**, under any profile.
- Digital silence means a whole-file A-weighted floor at or below −110 dBA. Real rooms measured so far sit between −80 and −95 dBA.
- The scanner passes over the silent take to the next noise-named file or the content scan, or else reads the folder as having no noise take.
- The file still shows as the noise take, and the report adds `Noise take not used: … digital silence`.
- The platform's own noise core accepts such a take, but live capture cannot be digitally silent; an intake folder has no such guarantee.

The floor:

- **v7 and Strict** use the take's whole-file A-weighted RMS, as v7 did. The noise core also checks the take, only to label the rows. A take it rejects still sets the floor for its folder, as in v7, and those rows say `low`.
- **Platform Band** uses the noise core's floor on the take's longest clean run. If the core rejects the take, every content file in its folder reads noise REJECT with the core's reason.

Every row says how far to trust its floor:

| `noise_floor_confidence` | When |
|---|---|
| `high` | A noise take found by name, accepted by the noise core |
| `medium` | A noise take found by content scan, accepted by the noise core |
| `low` | No noise take: the per-file estimate. Under v7 and Strict, also a take the noise core rejects. |
| `none` | No floor: an error, a skipped file, nothing measurable, a take of digital silence, or, under Platform Band, a take the core rejects |

`noise_floor_method` is `noise_take_clean_run` (Platform Band), `noise_take_full_file` (v7, Strict), `per_file_quiet_window` or `unavailable`.

### No noise take: the per-file estimate

With no noise take in a folder, each content file's floor is the quietest steady stretch of the file itself.
- The file is measured in 50 ms A-weighted frames, and the floor is the quietest 0.3 s whose frames vary by 2 dB or less.
- Digital silence, and the 50 ms either side of it, is ignored, so the gaps in an edited or gated file do not pass for the room.
- A file with no such stretch gets no estimate, and its SNR reads N/A.

The number is shown with `low` confidence. It never sorts a file (v7, Strict), never gives a band (Platform Band), and never decides a label.

**How close it is.** Against each room's own noise take:
- within 3 dB on 71 of the 74 test-corpus files it answered outside one room, and on 13 of 14 full audition takes recorded on the platform;
- now and then it misses by 8 to 14 dB, mostly reading the room noisier than it is;
- it is weaker on short files: 12 of 17 six-second phrases were within 3 dB;
- near a band line, a 2 dB error changes the band.

Until v8 the estimate was a relative silence gate. Its error grew with how quiet the room was, 14 to 19 dB high in quiet rooms, and no constant offset could fix it.

The fix is still a noise take in every session folder: at least 3 s with a continuous 3 s clean run, and 5 s is the target.

## Clipping

v7's clip finder oversampled only around samples at or above −0.2 dBFS, so an over whose samples all stay below that went unseen. v8 decides the clip with the level core's whole-file 4x true peak, against 0.0 dBTP, or −0.10 dBTP with the bias toggle. The finder still runs and gives the timecodes and the near-clip notes.

A clip the finder has no event for is listed without a timecode:

```
TRUE PEAK ✗
   --:--.--- → TP CLIP (+0.05 dBTP) — whole-file true peak, no timecode
```

A missing timecode means the over sits where the finder does not look: between samples that all stay below −0.2 dBFS, or at the first or last sample of a file that starts or ends abruptly. Whole-file oversampling reads an abrupt start or end near full scale as an over at the file's edge; the platform's level core reads it the same way.

**Near-clip.** A finder event at or above −0.5 dBTP that stays below the clip line is noted as `near-clip`. It does not change the sort.

**Bias toggle.** The checkbox, or `--bias`, moves the clip line to −0.10 dBTP under v7 and Strict. Under Platform Band it is disabled. It is marked for removal in a later version.

## Reports

Both files are written into the scanned folder, replacing the previous pair. File paths in them are relative to that folder.

### INTAKE_REPORT.txt

**v7 and Strict.** v7's report, with the changes listed above. Files are sorted reject, salvageable, pass, then noise takes, worst peak first within each.

**Platform Band.** The header gives the noise bands and the level-check lines, read from the cores with each policy's name and hash, then counts by noise band and by level verdict, then one line per noise take. Each file block opens with a `Platform:` line. Files that fail either stage come first, then noise WARN, CAUTION, PASS, files with no band, noise takes and errors.

### INTAKE_REPORT.jsonl

Strict JSON, one object per line, UTF-8. NaN and infinities are written as `null`.

**Line 1, the run record** (`"record": "run"`): `schema_version` (1.1), `scanner_version`, `scanned_at`, `profile`, `readout` (`editing` or `platform`), `policy`, `bias_db`, `file_count`, `vendored` (each vendored file's path, version, source commit and sha256), and the four not-applicable fields.

**Then one row per file** (`"record": "file"`). Every field is on every row; a field that does not apply is `null`.

| Group | Fields |
|---|---|
| Identity | `rel_path`; `role`: `content`, `noise_take`, `skipped` or `error` |
| Format | `bit_depth`, `channels`, `sample_rate`, `duration_sec`, `format_issues` |
| Measures | `sample_peak_dbfs`, `true_peak_dbtp`, `true_peak_events`, `integrated_lufs`, `crest_factor_db`, `rms_dbfs`, `speech_level_dba`, `noise_floor_dba`, `snr_db`, `peak_outlier` |
| Noise method | `noise_floor_method`, `noise_floor_confidence`, `noise_reference_rel_path`, `noise_reference_found_by` (`name` or `content_scan`), `noise_reference_gate_reason`, `valid_noise_sec`, `impulse_count` |
| Editing sort (v7, Strict) | `disposition` (`pass`, `salvageable`, `reject`, or `reference` for a noise take) and `flags` (the triggers, such as `snr_warn` or `tp_clip`) |
| Platform readout (Platform Band) | `level_verdict`, `peak_disposition`, `lufs_disposition`, `level_fail_reasons`, `noise_band`, `noise_verdict`, `noise_gate_reason` |
| Policy | `profile`, `scanner_config_sha256` (v7, Strict, custom), `level_config_sha256` and `band_config_sha256` (Platform Band, copied from the cores' records) |
| Not applicable | `rt60_status` and `drift_status`, always `n/a`, with `rt60_reason` and `drift_reason` |

Every policy hash is a sha256 over the resolved policy, serialized the same way in both cores, so it moves when a policy file or a core's defaults change.

## Finder labels

| Label | v7 and Strict | Platform Band |
|---|---|---|
| Red | reject, including a format skip and an unreadable file | level FAIL, noise REJECT, or an unreadable file |
| Orange | salvageable with an SNR warn, a crest-factor warn or a LUFS caution | noise WARN |
| Yellow | salvageable on an SNR caution alone | noise CAUTION, or a level flag (HOT or QUIET) |
| Blue | pass, with a peak outlier | — |
| Green | pass | level PASS and noise PASS, with no flag |
| Gray | noise take | noise take |
| None | — | level passes and the folder has no noise take |

Every scanned file gets a label, and None clears an older one. Under Platform Band a row with no answer never shows green.

## The app

### Window

- It opens on Platform Band. v7 and Strict are in the dropdown, with **Custom YAML...** for any other profile.
- Drop a folder on the window, or click **Select Folder → Scan**. The log fills while the scan runs, then the window shows the report in colour and where both files were saved.
- If an editing profile fails to load, the scan falls back to Strict's built-in numbers and says so. If Platform Band fails to load, the scan stops with the error.
- **The skin.** The window takes the platform's design tokens from `cores/platform_tokens/tokens.css`, following the macOS appearance once, at launch. The report stays a monospace log. If the tokens cannot be read, the window keeps a plain dark look and says why.

### Headless

```
python intake_scanner_gui.py --headless <folder> [--profile <name or path>] [--bias]
```

- No `--profile` runs Platform Band.
- `--profile` takes a path to a `.yaml` file, part of a built-in file name (`default`, `strict`, `platform`), or a profile's name (`v7`, `Strict`, `"Platform Band"`).
- `--bias` sets the −0.10 dBTP clip line. Platform Band ignores it.

"Headless" is a misnomer kept from v7: the app still opens its window, runs the scan there and stays open for review.

## Vendored cores

`cores/` holds copies of the platform's two Rec QC cores, the noise core's Platform Band policy, and the platform's design tokens.
- They were copied from the platform's private repositories and are not edited here, apart from comments removed for publication.
- `cores/VENDORED.json` records each file's source commit and sha256.
- The first import of `cores` compares every listed file with its recorded sha256. On a mismatch the scanner will not start, and it names the file.

## Requirements

Python 3.10 or newer, plus the packages in `requirements.txt`: numpy, scipy, soundfile, pydub, pyloudnorm, pyyaml and tkinterdnd2.
- pydub needs `ffmpeg` on `PATH` for the formats soundfile cannot read, such as m4a and mp3.
- Drag and drop needs a Python built against Tcl 8.6; a Python linked against Tcl 9 breaks `tkdnd`.
- The Finder labels use AppleScript and are macOS-only.

## The stack

- `scipy.signal.resample_poly` upsamples the whole file 4x for the true peak, so an inter-sample peak is caught wherever it sits.
- The A-weighting filter, built with `bilinear_zpk` and `zpk2sos` and run with `sosfilt`, comes from the noise core, so the scanner and the platform share one filter.
- `pyloudnorm` measures BS.1770-4 loudness, and `soundfile` reads WAV and FLAC, with `pydub` and ffmpeg as the fallback for other formats.
- `numpy` underlies the signal math, and `pyyaml` loads the profiles.

## How it's built

Through AI pair programming: I authored the requirements and audio specs, sketched the design, directed the build, and reviewed the results against the test corpus.

## Known limitations

- **Headless opens a window.**
- **Speed and memory.** v8 oversamples every whole file 4x, and Platform Band also runs both cores on every file. On a 113-file corpus the folder scan took about 17 s under the v7 profile and 35 s under Platform Band, against 7 s for v7. Memory grows with file length, roughly 1.4 GB at 10 minutes for the level core's call.
- **One calibration corpus.** The v7 and Strict numbers come from one April 2026 corpus. Another recording domain may need its own profile.
- **Spot problems are not detected.** Clicks, plosives and intermittent noise, such as a dog or a passing car, have to be heard.
- **A folder without a usable noise take** gets only the per-file estimate, which never sorts. Record a noise take in every session folder.
