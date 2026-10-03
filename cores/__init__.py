"""
cores - vendored copies of two Rec QC cores and the noise core's Platform Band policy.

    level_check_phrase_core.py                the platform's level-check core
    noise_profile_scanner_core.py             the platform's noise-profile core
    noise_profile_configs/platform_band.yaml  the noise core's Platform Band policy
    platform_tokens/tokens.css                the platform's design tokens (the GUI skin)

They are copied from the collection platform's private repositories and not
edited here, apart from comments removed for publication. VENDORED.json records
each file's source, the commit it was copied at, and its sha256.

This file and _vendor_check.py are the scanner's own code, not vendored.

The start check: the first import of this package compares every file listed in
VENDORED.json with its recorded sha256, before any vendored module can load. On a
mismatch it raises VendoredCoreError naming the file, and the scanner does not start.
"""
from pathlib import Path

from ._vendor_check import VendoredCoreError, check_vendored, load_manifest, require_vendored

CORES_DIR = Path(__file__).resolve().parent

require_vendored(CORES_DIR)
