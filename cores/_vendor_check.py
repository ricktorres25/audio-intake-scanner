"""
_vendor_check.py - the start check for the vendored cores.

Scanner code, not vendored. Standard library only, and nothing runs at import,
so a test can load this file by path even when the vendored copies are damaged.

cores/__init__.py calls require_vendored() the first time anything imports the
package, so a vendored file that differs from VENDORED.json stops the scanner
before any vendored code runs.

It also runs as a script:
    python3 _vendor_check.py write <cores dir>    rows on stdin -> VENDORED.json
    python3 _vendor_check.py check <cores dir>    exit 0 if every file matches, else 1
so the manifest is written and read in one place.
"""
import hashlib
import json
import sys
from pathlib import Path

MANIFEST_NAME = "VENDORED.json"

# Manifest fields, in the order of each stdin row.
ROW_FIELDS = ("path", "source_folder", "source_path", "repo", "commit")

MANIFEST_NOTE = ("Do not edit by hand. The scanner refuses to start if a file below "
                 "differs from its sha256.")


class VendoredCoreError(RuntimeError):
    """A vendored file is missing, or its bytes differ from VENDORED.json."""


def sha256_file(path):
    """sha256 of a file's bytes, as lowercase hex."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(cores_dir):
    """Read VENDORED.json from cores_dir. Raises OSError or ValueError if it cannot."""
    return json.loads((Path(cores_dir) / MANIFEST_NAME).read_text(encoding="utf-8"))


def check_vendored(cores_dir):
    """Compare every file listed in VENDORED.json with its recorded sha256.

    Returns a list of problems, one line each. An empty list means every listed
    file is present and byte-identical to what was vendored.
    """
    cores_dir = Path(cores_dir)
    try:
        entries = load_manifest(cores_dir)["files"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        return [f"cores/{MANIFEST_NAME} is missing or unreadable ({e})"]
    if not isinstance(entries, list) or not entries:
        return [f"cores/{MANIFEST_NAME} lists no files"]

    problems = []
    for entry in entries:
        if not isinstance(entry, dict):
            problems.append(f"cores/{MANIFEST_NAME} has a malformed entry")
            continue
        rel = str(entry.get("path", "?"))
        path = cores_dir / rel
        if not path.is_file():
            problems.append(f"cores/{rel} is missing")
            continue
        recorded = str(entry.get("sha256", ""))
        found = sha256_file(path)
        if found != recorded:
            problems.append(f"cores/{rel} differs from {MANIFEST_NAME} "
                            f"(sha256 recorded {recorded[:12]}, found {found[:12]})")
    return problems


def require_vendored(cores_dir):
    """Raise VendoredCoreError, naming every bad file, if check_vendored finds any."""
    problems = check_vendored(cores_dir)
    if problems:
        raise VendoredCoreError(
            "The vendored cores failed their start check, so the scanner will not run.\n  "
            + "\n  ".join(problems)
            + "\nFiles under cores/ are not edited by hand. "
              "Restore them with git.")


def write_manifest(cores_dir, rows):
    """Write VENDORED.json from rows (one 'a|b|c|d|e' line per file).

    The sha256 is taken here, from the file as it now sits under cores/.
    No timestamp is written, so a refresh with unchanged sources leaves the
    manifest byte-identical.
    """
    cores_dir = Path(cores_dir)
    files = []
    for line in rows:
        if not line.strip():
            continue
        values = line.rstrip("\n").split("|")
        if len(values) != len(ROW_FIELDS):
            raise ValueError(f"bad manifest row: {line!r}")
        entry = dict(zip(ROW_FIELDS, values))
        entry["sha256"] = sha256_file(cores_dir / entry["path"])
        files.append(entry)
    doc = {"written_by": "_vendor_check.py write", "note": MANIFEST_NOTE, "files": files}
    (cores_dir / MANIFEST_NAME).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("write", "check"):
        print("usage: _vendor_check.py write|check <cores dir>", file=sys.stderr)
        sys.exit(2)
    if sys.argv[1] == "write":
        write_manifest(sys.argv[2], sys.stdin.read().splitlines())
        sys.exit(0)
    found_problems = check_vendored(sys.argv[2])
    for p in found_problems:
        print(p, file=sys.stderr)
    sys.exit(1 if found_problems else 0)
