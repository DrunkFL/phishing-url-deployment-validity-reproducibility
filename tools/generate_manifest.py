from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "MANIFEST.csv"
EXCLUDED_PARTS = {".git", ".venv", "__pycache__", ".tldextract_cache"}
EXCLUDED_PREFIXES = (
    Path("code/part17_phreshphish_external_validation/data/metadata"),
)
TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".tex", ".txt", ".yaml", ".yml"}
TEXT_NAMES = {"LICENSE-CODE", "LICENSE-DOCUMENTATION"}


def canonical_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES or path.name in TEXT_NAMES:
        return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return data


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return (
        path != OUTPUT
        and not EXCLUDED_PARTS.intersection(relative.parts)
        and not any(relative.is_relative_to(prefix) for prefix in EXCLUDED_PREFIXES)
    )


def main() -> None:
    files = sorted(path for path in ROOT.rglob("*") if path.is_file() and included(path))
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["relative_path", "bytes", "sha256"])
        for path in files:
            data = canonical_bytes(path)
            writer.writerow(
                [
                    path.relative_to(ROOT).as_posix(),
                    len(data),
                    hashlib.sha256(data).hexdigest(),
                ]
            )
    print(f"Wrote {len(files)} entries to {OUTPUT}")


if __name__ == "__main__":
    main()
