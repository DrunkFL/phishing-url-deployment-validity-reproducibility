from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST.csv"
GITHUB_HARD_LIMIT = 100 * 1024 * 1024
EXCLUDED_PARTS = {".git", ".venv", "__pycache__"}
TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".tex", ".txt", ".yaml", ".yml"}
LOCAL_PATHS = ("C:" + "\\Users\\FL", "C:" + "/Users/FL")
BANNED_IDENTIFIER_COLUMNS = {"raw_url", "url", "host", "registrable_domain", "source_file"}
SMOKE_TESTS = (
    "code/part4_baseline_models/scripts/train_baselines.py",
    "code/part11_dcss_innovation/scripts/train_dcss_main_comparison.py",
    "code/part14_low_fpr_uncertainty/scripts/run_part14_low_fpr_uncertainty.py",
    "code/part15_statistical_inference_robustness/scripts/run_part15_robust_inference.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def repository_files() -> list[Path]:
    files = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path == MANIFEST:
            continue
        relative = path.relative_to(ROOT)
        if EXCLUDED_PARTS.intersection(relative.parts):
            continue
        files.append(path)
    return sorted(files)


def validate_parquet_schemas() -> int:
    try:
        import pyarrow.parquet as parquet
    except ImportError:
        print("NOTE: pyarrow is unavailable; Parquet schema checks were skipped.")
        return 0

    checked = 0
    for path in sorted((ROOT / "splits").glob("*.parquet")):
        columns = set(parquet.read_schema(path).names)
        require(not columns.intersection(BANNED_IDENTIFIER_COLUMNS), f"Plaintext identifier column in {path}")
        checked += 1
    feature_data = ROOT / "code" / "part2_url_normalization_leakage_audit" / "data"
    for path in sorted(feature_data.glob("*.parquet")):
        columns = set(parquet.read_schema(path).names)
        require(not columns.intersection(BANNED_IDENTIFIER_COLUMNS), f"Plaintext identifier column in {path}")
        require("raw_url_sha256" in columns, f"Missing hashed sample identifier in {path}")
        checked += 1
    return checked


def validate_entry_points() -> int:
    for relative_path in SMOKE_TESTS:
        path = ROOT / relative_path
        require(path.is_file(), f"Missing smoke-test entry point: {relative_path}")
        completed = subprocess.run(
            [sys.executable, str(path), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        require(
            completed.returncode == 0,
            f"Entry point failed: {relative_path}\n{completed.stdout}\n{completed.stderr}",
        )
    return len(SMOKE_TESTS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the public reproducibility repository")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="also import representative experiment entry points and run their --help commands",
    )
    args = parser.parse_args()

    required = [
        ROOT / "README.md",
        ROOT / "DATA_AVAILABILITY.md",
        ROOT / "LICENSE_NOT_INCLUDED.md",
        ROOT / "manuscript" / "main.md",
        ROOT / "manuscript" / "main.tex",
        ROOT / "manuscript" / "paper.pdf",
        ROOT / "environment" / "requirements.lock.txt",
        ROOT / "features" / "FEATURE_DEFINITIONS.md",
        ROOT / "protocols" / "split_protocol.md",
        ROOT / "REPRODUCE.md",
    ]
    for path in required:
        require(path.is_file(), f"Missing required file: {path.relative_to(ROOT)}")
    for number in range(1, 8):
        figure = ROOT / "manuscript" / "figures" / f"Figure_{number}.pdf"
        require(figure.is_file(), f"Missing manuscript figure: {figure.relative_to(ROOT)}")

    with MANIFEST.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    actual = repository_files()
    listed = {row["relative_path"] for row in rows}
    actual_names = {path.relative_to(ROOT).as_posix() for path in actual}
    require(listed == actual_names, f"Manifest membership mismatch: missing={actual_names-listed}, extra={listed-actual_names}")

    for row in rows:
        path = ROOT / row["relative_path"]
        require(path.stat().st_size == int(row["bytes"]), f"Size mismatch: {row['relative_path']}")
        require(sha256(path) == row["sha256"], f"SHA-256 mismatch: {row['relative_path']}")
        require(path.stat().st_size < GITHUB_HARD_LIMIT, f"File reaches GitHub's 100 MB limit: {row['relative_path']}")

    scanned_text = 0
    python_files = 0
    for path in actual:
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8")
        require(not any(marker in text for marker in LOCAL_PATHS), f"Local user path in {path.relative_to(ROOT)}")
        scanned_text += 1
        if path.suffix.lower() == ".py":
            ast.parse(text, filename=str(path))
            python_files += 1

    parquet_schemas = validate_parquet_schemas()
    entry_points = validate_entry_points() if args.smoke else 0
    summary = {
        "status": "PASS",
        "manifested_files": len(rows),
        "python_files_syntax_checked": python_files,
        "text_files_local_path_scanned": scanned_text,
        "parquet_schemas_checked": parquet_schemas,
        "entry_points_smoke_tested": entry_points,
        "largest_file_bytes": max(path.stat().st_size for path in actual),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
