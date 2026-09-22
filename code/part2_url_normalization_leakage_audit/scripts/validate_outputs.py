from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pandas as pd


PART2 = Path(__file__).resolve().parents[1]
DATA = PART2 / "data"
RESULTS = PART2 / "results"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_candidate(path: Path, expected_rows: int) -> None:
    frame = pd.read_parquet(path)
    assert len(frame) == expected_rows
    assert "raw_url" not in frame.columns
    assert frame["normalized_url_sha256"].is_unique
    assert frame["parse_status"].eq("ok").all()
    assert set(frame["label"].unique()).issubset({0, 1})


def validate_master(path: Path, expected_rows: int) -> None:
    frame = pd.read_parquet(path)
    assert len(frame) == expected_rows
    assert "raw_url" not in frame.columns
    assert frame["parse_status"].eq("ok").all()
    conflicts = frame.groupby("normalized_url_sha256")["label"].nunique()
    assert conflicts.le(1).all()


def main() -> None:
    validate_master(DATA / "phiusiil_conflict_cleaned_master.parquet", 235793)
    validate_master(
        DATA / "iscx_url2016_binary_conflict_cleaned_master.parquet", 45343
    )
    validate_candidate(DATA / "phiusiil_deduplicated_candidate.parquet", 234656)
    validate_candidate(
        DATA / "iscx_url2016_binary_deduplicated_candidate.parquet", 45224
    )

    conflicts = pd.read_csv(RESULTS / "conflicting_label_groups.csv")
    assert len(conflicts) == 1
    assert conflicts["label_count"].eq(2).all()

    with (RESULTS / "output_manifest.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        manifest = list(csv.DictReader(handle))
    for row in manifest:
        path = PART2 / row["path"]
        assert path.is_file()
        assert path.stat().st_size == int(row["bytes"])
        assert sha256_file(path) == row["sha256"]

    print("candidate_row_counts=pass")
    print("master_row_counts=pass")
    print("master_conflicting_labels=none")
    print("candidate_normalized_hashes_unique=pass")
    print("candidate_parse_status=pass")
    print("conflict_group_count=1")
    print(f"output_manifest_files={len(manifest)}")
    print("output_hashes=match")


if __name__ == "__main__":
    main()
