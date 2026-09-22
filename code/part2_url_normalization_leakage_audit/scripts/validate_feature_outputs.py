from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from feature_extraction import FEATURE_NAMES, FEATURE_SPECS, feature_vector_sha256


PART2 = Path(__file__).resolve().parents[1]
DATA = PART2 / "data"
RESULTS = PART2 / "results"
EXPECTED_TABLES = {
    DATA / "phiusiil_master_features.parquet": (235793, "master", False),
    DATA / "iscx_url2016_binary_master_features.parquet": (45343, "master", False),
    DATA / "phiusiil_deduplicated_features.parquet": (234656, "deduplicated", True),
    DATA / "iscx_url2016_binary_deduplicated_features.parquet": (
        45224,
        "deduplicated",
        True,
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_table(
    path: Path, expected_rows: int, corpus_version: str, must_be_deduplicated: bool
) -> None:
    table = pd.read_parquet(path)
    assert len(table) == expected_rows
    assert table["corpus_version"].eq(corpus_version).all()
    assert "raw_url" not in table.columns
    assert "normalized_url" not in table.columns
    assert set(FEATURE_NAMES).issubset(table.columns)
    values = table[list(FEATURE_NAMES)].to_numpy(dtype=float)
    assert not np.isnan(values).any()
    assert np.isfinite(values).all()
    if must_be_deduplicated:
        assert table["normalized_url_sha256"].is_unique

    sample_indices = sorted({0, len(table) // 2, len(table) - 1})
    integer_features = {spec.name for spec in FEATURE_SPECS if spec.dtype == "int"}
    for index in sample_indices:
        row = table.iloc[index]
        features = {
            name: int(row[name]) if name in integer_features else float(row[name])
            for name in FEATURE_NAMES
        }
        assert feature_vector_sha256(features) == row["feature_vector_sha256"]


def main() -> None:
    dictionary = pd.read_csv(RESULTS / "feature_dictionary.csv")
    assert dictionary["feature"].tolist() == list(FEATURE_NAMES)
    assert len(dictionary) == 35
    assert dictionary["requires_network"].eq(False).all()
    assert dictionary["uses_target_statistics"].eq(False).all()

    for path, expectations in EXPECTED_TABLES.items():
        validate_table(path, *expectations)

    quality = pd.read_csv(RESULTS / "feature_quality_audit.csv")
    class_stats = pd.read_csv(RESULTS / "class_feature_statistics.csv")
    shifts = pd.read_csv(RESULTS / "feature_distribution_shift.csv")
    vector_summary = pd.read_csv(RESULTS / "feature_vector_duplicate_summary.csv")
    review = pd.read_csv(PART2 / "review" / "feature_spotcheck_sample.csv")
    assert len(quality) == 4 * len(FEATURE_NAMES)
    assert int(quality["missing_count"].sum()) == 0
    assert int(quality["infinite_count"].sum()) == 0
    assert len(class_stats) == 4 * 2 * len(FEATURE_NAMES)
    assert len(shifts) == 2 * 3 * len(FEATURE_NAMES)
    assert len(vector_summary) == 4
    assert len(review) == 40
    assert "untrusted_url_text_do_not_visit" in review.columns
    assert (PART2 / "review" / "spotcheck_review.md").is_file()

    manifest_path = RESULTS / "feature_output_manifest.csv"
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    assert len(manifest) == 17
    for row in manifest:
        path = PART2 / row["path"]
        assert path.is_file()
        assert path.stat().st_size == int(row["bytes"])
        assert sha256_file(path) == row["sha256"]

    print("feature_count=35")
    print("feature_table_row_counts=pass")
    print("feature_table_values_finite=pass")
    print("feature_vector_sample_hashes=match")
    print("deduplicated_normalized_hashes_unique=pass")
    print("feature_audit_shapes=pass")
    print("feature_output_manifest_files=17")
    print("feature_output_hashes=match")


if __name__ == "__main__":
    main()
