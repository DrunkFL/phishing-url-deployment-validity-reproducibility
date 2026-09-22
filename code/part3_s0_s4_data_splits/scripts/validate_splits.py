from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pandas as pd


PART3 = Path(__file__).resolve().parents[1]
ASSIGNMENTS = PART3 / "data" / "assignments"
RESULTS = PART3 / "results"
SPLIT_COLUMNS = [f"split_r{index:02d}" for index in range(10)]
ROLE_COLUMNS = [f"role_r{index:02d}" for index in range(10)]
EXPECTED_ROWS = {
    ("PhiUSIIL", "master"): 235793,
    ("ISCX-URL2016-binary", "master"): 45343,
    ("PhiUSIIL", "deduplicated"): 234656,
    ("ISCX-URL2016-binary", "deduplicated"): 45224,
}
SCENARIO_ENTITY = {
    "S1-url": "normalized_url_sha256",
    "S2-host": "host_sha256",
    "S3-domain": "registrable_domain_sha256",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_internal(path: Path) -> tuple[tuple[str, str], set[str]]:
    table = pd.read_parquet(path)
    dataset = str(table["dataset"].iloc[0])
    corpus_version = str(table["corpus_version"].iloc[0])
    scenario = str(table["scenario"].iloc[0])
    assert len(table) == EXPECTED_ROWS[(dataset, corpus_version)]
    assert table["sample_id"].is_unique
    assert set(SPLIT_COLUMNS).issubset(table.columns)
    if corpus_version == "deduplicated":
        assert table["normalized_url_sha256"].is_unique

    for column in SPLIT_COLUMNS:
        assert set(table[column].astype(str).unique()) == {
            "train",
            "validation",
            "test",
        }
        entity_column = SCENARIO_ENTITY.get(scenario)
        if entity_column:
            memberships = table.groupby(entity_column)[column].nunique()
            assert memberships.eq(1).all()
    return (dataset, corpus_version), set(table["sample_id"])


def validate_s4(path: Path) -> None:
    table = pd.read_parquet(path)
    corpus_version = str(table["corpus_version"].iloc[0])
    expected_rows = sum(
        count for (dataset, version), count in EXPECTED_ROWS.items() if version == corpus_version
    )
    assert len(table) == expected_rows
    assert table["sample_id"].is_unique
    assert set(ROLE_COLUMNS).issubset(table.columns)
    source = table["origin"].eq("source")
    target = table["origin"].eq("target")
    for column in ROLE_COLUMNS:
        roles = set(table[column].astype(str).unique())
        assert roles == {
            "source_train",
            "source_validation",
            "source_unused_internal_test",
            "target_external_primary",
            "target_excluded_domain_overlap",
        }
        source_fit_domains = set(
            table.loc[
                source
                & table[column].astype(str).isin(
                    ["source_train", "source_validation"]
                ),
                "registrable_domain_sha256",
            ]
        )
        primary_target_domains = set(
            table.loc[
                target & table[column].astype(str).eq("target_external_primary"),
                "registrable_domain_sha256",
            ]
        )
        assert source_fit_domains.isdisjoint(primary_target_domains)


def main() -> None:
    internal_paths = sorted(ASSIGNMENTS.glob("*s[0-3]_*_assignments.parquet"))
    s4_paths = sorted(ASSIGNMENTS.glob("s4_*_assignments.parquet"))
    assert len(internal_paths) == 16
    assert len(s4_paths) == 4

    sample_sets: dict[tuple[str, str], set[str]] = {}
    for path in internal_paths:
        key, sample_ids = validate_internal(path)
        previous = sample_sets.setdefault(key, sample_ids)
        assert previous == sample_ids
    for path in s4_paths:
        validate_s4(path)

    split_summary = pd.read_csv(RESULTS / "split_summary.csv")
    overlap = pd.read_csv(RESULTS / "entity_overlap_audit.csv")
    constraints = pd.read_csv(RESULTS / "constraint_validation.csv")
    differences = pd.read_csv(RESULTS / "partition_difference.csv")
    s4_summary = pd.read_csv(RESULTS / "s4_summary.csv")
    s4_overlap = pd.read_csv(RESULTS / "s4_overlap_audit.csv")
    seeds = pd.read_csv(RESULTS / "seed_registry.csv")
    assert len(split_summary) == 480
    assert len(overlap) == 1440
    assert len(constraints) == 360
    assert constraints["constraint_pass"].eq(True).all()
    assert len(differences) == 160
    assert differences.loc[
        differences["repetition"] > 0, "fraction_different_from_r00"
    ].gt(0).all()
    assert len(s4_summary) == 40
    assert len(s4_overlap) == 240
    assert s4_overlap.loc[
        s4_overlap["target_cohort"] == "primary_domain_filtered",
        "overlap_entities",
    ].eq(0).all()
    assert len(seeds) == 10
    assert seeds["seed"].is_unique
    assert seeds["primary"].sum() == 1

    manifest_path = RESULTS / "split_output_manifest.csv"
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    assert len(manifest) == 29
    for row in manifest:
        path = PART3 / row["path"]
        assert path.is_file()
        assert path.stat().st_size == int(row["bytes"])
        assert sha256_file(path) == row["sha256"]

    print("internal_assignment_files=16")
    print("s4_assignment_files=4")
    print("assignment_row_counts=pass")
    print("sample_ids_unique_and_consistent=pass")
    print("ten_repetitions_present=pass")
    print("internal_group_constraints=pass")
    print("s4_primary_domain_constraints=pass")
    print("audit_table_shapes=pass")
    print("split_output_manifest_files=29")
    print("split_output_hashes=match")


if __name__ == "__main__":
    main()
