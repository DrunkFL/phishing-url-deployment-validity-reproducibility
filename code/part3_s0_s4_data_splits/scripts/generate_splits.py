from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from split_utils import (
    SEEDS,
    SPLIT_NAMES,
    TARGET_PROPORTIONS,
    assert_group_disjoint,
    grouped_stratified_assignment,
    row_stratified_assignment,
    stable_sha256,
)


PART3 = Path(__file__).resolve().parents[1]
ROOT = PART3.parent
PART2 = ROOT / "part2_url_normalization_leakage_audit"
PART2_DATA = PART2 / "data"
PART2_MANIFEST = PART2 / "results" / "feature_output_manifest.csv"
ASSIGNMENTS = PART3 / "data" / "assignments"
RESULTS = PART3 / "results"
LOGS = PART3 / "logs"
INPUTS = {
    ("PhiUSIIL", "master"): PART2_DATA / "phiusiil_master_features.parquet",
    ("ISCX-URL2016-binary", "master"): PART2_DATA / "iscx_url2016_binary_master_features.parquet",
    ("PhiUSIIL", "deduplicated"): PART2_DATA / "phiusiil_deduplicated_features.parquet",
    ("ISCX-URL2016-binary", "deduplicated"): PART2_DATA / "iscx_url2016_binary_deduplicated_features.parquet",
}
SCENARIOS = {
    "S0-row": None,
    "S1-url": "normalized_url_sha256",
    "S2-host": "host_sha256",
    "S3-domain": "registrable_domain_sha256",
}
ENTITY_COLUMNS = {
    "normalized_url": "normalized_url_sha256",
    "host": "host_sha256",
    "registrable_domain": "registrable_domain_sha256",
}
SPLIT_PAIRS = (
    ("train", "validation"),
    ("train", "test"),
    ("validation", "test"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_inputs() -> None:
    manifest = pd.read_csv(PART2_MANIFEST)
    expected = {PART2 / row.path: row.sha256 for row in manifest.itertuples()}
    for path in INPUTS.values():
        if not path.is_file():
            raise FileNotFoundError(path)
        if expected.get(path) != sha256_file(path):
            raise ValueError(f"Part 2 feature-table hash mismatch: {path}")


def hash_series(values: pd.Series) -> pd.Series:
    unique_values = values.astype(str).drop_duplicates()
    mapping = {value: stable_sha256(value) for value in unique_values}
    return values.astype(str).map(mapping)


def prepare_metadata(path: Path) -> pd.DataFrame:
    columns = [
        "dataset",
        "source_row",
        "label",
        "raw_url_sha256",
        "normalized_url_sha256",
        "host",
        "registrable_domain",
        "feature_vector_sha256",
    ]
    available = pq.ParquetFile(path).schema.names
    if "source_file" in available:
        columns.insert(2, "source_file")
    table = pd.read_parquet(path, columns=columns)
    if "source_file" not in table.columns:
        table.insert(2, "source_file", "")
    table["source_file"] = table["source_file"].fillna("").astype(str)
    table["host_sha256"] = hash_series(table["host"])
    table["registrable_domain_sha256"] = hash_series(table["registrable_domain"])
    provenance = (
        table["dataset"].astype(str)
        + "|"
        + table["source_file"]
        + "|"
        + table["source_row"].astype(str)
        + "|"
        + table["raw_url_sha256"]
    )
    table.insert(0, "sample_id", provenance.map(stable_sha256))
    if not table["sample_id"].is_unique:
        raise ValueError(f"sample_id is not unique for {path}")
    return table.drop(columns=["host", "registrable_domain"])


def safe_dataset_name(dataset: str) -> str:
    return dataset.lower().replace("-", "_")


def assignment_column(repetition: int) -> str:
    return f"split_r{repetition:02d}"


def generate_internal_assignments(
    table: pd.DataFrame,
    dataset: str,
    corpus_version: str,
    scenario: str,
) -> pd.DataFrame:
    result = table.copy()
    labels = result["label"].to_numpy(dtype=np.int8)
    group_column = SCENARIOS[scenario]
    groups = None if group_column is None else result[group_column].to_numpy()
    for repetition, seed in enumerate(SEEDS):
        if scenario == "S0-row":
            assignment = row_stratified_assignment(labels, seed)
        else:
            assignment = grouped_stratified_assignment(labels, groups, seed)
            assert_group_disjoint(assignment, groups)
        result[assignment_column(repetition)] = pd.Categorical(
            assignment, categories=SPLIT_NAMES
        )
    result.insert(1, "corpus_version", corpus_version)
    result.insert(2, "scenario", scenario)
    return result


def summarize_internal(
    assignments: pd.DataFrame,
    dataset: str,
    corpus_version: str,
    scenario: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    summary_rows: list[dict[str, object]] = []
    overlap_rows: list[dict[str, object]] = []
    difference_rows: list[dict[str, object]] = []
    primary = assignments[assignment_column(0)].astype(str).to_numpy()

    for repetition, seed in enumerate(SEEDS):
        column = assignment_column(repetition)
        assigned = assignments[column].astype(str)
        difference_rows.append(
            {
                "dataset": dataset,
                "corpus_version": corpus_version,
                "scenario": scenario,
                "repetition": repetition,
                "seed": seed,
                "fraction_different_from_r00": float(
                    np.mean(assigned.to_numpy() != primary)
                ),
            }
        )
        for split in SPLIT_NAMES:
            subset = assignments.loc[assigned == split]
            row_count = len(subset)
            summary_rows.append(
                {
                    "dataset": dataset,
                    "corpus_version": corpus_version,
                    "scenario": scenario,
                    "repetition": repetition,
                    "seed": seed,
                    "split": split,
                    "row_count": row_count,
                    "row_fraction": row_count / len(assignments),
                    "target_fraction": TARGET_PROPORTIONS[split],
                    "fraction_deviation": row_count / len(assignments)
                    - TARGET_PROPORTIONS[split],
                    "benign_rows": int((subset["label"] == 0).sum()),
                    "phishing_rows": int((subset["label"] == 1).sum()),
                    "phishing_fraction": float(subset["label"].mean()),
                    "unique_normalized_urls": int(
                        subset["normalized_url_sha256"].nunique()
                    ),
                    "unique_hosts": int(subset["host_sha256"].nunique()),
                    "unique_registrable_domains": int(
                        subset["registrable_domain_sha256"].nunique()
                    ),
                }
            )

        for left_split, right_split in SPLIT_PAIRS:
            left_rows = assignments.loc[assigned == left_split]
            right_rows = assignments.loc[assigned == right_split]
            for entity, entity_column in ENTITY_COLUMNS.items():
                left = set(left_rows[entity_column])
                right = set(right_rows[entity_column])
                overlap = left & right
                overlap_rows.append(
                    {
                        "dataset": dataset,
                        "corpus_version": corpus_version,
                        "scenario": scenario,
                        "repetition": repetition,
                        "seed": seed,
                        "left_split": left_split,
                        "right_split": right_split,
                        "entity": entity,
                        "left_unique_entities": len(left),
                        "right_unique_entities": len(right),
                        "overlap_entities": len(overlap),
                        "overlap_rate_min_denominator": (
                            len(overlap) / min(len(left), len(right))
                            if left and right
                            else 0.0
                        ),
                    }
                )
    return summary_rows, overlap_rows, difference_rows


def role_column(repetition: int) -> str:
    return f"role_r{repetition:02d}"


def generate_s4_assignments(
    source: pd.DataFrame,
    target: pd.DataFrame,
    source_s3: pd.DataFrame,
    source_dataset: str,
    target_dataset: str,
    corpus_version: str,
) -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    source_base = source.copy()
    target_base = target.copy()
    source_base.insert(1, "origin", "source")
    target_base.insert(1, "origin", "target")
    combined = pd.concat([source_base, target_base], ignore_index=True)
    summary_rows: list[dict[str, object]] = []
    overlap_rows: list[dict[str, object]] = []

    for repetition, seed in enumerate(SEEDS):
        split_col = assignment_column(repetition)
        source_split = source_s3.set_index("sample_id")[split_col].astype(str)
        source_roles = source["sample_id"].map(source_split).map(
            {
                "train": "source_train",
                "validation": "source_validation",
                "test": "source_unused_internal_test",
            }
        )
        source_fit_domains = set(
            source.loc[
                source_roles.isin(["source_train", "source_validation"]),
                "registrable_domain_sha256",
            ]
        )
        target_included = ~target["registrable_domain_sha256"].isin(
            source_fit_domains
        )
        target_roles = np.where(
            target_included,
            "target_external_primary",
            "target_excluded_domain_overlap",
        )
        combined[role_column(repetition)] = pd.Categorical(
            pd.concat(
                [
                    source_roles.reset_index(drop=True),
                    pd.Series(target_roles),
                ],
                ignore_index=True,
            )
        )

        source_train = int((source_roles == "source_train").sum())
        source_validation = int((source_roles == "source_validation").sum())
        source_unused = int((source_roles == "source_unused_internal_test").sum())
        target_primary = target.loc[target_included]
        target_excluded = target.loc[~target_included]
        summary_rows.append(
            {
                "source_dataset": source_dataset,
                "target_dataset": target_dataset,
                "corpus_version": corpus_version,
                "repetition": repetition,
                "seed": seed,
                "source_train_rows": source_train,
                "source_validation_rows": source_validation,
                "source_unused_internal_test_rows": source_unused,
                "target_unfiltered_rows": len(target),
                "target_primary_rows": len(target_primary),
                "target_excluded_domain_overlap_rows": len(target_excluded),
                "target_primary_benign_rows": int(
                    (target_primary["label"] == 0).sum()
                ),
                "target_primary_phishing_rows": int(
                    (target_primary["label"] == 1).sum()
                ),
                "target_excluded_unique_domains": int(
                    target_excluded["registrable_domain_sha256"].nunique()
                ),
            }
        )

        source_fit = source.loc[
            source_roles.isin(["source_train", "source_validation"])
        ]
        for cohort_name, target_cohort in (
            ("unfiltered", target),
            ("primary_domain_filtered", target_primary),
        ):
            for entity, entity_column in ENTITY_COLUMNS.items():
                source_entities = set(source_fit[entity_column])
                target_entities = set(target_cohort[entity_column])
                overlap = source_entities & target_entities
                overlap_rows.append(
                    {
                        "source_dataset": source_dataset,
                        "target_dataset": target_dataset,
                        "corpus_version": corpus_version,
                        "repetition": repetition,
                        "seed": seed,
                        "target_cohort": cohort_name,
                        "entity": entity,
                        "source_fit_unique_entities": len(source_entities),
                        "target_unique_entities": len(target_entities),
                        "overlap_entities": len(overlap),
                        "overlap_rate_min_denominator": (
                            len(overlap)
                            / min(len(source_entities), len(target_entities))
                            if source_entities and target_entities
                            else 0.0
                        ),
                    }
                )
    combined.insert(2, "corpus_version", corpus_version)
    combined.insert(3, "direction", f"{source_dataset}_to_{target_dataset}")
    return combined, summary_rows, overlap_rows


def write_report(
    split_summary: pd.DataFrame,
    overlap: pd.DataFrame,
    constraints: pd.DataFrame,
    differences: pd.DataFrame,
    s4_summary: pd.DataFrame,
    s4_overlap: pd.DataFrame,
    path: Path,
) -> None:
    primary_summary = split_summary.loc[
        (split_summary["corpus_version"] == "master")
        & (split_summary["repetition"] == 0)
    ]
    primary_overlap = overlap.loc[
        (overlap["corpus_version"] == "master")
        & (overlap["repetition"] == 0)
        & (overlap["left_split"] == "train")
        & (overlap["right_split"] == "test")
    ]
    lines = []
    for dataset in ("PhiUSIIL", "ISCX-URL2016-binary"):
        for scenario in SCENARIOS:
            rows = primary_summary.loc[
                (primary_summary["dataset"] == dataset)
                & (primary_summary["scenario"] == scenario)
            ]
            counts = {row.split: int(row.row_count) for row in rows.itertuples()}
            overlaps = primary_overlap.loc[
                (primary_overlap["dataset"] == dataset)
                & (primary_overlap["scenario"] == scenario)
            ]
            entity_counts = {
                row.entity: int(row.overlap_entities) for row in overlaps.itertuples()
            }
            lines.append(
                f"| {dataset} | {scenario} | {counts['train']} | {counts['validation']} | "
                f"{counts['test']} | {entity_counts['normalized_url']} | "
                f"{entity_counts['host']} | {entity_counts['registrable_domain']} |"
            )
    s4_primary = s4_summary.loc[
        (s4_summary["corpus_version"] == "master")
        & (s4_summary["repetition"] == 0)
    ]
    s4_lines = []
    for row in s4_primary.itertuples():
        direction_overlap = s4_overlap.loc[
            (s4_overlap["source_dataset"] == row.source_dataset)
            & (s4_overlap["target_dataset"] == row.target_dataset)
            & (s4_overlap["corpus_version"] == "master")
            & (s4_overlap["repetition"] == 0)
            & (s4_overlap["target_cohort"] == "unfiltered")
        ]
        overlap_counts = {
            item.entity: int(item.overlap_entities)
            for item in direction_overlap.itertuples()
        }
        excluded_fraction = (
            row.target_excluded_domain_overlap_rows / row.target_unfiltered_rows
        )
        s4_lines.append(
            f"| {row.source_dataset} -> {row.target_dataset} | {int(row.target_unfiltered_rows)} | "
            f"{overlap_counts['host']} | {overlap_counts['registrable_domain']} | "
            f"{int(row.target_excluded_domain_overlap_rows)} ({excluded_fraction:.2%}) | "
            f"{int(row.target_primary_rows)} |"
        )
    filtered_s4 = s4_overlap.loc[
        s4_overlap["target_cohort"] == "primary_domain_filtered"
    ]
    report = f"""# S0-S4 Split Report

## Protocol

- Ten prespecified partition seeds: {', '.join(map(str, SEEDS))}.
- Primary repetition: r00, seed {SEEDS[0]}.
- S0-S3 target proportions: 70% train, 15% validation, 15% test.
- Grouped splits use prespecified weighted greedy group stratification.
- Models were not trained in Part 3.

## Primary Master-Corpus Assignments

| Dataset | Scenario | Train | Validation | Test | Train-test URL overlap | Host overlap | Registrable-domain overlap |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

All {len(constraints)} prespecified zero-overlap constraint checks passed: {bool(constraints['constraint_pass'].all())}.
The largest absolute row-fraction deviation across all internal assignments was {split_summary['fraction_deviation'].abs().max():.4f}.
All non-primary repetitions changed at least one row assignment: {bool((differences.loc[differences['repetition'] > 0, 'fraction_different_from_r00'] > 0).all())}.

## Primary External Transfer Cohorts

| Direction | Unfiltered target rows | Shared hosts before filtering | Shared domains before filtering | Rows excluded | Primary target rows |
|---|---:|---:|---:|---:|---:|
{chr(10).join(s4_lines)}

After domain filtering, the maximum source-fit/target overlap count across normalized URL, host, registrable domain, both directions, both corpus versions, and all repetitions was {int(filtered_s4['overlap_entities'].max())}.

## Interpretation

- S0-S3 use the same conflict-cleaned observations within each dataset; only the assignment rule changes.
- The fully deduplicated files are sensitivity-analysis assignments and do not replace the master-corpus analysis.
- Grouped split proportions are approximate because complete groups cannot be divided.
- S4 unfiltered and domain-filtered cohorts are both retained. Target labels must remain unavailable to tuning and threshold selection.
"""
    path.write_text(report, encoding="utf-8")


def write_manifest(paths: list[Path], output: Path) -> None:
    rows = [
        {
            "path": str(path.relative_to(PART3)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(paths)
    ]
    pd.DataFrame(rows).to_csv(output, index=False)


def main() -> None:
    ASSIGNMENTS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    verify_inputs()
    print("part2_feature_hash_check=pass", flush=True)

    metadata_tables = {key: prepare_metadata(path) for key, path in INPUTS.items()}
    internal_tables: dict[tuple[str, str, str], pd.DataFrame] = {}
    output_paths: list[Path] = []
    summary_rows: list[dict[str, object]] = []
    overlap_rows: list[dict[str, object]] = []
    difference_rows: list[dict[str, object]] = []

    for (dataset, corpus_version), table in metadata_tables.items():
        for scenario in SCENARIOS:
            print(
                f"generating={dataset}|{corpus_version}|{scenario}", flush=True
            )
            assignments = generate_internal_assignments(
                table, dataset, corpus_version, scenario
            )
            internal_tables[(dataset, corpus_version, scenario)] = assignments
            path = ASSIGNMENTS / (
                f"{safe_dataset_name(dataset)}_{corpus_version}_"
                f"{scenario.lower().replace('-', '_')}_assignments.parquet"
            )
            assignments.to_parquet(path, index=False)
            output_paths.append(path)
            summary, overlaps, differences = summarize_internal(
                assignments, dataset, corpus_version, scenario
            )
            summary_rows.extend(summary)
            overlap_rows.extend(overlaps)
            difference_rows.extend(differences)

    s4_summary_rows: list[dict[str, object]] = []
    s4_overlap_rows: list[dict[str, object]] = []
    for corpus_version in ("master", "deduplicated"):
        for source_dataset, target_dataset in (
            ("PhiUSIIL", "ISCX-URL2016-binary"),
            ("ISCX-URL2016-binary", "PhiUSIIL"),
        ):
            source = metadata_tables[(source_dataset, corpus_version)]
            target = metadata_tables[(target_dataset, corpus_version)]
            source_s3 = internal_tables[
                (source_dataset, corpus_version, "S3-domain")
            ]
            assignments, summary, overlaps = generate_s4_assignments(
                source,
                target,
                source_s3,
                source_dataset,
                target_dataset,
                corpus_version,
            )
            path = ASSIGNMENTS / (
                f"s4_{safe_dataset_name(source_dataset)}_to_"
                f"{safe_dataset_name(target_dataset)}_{corpus_version}_assignments.parquet"
            )
            assignments.to_parquet(path, index=False)
            output_paths.append(path)
            s4_summary_rows.extend(summary)
            s4_overlap_rows.extend(overlaps)
            print(
                f"generated_s4={source_dataset}->{target_dataset}|{corpus_version}",
                flush=True,
            )

    split_summary = pd.DataFrame(summary_rows)
    overlap = pd.DataFrame(overlap_rows)
    differences = pd.DataFrame(difference_rows)
    constraints = overlap.copy()
    expected_entity = {
        "S0-row": None,
        "S1-url": "normalized_url",
        "S2-host": "host",
        "S3-domain": "registrable_domain",
    }
    constraints = constraints.loc[
        constraints.apply(
            lambda row: expected_entity[row["scenario"]] == row["entity"], axis=1
        )
    ].copy()
    constraints["expected_overlap"] = 0
    constraints["constraint_pass"] = constraints["overlap_entities"].eq(0)
    s4_summary = pd.DataFrame(s4_summary_rows)
    s4_overlap = pd.DataFrame(s4_overlap_rows)

    summary_path = RESULTS / "split_summary.csv"
    overlap_path = RESULTS / "entity_overlap_audit.csv"
    constraint_path = RESULTS / "constraint_validation.csv"
    difference_path = RESULTS / "partition_difference.csv"
    s4_summary_path = RESULTS / "s4_summary.csv"
    s4_overlap_path = RESULTS / "s4_overlap_audit.csv"
    seed_path = RESULTS / "seed_registry.csv"
    report_path = RESULTS / "split_report.md"
    metadata_path = RESULTS / "split_run_metadata.json"
    manifest_path = RESULTS / "split_output_manifest.csv"

    split_summary.to_csv(summary_path, index=False)
    overlap.to_csv(overlap_path, index=False)
    constraints.to_csv(constraint_path, index=False)
    differences.to_csv(difference_path, index=False)
    s4_summary.to_csv(s4_summary_path, index=False)
    s4_overlap.to_csv(s4_overlap_path, index=False)
    pd.DataFrame(
        {
            "repetition": range(len(SEEDS)),
            "seed": SEEDS,
            "primary": [True] + [False] * (len(SEEDS) - 1),
        }
    ).to_csv(seed_path, index=False)
    write_report(
        split_summary,
        overlap,
        constraints,
        differences,
        s4_summary,
        s4_overlap,
        report_path,
    )
    metadata_path.write_text(
        json.dumps(
            {
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "seeds": list(SEEDS),
                "primary_repetition": 0,
                "target_proportions": TARGET_PROPORTIONS,
                "internal_scenarios": SCENARIOS,
                "grouped_split_method": (
                    "seed-shuffled size-ordered complete groups assigned by "
                    "incremental squared relative row/class target error"
                ),
                "s4_primary_filter": (
                    "exclude target rows whose registrable domain appears in "
                    "source train or validation"
                ),
                "model_training_performed": False,
                "raw_urls_were_visited": False,
                "elapsed_seconds": time.perf_counter() - started,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    output_paths.extend(
        [
            summary_path,
            overlap_path,
            constraint_path,
            difference_path,
            s4_summary_path,
            s4_overlap_path,
            seed_path,
            report_path,
            metadata_path,
        ]
    )
    write_manifest(output_paths, manifest_path)

    if not constraints["constraint_pass"].all():
        raise AssertionError("one or more internal group constraints failed")
    filtered = s4_overlap.loc[
        s4_overlap["target_cohort"] == "primary_domain_filtered"
    ]
    if not filtered["overlap_entities"].eq(0).all():
        raise AssertionError("S4 filtered cohort still has source-fit overlap")
    nonprimary = differences.loc[differences["repetition"] > 0]
    if not nonprimary["fraction_different_from_r00"].gt(0).all():
        raise AssertionError("one or more repeated partitions equal r00")

    print(f"internal_assignment_files={len(internal_tables)}", flush=True)
    print("s4_assignment_files=4", flush=True)
    print(f"constraint_checks={len(constraints)}", flush=True)
    print("constraint_checks_passed=true", flush=True)
    print(f"elapsed_seconds={time.perf_counter() - started:.1f}", flush=True)
    print("model_training_performed=false", flush=True)
    print("part3_status=complete", flush=True)


if __name__ == "__main__":
    main()
