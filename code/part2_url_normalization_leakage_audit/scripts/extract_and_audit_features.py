from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from feature_extraction import (
    FEATURE_NAMES,
    FEATURE_SPECS,
    extract_url_features,
    feature_vector_sha256,
)
from normalize_and_audit import load_iscx_binary, load_phiusiil, sha256_bytes
from url_normalization import PSL_SNAPSHOT


PART2 = Path(__file__).resolve().parents[1]
DATA = PART2 / "data"
RESULTS = PART2 / "results"
LOGS = PART2 / "logs"
REVIEW = PART2 / "review"
CORRELATION_THRESHOLD = 0.95
CORPORA = {
    ("PhiUSIIL", "master"): DATA / "phiusiil_conflict_cleaned_master.parquet",
    ("ISCX-URL2016-binary", "master"): DATA / "iscx_url2016_binary_conflict_cleaned_master.parquet",
    ("PhiUSIIL", "deduplicated"): DATA / "phiusiil_deduplicated_candidate.parquet",
    ("ISCX-URL2016-binary", "deduplicated"): DATA / "iscx_url2016_binary_deduplicated_candidate.parquet",
}
FEATURE_TABLE_PATHS = {
    ("PhiUSIIL", "master"): DATA / "phiusiil_master_features.parquet",
    ("ISCX-URL2016-binary", "master"): DATA / "iscx_url2016_binary_master_features.parquet",
    ("PhiUSIIL", "deduplicated"): DATA / "phiusiil_deduplicated_features.parquet",
    ("ISCX-URL2016-binary", "deduplicated"): DATA / "iscx_url2016_binary_deduplicated_features.parquet",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_raw_url_lookup() -> dict[str, str]:
    raw = pd.concat([load_phiusiil(), load_iscx_binary()], ignore_index=True)
    raw["raw_url_sha256"] = raw["raw_url"].map(sha256_bytes)
    collision_check = raw.groupby("raw_url_sha256")["raw_url"].nunique()
    if not collision_check.le(1).all():
        raise ValueError("Unexpected SHA-256 collision in raw URLs")
    return (
        raw.drop_duplicates("raw_url_sha256")
        .set_index("raw_url_sha256")["raw_url"]
        .to_dict()
    )


def extract_unique_feature_map(
    required_hashes: set[str], raw_lookup: dict[str, str]
) -> dict[str, dict[str, int | float | str]]:
    missing = required_hashes - raw_lookup.keys()
    if missing:
        raise ValueError(f"Raw URL lookup is missing {len(missing)} hashes")

    extracted: dict[str, dict[str, int | float | str]] = {}
    total = len(required_hashes)
    start = time.perf_counter()
    for index, raw_hash in enumerate(sorted(required_hashes), start=1):
        features = extract_url_features(raw_lookup[raw_hash])
        extracted[raw_hash] = {
            **features,
            "feature_vector_sha256": feature_vector_sha256(features),
        }
        if index % 50000 == 0 or index == total:
            elapsed = time.perf_counter() - start
            print(
                f"feature_progress={index}/{total}|elapsed_seconds={elapsed:.1f}",
                flush=True,
            )
    return extracted


def build_feature_table(
    corpus: pd.DataFrame,
    feature_map: dict[str, dict[str, int | float | str]],
    corpus_version: str,
) -> pd.DataFrame:
    records = [feature_map[value] for value in corpus["raw_url_sha256"]]
    features = pd.DataFrame.from_records(records, index=corpus.index)
    integer_features = [spec.name for spec in FEATURE_SPECS if spec.dtype == "int"]
    float_features = [spec.name for spec in FEATURE_SPECS if spec.dtype == "float"]
    features[integer_features] = features[integer_features].astype("int32")
    features[float_features] = features[float_features].astype("float64")

    metadata_columns = [
        "dataset",
        "source_row",
        "label",
        "label_name",
        "raw_url_sha256",
        "normalized_url_sha256",
        "host",
        "registrable_domain",
    ]
    if "source_file" in corpus.columns:
        metadata_columns.insert(2, "source_file")
    table = pd.concat(
        [corpus[metadata_columns].reset_index(drop=True), features.reset_index(drop=True)],
        axis=1,
    )
    table.insert(1, "corpus_version", corpus_version)
    return table


def feature_dictionary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature_order": index,
                "feature": spec.name,
                "group": spec.group,
                "dtype": spec.dtype,
                "definition": spec.definition,
                "requires_network": False,
                "uses_target_statistics": False,
            }
            for index, spec in enumerate(FEATURE_SPECS, start=1)
        ]
    )


def quality_audit(
    table: pd.DataFrame, dataset: str, corpus_version: str
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for feature in FEATURE_NAMES:
        values = table[feature].astype(float)
        counts = values.value_counts(dropna=False)
        rows.append(
            {
                "dataset": dataset,
                "corpus_version": corpus_version,
                "feature": feature,
                "rows": len(values),
                "missing_count": int(values.isna().sum()),
                "infinite_count": int(np.isinf(values).sum()),
                "unique_count": int(values.nunique(dropna=True)),
                "constant": int(values.nunique(dropna=True) <= 1),
                "zero_fraction": float(values.eq(0).mean()),
                "most_frequent_value_fraction": float(counts.iloc[0] / len(values)),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
                "min": float(values.min()),
                "q25": float(values.quantile(0.25)),
                "median": float(values.median()),
                "q75": float(values.quantile(0.75)),
                "max": float(values.max()),
            }
        )
    return rows


def class_statistics(
    table: pd.DataFrame, dataset: str, corpus_version: str
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label, label_name in ((0, "benign"), (1, "phishing")):
        subset = table.loc[table["label"] == label]
        for feature in FEATURE_NAMES:
            values = subset[feature].astype(float)
            rows.append(
                {
                    "dataset": dataset,
                    "corpus_version": corpus_version,
                    "label": label,
                    "label_name": label_name,
                    "feature": feature,
                    "rows": len(values),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)),
                    "q25": float(values.quantile(0.25)),
                    "median": float(values.median()),
                    "q75": float(values.quantile(0.75)),
                }
            )
    return rows


def correlation_audit(
    table: pd.DataFrame, dataset: str, corpus_version: str
) -> list[dict[str, object]]:
    correlation = table[list(FEATURE_NAMES)].corr(method="spearman")
    rows: list[dict[str, object]] = []
    for left_index, left in enumerate(FEATURE_NAMES):
        for right in FEATURE_NAMES[left_index + 1 :]:
            value = correlation.loc[left, right]
            if pd.notna(value) and abs(value) >= CORRELATION_THRESHOLD:
                rows.append(
                    {
                        "dataset": dataset,
                        "corpus_version": corpus_version,
                        "feature_a": left,
                        "feature_b": right,
                        "spearman": float(value),
                        "absolute_spearman": float(abs(value)),
                    }
                )
    return rows


def vector_duplicate_audit(
    table: pd.DataFrame, dataset: str, corpus_version: str
) -> tuple[dict[str, object], pd.DataFrame]:
    groups = (
        table.groupby("feature_vector_sha256")
        .agg(
            row_count=("label", "size"),
            label_count=("label", "nunique"),
            labels=("label", lambda values: ",".join(map(str, sorted(set(values))))),
            benign_rows=("label", lambda values: int((values == 0).sum())),
            phishing_rows=("label", lambda values: int((values == 1).sum())),
            normalized_url_count=("normalized_url_sha256", "nunique"),
        )
        .reset_index()
    )
    duplicate_groups = groups[groups["row_count"] > 1].copy()
    duplicate_groups.insert(0, "dataset", dataset)
    duplicate_groups.insert(1, "corpus_version", corpus_version)
    summary = {
        "dataset": dataset,
        "corpus_version": corpus_version,
        "rows": len(table),
        "unique_feature_vectors": int(groups.shape[0]),
        "duplicate_extra_rows": int(len(table) - groups.shape[0]),
        "duplicate_groups": int(len(duplicate_groups)),
        "conflicting_vector_groups": int(
            (duplicate_groups["label_count"] > 1).sum()
        ),
        "conflicting_vector_rows": int(
            duplicate_groups.loc[
                duplicate_groups["label_count"] > 1, "row_count"
            ].sum()
        ),
    }
    return summary, duplicate_groups


def standardized_mean_difference(left: np.ndarray, right: np.ndarray) -> float:
    pooled = math.sqrt((float(np.var(left, ddof=1)) + float(np.var(right, ddof=1))) / 2)
    difference = float(np.mean(left) - np.mean(right))
    if pooled == 0:
        return 0.0 if difference == 0 else float("nan")
    return difference / pooled


def distribution_shift(
    phi: pd.DataFrame,
    iscx: pd.DataFrame,
    corpus_version: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    scopes = (("all", None), ("benign", 0), ("phishing", 1))
    for scope, label in scopes:
        left_frame = phi if label is None else phi.loc[phi["label"] == label]
        right_frame = iscx if label is None else iscx.loc[iscx["label"] == label]
        for feature in FEATURE_NAMES:
            left = left_frame[feature].to_numpy(dtype=float)
            right = right_frame[feature].to_numpy(dtype=float)
            ks_result = ks_2samp(left, right, alternative="two-sided", method="asymp")
            smd = standardized_mean_difference(left, right)
            rows.append(
                {
                    "corpus_version": corpus_version,
                    "scope": scope,
                    "feature": feature,
                    "phi_rows": len(left),
                    "iscx_rows": len(right),
                    "phi_mean": float(np.mean(left)),
                    "iscx_mean": float(np.mean(right)),
                    "standardized_mean_difference_phi_minus_iscx": smd,
                    "absolute_smd": abs(smd) if math.isfinite(smd) else float("nan"),
                    "wasserstein_distance": float(wasserstein_distance(left, right)),
                    "ks_statistic": float(ks_result.statistic),
                    "ks_pvalue": float(ks_result.pvalue),
                }
            )
    return rows


def cross_dataset_vector_overlap(
    phi: pd.DataFrame, iscx: pd.DataFrame, corpus_version: str
) -> pd.DataFrame:
    combined = pd.concat([phi, iscx], ignore_index=True)
    groups = (
        combined.groupby("feature_vector_sha256")
        .agg(
            dataset_count=("dataset", "nunique"),
            row_count=("label", "size"),
            label_count=("label", "nunique"),
        )
        .reset_index()
    )
    groups = groups[groups["dataset_count"] > 1].copy()
    groups.insert(0, "corpus_version", corpus_version)
    return groups


def write_report(
    quality: pd.DataFrame,
    correlations: pd.DataFrame,
    vector_summary: pd.DataFrame,
    shifts: pd.DataFrame,
    cross_vectors: pd.DataFrame,
    path: Path,
) -> None:
    master_summary = vector_summary[vector_summary["corpus_version"] == "master"]
    phi_summary = master_summary.loc[master_summary["dataset"] == "PhiUSIIL"].iloc[0]
    iscx_summary = master_summary.loc[
        master_summary["dataset"] == "ISCX-URL2016-binary"
    ].iloc[0]
    master_quality = quality[quality["corpus_version"] == "master"]
    constant_count = int(master_quality["constant"].sum())
    high_corr_count = int(
        (correlations["corpus_version"] == "master").sum()
    ) if not correlations.empty else 0
    master_shifts = shifts[shifts["corpus_version"] == "master"].copy()
    top_shift = (
        master_shifts.sort_values("absolute_smd", ascending=False)
        .groupby("scope", sort=False)
        .head(5)
    )
    shift_lines = "\n".join(
        f"- {row.scope}: `{row.feature}` (absolute SMD {row.absolute_smd:.3f})"
        for row in top_shift.itertuples()
    )
    report = f"""# Feature Extraction and Audit Report

## Feature Space

- Core features: {len(FEATURE_NAMES)} deterministic lexical and structural features in five groups.
- Network-dependent features: 0.
- Features derived from target labels or complete-dataset vocabulary: 0.
- Public Suffix List: frozen local snapshot, private suffixes disabled.
- Feature tables contain hashes and grouping identifiers but no raw URL column.

## Feature Tables

| Dataset | Master rows | Unique feature vectors | Duplicate extra rows | Conflicting vector groups | Rows in conflicting vectors |
|---|---:|---:|---:|---:|---:|
| PhiUSIIL | {int(phi_summary['rows'])} | {int(phi_summary['unique_feature_vectors'])} | {int(phi_summary['duplicate_extra_rows'])} | {int(phi_summary['conflicting_vector_groups'])} | {int(phi_summary['conflicting_vector_rows'])} |
| ISCX-URL2016 binary | {int(iscx_summary['rows'])} | {int(iscx_summary['unique_feature_vectors'])} | {int(iscx_summary['duplicate_extra_rows'])} | {int(iscx_summary['conflicting_vector_groups'])} | {int(iscx_summary['conflicting_vector_rows'])} |

- Missing or infinite feature values: {int(master_quality['missing_count'].sum() + master_quality['infinite_count'].sum())}.
- Constant dataset-feature combinations in the master corpora: {constant_count}.
- Feature pairs with absolute Spearman correlation at least {CORRELATION_THRESHOLD}: {high_corr_count} across the two master corpora.
- Identical feature vectors shared across the two master corpora: {len(cross_vectors[cross_vectors['corpus_version'] == 'master'])} groups.

## Largest Source Shifts

The following are the five largest standardized mean differences within each comparison scope. Positive signs in the CSV mean a larger PhiUSIIL mean; this list reports magnitudes only.

{shift_lines}

## Interpretation Limits

- Distribution differences describe source shift; they do not establish which source is more realistic.
- Correlation does not justify deleting a feature before the training-only selection stage. The audit only flags redundancy for later sensitivity analysis.
- Identical feature vectors with conflicting labels are reported, not automatically relabeled.
- A 40-row deterministic sample was inspected offline for parsing/count consistency with no mismatch observed. This did not verify label truth and no URL was opened or visited.
"""
    path.write_text(report, encoding="utf-8")


def write_manifest(paths: list[Path], path: Path) -> None:
    rows = [
        {
            "path": str(item.relative_to(PART2)).replace("\\", "/"),
            "bytes": item.stat().st_size,
            "sha256": sha256_file(item),
        }
        for item in sorted(paths)
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    REVIEW.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    corpora = {key: pd.read_parquet(path) for key, path in CORPORA.items()}
    required_hashes = set().union(
        *(set(frame["raw_url_sha256"]) for frame in corpora.values())
    )
    print(f"required_unique_raw_urls={len(required_hashes)}", flush=True)
    raw_lookup = load_raw_url_lookup()
    feature_map = extract_unique_feature_map(required_hashes, raw_lookup)

    feature_tables: dict[tuple[str, str], pd.DataFrame] = {}
    output_paths: list[Path] = []
    for key, corpus in corpora.items():
        dataset, corpus_version = key
        table = build_feature_table(corpus, feature_map, corpus_version)
        path = FEATURE_TABLE_PATHS[key]
        table.to_parquet(path, index=False)
        feature_tables[key] = table
        output_paths.append(path)
        print(f"feature_table={path.name}|rows={len(table)}", flush=True)

    dictionary_path = RESULTS / "feature_dictionary.csv"
    quality_path = RESULTS / "feature_quality_audit.csv"
    class_stats_path = RESULTS / "class_feature_statistics.csv"
    correlations_path = RESULTS / "high_correlation_pairs.csv"
    vector_summary_path = RESULTS / "feature_vector_duplicate_summary.csv"
    vector_groups_path = RESULTS / "feature_vector_duplicate_groups.csv"
    shift_path = RESULTS / "feature_distribution_shift.csv"
    cross_vector_path = RESULTS / "cross_dataset_feature_vector_overlap.csv"
    report_path = RESULTS / "feature_audit_report.md"
    metadata_path = RESULTS / "feature_run_metadata.json"
    review_path = REVIEW / "feature_spotcheck_sample.csv"
    review_note_path = REVIEW / "spotcheck_review.md"
    manifest_path = RESULTS / "feature_output_manifest.csv"

    feature_dictionary_frame().to_csv(dictionary_path, index=False)
    quality_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    correlation_rows: list[dict[str, object]] = []
    vector_summaries: list[dict[str, object]] = []
    vector_groups: list[pd.DataFrame] = []
    for (dataset, corpus_version), table in feature_tables.items():
        quality_rows.extend(quality_audit(table, dataset, corpus_version))
        class_rows.extend(class_statistics(table, dataset, corpus_version))
        correlation_rows.extend(correlation_audit(table, dataset, corpus_version))
        summary, groups = vector_duplicate_audit(table, dataset, corpus_version)
        vector_summaries.append(summary)
        vector_groups.append(groups)

    quality = pd.DataFrame(quality_rows)
    class_stats = pd.DataFrame(class_rows)
    correlations = pd.DataFrame(correlation_rows)
    vector_summary = pd.DataFrame(vector_summaries)
    vector_group_table = pd.concat(vector_groups, ignore_index=True)
    shift_rows: list[dict[str, object]] = []
    cross_vectors: list[pd.DataFrame] = []
    for corpus_version in ("master", "deduplicated"):
        phi = feature_tables[("PhiUSIIL", corpus_version)]
        iscx = feature_tables[("ISCX-URL2016-binary", corpus_version)]
        shift_rows.extend(distribution_shift(phi, iscx, corpus_version))
        cross_vectors.append(cross_dataset_vector_overlap(phi, iscx, corpus_version))
    shifts = pd.DataFrame(shift_rows)
    cross_vector_table = pd.concat(cross_vectors, ignore_index=True)

    quality.to_csv(quality_path, index=False)
    class_stats.to_csv(class_stats_path, index=False)
    correlations.to_csv(correlations_path, index=False)
    vector_summary.to_csv(vector_summary_path, index=False)
    vector_group_table.to_csv(vector_groups_path, index=False)
    shifts.to_csv(shift_path, index=False)
    cross_vector_table.to_csv(cross_vector_path, index=False)

    review_frames = []
    for (dataset, corpus_version), table in feature_tables.items():
        if corpus_version != "deduplicated":
            continue
        for label in (0, 1):
            sample = (
                table.loc[table["label"] == label]
                .sort_values("raw_url_sha256")
                .head(10)
                .copy()
            )
            sample.insert(
                0,
                "untrusted_url_text_do_not_visit",
                sample["raw_url_sha256"].map(raw_lookup),
            )
            review_frames.append(sample)
    pd.concat(review_frames, ignore_index=True).to_csv(review_path, index=False)
    review_note_path.write_text(
        """# Offline Feature Spot Check

- Scope: 40 deterministic records, with 10 records from each dataset/class combination.
- Checked fields: URL, hostname, path, query and fragment lengths; subdomain depth; path segments; query parameters; IP-host flag.
- Result: no parsing or count inconsistency was observed in the displayed sample.
- Safety: URL strings were inspected as offline text only. No URL was opened, resolved, or visited.
- Limit: this check does not verify whether the source label is factually correct and is not a substitute for model validation.
""",
        encoding="utf-8",
    )

    metadata = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_count": len(FEATURE_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "feature_source": "feature-ready conservative normalized URL with fragment retained",
        "raw_urls_were_visited": False,
        "psl_snapshot": str(PSL_SNAPSHOT.relative_to(PART2)).replace("\\", "/"),
        "psl_sha256": sha256_file(PSL_SNAPSHOT),
        "include_psl_private_domains": False,
        "correlation_threshold": CORRELATION_THRESHOLD,
        "spotcheck_status": "40-row offline assistant inspection completed; user review optional",
        "elapsed_seconds": time.perf_counter() - started,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(
        quality,
        correlations,
        vector_summary,
        shifts,
        cross_vector_table,
        report_path,
    )
    output_paths.extend(
        [
            dictionary_path,
            quality_path,
            class_stats_path,
            correlations_path,
            vector_summary_path,
            vector_groups_path,
            shift_path,
            cross_vector_path,
            report_path,
            metadata_path,
            review_path,
            review_note_path,
            PSL_SNAPSHOT,
        ]
    )
    write_manifest(output_paths, manifest_path)
    print(f"feature_count={len(FEATURE_NAMES)}", flush=True)
    print(f"missing_feature_values={int(quality['missing_count'].sum())}", flush=True)
    print(f"infinite_feature_values={int(quality['infinite_count'].sum())}", flush=True)
    print(f"elapsed_seconds={time.perf_counter() - started:.1f}", flush=True)
    print("raw_urls_were_visited=false", flush=True)
    print("feature_stage_status=complete", flush=True)


if __name__ == "__main__":
    main()
