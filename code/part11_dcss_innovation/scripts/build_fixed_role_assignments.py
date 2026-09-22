from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT.parent
PART2 = EXPERIMENT_ROOT / "part2_url_normalization_leakage_audit"
PART3 = EXPERIMENT_ROOT / "part3_s0_s4_data_splits"
OUTPUT = ROOT / "data" / "processed" / "fixed_role_assignments"
RESULTS = ROOT / "results"

DATASETS = ["iscx_url2016_binary", "phiusiil"]
SCENARIOS = ["s0_row", "s3_domain"]
REPETITIONS = [f"r{i:02d}" for i in range(10)]
SPLIT_COLUMNS = [f"split_{repetition}" for repetition in REPETITIONS]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def build_one(dataset: str, scenario: str) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    dedup_path = PART2 / "data" / f"{dataset}_deduplicated_features.parquet"
    master_path = (
        PART3
        / "data"
        / "assignments"
        / f"{dataset}_master_{scenario}_assignments.parquet"
    )
    legacy_path = (
        PART3
        / "data"
        / "assignments"
        / f"{dataset}_deduplicated_{scenario}_assignments.parquet"
    )
    dedup = pd.read_parquet(
        dedup_path,
        columns=[
            "source_row",
            "raw_url_sha256",
            "normalized_url_sha256",
            "feature_vector_sha256",
            "label",
        ],
    )
    master = pd.read_parquet(master_path)
    legacy = pd.read_parquet(
        legacy_path,
        columns=["source_row", "raw_url_sha256", *SPLIT_COLUMNS],
    )

    require(not dedup.duplicated(["source_row", "raw_url_sha256"]).any(), f"Duplicate dedup key: {dataset}")
    require(dedup["normalized_url_sha256"].is_unique, f"Deduplicated URLs are not unique: {dataset}")
    require(not master.duplicated(["source_row", "raw_url_sha256"]).any(), f"Duplicate master key: {dataset}/{scenario}")

    fixed = master.merge(
        dedup,
        on=["source_row", "raw_url_sha256"],
        how="inner",
        suffixes=("_master", "_dedup"),
        validate="one_to_one",
    )
    require(len(fixed) == len(dedup), f"Not every deduplicated row matched master: {dataset}/{scenario}")
    for field in ("label", "normalized_url_sha256", "feature_vector_sha256"):
        require(
            fixed[f"{field}_master"].equals(fixed[f"{field}_dedup"]),
            f"{field} mismatch: {dataset}/{scenario}",
        )
        fixed[field] = fixed.pop(f"{field}_master")
        fixed = fixed.drop(columns=f"{field}_dedup")

    fixed["corpus_version"] = "deduplicated_fixed_role"
    fixed = fixed[master.columns]
    output_path = OUTPUT / f"{dataset}_deduplicated_fixed_role_{scenario}_assignments.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fixed.to_parquet(output_path, index=False, compression="zstd")

    reread = pd.read_parquet(output_path)
    require(len(reread) == len(dedup), f"Written row count mismatch: {dataset}/{scenario}")
    require(reread["normalized_url_sha256"].is_unique, f"Written URLs are not unique: {dataset}/{scenario}")
    require(set(reread["label"].unique()) == {0, 1}, f"Unexpected labels: {dataset}/{scenario}")

    fixed_master = fixed.merge(
        master[["source_row", "raw_url_sha256", *SPLIT_COLUMNS]],
        on=["source_row", "raw_url_sha256"],
        suffixes=("_fixed", "_master"),
        validate="one_to_one",
    )
    legacy_master = legacy.merge(
        master[["source_row", "raw_url_sha256", *SPLIT_COLUMNS]],
        on=["source_row", "raw_url_sha256"],
        suffixes=("_legacy", "_master"),
        validate="one_to_one",
    )

    audit_rows: list[dict[str, object]] = []
    count_rows: list[dict[str, object]] = []
    for repetition, split_column in zip(REPETITIONS, SPLIT_COLUMNS):
        fixed_agreement = (
            fixed_master[f"{split_column}_fixed"]
            == fixed_master[f"{split_column}_master"]
        ).mean()
        legacy_agreement = (
            legacy_master[f"{split_column}_legacy"]
            == legacy_master[f"{split_column}_master"]
        ).mean()
        require(fixed_agreement == 1.0, f"Fixed role mismatch: {dataset}/{scenario}/{repetition}")

        group_overlap = 0
        if scenario == "s3_domain":
            group_sets = {
                role: set(
                    fixed.loc[fixed[split_column] == role, "registrable_domain_sha256"]
                )
                for role in ("train", "validation", "test")
            }
            group_overlap = sum(
                len(group_sets[left] & group_sets[right])
                for left, right in (
                    ("train", "validation"),
                    ("train", "test"),
                    ("validation", "test"),
                )
            )
            require(group_overlap == 0, f"S3 domain overlap: {dataset}/{repetition}")

        audit_rows.append(
            {
                "dataset": dataset,
                "scenario": scenario,
                "repetition": repetition,
                "n_retained": len(fixed),
                "fixed_vs_master_role_agreement": fixed_agreement,
                "legacy_vs_master_role_agreement": legacy_agreement,
                "unique_normalized_urls": fixed["normalized_url_sha256"].nunique(),
                "s3_pairwise_domain_overlap_total": group_overlap,
            }
        )
        for role in ("train", "validation", "test"):
            mask = fixed[split_column] == role
            count_rows.append(
                {
                    "dataset": dataset,
                    "scenario": scenario,
                    "repetition": repetition,
                    "role": role,
                    "n_rows": int(mask.sum()),
                    "n_benign": int((fixed.loc[mask, "label"] == 0).sum()),
                    "n_phishing": int((fixed.loc[mask, "label"] == 1).sum()),
                    "n_domains": int(fixed.loc[mask, "registrable_domain_sha256"].nunique()),
                }
            )

    manifest_row = {
        "dataset": dataset,
        "scenario": scenario,
        "path": str(output_path),
        "rows": len(fixed),
        "bytes": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
        "source_deduplicated_features_sha256": sha256_file(dedup_path),
        "source_master_assignment_sha256": sha256_file(master_path),
    }
    return audit_rows, count_rows, manifest_row


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    audit_rows: list[dict[str, object]] = []
    count_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        for scenario in SCENARIOS:
            audit, counts, manifest = build_one(dataset, scenario)
            audit_rows.extend(audit)
            count_rows.extend(counts)
            manifest_rows.append(manifest)
            print(f"built={dataset}/{scenario}", flush=True)

    audit = pd.DataFrame(audit_rows).sort_values(["dataset", "scenario", "repetition"])
    counts = pd.DataFrame(count_rows).sort_values(["dataset", "scenario", "repetition", "role"])
    manifest = pd.DataFrame(manifest_rows).sort_values(["dataset", "scenario"])
    audit.to_csv(RESULTS / "fixed_role_assignment_audit.csv", index=False)
    counts.to_csv(RESULTS / "fixed_role_split_counts.csv", index=False)
    manifest.to_csv(RESULTS / "fixed_role_assignment_manifest.csv", index=False)

    summary = {
        "stage": "11B",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "assignment_files": len(manifest),
        "audit_rows": len(audit),
        "fixed_role_agreement_min": float(audit["fixed_vs_master_role_agreement"].min()),
        "fixed_role_agreement_max": float(audit["fixed_vs_master_role_agreement"].max()),
        "legacy_role_agreement_min": float(audit["legacy_vs_master_role_agreement"].min()),
        "legacy_role_agreement_max": float(audit["legacy_vs_master_role_agreement"].max()),
        "s3_domain_overlap_max": int(audit["s3_pairwise_domain_overlap_total"].max()),
    }
    (RESULTS / "fixed_role_assignment_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
