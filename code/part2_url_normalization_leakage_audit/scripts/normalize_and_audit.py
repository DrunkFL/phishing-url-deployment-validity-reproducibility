from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from url_normalization import normalize_url


PART2 = Path(__file__).resolve().parents[1]
ROOT = PART2.parent
RAW = ROOT / "data" / "raw"
OUTPUT_DATA = PART2 / "data"
RESULTS = PART2 / "results"
PHI_PATH = (
    RAW
    / "phiusiil_uci"
    / "extracted"
    / "PhiUSIIL_Phishing_URL_Dataset.csv"
)
ISCX_DIR = (
    RAW
    / "iscx_url2016_repository_mirror"
    / "extracted_url_lists"
)
MANIFEST = ROOT / "data" / "metadata" / "data_manifest.csv"


def sha256_bytes(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="surrogatepass")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_inputs() -> None:
    required = {
        PHI_PATH,
        ISCX_DIR / "Benign_list_big_final.csv",
        ISCX_DIR / "phishing_dataset.csv",
    }
    with MANIFEST.open("r", encoding="utf-8", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    expected = {ROOT / row["local_path"]: row["sha256"] for row in manifest_rows}
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
        if expected.get(path) != sha256_file(path):
            raise ValueError(f"Input hash mismatch: {path}")


def load_phiusiil() -> pd.DataFrame:
    frame = pd.read_csv(PHI_PATH, usecols=["URL", "label"], low_memory=False)
    if not set(frame["label"].dropna().unique()).issubset({0, 1}):
        raise ValueError("Unexpected PhiUSIIL label")
    return pd.DataFrame(
        {
            "dataset": "PhiUSIIL",
            "source_row": frame.index + 2,
            "raw_url": frame["URL"].astype(str),
            "label": (frame["label"] == 0).astype("int8"),
        }
    )


def read_url_list(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", errors="replace", newline=None) as handle:
        return [line.strip() for line in handle if line.strip()]


def load_iscx_binary() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for filename, label in (
        ("Benign_list_big_final.csv", 0),
        ("phishing_dataset.csv", 1),
    ):
        for source_row, url in enumerate(read_url_list(ISCX_DIR / filename), start=1):
            records.append(
                {
                    "dataset": "ISCX-URL2016-binary",
                    "source_row": source_row,
                    "source_file": filename,
                    "raw_url": url,
                    "label": label,
                }
            )
    return pd.DataFrame.from_records(records)


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    unique_urls = frame["raw_url"].drop_duplicates().tolist()
    normalized = {url: normalize_url(url) for url in unique_urls}
    result = frame.copy()
    result["label_name"] = result["label"].map({0: "benign", 1: "phishing"})
    result["raw_url_sha256"] = result["raw_url"].map(sha256_bytes)
    result["normalized_url"] = result["raw_url"].map(
        lambda value: normalized[value].value
    )
    result["normalized_url_sha256"] = result["normalized_url"].map(sha256_bytes)
    result["host"] = result["raw_url"].map(lambda value: normalized[value].host)
    result["registrable_domain"] = result["raw_url"].map(
        lambda value: normalized[value].registrable_domain
    )
    result["parse_status"] = result["raw_url"].map(
        lambda value: normalized[value].parse_status
    )
    result["normalization_changed"] = result["raw_url"].map(
        lambda value: normalized[value].changed
    )
    return result


def group_table(frame: pd.DataFrame, key: str, duplicate_type: str) -> pd.DataFrame:
    grouped = (
        frame.groupby(["dataset", key], dropna=False)
        .agg(
            row_count=("label", "size"),
            label_count=("label", "nunique"),
            labels=("label", lambda values: ",".join(map(str, sorted(set(values))))),
        )
        .reset_index()
    )
    grouped = grouped[grouped["row_count"] > 1].copy()
    grouped.insert(1, "duplicate_type", duplicate_type)
    return grouped.rename(columns={key: "group_sha256"})


def audit_dataset(
    frame: pd.DataFrame,
) -> tuple[list[dict[str, object]], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset = str(frame["dataset"].iloc[0])
    exact_groups = group_table(frame, "raw_url_sha256", "exact_raw")
    normalized_groups = group_table(
        frame, "normalized_url_sha256", "normalized"
    )
    conflict_hashes = set(
        normalized_groups.loc[
            normalized_groups["label_count"] > 1, "group_sha256"
        ]
    )
    master = frame[frame["parse_status"] == "ok"].copy()
    master = master[
        ~master["normalized_url_sha256"].isin(conflict_hashes)
    ]
    clean = master.drop_duplicates("normalized_url_sha256", keep="first").copy()

    metrics = {
        "input_rows": len(frame),
        "benign_rows": int((frame["label"] == 0).sum()),
        "phishing_rows": int((frame["label"] == 1).sum()),
        "unique_raw_urls": frame["raw_url_sha256"].nunique(),
        "exact_duplicate_extra_rows": len(frame)
        - frame["raw_url_sha256"].nunique(),
        "unique_normalized_urls": frame["normalized_url_sha256"].nunique(),
        "normalized_duplicate_extra_rows": len(frame)
        - frame["normalized_url_sha256"].nunique(),
        "conflicting_normalized_groups": len(conflict_hashes),
        "conflicting_rows": int(
            frame["normalized_url_sha256"].isin(conflict_hashes).sum()
        ),
        "unparsed_rows": int((frame["parse_status"] != "ok").sum()),
        "normalization_changed_rows": int(frame["normalization_changed"].sum()),
        "unique_registrable_domains": frame.loc[
            frame["registrable_domain"] != "", "registrable_domain"
        ].nunique(),
        "conflict_cleaned_master_rows": len(master),
        "deduplicated_candidate_rows": len(clean),
        "deduplicated_candidate_benign_rows": int((clean["label"] == 0).sum()),
        "deduplicated_candidate_phishing_rows": int((clean["label"] == 1).sum()),
    }
    rows = [
        {"dataset": dataset, "metric": metric, "value": int(value)}
        for metric, value in metrics.items()
    ]
    duplicate_groups = pd.concat(
        [exact_groups, normalized_groups], ignore_index=True
    )
    return rows, master, clean, duplicate_groups


def write_output_manifest(paths: list[Path]) -> None:
    records = [
        {
            "path": str(path.relative_to(PART2)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(paths)
    ]
    pd.DataFrame(records).to_csv(RESULTS / "output_manifest.csv", index=False)


def metric_value(summary: pd.DataFrame, dataset: str, metric: str) -> int:
    values = summary.loc[
        (summary["dataset"] == dataset) & (summary["metric"] == metric), "value"
    ]
    if len(values) != 1:
        raise ValueError(f"Missing or repeated summary metric: {dataset}/{metric}")
    return int(values.iloc[0])


def write_audit_report(summary: pd.DataFrame, path: Path) -> None:
    phi = "PhiUSIIL"
    iscx = "ISCX-URL2016-binary"
    report = f"""# Part 2 Audit Report

## Scope

This stage normalized URL strings and audited exact duplicates, normalized duplicates, conflicting labels, and cross-dataset overlap. It did not split data, train a model, compute features, or visit any URL.

## Results

| Dataset | Input rows | Exact duplicate extra rows | Normalized duplicate extra rows | Conflicting groups | Clean benign | Clean phishing | Clean total |
|---|---:|---:|---:|---:|---:|---:|---:|
| PhiUSIIL | {metric_value(summary, phi, 'input_rows')} | {metric_value(summary, phi, 'exact_duplicate_extra_rows')} | {metric_value(summary, phi, 'normalized_duplicate_extra_rows')} | {metric_value(summary, phi, 'conflicting_normalized_groups')} | {metric_value(summary, phi, 'deduplicated_candidate_benign_rows')} | {metric_value(summary, phi, 'deduplicated_candidate_phishing_rows')} | {metric_value(summary, phi, 'deduplicated_candidate_rows')} |
| ISCX-URL2016 binary | {metric_value(summary, iscx, 'input_rows')} | {metric_value(summary, iscx, 'exact_duplicate_extra_rows')} | {metric_value(summary, iscx, 'normalized_duplicate_extra_rows')} | {metric_value(summary, iscx, 'conflicting_normalized_groups')} | {metric_value(summary, iscx, 'deduplicated_candidate_benign_rows')} | {metric_value(summary, iscx, 'deduplicated_candidate_phishing_rows')} | {metric_value(summary, iscx, 'deduplicated_candidate_rows')} |

- Cross-dataset identical normalized URL groups: {metric_value(summary, 'cross_dataset', 'overlapping_normalized_url_groups')}.
- Cross-dataset overlapping registrable domains: {metric_value(summary, 'cross_dataset', 'overlapping_registrable_domains')}.
- PhiUSIIL original labels were recoded from `1 = legitimate, 0 = phishing` to `0 = benign, 1 = phishing`.
- ISCX includes only the benign and phishing source lists in this binary analysis.

## Interpretation

Normalization exposed additional duplicate records beyond exact string matching. The clean candidates remove all within-dataset conflicting normalized groups and keep one deterministic source row per normalized URL. They are intermediate inputs, not final train/test sets.

The shared registrable domains show that a future random row split can place related URLs on both sides of the evaluation. Part 3 should therefore compare a conventional stratified split with a registrable-domain group split. No predictive-performance claim can be made from Part 2 alone.

## Limitations

- The ISCX raw URL files came from a pinned public repository mirror because the official registration endpoint was unavailable; identity with the official archive is not verified.
- The rules are conservative and do not perform approximate edit-distance or semantic near-duplicate detection.
- Public Suffix List network updates were disabled; the bundled `tldextract` snapshot was used for reproducibility.
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    OUTPUT_DATA.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    verify_inputs()
    print("input_hash_check=pass")

    frames = [normalize_frame(load_phiusiil()), normalize_frame(load_iscx_binary())]
    summary_rows: list[dict[str, object]] = []
    clean_frames: list[pd.DataFrame] = []
    duplicate_tables: list[pd.DataFrame] = []
    output_paths: list[Path] = []

    for frame in frames:
        dataset = str(frame["dataset"].iloc[0])
        summary, master, clean, duplicates = audit_dataset(frame)
        summary_rows.extend(summary)
        clean_frames.append(clean.assign(dataset=dataset))
        duplicate_tables.append(duplicates)

        safe_name = dataset.lower().replace("-", "_")
        normalized_path = OUTPUT_DATA / f"{safe_name}_normalized.parquet"
        master_path = OUTPUT_DATA / f"{safe_name}_conflict_cleaned_master.parquet"
        clean_path = OUTPUT_DATA / f"{safe_name}_deduplicated_candidate.parquet"
        export_columns = [column for column in frame.columns if column != "raw_url"]
        frame[export_columns].to_parquet(normalized_path, index=False)
        master[export_columns].to_parquet(master_path, index=False)
        clean[export_columns].to_parquet(clean_path, index=False)
        output_paths.extend([normalized_path, master_path, clean_path])

    combined = pd.concat(frames, ignore_index=True)
    cross_url = (
        combined.groupby("normalized_url_sha256")
        .agg(dataset_count=("dataset", "nunique"), row_count=("label", "size"))
        .reset_index()
    )
    cross_url = cross_url[cross_url["dataset_count"] > 1]
    cross_domain = (
        combined.loc[combined["registrable_domain"] != ""]
        .groupby("registrable_domain")
        .agg(dataset_count=("dataset", "nunique"), row_count=("label", "size"))
        .reset_index()
    )
    cross_domain = cross_domain[cross_domain["dataset_count"] > 1]

    summary_rows.extend(
        [
            {
                "dataset": "cross_dataset",
                "metric": "overlapping_normalized_url_groups",
                "value": len(cross_url),
            },
            {
                "dataset": "cross_dataset",
                "metric": "overlapping_registrable_domains",
                "value": len(cross_domain),
            },
        ]
    )

    summary_path = RESULTS / "audit_summary.csv"
    duplicates_path = RESULTS / "duplicate_groups.csv"
    conflicts_path = RESULTS / "conflicting_label_groups.csv"
    cross_url_path = RESULTS / "cross_dataset_url_overlap.csv"
    cross_domain_path = RESULTS / "cross_dataset_domain_overlap.csv"
    run_metadata_path = RESULTS / "run_metadata.json"
    report_path = RESULTS / "audit_report.md"

    summary_frame = pd.DataFrame(summary_rows)
    duplicate_frame = pd.concat(duplicate_tables, ignore_index=True)
    conflict_frame = duplicate_frame[
        (duplicate_frame["duplicate_type"] == "normalized")
        & (duplicate_frame["label_count"] > 1)
    ]
    summary_frame.to_csv(summary_path, index=False)
    duplicate_frame.to_csv(duplicates_path, index=False)
    conflict_frame.to_csv(conflicts_path, index=False)
    cross_url.to_csv(cross_url_path, index=False)
    cross_domain.to_csv(cross_domain_path, index=False)
    run_metadata_path.write_text(
        json.dumps(
            {
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "raw_urls_were_visited": False,
                "phiusiil_label_mapping": {"original_1": 0, "original_0": 1},
                "iscx_included_classes": ["benign", "phishing"],
                "public_suffix_updates_disabled": True,
                "clean_candidate_policy": (
                    "parse_status ok; remove all within-dataset conflicting normalized "
                    "groups; keep first source row per normalized URL"
                ),
                "master_corpus_policy": (
                    "parse_status ok; remove all within-dataset conflicting normalized "
                    "groups; retain same-label duplicate rows"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_audit_report(summary_frame, report_path)
    output_paths.extend(
        [
            summary_path,
            duplicates_path,
            conflicts_path,
            cross_url_path,
            cross_domain_path,
            run_metadata_path,
            report_path,
        ]
    )
    write_output_manifest(output_paths)

    print(summary_frame.to_string(index=False))
    print("raw_urls_were_visited=false")
    print("part2_status=complete")


if __name__ == "__main__":
    main()
