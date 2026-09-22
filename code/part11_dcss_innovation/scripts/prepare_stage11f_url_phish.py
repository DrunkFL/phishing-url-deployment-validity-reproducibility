from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT.parent
PART2 = EXPERIMENT_ROOT / "part2_url_normalization_leakage_audit"
sys.path.insert(0, str(PART2 / "scripts"))

from feature_extraction import FEATURE_NAMES, extract_url_features  # noqa: E402
from url_normalization import normalize_url  # noqa: E402


SNAPSHOT = ROOT / "data" / "external_snapshot" / "url_phish_mendeley_v1"
CSV_PATH = (
    SNAPSHOT
    / "extracted"
    / "URL-Phish A Feature-Engineered Dataset for Phishin"
    / "Dataset.csv"
)
OUTPUT = ROOT / "data" / "processed" / "stage11f_url_phish_v1.parquet"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def source_sets() -> tuple[set[str], set[str], dict[str, dict[str, int]]]:
    url_hashes: set[str] = set()
    domain_hashes: set[str] = set()
    counts: dict[str, dict[str, int]] = {}
    for dataset in ("phiusiil", "iscx_url2016_binary"):
        path = PART2 / "data" / f"{dataset}_conflict_cleaned_master.parquet"
        frame = pd.read_parquet(path, columns=["normalized_url_sha256", "registrable_domain"])
        dataset_urls = set(frame["normalized_url_sha256"].astype(str))
        dataset_domains = {sha256_text(value) for value in frame["registrable_domain"].astype(str)}
        url_hashes.update(dataset_urls)
        domain_hashes.update(dataset_domains)
        counts[dataset] = {
            "unique_normalized_urls": len(dataset_urls),
            "unique_registrable_domains": len(dataset_domains),
        }
    return url_hashes, domain_hashes, counts


def main() -> None:
    raw = pd.read_csv(CSV_PATH)
    required = {"url", "label"}
    if not required <= set(raw.columns):
        raise ValueError(f"Missing columns: {required - set(raw.columns)}")
    if set(raw["label"].dropna().unique()) != {0, 1}:
        raise ValueError("URL-Phish labels must be exactly {0, 1}")

    records = []
    for source_row, row in enumerate(raw[["url", "label"]].itertuples(index=False)):
        raw_url = str(row.url)
        normalized = normalize_url(raw_url)
        records.append(
            {
                "source_row": source_row,
                "label": int(row.label),
                "raw_url_sha256": sha256_text(raw_url),
                "normalized_url": normalized.value,
                "normalized_url_sha256": sha256_text(normalized.value),
                "host": normalized.host,
                "registrable_domain": normalized.registrable_domain,
                "registrable_domain_sha256": sha256_text(normalized.registrable_domain),
                "parse_status": normalized.parse_status,
            }
        )
    normalized = pd.DataFrame(records)

    valid = normalized.loc[normalized["parse_status"] == "ok"].copy()
    conflict_hashes = set(
        valid.groupby("normalized_url_sha256")["label"].nunique().loc[lambda x: x > 1].index
    )
    valid["normalized_label_conflict"] = valid["normalized_url_sha256"].isin(conflict_hashes)
    clean = valid.loc[~valid["normalized_label_conflict"]].copy()
    clean["duplicate_normalized_url"] = clean.duplicated("normalized_url_sha256", keep="first")
    retained = clean.loc[~clean["duplicate_normalized_url"]].copy()

    features = [extract_url_features(value) for value in retained["normalized_url"]]
    feature_frame = pd.DataFrame(features, index=retained.index)
    if list(feature_frame.columns) != list(FEATURE_NAMES):
        raise ValueError("Part 2 feature order changed")
    retained = pd.concat([retained, feature_frame], axis=1)

    source_url_hashes, source_domain_hashes, source_counts = source_sets()
    retained["overlap_source_normalized_url"] = retained["normalized_url_sha256"].isin(
        source_url_hashes
    )
    retained["overlap_source_registrable_domain"] = retained[
        "registrable_domain_sha256"
    ].isin(source_domain_hashes)
    retained["included_source_domain_filtered"] = ~retained[
        "overlap_source_registrable_domain"
    ]
    retained.insert(
        0,
        "sample_id",
        [
            sha256_text(f"url_phish_v1|{row.source_row}|{row.raw_url_sha256}")
            for row in retained[["source_row", "raw_url_sha256"]].itertuples(index=False)
        ],
    )
    retained.insert(1, "dataset", "url_phish_mendeley_v1")

    if retained["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample_id after cleaning")
    for flag in (np.ones(len(retained), dtype=bool), retained["included_source_domain_filtered"]):
        if set(retained.loc[flag, "label"].unique()) != {0, 1}:
            raise ValueError("A frozen Stage 11F cohort lacks one class")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    retained.drop(columns=["normalized_url"]).to_parquet(
        OUTPUT, index=False, compression="zstd"
    )

    count_rows = []
    for name, frame in (
        ("raw", normalized),
        ("valid", valid),
        ("conflict_free", clean),
        ("unfiltered_deduplicated", retained),
        (
            "source_domain_filtered",
            retained.loc[retained["included_source_domain_filtered"]],
        ),
    ):
        for label in (0, 1):
            count_rows.append(
                {"cohort": name, "label": label, "n_rows": int((frame["label"] == label).sum())}
            )
    pd.DataFrame(count_rows).to_csv(ROOT / "results" / "stage11f_cohort_counts.csv", index=False)

    structure = raw.groupby("label").agg(
        n_rows=("url", "size"),
        url_length_mean=("url_len", "mean"),
        path_length_mean=("path_len", "mean"),
        query_length_mean=("query_len", "mean"),
        path_positive_rate=("path_len", lambda values: float((values > 1).mean())),
        query_positive_rate=("query_len", lambda values: float((values > 0).mean())),
    )
    structure.reset_index().to_csv(
        ROOT / "results" / "stage11f_class_structure_audit.csv", index=False
    )

    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_doi": "10.17632/65z9twcx3r.1",
        "license": "CC BY 4.0",
        "zip_sha256": sha256_file(SNAPSHOT / "65z9twcx3r-1.zip"),
        "csv_sha256": sha256_file(CSV_PATH),
        "output_sha256": sha256_file(OUTPUT),
        "advertised_rows": 111660,
        "advertised_benign": 100000,
        "advertised_phishing": 11660,
        "raw_rows": len(raw),
        "raw_benign": int((raw["label"] == 0).sum()),
        "raw_phishing": int((raw["label"] == 1).sum()),
        "raw_exact_row_duplicates": int(raw.duplicated().sum()),
        "raw_exact_url_duplicates": int(raw.duplicated("url").sum()),
        "parse_failures": int((normalized["parse_status"] != "ok").sum()),
        "normalized_conflict_hashes": len(conflict_hashes),
        "normalized_duplicate_rows_removed": int(clean["duplicate_normalized_url"].sum()),
        "retained_rows": len(retained),
        "retained_benign": int((retained["label"] == 0).sum()),
        "retained_phishing": int((retained["label"] == 1).sum()),
        "normalized_url_overlap_rows": int(retained["overlap_source_normalized_url"].sum()),
        "registrable_domain_overlap_rows": int(
            retained["overlap_source_registrable_domain"].sum()
        ),
        "source_domain_filtered_rows": int(retained["included_source_domain_filtered"].sum()),
        "source_corpus_counts": source_counts,
        "feature_count": len(FEATURE_NAMES),
        "active_url_requests": 0,
    }
    write_json(ROOT / "results" / "stage11f_source_audit.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
