from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from main_comparison_utils import extended_metrics


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    checks = {}
    audit = json.loads((RESULTS / "stage11f_source_audit.json").read_text(encoding="utf-8"))
    processed_path = ROOT / "data" / "processed" / "stage11f_url_phish_v1.parquet"
    target = pd.read_parquet(processed_path)
    checks["processed_hash"] = sha256_file(processed_path) == audit["output_sha256"]
    checks["processed_count"] = len(target) == audit["retained_rows"] == 115037
    checks["unique_ids_and_urls"] = bool(
        target["sample_id"].is_unique and target["normalized_url_sha256"].is_unique
    )
    filtered = target.loc[target["included_source_domain_filtered"]]
    checks["filtered_count_and_classes"] = bool(
        len(filtered) == audit["source_domain_filtered_rows"] and set(filtered["label"]) == {0, 1}
    )
    checks["filtered_has_no_source_domain_overlap"] = bool(
        not filtered["overlap_source_registrable_domain"].any()
    )

    complete_paths = list((ROOT / "runs" / "stage11f").glob("*/r??/*/complete.json"))
    prediction_paths = list((ROOT / "predictions" / "stage11f").glob("*/r??/*.parquet"))
    metrics = pd.read_csv(RESULTS / "stage11f_external_metrics.csv")
    checks["artifact_counts"] = (
        len(complete_paths) == 60 and len(prediction_paths) == 60 and len(metrics) == 480
    )
    checks["metric_key_uniqueness"] = not metrics.duplicated(
        ["source_dataset", "repetition", "model", "feature_set", "cohort"]
    ).any()
    checks["model_class_orders"] = set(metrics["model_classes_json"]) == {"[0, 1]"}

    maximum_error = 0.0
    for path in prediction_paths:
        source, repetition, model = path.parts[-3], path.parts[-2], path.stem
        prediction = pd.read_parquet(path)
        if prediction["sample_id"].tolist() != target["sample_id"].tolist():
            raise AssertionError(f"Prediction row mismatch: {path}")
        for method, column in (
            ("f_stable", "probability_f_stable"),
            ("f_dcss_15", "probability_f_dcss_15"),
        ):
            if prediction[column].dtype != np.float64:
                raise AssertionError(f"Non-float64 probabilities: {path}/{column}")
            for cohort, mask in (
                ("unfiltered_deduplicated", np.ones(len(prediction), dtype=bool)),
                (
                    "source_domain_filtered",
                    prediction["included_source_domain_filtered"].to_numpy(dtype=bool),
                ),
            ):
                expected = metrics.loc[
                    (metrics["source_dataset"] == source)
                    & (metrics["repetition"] == repetition)
                    & (metrics["model"] == model)
                    & (metrics["feature_set"] == method)
                    & (metrics["cohort"] == cohort)
                ].iloc[0]
                reproduced = extended_metrics(
                    prediction.loc[mask, "label"].to_numpy(),
                    prediction.loc[mask, column].to_numpy(),
                    float(expected["threshold"]),
                )
                for metric in (
                    "roc_auc", "pr_auc", "macro_f1", "recall", "fpr", "brier_score",
                    "log_loss", "ece_15",
                ):
                    maximum_error = max(maximum_error, abs(float(reproduced[metric]) - float(expected[metric])))
    checks["saved_prediction_metric_reproduction"] = maximum_error <= 1e-12
    checks["analysis_outputs"] = all(
        (RESULTS / name).exists()
        for name in (
            "stage11f_primary_auc_pairs.csv",
            "stage11f_primary_auc_by_model.csv",
            "stage11f_primary_auc_by_direction.csv",
            "stage11f_performance_summary.csv",
            "stage11f_analysis_summary.json",
        )
    )
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "maximum_metric_reproduction_error": maximum_error,
    }
    (RESULTS / "stage11f_validation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
