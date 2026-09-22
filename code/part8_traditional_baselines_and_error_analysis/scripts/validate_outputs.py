from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT.parent
PART2 = EXPERIMENT_ROOT / "part2_url_normalization_leakage_audit"
PART3 = EXPERIMENT_ROOT / "part3_s0_s4_data_splits"
PART4 = EXPERIMENT_ROOT / "part4_baseline_models"
RESULTS = ROOT / "results"
SOURCES = ["iscx_url2016_binary", "phiusiil"]
MODELS = ["lr", "rf", "xgb"]
REPETITIONS = [f"r{index:02d}" for index in range(10)]
FEATURE_SETS = ["f_mi", "f_permutation"]
ASSIGNMENTS = {
    "iscx_url2016_binary": "iscx_url2016_binary_master_s3_domain_assignments.parquet",
    "phiusiil": "phiusiil_master_s3_domain_assignments.parquet",
}


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recompute(frame: pd.DataFrame) -> dict[str, float | int]:
    labels = frame["label"].to_numpy(dtype=int)
    predictions = frame["prediction"].to_numpy(dtype=int)
    probabilities = frame["probability_phishing"].to_numpy(dtype=float)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "fpr": fp / (fp + tn),
        "roc_auc": roc_auc_score(labels, probabilities),
        "pr_auc": average_precision_score(labels, probabilities),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def assert_metrics(actual: dict, expected: dict, context: str) -> None:
    for key, value in actual.items():
        target = expected[key]
        if isinstance(value, int):
            if value != int(target):
                raise AssertionError(f"{context}: {key}={value} != {target}")
        elif not np.isclose(value, float(target), atol=1e-12, rtol=1e-12):
            raise AssertionError(f"{context}: {key}={value} != {target}")


def validate_selection_scope(feature_names: set[str]) -> None:
    for dataset in SOURCES:
        assignment = pd.read_parquet(
            PART3 / "data" / "assignments" / ASSIGNMENTS[dataset],
            columns=["sample_id"] + [f"split_{rep}" for rep in REPETITIONS],
        )
        for repetition in REPETITIONS:
            outer_train = set(
                assignment.loc[assignment[f"split_{repetition}"] == "train", "sample_id"]
            )
            scope = ROOT / "selection_scope" / dataset / "s3" / repetition
            train = pd.read_parquet(scope / "i00_train_ids.parquet")
            cohort = pd.read_parquet(scope / "i00_validation_cohort_ids.parquet")
            train_ids = set(train["sample_id"])
            cohort_ids = set(cohort["sample_id"])
            if not train_ids <= outer_train or not cohort_ids <= outer_train:
                raise AssertionError(f"Selection escaped outer training: {dataset}/{repetition}")
            if train_ids & cohort_ids:
                raise AssertionError(f"Inner train/cohort overlap: {dataset}/{repetition}")
            if len(cohort) > 2000:
                raise AssertionError(f"Permutation cohort too large: {dataset}/{repetition}")
            for model in MODELS:
                path = (
                    ROOT / "feature_sets" / dataset / "s3" / repetition / model
                    / "traditional_feature_sets.json"
                )
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload["selection_scope"] != "outer_train_i00_only":
                    raise AssertionError(f"Unexpected selection scope: {path}")
                for feature_set in FEATURE_SETS:
                    selected = payload[feature_set]
                    if len(selected) != 15 or len(set(selected)) != 15:
                        raise AssertionError(f"Invalid selected feature count: {path}/{feature_set}")
                    if not set(selected) <= feature_names:
                        raise AssertionError(f"Unknown feature: {path}/{feature_set}")


def validate_predictions() -> None:
    for dataset in SOURCES:
        for repetition in REPETITIONS:
            for model in MODELS:
                internal_reference = pd.read_parquet(
                    PART4 / "predictions" / "internal" / dataset / "s3" / repetition
                    / f"{model}.parquet",
                    columns=["sample_id", "source_row", "label"],
                )
                external_reference = pd.read_parquet(
                    PART4 / "predictions" / "s4" / dataset / repetition / model
                    / "s4_predictions.parquet",
                    columns=["sample_id", "source_row", "label", "included_in_primary"],
                )
                for feature_set in FEATURE_SETS:
                    run = ROOT / "runs" / dataset / "s3" / repetition / model / feature_set
                    if not (run / "complete.json").exists():
                        raise AssertionError(f"Missing complete marker: {run}")
                    internal = pd.read_parquet(
                        ROOT / "predictions" / "internal" / dataset / "s3" / repetition
                        / model / f"{feature_set}.parquet"
                    )
                    external = pd.read_parquet(
                        ROOT / "predictions" / "external" / dataset / repetition
                        / model / f"{feature_set}.parquet"
                    )
                    if not internal_reference.equals(
                        internal[["sample_id", "source_row", "label"]]
                    ):
                        raise AssertionError(f"Internal cohort mismatch: {dataset}/{repetition}/{model}/{feature_set}")
                    if not external_reference.equals(
                        external[["sample_id", "source_row", "label", "included_in_primary"]]
                    ):
                        raise AssertionError(f"External cohort mismatch: {dataset}/{repetition}/{model}/{feature_set}")
                    expected_internal = json.loads(
                        (run / "internal_metrics.json").read_text(encoding="utf-8")
                    )
                    assert_metrics(
                        recompute(internal), expected_internal,
                        f"internal/{dataset}/{repetition}/{model}/{feature_set}",
                    )
                    expected_external = {
                        row["cohort"]: row
                        for row in json.loads((run / "external_metrics.json").read_text(encoding="utf-8"))
                    }
                    assert_metrics(
                        recompute(external), expected_external["unfiltered"],
                        f"external-all/{dataset}/{repetition}/{model}/{feature_set}",
                    )
                    primary = external.loc[external["included_in_primary"]]
                    assert_metrics(
                        recompute(primary), expected_external["primary_domain_filtered"],
                        f"external-primary/{dataset}/{repetition}/{model}/{feature_set}",
                    )


def validate_aggregate() -> None:
    expected_rows = {
        "internal_metrics_all_five.csv": 300,
        "external_metrics_all_five.csv": 600,
        "internal_performance_summary.csv": 180,
        "external_performance_summary.csv": 360,
        "internal_paired_deltas.csv": 1440,
        "external_paired_deltas.csv": 2880,
        "internal_paired_summary.csv": 144,
        "external_paired_summary.csv": 288,
        "traditional_outer_agreement.csv": 540,
        "traditional_cross_source_overlap.csv": 60,
        "stable_vs_traditional_overlap.csv": 120,
        "external_error_disagreement.csv": 360,
        "false_positive_domain_concentration.csv": 300,
    }
    for name, count in expected_rows.items():
        found = len(pd.read_csv(RESULTS / name))
        if found != count:
            raise AssertionError(f"{name}: expected {count}, found {found}")
    manifest = pd.read_csv(RESULTS / "result_manifest.csv")
    if len(manifest) != len(expected_rows):
        raise AssertionError("Result manifest row count mismatch")
    for row in manifest.itertuples(index=False):
        path = ROOT / row.path
        if path.stat().st_size != row.bytes or hash_file(path) != row.sha256:
            raise AssertionError(f"Manifest mismatch: {path}")


def main() -> None:
    feature_names = set(
        pd.read_csv(PART2 / "results" / "feature_dictionary.csv")["feature"]
    )
    validate_selection_scope(feature_names)
    validate_predictions()
    validate_aggregate()
    report = {
        "status": "PASS",
        "selection_groups": 60,
        "training_runs": 120,
        "target_feedback_used_for_selection": False,
        "internal_prediction_alignment": "PASS",
        "external_prediction_alignment": "PASS",
        "stored_metric_recalculation": "PASS",
        "aggregate_row_counts": "PASS",
        "manifest_integrity": "PASS",
    }
    (RESULTS / "validation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
