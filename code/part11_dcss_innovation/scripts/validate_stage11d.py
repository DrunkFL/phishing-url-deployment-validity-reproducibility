from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT.parent
PART3 = EXPERIMENT_ROOT / "part3_s0_s4_data_splits"
sys.path.insert(0, str(ROOT / "scripts"))

from main_comparison_utils import extended_metrics  # noqa: E402


RESULTS = ROOT / "results"
DATASETS = {
    "phiusiil": {
        "assignment": "phiusiil_master_s3_domain_assignments.parquet",
        "target": "iscx_url2016_binary",
        "s4": "s4_phiusiil_to_iscx_url2016_binary_master_assignments.parquet",
    },
    "iscx_url2016_binary": {
        "assignment": "iscx_url2016_binary_master_s3_domain_assignments.parquet",
        "target": "phiusiil",
        "s4": "s4_iscx_url2016_binary_to_phiusiil_master_assignments.parquet",
    },
}
MODELS = ("lr", "rf", "xgb")
REPETITIONS = tuple(f"r{index:02d}" for index in range(10))
DCSS_SETS = ("f_dcss_10", "f_dcss_15", "f_dcss_20")
BASELINES = ("f_all", "f_single", "f_stable", "f_mi", "f_permutation")
METRICS = (
    "n_samples",
    "n_benign",
    "n_phishing",
    "threshold",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "precision",
    "recall",
    "fpr",
    "roc_auc",
    "inverted_roc_auc",
    "pr_auc",
    "brier_score",
    "log_loss",
    "ece_15",
    "tn",
    "fp",
    "fn",
    "tp",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def assert_metric_row(frame: pd.DataFrame, row: dict, context: str) -> None:
    values = extended_metrics(
        frame["label"].to_numpy(np.int8),
        frame["probability_phishing"].to_numpy(float),
        float(row["threshold"]),
    )
    for metric in METRICS:
        actual = values[metric]
        expected = row[metric]
        if isinstance(actual, int):
            require(actual == int(expected), f"{context}: {metric} mismatch")
        else:
            require(
                np.isclose(actual, float(expected), atol=1e-12, rtol=1e-12),
                f"{context}: {metric} mismatch {actual} != {expected}",
            )
    stored = frame["prediction"].to_numpy(np.int8)
    regenerated = (
        frame["probability_phishing"].to_numpy(float) >= float(row["threshold"])
    ).astype(np.int8)
    require(np.array_equal(stored, regenerated), f"{context}: threshold prediction mismatch")


def expected_internal(dataset: str, repetition: str) -> pd.DataFrame:
    role = f"split_{repetition}"
    return pd.read_parquet(
        PART3 / "data" / "assignments" / DATASETS[dataset]["assignment"],
        columns=["sample_id", "source_row", "registrable_domain_sha256", "label", role],
        filters=[(role, "==", "test")],
    )[["sample_id", "source_row", "registrable_domain_sha256", "label"]].reset_index(drop=True)


def expected_external(source: str, repetition: str) -> pd.DataFrame:
    role = f"role_{repetition}"
    frame = pd.read_parquet(
        PART3 / "data" / "assignments" / DATASETS[source]["s4"],
        columns=[
            "sample_id",
            "origin",
            "source_row",
            "registrable_domain_sha256",
            "label",
            role,
        ],
        filters=[("origin", "==", "target")],
    )
    frame = frame[
        ["sample_id", "source_row", "registrable_domain_sha256", "label", role]
    ].reset_index(drop=True)
    frame["included_in_primary"] = frame[role].astype(str) == "target_external_primary"
    return frame.drop(columns=role)


def validate_new_runs() -> None:
    internal_metrics = pd.read_csv(RESULTS / "dcss_main_internal_metrics.csv")
    external_metrics = pd.read_csv(RESULTS / "dcss_main_external_metrics.csv")
    key = ["dataset", "repetition", "model", "feature_set"]
    require(len(internal_metrics) == 180, "Expected 180 internal metric rows")
    require(not internal_metrics.duplicated(key).any(), "Duplicate internal metric key")
    require(len(external_metrics) == 360, "Expected 360 external metric rows")
    require(
        not external_metrics.duplicated(
            ["source_dataset", "repetition", "model", "feature_set", "cohort"]
        ).any(),
        "Duplicate external metric key",
    )
    require(set(internal_metrics["model_classes_json"]) == {"[0, 1]"}, "Class-order metric failure")

    for dataset in DATASETS:
        for repetition in REPETITIONS:
            internal_reference = expected_internal(dataset, repetition)
            external_reference = expected_external(dataset, repetition)
            for model in MODELS:
                feature_payload = json.loads(
                    (
                        ROOT
                        / "feature_sets"
                        / "dcss"
                        / dataset
                        / "s3"
                        / repetition
                        / model
                        / "feature_sets.json"
                    ).read_text(encoding="utf-8")
                )
                for feature_set in DCSS_SETS:
                    run = ROOT / "runs" / "main_comparison" / dataset / "s3" / repetition / model / feature_set
                    require((run / "complete.json").is_file(), f"Missing complete marker: {run}")
                    model_path = ROOT / "models" / "main_comparison" / dataset / "s3" / repetition / model / f"{feature_set}.joblib"
                    estimator = joblib.load(model_path)
                    require(list(estimator.classes_) == [0, 1], f"Class order failure: {model_path}")
                    require(
                        int(estimator.n_features_in_) == len(feature_payload[feature_set]),
                        f"Model feature count failure: {model_path}",
                    )

                    internal_path = ROOT / "predictions" / "main_comparison" / "internal" / dataset / "s3" / repetition / model / f"{feature_set}.parquet"
                    external_path = ROOT / "predictions" / "main_comparison" / "external" / dataset / repetition / model / f"{feature_set}.parquet"
                    internal_prediction = pd.read_parquet(internal_path)
                    external_prediction = pd.read_parquet(external_path)
                    require(internal_prediction["probability_phishing"].dtype == np.float64, "Internal probability precision failure")
                    require(external_prediction["probability_phishing"].dtype == np.float64, "External probability precision failure")
                    require(
                        internal_reference.equals(
                            internal_prediction[
                                ["sample_id", "source_row", "registrable_domain_sha256", "label"]
                            ]
                        ),
                        f"Internal cohort mismatch: {dataset}/{repetition}/{model}/{feature_set}",
                    )
                    require(
                        external_reference.equals(
                            external_prediction[
                                [
                                    "sample_id",
                                    "source_row",
                                    "registrable_domain_sha256",
                                    "label",
                                    "included_in_primary",
                                ]
                            ]
                        ),
                        f"External cohort mismatch: {dataset}/{repetition}/{model}/{feature_set}",
                    )

                    internal_row = internal_metrics.loc[
                        (internal_metrics["dataset"] == dataset)
                        & (internal_metrics["repetition"] == repetition)
                        & (internal_metrics["model"] == model)
                        & (internal_metrics["feature_set"] == feature_set)
                    ].iloc[0].to_dict()
                    require(
                        json.loads(internal_row["feature_names_json"])
                        == feature_payload[feature_set],
                        "Internal feature list mismatch",
                    )
                    assert_metric_row(
                        internal_prediction,
                        internal_row,
                        f"internal/{dataset}/{repetition}/{model}/{feature_set}",
                    )
                    for cohort in ("unfiltered", "primary_domain_filtered"):
                        external_row = external_metrics.loc[
                            (external_metrics["source_dataset"] == dataset)
                            & (external_metrics["repetition"] == repetition)
                            & (external_metrics["model"] == model)
                            & (external_metrics["feature_set"] == feature_set)
                            & (external_metrics["cohort"] == cohort)
                        ].iloc[0].to_dict()
                        cohort_prediction = (
                            external_prediction
                            if cohort == "unfiltered"
                            else external_prediction.loc[
                                external_prediction["included_in_primary"].astype(bool)
                            ]
                        )
                        assert_metric_row(
                            cohort_prediction,
                            external_row,
                            f"external/{dataset}/{repetition}/{model}/{feature_set}/{cohort}",
                        )


def validate_analysis() -> None:
    internal = pd.read_csv(RESULTS / "main_internal_metrics_all.csv")
    external = pd.read_csv(RESULTS / "main_external_metrics_all.csv")
    require(len(internal) == 480, "Expected 480 combined internal rows")
    require(len(external) == 960, "Expected 960 combined external rows")
    require(set(internal["feature_set"]) == set(BASELINES + DCSS_SETS), "Internal method set failure")
    require(set(external["feature_set"]) == set(BASELINES + DCSS_SETS), "External method set failure")
    require(
        external.loc[external["feature_set"].isin(DCSS_SETS), "threshold_prediction_mismatches_after_persistence"].sum() == 0,
        "DCSS persistence changed a threshold prediction",
    )
    require(len(pd.read_csv(RESULTS / "h1_internal_noninferiority_pairs.csv")) == 60, "H1 pair count failure")
    require(len(pd.read_csv(RESULTS / "h1_internal_noninferiority_summary.csv")) == 6, "H1 summary count failure")
    require(len(pd.read_csv(RESULTS / "h2_external_auc_pairs.csv")) == 60, "H2 pair count failure")
    require(len(pd.read_csv(RESULTS / "h2_external_auc_by_model.csv")) == 6, "H2 model count failure")
    require(len(pd.read_csv(RESULTS / "h2_external_auc_by_direction.csv")) == 2, "H2 direction count failure")
    summary = json.loads(
        (RESULTS / "stage11d_analysis_summary.json").read_text(encoding="utf-8")
    )
    require(summary["status"] == "ANALYZED", "Stage 11D analysis status failure")


def main() -> None:
    require((ROOT / "STAGE11D_ANALYSIS_LOCK.md").is_file(), "Missing Stage 11D lock")
    validate_new_runs()
    validate_analysis()
    report = {
        "stage": "11D",
        "status": "PASS",
        "checks": {
            "analysis_lock": "PASS",
            "run_completeness": "PASS",
            "model_class_order": "PASS",
            "feature_list_alignment": "PASS",
            "internal_cohort_alignment": "PASS",
            "external_cohort_alignment": "PASS",
            "float64_probability_persistence": "PASS",
            "metric_reproduction": "PASS",
            "combined_baseline_table": "PASS",
            "h1_h2_pairing": "PASS",
        },
        "completed_dcss_runs": 180,
        "dcss_internal_predictions": 180,
        "dcss_external_predictions": 180,
        "combined_internal_metric_rows": 480,
        "combined_external_metric_rows": 960,
        "runtime_warnings": 0,
    }
    (RESULTS / "stage11d_validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
