from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT.parent
PART2 = EXPERIMENT_ROOT / "part2_url_normalization_leakage_audit"
PART3 = EXPERIMENT_ROOT / "part3_s0_s4_data_splits"
PART4 = EXPERIMENT_ROOT / "part4_baseline_models"
PART6 = EXPERIMENT_ROOT / "part6_stable_feature_selection"
PART7 = EXPERIMENT_ROOT / "part7_external_transfer"
PART8 = EXPERIMENT_ROOT / "part8_traditional_baselines_and_error_analysis"
sys.path.insert(0, str(PART4 / "scripts"))

from model_utils import select_model  # noqa: E402


DATASETS = {
    "phiusiil": {
        "target": "iscx_url2016_binary",
        "s4_file": "s4_phiusiil_to_iscx_url2016_binary_master_assignments.parquet",
    },
    "iscx_url2016_binary": {
        "target": "phiusiil",
        "s4_file": "s4_iscx_url2016_binary_to_phiusiil_master_assignments.parquet",
    },
}
MODELS = ["lr", "rf", "xgb"]
FEATURE_SETS = ["f_all", "f_single", "f_stable", "f_mi", "f_permutation"]
REPETITIONS = [f"r{i:02d}" for i in range(10)]
MODEL_SEEDS = [20261001 + i for i in range(10)]
METRIC_COLUMNS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "precision",
    "recall",
    "fpr",
    "roc_auc",
    "pr_auc",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit external class and probability orientation")
    parser.add_argument("--n-jobs", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--force-reconstruction", action="store_true")
    return parser.parse_args()


def feature_names() -> list[str]:
    frame = pd.read_csv(PART2 / "results" / "feature_dictionary.csv").sort_values("feature_order")
    names = frame["feature"].tolist()
    if len(names) != 35 or len(set(names)) != 35:
        raise ValueError("Expected 35 frozen features")
    return names


def feature_table(dataset: str, names: list[str]) -> pd.DataFrame:
    frame = pd.read_parquet(
        PART2 / "data" / f"{dataset}_master_features.parquet",
        columns=["source_row", "raw_url_sha256", "label", *names],
    )
    if frame[["source_row", "raw_url_sha256"]].duplicated().any():
        raise ValueError(f"Duplicate feature key: {dataset}")
    if set(frame["label"].unique()) != {0, 1}:
        raise ValueError(f"Unexpected label set: {dataset}")
    return frame


def source_s3(dataset: str, features: pd.DataFrame) -> pd.DataFrame:
    assignment = pd.read_parquet(
        PART3 / "data" / "assignments" / f"{dataset}_master_s3_domain_assignments.parquet",
        columns=[
            "sample_id",
            "source_row",
            "raw_url_sha256",
            "registrable_domain_sha256",
            *[f"split_{repetition}" for repetition in REPETITIONS],
        ],
    )
    merged = assignment.merge(
        features,
        on=["source_row", "raw_url_sha256"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(features):
        raise ValueError(f"Source feature-assignment mismatch: {dataset}")
    return merged


def external_target(
    source: str,
    repetition: str,
    target_features: pd.DataFrame,
) -> pd.DataFrame:
    assignment = pd.read_parquet(
        PART3 / "data" / "assignments" / DATASETS[source]["s4_file"],
        columns=[
            "sample_id",
            "origin",
            "source_row",
            "raw_url_sha256",
            "label",
            f"role_{repetition}",
        ],
    )
    assignment = assignment.loc[assignment["origin"] == "target"].copy()
    target = assignment.merge(
        target_features.drop(columns="label"),
        on=["source_row", "raw_url_sha256"],
        how="inner",
        validate="one_to_one",
    )
    if len(target) != len(target_features):
        raise ValueError(f"Target feature-assignment mismatch: {source}/{repetition}")
    if set(target["label"].unique()) != {0, 1}:
        raise ValueError(f"Unexpected target labels: {source}/{repetition}")
    return target


def classes_of(model: object) -> list[int]:
    classes = np.asarray(getattr(model, "classes_", []))
    if not np.array_equal(classes, np.array([0, 1])):
        raise ValueError(f"Unexpected model classes: {classes.tolist()}")
    return [int(value) for value in classes]


def save_model(path: Path, model: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(model, temporary, compress=3)
    temporary.replace(path)


def historical_prediction_path(
    source: str, repetition: str, model: str, feature_set: str
) -> Path:
    if feature_set == "f_all":
        return (
            PART4
            / "predictions"
            / "s4"
            / source
            / repetition
            / model
            / "s4_predictions.parquet"
        )
    if feature_set in {"f_single", "f_stable"}:
        return PART7 / "predictions" / source / repetition / model / f"{feature_set}.parquet"
    return (
        PART8
        / "predictions"
        / "external"
        / source
        / repetition
        / model
        / f"{feature_set}.parquet"
    )


def historical_model_path(
    source: str, repetition: str, model: str, feature_set: str
) -> Path:
    if feature_set in {"f_single", "f_stable"}:
        return PART6 / "models" / source / "s3" / repetition / model / f"{feature_set}.joblib"
    if feature_set in {"f_mi", "f_permutation"}:
        return PART8 / "models" / source / "s3" / repetition / model / f"{feature_set}.joblib"
    return (
        ROOT
        / "models"
        / "validity_repair"
        / "reconstructed_f_all"
        / source
        / "s3"
        / repetition
        / f"{model}.joblib"
    )


def reconstruct_f_all(
    source: str,
    repetition: str,
    model_name: str,
    source_data: pd.DataFrame,
    target: pd.DataFrame,
    names: list[str],
    reported: pd.DataFrame,
    n_jobs: int,
    force: bool,
) -> tuple[object, dict[str, object]]:
    model_path = historical_model_path(source, repetition, model_name, "f_all")
    metadata_path = model_path.with_suffix(".json")
    if model_path.exists() and metadata_path.exists() and not force:
        model = joblib.load(model_path)
        metadata_row = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        split = source_data[f"split_{repetition}"].astype(str).to_numpy()
        train = split == "train"
        validation = split == "validation"
        X = source_data[names].to_numpy(dtype=np.float32)
        y = source_data["label"].to_numpy(dtype=np.int8)
        selected = select_model(
            model_name,
            X[train],
            y[train],
            X[validation],
            y[validation],
            MODEL_SEEDS[int(repetition[1:])],
            n_jobs,
        )
        classes_of(selected.model)
        model = selected.model
        metadata_row = {
            "source_dataset": source,
            "repetition": repetition,
            "model": model_name,
            "selected_candidate": selected.selected_candidate,
            "selected_params_json": json.dumps(selected.selected_params, sort_keys=True),
            "threshold": float(selected.threshold),
            "model_classes": classes_of(selected.model),
        }
        save_model(model_path, model)
        metadata_path.write_text(
            json.dumps(metadata_row, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    classes_of(model)
    expected = reported.loc[
        (reported["source_dataset"] == source)
        & (reported["repetition"] == repetition)
        & (reported["model"] == model_name)
        & (reported["feature_set"] == "f_all")
        & (reported["cohort"] == "primary_domain_filtered")
    ]
    if len(expected) != 1:
        raise ValueError(f"Missing reported F-All row: {source}/{repetition}/{model_name}")
    expected_row = expected.iloc[0]
    candidate_match = metadata_row["selected_candidate"] == expected_row["selected_candidate"]
    threshold_difference = abs(float(metadata_row["threshold"]) - float(expected_row["threshold"]))
    if not candidate_match or threshold_difference > 1e-12:
        raise AssertionError(f"F-All reconstruction selection mismatch: {source}/{repetition}/{model_name}")

    reconstructed = model.predict_proba(target[names].to_numpy(dtype=np.float32))[:, 1]
    historical = pd.read_parquet(
        historical_prediction_path(source, repetition, model_name, "f_all"),
        columns=["sample_id", "probability_phishing"],
    )
    comparison = target[["sample_id"]].copy()
    comparison["reconstructed_probability"] = reconstructed
    comparison = comparison.merge(historical, on="sample_id", validate="one_to_one")
    max_difference = float(
        np.max(
            np.abs(
                comparison["reconstructed_probability"].to_numpy(float)
                - comparison["probability_phishing"].to_numpy(float)
            )
        )
    )
    if max_difference > 2e-6:
        raise AssertionError(
            f"F-All reconstruction probability mismatch {max_difference}: "
            f"{source}/{repetition}/{model_name}"
        )
    return model, {
        "source_dataset": source,
        "target_dataset": DATASETS[source]["target"],
        "repetition": repetition,
        "model": model_name,
        "selected_candidate": metadata_row["selected_candidate"],
        "candidate_match": candidate_match,
        "threshold_difference": threshold_difference,
        "max_probability_abs_difference": max_difference,
        "n_predictions": len(comparison),
        "model_classes_json": json.dumps(classes_of(model)),
    }


def recompute_metrics(
    frame: pd.DataFrame, probabilities: np.ndarray | None = None
) -> dict[str, float | int]:
    y = frame["label"].to_numpy(dtype=np.int8)
    p = (
        frame["probability_phishing"].to_numpy(dtype=float)
        if probabilities is None
        else np.asarray(probabilities, dtype=float)
    )
    prediction = frame["prediction"].to_numpy(dtype=np.int8)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "n_samples": int(len(frame)),
        "n_benign": int((y == 0).sum()),
        "n_phishing": int((y == 1).sum()),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "fpr": float(fp / (fp + tn)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "inverted_roc_auc": float(roc_auc_score(y, 1.0 - p)),
    }


def sign(value: float) -> int:
    if not math.isfinite(value) or value == 0:
        return 0
    return 1 if value > 0 else -1


def safe_spearman(values: np.ndarray, labels: np.ndarray) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        result = spearmanr(values, labels).statistic
    return float(result) if math.isfinite(float(result)) else math.nan


def main() -> None:
    args = parse_args()
    names = feature_names()
    reported = pd.read_csv(PART8 / "results" / "external_metrics_all_five.csv")
    features = {dataset: feature_table(dataset, names) for dataset in DATASETS}
    sources = {dataset: source_s3(dataset, features[dataset]) for dataset in DATASETS}

    reconstruction_rows: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    direction_rows: list[dict[str, object]] = []
    target_cache: dict[tuple[str, str], pd.DataFrame] = {}
    started = time.perf_counter()

    for source in DATASETS:
        target_dataset = DATASETS[source]["target"]
        for repetition in REPETITIONS:
            target = external_target(source, repetition, features[target_dataset])
            target_cache[(source, repetition)] = target
            for model_name in MODELS:
                f_all_model, reconstruction = reconstruct_f_all(
                    source,
                    repetition,
                    model_name,
                    sources[source],
                    target,
                    names,
                    reported,
                    args.n_jobs,
                    args.force_reconstruction,
                )
                reconstruction_rows.append(reconstruction)

                for feature_set in FEATURE_SETS:
                    if feature_set == "f_all":
                        model = f_all_model
                        model_path = historical_model_path(
                            source, repetition, model_name, feature_set
                        )
                    else:
                        model_path = historical_model_path(
                            source, repetition, model_name, feature_set
                        )
                        if not model_path.is_file():
                            raise FileNotFoundError(model_path)
                        model = joblib.load(model_path)
                    model_class_values = classes_of(model)
                    model_rows.append(
                        {
                            "source_dataset": source,
                            "repetition": repetition,
                            "model": model_name,
                            "feature_set": feature_set,
                            "model_path": str(model_path),
                            "model_classes_json": json.dumps(model_class_values),
                            "n_features_in": int(getattr(model, "n_features_in_", -1)),
                            "class_order_pass": True,
                        }
                    )

                    prediction_path = historical_prediction_path(
                        source, repetition, model_name, feature_set
                    )
                    prediction = pd.read_parquet(prediction_path)
                    if prediction["sample_id"].duplicated().any():
                        raise AssertionError(f"Duplicate prediction IDs: {prediction_path}")
                    if set(prediction["label"].unique()) != {0, 1}:
                        raise AssertionError(f"Unexpected prediction labels: {prediction_path}")
                    probabilities = prediction["probability_phishing"].to_numpy(float)
                    if not np.isfinite(probabilities).all() or not (
                        (probabilities >= 0) & (probabilities <= 1)
                    ).all():
                        raise AssertionError(f"Invalid prediction probabilities: {prediction_path}")

                    feature_rows = reported.loc[
                        (reported["source_dataset"] == source)
                        & (reported["repetition"] == repetition)
                        & (reported["model"] == model_name)
                        & (reported["feature_set"] == feature_set)
                    ]
                    serialized_feature_names = feature_rows["feature_names_json"].dropna().unique()
                    if len(serialized_feature_names) != 1:
                        raise AssertionError(
                            f"Ambiguous feature list: {source}/{repetition}/{model_name}/{feature_set}"
                        )
                    selected_names = json.loads(serialized_feature_names[0])
                    audit_probabilities = model.predict_proba(
                        target[selected_names].to_numpy(dtype=np.float32)
                    )[:, 1]
                    audit_probability_frame = target[["sample_id"]].copy()
                    audit_probability_frame["audit_probability"] = audit_probabilities
                    prediction = prediction.merge(
                        audit_probability_frame, on="sample_id", validate="one_to_one"
                    )
                    persisted_probability_difference = np.abs(
                        prediction["audit_probability"].to_numpy(float)
                        - prediction["probability_phishing"].to_numpy(float)
                    )
                    max_persisted_probability_difference = float(
                        persisted_probability_difference.max()
                    )

                    for cohort in ("unfiltered", "primary_domain_filtered"):
                        if cohort == "unfiltered":
                            cohort_frame = prediction
                        else:
                            cohort_frame = prediction.loc[
                                prediction["included_in_primary"].astype(bool)
                            ]
                        cohort_audit_probabilities = cohort_frame[
                            "audit_probability"
                        ].to_numpy(float)
                        values = recompute_metrics(
                            cohort_frame, probabilities=cohort_audit_probabilities
                        )
                        expected = reported.loc[
                            (reported["source_dataset"] == source)
                            & (reported["target_dataset"] == target_dataset)
                            & (reported["repetition"] == repetition)
                            & (reported["model"] == model_name)
                            & (reported["feature_set"] == feature_set)
                            & (reported["cohort"] == cohort)
                        ]
                        if len(expected) != 1:
                            raise AssertionError(
                                f"Missing reported metric row: {source}/{repetition}/{model_name}/{feature_set}/{cohort}"
                            )
                        expected_row = expected.iloc[0]
                        differences = {
                            metric: abs(float(values[metric]) - float(expected_row[metric]))
                            for metric in METRIC_COLUMNS
                        }
                        max_metric_difference = max(differences.values())
                        metric_tolerance = 1e-4
                        if max_metric_difference > metric_tolerance:
                            raise AssertionError(
                                f"Metric mismatch {max_metric_difference}: "
                                f"{source}/{repetition}/{model_name}/{feature_set}/{cohort}"
                            )
                        threshold = float(expected_row["threshold"])
                        threshold_predictions = (
                            cohort_audit_probabilities >= threshold
                        ).astype(np.int8)
                        stored_predictions = cohort_frame["prediction"].to_numpy(np.int8)
                        threshold_mismatches = int(
                            np.count_nonzero(threshold_predictions != stored_predictions)
                        )
                        if threshold_mismatches:
                            raise AssertionError(f"Stored threshold prediction mismatch: {prediction_path}")

                        metric_rows.append(
                            {
                                "source_dataset": source,
                                "target_dataset": target_dataset,
                                "repetition": repetition,
                                "model": model_name,
                                "feature_set": feature_set,
                                "cohort": cohort,
                                "model_classes_json": json.dumps(model_class_values),
                                "label_values_json": "[0, 1]",
                                "probability_min": float(cohort_frame["probability_phishing"].min()),
                                "probability_max": float(cohort_frame["probability_phishing"].max()),
                                "max_persisted_probability_abs_difference": max_persisted_probability_difference,
                                "roc_auc": values["roc_auc"],
                                "inverted_roc_auc": values["inverted_roc_auc"],
                                "pr_auc": values["pr_auc"],
                                "macro_f1": values["macro_f1"],
                                "fpr": values["fpr"],
                                "max_reported_metric_abs_difference": max_metric_difference,
                                "reported_metric_tolerance": metric_tolerance,
                                "threshold_prediction_mismatches": threshold_mismatches,
                                "n_samples": values["n_samples"],
                            }
                        )
                        for label in (0, 1):
                            label_scores = cohort_frame.loc[
                                cohort_frame["label"] == label, "audit_probability"
                            ].to_numpy(float)
                            score_rows.append(
                                {
                                    "source_dataset": source,
                                    "target_dataset": target_dataset,
                                    "repetition": repetition,
                                    "model": model_name,
                                    "feature_set": feature_set,
                                    "cohort": cohort,
                                    "label": label,
                                    "n": len(label_scores),
                                    "mean": float(np.mean(label_scores)),
                                    "median": float(np.median(label_scores)),
                                    "p05": float(np.quantile(label_scores, 0.05)),
                                    "p95": float(np.quantile(label_scores, 0.95)),
                                }
                            )
                print(
                    f"audited={source}/{repetition}/{model_name}", flush=True
                )

            source_train = sources[source].loc[
                sources[source][f"split_{repetition}"] == "train"
            ]
            target_primary = target.loc[
                target[f"role_{repetition}"] == "target_external_primary"
            ]
            source_labels = source_train["label"].to_numpy(np.int8)
            target_labels = target_primary["label"].to_numpy(np.int8)
            for feature in names:
                source_values = source_train[feature].to_numpy(float)
                target_values = target_primary[feature].to_numpy(float)
                source_rho = safe_spearman(source_values, source_labels)
                target_rho = safe_spearman(target_values, target_labels)
                source_sign = sign(source_rho)
                target_sign = sign(target_rho)
                direction_rows.append(
                    {
                        "source_dataset": source,
                        "target_dataset": target_dataset,
                        "repetition": repetition,
                        "feature": feature,
                        "source_train_spearman_with_label": source_rho,
                        "target_primary_spearman_with_label": target_rho,
                        "source_direction": source_sign,
                        "target_direction": target_sign,
                        "nonzero_direction_flip": bool(
                            source_sign != 0
                            and target_sign != 0
                            and source_sign != target_sign
                        ),
                        "source_benign_median": float(
                            np.median(source_values[source_labels == 0])
                        ),
                        "source_phishing_median": float(
                            np.median(source_values[source_labels == 1])
                        ),
                        "target_benign_median": float(
                            np.median(target_values[target_labels == 0])
                        ),
                        "target_phishing_median": float(
                            np.median(target_values[target_labels == 1])
                        ),
                    }
                )

    results = ROOT / "results"
    pd.DataFrame(reconstruction_rows).to_csv(
        results / "f_all_reconstruction_audit.csv", index=False
    )
    pd.DataFrame(model_rows).to_csv(
        results / "external_model_class_order_audit.csv", index=False
    )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(results / "external_probability_orientation_audit.csv", index=False)
    pd.DataFrame(score_rows).to_csv(
        results / "external_class_conditional_score_summary.csv", index=False
    )
    directions = pd.DataFrame(direction_rows)
    directions.to_csv(results / "source_target_feature_direction_audit.csv", index=False)
    direction_summary = (
        directions.groupby(["source_dataset", "target_dataset", "repetition"], observed=True)
        .agg(
            n_features=("feature", "size"),
            nonzero_direction_flips=("nonzero_direction_flip", "sum"),
        )
        .reset_index()
    )
    direction_summary.to_csv(
        results / "source_target_feature_direction_summary.csv", index=False
    )

    primary = metrics.loc[metrics["cohort"] == "primary_domain_filtered"]
    reverse_xgb = primary.loc[
        (primary["source_dataset"] == "phiusiil")
        & (primary["model"] == "xgb")
    ]
    summary = {
        "status": "PASS",
        "reconstructed_f_all_models": len(reconstruction_rows),
        "audited_model_files": len(model_rows),
        "audited_prediction_configurations": len(primary),
        "audited_metric_rows_including_two_cohorts": len(metrics),
        "class_order_failures": int((~pd.DataFrame(model_rows)["class_order_pass"]).sum()),
        "maximum_reconstruction_probability_abs_difference": float(
            pd.DataFrame(reconstruction_rows)["max_probability_abs_difference"].max()
        ),
        "maximum_reported_metric_abs_difference": float(
            metrics["max_reported_metric_abs_difference"].max()
        ),
        "primary_auc_below_0_5_count": int((primary["roc_auc"] < 0.5).sum()),
        "primary_auc_below_0_1_count": int((primary["roc_auc"] < 0.1).sum()),
        "reverse_xgb_auc_min": float(reverse_xgb["roc_auc"].min()),
        "reverse_xgb_auc_max": float(reverse_xgb["roc_auc"].max()),
        "reverse_xgb_inverted_auc_min": float(reverse_xgb["inverted_roc_auc"].min()),
        "reverse_xgb_inverted_auc_max": float(reverse_xgb["inverted_roc_auc"].max()),
        "feature_direction_audit_rows": len(directions),
        "mean_nonzero_direction_flips_by_direction": {
            f"{row.source_dataset}_to_{row.target_dataset}": float(row.nonzero_direction_flips)
            for row in (
                direction_summary.groupby(
                    ["source_dataset", "target_dataset"], observed=True
                )["nonzero_direction_flips"]
                .mean()
                .reset_index()
                .itertuples(index=False)
            )
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    (results / "probability_orientation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
