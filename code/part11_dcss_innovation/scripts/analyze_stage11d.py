from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from main_comparison_utils import extended_metrics


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT.parent
PART4 = EXPERIMENT_ROOT / "part4_baseline_models"
PART6 = EXPERIMENT_ROOT / "part6_stable_feature_selection"
PART7 = EXPERIMENT_ROOT / "part7_external_transfer"
PART8 = EXPERIMENT_ROOT / "part8_traditional_baselines_and_error_analysis"
RESULTS = ROOT / "results"
BASELINES = ("f_all", "f_single", "f_stable", "f_mi", "f_permutation")
DCSS_SETS = ("f_dcss_10", "f_dcss_15", "f_dcss_20")
METRICS = (
    "macro_f1",
    "roc_auc",
    "pr_auc",
    "balanced_accuracy",
    "recall",
    "fpr",
    "brier_score",
    "log_loss",
    "ece_15",
)


def stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def bootstrap_ci(values: np.ndarray, *key: str) -> tuple[float, float]:
    rng = np.random.default_rng(stable_seed(*key))
    draws = rng.choice(values, size=(20_000, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def prediction_path(scope: str, dataset: str, repetition: str, model: str, feature_set: str) -> Path:
    if scope == "internal":
        if feature_set == "f_all":
            return PART4 / "predictions" / "internal" / dataset / "s3" / repetition / f"{model}.parquet"
        if feature_set in {"f_single", "f_stable"}:
            return PART6 / "predictions" / dataset / "s3" / repetition / model / f"{feature_set}.parquet"
        if feature_set in {"f_mi", "f_permutation"}:
            return PART8 / "predictions" / "internal" / dataset / "s3" / repetition / model / f"{feature_set}.parquet"
        return ROOT / "predictions" / "main_comparison" / "internal" / dataset / "s3" / repetition / model / f"{feature_set}.parquet"
    if feature_set == "f_all":
        return PART4 / "predictions" / "s4" / dataset / repetition / model / "s4_predictions.parquet"
    if feature_set in {"f_single", "f_stable"}:
        return PART7 / "predictions" / dataset / repetition / model / f"{feature_set}.parquet"
    if feature_set in {"f_mi", "f_permutation"}:
        return PART8 / "predictions" / "external" / dataset / repetition / model / f"{feature_set}.parquet"
    return ROOT / "predictions" / "main_comparison" / "external" / dataset / repetition / model / f"{feature_set}.parquet"


def add_prediction_metrics(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    rows = []
    for row in frame.to_dict(orient="records"):
        dataset = row["dataset"] if scope == "internal" else row["source_dataset"]
        path = prediction_path(
            scope, dataset, row["repetition"], row["model"], row["feature_set"]
        )
        prediction = pd.read_parquet(path)
        if scope == "external" and row["cohort"] == "primary_domain_filtered":
            prediction = prediction.loc[prediction["included_in_primary"].astype(bool)]
        values = extended_metrics(
            prediction["label"].to_numpy(np.int8),
            prediction["probability_phishing"].to_numpy(float),
            float(row["threshold"]),
        )
        labels = prediction["label"].to_numpy(np.int8)
        stored_predictions = prediction["prediction"].to_numpy(np.int8)
        tn, fp, fn, tp = confusion_matrix(
            labels, stored_predictions, labels=[0, 1]
        ).ravel()
        stored_discrete = {
            "accuracy": float(accuracy_score(labels, stored_predictions)),
            "balanced_accuracy": float(
                balanced_accuracy_score(labels, stored_predictions)
            ),
            "macro_f1": float(
                f1_score(labels, stored_predictions, average="macro", zero_division=0)
            ),
            "precision": float(
                precision_score(labels, stored_predictions, zero_division=0)
            ),
            "recall": float(recall_score(labels, stored_predictions, zero_division=0)),
            "fpr": float(fp / (fp + tn)),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }
        discrete_metrics = (
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "precision",
            "recall",
            "fpr",
            "tn",
            "fp",
            "fn",
            "tp",
        )
        discrete_error = max(
            abs(float(stored_discrete[name]) - float(row[name]))
            for name in discrete_metrics
        )
        if discrete_error > 1e-12:
            raise AssertionError(f"Discrete metric mismatch {discrete_error}: {path}")
        row["brier_score"] = values["brier_score"]
        row["log_loss"] = values["log_loss"]
        row["ece_15"] = values["ece_15"]
        row["inverted_roc_auc"] = values["inverted_roc_auc"]
        row["persisted_roc_auc"] = values["roc_auc"]
        row["persisted_pr_auc"] = values["pr_auc"]
        row["roc_auc_persistence_difference"] = values["roc_auc"] - float(row["roc_auc"])
        row["pr_auc_persistence_difference"] = values["pr_auc"] - float(row["pr_auc"])
        row["threshold_prediction_mismatches_after_persistence"] = int(
            np.count_nonzero(
                (
                    prediction["probability_phishing"].to_numpy(float)
                    >= float(row["threshold"])
                ).astype(np.int8)
                != stored_predictions
            )
        )
        row["prediction_path"] = str(path)
        rows.append(row)
    return pd.DataFrame(rows)


def combine_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_internal = pd.read_csv(PART8 / "results" / "internal_metrics_all_five.csv")
    baseline_external = pd.read_csv(PART8 / "results" / "external_metrics_all_five.csv")
    dcss_internal = pd.read_csv(RESULTS / "dcss_main_internal_metrics.csv")
    dcss_external = pd.read_csv(RESULTS / "dcss_main_external_metrics.csv")
    internal = pd.concat([baseline_internal, dcss_internal], ignore_index=True, sort=False)
    external = pd.concat([baseline_external, dcss_external], ignore_index=True, sort=False)
    internal = add_prediction_metrics(internal, "internal").sort_values(
        ["dataset", "repetition", "model", "feature_set"]
    )
    external = add_prediction_metrics(external, "external").sort_values(
        ["source_dataset", "repetition", "model", "feature_set", "cohort"]
    )
    internal.to_csv(RESULTS / "main_internal_metrics_all.csv", index=False)
    external.to_csv(RESULTS / "main_external_metrics_all.csv", index=False)
    return internal, external


def performance_summary(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    keys = (
        ["dataset", "model", "feature_set"]
        if scope == "internal"
        else ["source_dataset", "target_dataset", "cohort", "model", "feature_set"]
    )
    rows = []
    for group_key, group in frame.groupby(keys, sort=True):
        for metric in METRICS:
            values = group[metric].to_numpy(float)
            low, high = bootstrap_ci(values, scope, *map(str, group_key), metric)
            rows.append(
                {
                    **dict(zip(keys, group_key)),
                    "metric": metric,
                    "n_runs": len(values),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)),
                    "median": float(np.median(values)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "bootstrap_95_ci_low": low,
                    "bootstrap_95_ci_high": high,
                }
            )
    return pd.DataFrame(rows)


def h1_analysis(internal: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = ["dataset", "repetition", "model"]
    dcss = internal.loc[internal["feature_set"] == "f_dcss_15", key + ["macro_f1"]]
    baseline = internal.loc[internal["feature_set"] == "f_all", key + ["macro_f1"]]
    paired = dcss.merge(baseline, on=key, suffixes=("_dcss15", "_all"), validate="one_to_one")
    paired["delta_dcss15_minus_all"] = paired["macro_f1_dcss15"] - paired["macro_f1_all"]
    rows = []
    for group_key, group in paired.groupby(["dataset", "model"], sort=True):
        values = group["delta_dcss15_minus_all"].to_numpy(float)
        low, high = bootstrap_ci(values, "h1", *group_key)
        rows.append(
            {
                "dataset": group_key[0],
                "model": group_key[1],
                "n_paired_runs": len(values),
                "mean_delta": float(values.mean()),
                "std_delta": float(values.std(ddof=1)),
                "min_delta": float(values.min()),
                "max_delta": float(values.max()),
                "bootstrap_95_ci_low": low,
                "bootstrap_95_ci_high": high,
                "mean_within_noninferiority_margin": bool(values.mean() >= -0.01),
                "runs_within_margin": int((values >= -0.01).sum()),
            }
        )
    return paired, pd.DataFrame(rows)


def h2_analysis(external: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary = external.loc[external["cohort"] == "primary_domain_filtered"]
    key = ["source_dataset", "target_dataset", "repetition", "model"]
    dcss = primary.loc[primary["feature_set"] == "f_dcss_15", key + ["roc_auc"]]
    stable = primary.loc[primary["feature_set"] == "f_stable", key + ["roc_auc"]]
    paired = dcss.merge(stable, on=key, suffixes=("_dcss15", "_stable"), validate="one_to_one")
    paired["delta_dcss15_minus_stable"] = paired["roc_auc_dcss15"] - paired["roc_auc_stable"]

    model_rows = []
    for group_key, group in paired.groupby(["source_dataset", "target_dataset", "model"], sort=True):
        values = group["delta_dcss15_minus_stable"].to_numpy(float)
        low, high = bootstrap_ci(values, "h2-model", *group_key)
        model_rows.append(
            {
                "source_dataset": group_key[0],
                "target_dataset": group_key[1],
                "model": group_key[2],
                "n_paired_runs": len(values),
                "mean_delta": float(values.mean()),
                "std_delta": float(values.std(ddof=1)),
                "bootstrap_95_ci_low": low,
                "bootstrap_95_ci_high": high,
                "wins": int((values > 0).sum()),
                "ties": int((values == 0).sum()),
                "losses": int((values < 0).sum()),
            }
        )

    repetition_means = (
        paired.groupby(["source_dataset", "target_dataset", "repetition"], sort=True)[
            "delta_dcss15_minus_stable"
        ]
        .mean()
        .reset_index(name="mean_delta_across_models")
    )
    direction_rows = []
    for group_key, group in repetition_means.groupby(
        ["source_dataset", "target_dataset"], sort=True
    ):
        values = group["mean_delta_across_models"].to_numpy(float)
        low, high = bootstrap_ci(values, "h2-direction", *group_key)
        direction_rows.append(
            {
                "source_dataset": group_key[0],
                "target_dataset": group_key[1],
                "n_outer_repetitions": len(values),
                "mean_delta_across_models": float(values.mean()),
                "std_delta_across_repetitions": float(values.std(ddof=1)),
                "bootstrap_95_ci_low": low,
                "bootstrap_95_ci_high": high,
                "positive_mean": bool(values.mean() > 0),
                "positive_repetitions": int((values > 0).sum()),
            }
        )
    return paired, pd.DataFrame(model_rows), pd.DataFrame(direction_rows)


def main() -> None:
    internal, external = combine_metrics()
    performance_summary(internal, "internal").to_csv(
        RESULTS / "main_internal_performance_summary.csv", index=False
    )
    performance_summary(external, "external").to_csv(
        RESULTS / "main_external_performance_summary.csv", index=False
    )
    h1_paired, h1_summary = h1_analysis(internal)
    h2_paired, h2_model, h2_direction = h2_analysis(external)
    h1_paired.to_csv(RESULTS / "h1_internal_noninferiority_pairs.csv", index=False)
    h1_summary.to_csv(RESULTS / "h1_internal_noninferiority_summary.csv", index=False)
    h2_paired.to_csv(RESULTS / "h2_external_auc_pairs.csv", index=False)
    h2_model.to_csv(RESULTS / "h2_external_auc_by_model.csv", index=False)
    h2_direction.to_csv(RESULTS / "h2_external_auc_by_direction.csv", index=False)
    summary = {
        "stage": "11D",
        "status": "ANALYZED",
        "internal_metric_rows": len(internal),
        "external_metric_rows": len(external),
        "h1_groups_with_mean_within_margin": int(
            h1_summary["mean_within_noninferiority_margin"].sum()
        ),
        "h1_total_groups": len(h1_summary),
        "h2_positive_directions": int(h2_direction["positive_mean"].sum()),
        "h2_total_directions": len(h2_direction),
        "max_absolute_roc_auc_persistence_difference": float(
            external["roc_auc_persistence_difference"].abs().max()
        ),
        "max_absolute_pr_auc_persistence_difference": float(
            external["pr_auc_persistence_difference"].abs().max()
        ),
    }
    (RESULTS / "stage11d_analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
