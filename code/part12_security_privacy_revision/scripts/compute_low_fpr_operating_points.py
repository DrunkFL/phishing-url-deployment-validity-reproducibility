from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


PART12_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = PART12_ROOT.parent
PART4_ROOT = EXPERIMENT_ROOT / "part4_baseline_models"
sys.path.insert(0, str(PART4_ROOT / "scripts"))

from model_utils import build_model, calculate_metrics  # noqa: E402
from train_baselines import (  # noqa: E402
    DATASETS,
    MODEL_SEEDS,
    MODEL_NAMES,
    PART3_ROOT,
    REPETITIONS,
    load_feature_names,
    load_feature_table,
    merge_assignment,
)


BUDGETS = (0.001, 0.01)
RESULTS_DIR = PART12_ROOT / "results"
LOGS_DIR = PART12_ROOT / "logs"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_fpr_constrained_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    budget: float,
) -> tuple[float, dict[str, float | int]]:
    """Select the highest-TPR validation threshold whose empirical FPR is within budget."""
    fpr, tpr, thresholds = roc_curve(labels, probabilities, drop_intermediate=False)
    valid = np.flatnonzero(fpr <= budget + 1e-15)
    if len(valid) == 0:
        threshold = float(np.nextafter(np.max(probabilities), np.inf))
    else:
        best_tpr = np.max(tpr[valid])
        tied = valid[np.isclose(tpr[valid], best_tpr, rtol=0.0, atol=1e-15)]
        threshold = float(np.min(thresholds[tied]))

    metrics = calculate_metrics(labels, probabilities, threshold)
    if metrics["fpr"] > budget + 1e-15:
        raise AssertionError(
            f"Selected threshold violates validation FPR budget: {metrics['fpr']} > {budget}"
        )
    return threshold, metrics


def metric_row(
    source: str,
    target: str,
    repetition: str,
    model_name: str,
    budget: float,
    threshold: float,
    scope: str,
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, object]:
    metrics = calculate_metrics(labels, probabilities, threshold)
    return {
        "source_dataset": source,
        "target_dataset": target,
        "repetition": repetition,
        "model": model_name,
        "fpr_budget": budget,
        "threshold_selected_on_source_validation": threshold,
        "evaluation_scope": scope,
        "alerts_per_10000_benign": metrics["fpr"] * 10000.0,
        **metrics,
    }


def load_external_target(
    source: str,
    repetition: str,
    feature_tables: dict[str, pd.DataFrame],
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = DATASETS[source]["s4_target"]
    role_column = f"role_{repetition}"
    assignment_path = (
        PART3_ROOT / "data" / "assignments" / DATASETS[source]["s4_file"]
    )
    assignment = pd.read_parquet(
        assignment_path,
        columns=[
            "sample_id",
            "origin",
            "source_row",
            "raw_url_sha256",
            "label",
            role_column,
        ],
    )
    assignment = assignment.loc[assignment["origin"] == "target"].copy()
    merged = assignment.merge(
        feature_tables[target].drop(columns="label"),
        on=["source_row", "raw_url_sha256"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(feature_tables[target]):
        raise ValueError(f"External assignment mismatch for {source}/{repetition}")
    primary = (merged[role_column] == "target_external_primary").to_numpy()
    return (
        merged[feature_names].to_numpy(dtype=np.float32),
        merged["label"].to_numpy(dtype=np.int8),
        primary,
    )


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    groups = [
        "source_dataset",
        "target_dataset",
        "model",
        "fpr_budget",
        "evaluation_scope",
    ]
    summary = (
        rows.groupby(groups, observed=True)
        .agg(
            repetitions=("repetition", "nunique"),
            threshold_mean=("threshold_selected_on_source_validation", "mean"),
            threshold_min=("threshold_selected_on_source_validation", "min"),
            threshold_max=("threshold_selected_on_source_validation", "max"),
            tpr_mean=("recall", "mean"),
            tpr_sd=("recall", "std"),
            fpr_mean=("fpr", "mean"),
            fpr_sd=("fpr", "std"),
            alerts_per_10000_mean=("alerts_per_10000_benign", "mean"),
            alerts_per_10000_sd=("alerts_per_10000_benign", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            pr_auc_mean=("pr_auc", "mean"),
        )
        .reset_index()
    )
    return summary


def main() -> None:
    started = time.perf_counter()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    feature_names = load_feature_names()
    feature_tables = {
        dataset: load_feature_table(dataset, feature_names) for dataset in DATASETS
    }
    merged_s3 = {
        dataset: merge_assignment(dataset, "s3", feature_tables[dataset])
        for dataset in DATASETS
    }

    output_rows: list[dict[str, object]] = []
    issues: list[dict[str, object]] = []
    input_files: set[Path] = set()

    for source in DATASETS:
        target = DATASETS[source]["s4_target"]
        merged = merged_s3[source]
        X_all = merged[feature_names].to_numpy(dtype=np.float32)
        y_all = merged["label"].to_numpy(dtype=np.int8)

        for repetition in REPETITIONS:
            split = merged[f"split_{repetition}"].astype(str).to_numpy()
            train_mask = split == "train"
            validation_mask = split == "validation"
            test_mask = split == "test"
            X_target, y_target, primary_mask = load_external_target(
                source, repetition, feature_tables, feature_names
            )

            for model_name in MODEL_NAMES:
                metrics_path = (
                    PART4_ROOT
                    / "runs"
                    / "internal"
                    / source
                    / "s3"
                    / repetition
                    / model_name
                    / "internal_metrics.json"
                )
                input_files.add(metrics_path)
                locked = json.loads(metrics_path.read_text(encoding="utf-8"))
                params = json.loads(locked["selected_params_json"])
                model = build_model(
                    model_name,
                    params,
                    MODEL_SEEDS[int(repetition[1:])],
                    n_jobs=6,
                )
                model.fit(X_all[train_mask], y_all[train_mask])
                validation_probabilities = model.predict_proba(X_all[validation_mask])[:, 1]
                test_probabilities = model.predict_proba(X_all[test_mask])[:, 1]
                target_probabilities = model.predict_proba(X_target)[:, 1]

                for budget in BUDGETS:
                    threshold, validation_metrics = select_fpr_constrained_threshold(
                        y_all[validation_mask], validation_probabilities, budget
                    )
                    output_rows.append(
                        metric_row(
                            source,
                            target,
                            repetition,
                            model_name,
                            budget,
                            threshold,
                            "source_validation",
                            y_all[validation_mask],
                            validation_probabilities,
                        )
                    )
                    output_rows.append(
                        metric_row(
                            source,
                            target,
                            repetition,
                            model_name,
                            budget,
                            threshold,
                            "source_s3_test",
                            y_all[test_mask],
                            test_probabilities,
                        )
                    )
                    output_rows.append(
                        metric_row(
                            source,
                            target,
                            repetition,
                            model_name,
                            budget,
                            threshold,
                            "external_primary",
                            y_target[primary_mask],
                            target_probabilities[primary_mask],
                        )
                    )
                    if validation_metrics["n_benign"] * budget < 10:
                        issues.append(
                            {
                                "type": "low_expected_false_positive_count",
                                "source_dataset": source,
                                "repetition": repetition,
                                "model": model_name,
                                "fpr_budget": budget,
                                "n_validation_benign": validation_metrics["n_benign"],
                                "interpretation": (
                                    "The empirical operating point is valid for the frozen "
                                    "validation cohort but is estimated from fewer than ten "
                                    "expected false positives at the requested budget."
                                ),
                            }
                        )

    runs = pd.DataFrame(output_rows)
    summary = summarize(runs)
    runs_path = RESULTS_DIR / "low_fpr_operating_point_runs.csv"
    summary_path = RESULTS_DIR / "low_fpr_operating_point_summary.csv"
    runs.to_csv(runs_path, index=False)
    summary.to_csv(summary_path, index=False)

    manifest = {
        "completed_utc": utc_now(),
        "elapsed_seconds": time.perf_counter() - started,
        "design": (
            "Thresholds selected only on each source S3 validation split, then frozen for "
            "source S3 test and primary domain-filtered external evaluation."
        ),
        "fpr_budgets": list(BUDGETS),
        "rows": len(runs),
        "expected_rows": len(DATASETS) * len(REPETITIONS) * len(MODEL_NAMES) * len(BUDGETS) * 3,
        "input_sha256": {str(path): sha256_file(path) for path in sorted(input_files)},
        "output_sha256": {
            str(runs_path): sha256_file(runs_path),
            str(summary_path): sha256_file(summary_path),
        },
    }
    (RESULTS_DIR / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (LOGS_DIR / "issues.jsonl").open("w", encoding="utf-8") as handle:
        for issue in issues:
            handle.write(json.dumps(issue, sort_keys=True) + "\n")

    if len(runs) != manifest["expected_rows"]:
        raise AssertionError(f"Expected {manifest['expected_rows']} rows, found {len(runs)}")
    if not np.isfinite(runs.select_dtypes(include=[np.number]).to_numpy()).all():
        raise AssertionError("Non-finite numeric value in operating-point outputs")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
