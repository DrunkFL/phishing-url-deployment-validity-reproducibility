from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT.parent
PART2 = EXPERIMENT_ROOT / "part2_url_normalization_leakage_audit"
PART3 = EXPERIMENT_ROOT / "part3_s0_s4_data_splits"
PART4 = EXPERIMENT_ROOT / "part4_baseline_models"
PART5 = EXPERIMENT_ROOT / "part5_shap_stability"
for scripts_path in (PART4 / "scripts", PART5 / "scripts"):
    sys.path.insert(0, str(scripts_path))

from model_utils import calculate_metrics, select_model  # noqa: E402
from shap_utils import (  # noqa: E402
    build_explainer,
    deterministic_stratified_sample,
    explain,
    global_feature_summary,
)
from selection_utils import validate_feature_names  # noqa: E402


DATASETS = {
    "phiusiil": {
        "features": "phiusiil_master_features.parquet",
        "assignment": "phiusiil_master_s3_domain_assignments.parquet",
    },
    "iscx_url2016_binary": {
        "features": "iscx_url2016_binary_master_features.parquet",
        "assignment": "iscx_url2016_binary_master_s3_domain_assignments.parquet",
    },
}
MODELS = ["lr", "rf", "xgb"]
FEATURE_SETS = ["f_single", "f_stable"]
REPETITIONS = [f"r{index:02d}" for index in range(10)]
MODEL_SEEDS = [20261001 + index for index in range(10)]
SHAP_SIZE = 200
FEATURE_DICTIONARY = PART2 / "results" / "feature_dictionary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Part 6 reduced feature models")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--repetitions", nargs="+", choices=REPETITIONS, default=REPETITIONS)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--feature-sets", nargs="+", choices=FEATURE_SETS, default=FEATURE_SETS)
    parser.add_argument("--n-jobs", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def append_issue(payload: dict[str, Any]) -> None:
    path = ROOT / "logs" / "runtime_issues.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time_utc": utc_now(), **payload}, sort_keys=True) + "\n")


def feature_names() -> list[str]:
    frame = pd.read_csv(FEATURE_DICTIONARY).sort_values("feature_order")
    names = frame["feature"].tolist()
    if len(names) != 35 or len(names) != len(set(names)):
        raise ValueError("Expected 35 unique features")
    return names


def load_internal(dataset: str, names: list[str]) -> pd.DataFrame:
    features = pd.read_parquet(
        PART2 / "data" / DATASETS[dataset]["features"],
        columns=["source_row", "raw_url_sha256", "label", *names],
    )
    assignment = pd.read_parquet(
        PART3 / "data" / "assignments" / DATASETS[dataset]["assignment"],
        columns=[
            "sample_id", "source_row", "raw_url_sha256",
            *[f"split_{rep}" for rep in REPETITIONS],
        ],
    )
    merged = assignment.merge(
        features, on=["source_row", "raw_url_sha256"], how="inner", validate="one_to_one"
    )
    if len(merged) != len(features) or len(merged) != len(assignment):
        raise ValueError(f"Assignment-feature mismatch for {dataset}")
    return merged


def predict_probabilities(model: Any, X: np.ndarray) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    probabilities = model.predict_proba(X)[:, 1]
    return probabilities, time.perf_counter() - started


def run_one(
    dataset: str,
    repetition: str,
    model_name: str,
    feature_set: str,
    internal: pd.DataFrame,
    all_names: list[str],
    n_jobs: int,
    force: bool,
) -> None:
    output = ROOT / "runs" / dataset / "s3" / repetition / model_name / feature_set
    complete = output / "complete.json"
    if complete.exists() and not force:
        print(f"SKIP train {dataset} {repetition} {model_name} {feature_set}", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    set_path = ROOT / "feature_sets" / dataset / "s3" / repetition / model_name / "feature_sets.json"
    payload = json.loads(set_path.read_text(encoding="utf-8"))
    names = validate_feature_names(payload[feature_set], all_names)

    split = internal[f"split_{repetition}"].astype(str).to_numpy()
    masks = {name: split == name for name in ("train", "validation", "test")}
    X_train = internal.loc[masks["train"], names].to_numpy(dtype=np.float32)
    y_train = internal.loc[masks["train"], "label"].to_numpy(dtype=np.int8)
    X_validation = internal.loc[masks["validation"], names].to_numpy(dtype=np.float32)
    y_validation = internal.loc[masks["validation"], "label"].to_numpy(dtype=np.int8)
    X_test = internal.loc[masks["test"], names].to_numpy(dtype=np.float32)
    y_test = internal.loc[masks["test"], "label"].to_numpy(dtype=np.int8)
    model_seed = MODEL_SEEDS[int(repetition[1:])]
    started = time.perf_counter()
    selection = select_model(
        model_name, X_train, y_train, X_validation, y_validation, model_seed, n_jobs
    )
    probabilities, inference_seconds = predict_probabilities(selection.model, X_test)
    metrics = calculate_metrics(y_test, probabilities, selection.threshold)

    sample_tag = f"partition:{dataset}:s3:{repetition}"
    train_frame = internal.loc[masks["train"]]
    test_frame = internal.loc[masks["test"]]
    background = deterministic_stratified_sample(
        train_frame, SHAP_SIZE, f"{sample_tag}:background"
    )
    cohort = deterministic_stratified_sample(
        test_frame, SHAP_SIZE, f"{sample_tag}:cohort"
    )
    bundle = build_explainer(
        model_name, selection.model, background[names].to_numpy(dtype=np.float32)
    )
    X_cohort = cohort[names].to_numpy(dtype=np.float32)
    shap_values, shap_seconds, shap_warnings = explain(bundle, X_cohort)
    global_feature_summary(X_cohort, shap_values, names).to_csv(
        output / "test_global_importance.csv", index=False
    )
    cohort[["sample_id", "source_row", "label"]].to_parquet(
        output / "test_cohort_ids.parquet", index=False, compression="zstd"
    )
    if shap_warnings:
        write_json(output / "shap_warnings.json", shap_warnings)
        for warning in shap_warnings:
            append_issue({
                "type": "test_shap_warning", "dataset": dataset,
                "repetition": repetition, "model": model_name,
                "feature_set": feature_set, **warning,
            })
    if selection.warnings:
        write_json(output / "model_warnings.json", selection.warnings)
        for warning in selection.warnings:
            append_issue({
                "type": "model_warning", "dataset": dataset,
                "repetition": repetition, "model": model_name,
                "feature_set": feature_set, **warning,
            })

    metric_row = {
        "dataset": dataset,
        "scenario": "S3-domain",
        "scenario_key": "s3",
        "repetition": repetition,
        "partition_seed": 20260902 + int(repetition[1:]),
        "model": model_name,
        "model_seed": model_seed,
        "feature_set": feature_set,
        "n_features": len(names),
        "feature_names_json": json.dumps(names),
        "selected_candidate": selection.selected_candidate,
        "selected_params_json": json.dumps(selection.selected_params, sort_keys=True),
        "validation_macro_f1": selection.validation_metrics["macro_f1"],
        "validation_roc_auc": selection.validation_metrics["roc_auc"],
        "tuning_seconds": selection.tuning_seconds,
        "inference_seconds": inference_seconds,
        "inference_seconds_per_sample": inference_seconds / len(y_test),
        "shap_background_size": len(background),
        "shap_cohort_size": len(cohort),
        "shap_setup_seconds": bundle.setup_seconds,
        "shap_explain_seconds": shap_seconds,
        "shap_seconds_per_sample": shap_seconds / len(cohort),
        "model_output_scale": bundle.model_output_scale,
        "total_run_seconds": time.perf_counter() - started,
        **metrics,
    }
    write_json(output / "internal_metrics.json", metric_row)
    pd.DataFrame(selection.tuning_rows).to_csv(output / "tuning_results.csv", index=False)
    predictions = pd.DataFrame({
        "sample_id": internal.loc[masks["test"], "sample_id"].to_numpy(),
        "source_row": internal.loc[masks["test"], "source_row"].to_numpy(dtype=np.int64),
        "label": y_test,
        "probability_phishing": probabilities.astype(np.float32),
        "prediction": (probabilities >= selection.threshold).astype(np.int8),
    })
    prediction_path = ROOT / "predictions" / dataset / "s3" / repetition / model_name
    prediction_path.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(
        prediction_path / f"{feature_set}.parquet", index=False, compression="zstd"
    )
    model_path = ROOT / "models" / dataset / "s3" / repetition / model_name
    model_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(selection.model, model_path / f"{feature_set}.joblib", compress=3)
    write_json(complete, {
        "completed_utc": utc_now(),
        "metrics": str(output / "internal_metrics.json"),
        "prediction": str(prediction_path / f"{feature_set}.parquet"),
        "model": str(model_path / f"{feature_set}.joblib"),
    })
    print(
        f"DONE train {dataset} {repetition} {model_name} {feature_set} "
        f"n={len(names)} f1={metrics['macro_f1']:.4f} shap={shap_seconds:.1f}s",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    names = feature_names()
    for dataset in args.datasets:
        internal = load_internal(dataset, names)
        for repetition in args.repetitions:
            for model_name in args.models:
                for feature_set in args.feature_sets:
                    run_one(
                        dataset, repetition, model_name, feature_set, internal,
                        names, args.n_jobs, args.force,
                    )


if __name__ == "__main__":
    main()
