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
sys.path.insert(0, str(PART4 / "scripts"))

from model_utils import calculate_metrics, select_model  # noqa: E402


DATASETS = {
    "phiusiil": {
        "features": "phiusiil_master_features.parquet",
        "assignment": "phiusiil_master_s3_domain_assignments.parquet",
        "target": "iscx_url2016_binary",
        "s4": "s4_phiusiil_to_iscx_url2016_binary_master_assignments.parquet",
    },
    "iscx_url2016_binary": {
        "features": "iscx_url2016_binary_master_features.parquet",
        "assignment": "iscx_url2016_binary_master_s3_domain_assignments.parquet",
        "target": "phiusiil",
        "s4": "s4_iscx_url2016_binary_to_phiusiil_master_assignments.parquet",
    },
}
MODELS = ["lr", "rf", "xgb"]
FEATURE_SETS = ["f_mi", "f_permutation"]
REPETITIONS = [f"r{index:02d}" for index in range(10)]
MODEL_SEEDS = [20261001 + index for index in range(10)]
FEATURE_DICTIONARY = PART2 / "results" / "feature_dictionary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate Part 8 baselines")
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


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
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
        raise ValueError("Expected 35 unique frozen features")
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
            *[f"split_{repetition}" for repetition in REPETITIONS],
        ],
    )
    merged = assignment.merge(
        features,
        on=["source_row", "raw_url_sha256"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(features) or len(merged) != len(assignment):
        raise ValueError(f"Internal assignment-feature mismatch for {dataset}")
    return merged


def load_target(source: str, names: list[str]) -> pd.DataFrame:
    target_key = DATASETS[source]["target"]
    target_features = pd.read_parquet(
        PART2 / "data" / DATASETS[target_key]["features"],
        columns=["source_row", "raw_url_sha256", "label", *names],
    ).rename(columns={"label": "feature_label"})
    assignment = pd.read_parquet(
        PART3 / "data" / "assignments" / DATASETS[source]["s4"],
        columns=[
            "sample_id", "origin", "source_row", "raw_url_sha256", "label",
            *[f"role_{repetition}" for repetition in REPETITIONS],
        ],
    )
    target_assignment = assignment.loc[assignment["origin"] == "target"].copy()
    target = target_assignment.merge(
        target_features,
        on=["source_row", "raw_url_sha256"],
        how="inner",
        validate="one_to_one",
    )
    if len(target) != len(target_features) or len(target) != len(target_assignment):
        raise ValueError(f"External assignment-feature mismatch for source {source}")
    if not np.array_equal(target["label"], target["feature_label"]):
        raise ValueError(f"External label mismatch for source {source}")
    return target.drop(columns="feature_label")


def prediction_frame(
    frame: pd.DataFrame,
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    primary: np.ndarray | None = None,
) -> pd.DataFrame:
    result = pd.DataFrame({
        "sample_id": frame["sample_id"].to_numpy(),
        "source_row": frame["source_row"].to_numpy(dtype=np.int64),
        "label": labels.astype(np.int8),
        "probability_phishing": probabilities.astype(np.float64),
        "prediction": (probabilities >= threshold).astype(np.int8),
    })
    if primary is not None:
        result["included_in_primary"] = primary.astype(bool)
    return result


def run_one(
    dataset: str,
    repetition: str,
    model_name: str,
    feature_set: str,
    internal: pd.DataFrame,
    target: pd.DataFrame,
    all_names: list[str],
    n_jobs: int,
    force: bool,
) -> None:
    run_dir = ROOT / "runs" / dataset / "s3" / repetition / model_name / feature_set
    complete = run_dir / "complete.json"
    if complete.exists() and not force:
        print(f"SKIP train {dataset} {repetition} {model_name} {feature_set}", flush=True)
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    set_payload = json.loads(
        (
            ROOT / "feature_sets" / dataset / "s3" / repetition
            / model_name / "traditional_feature_sets.json"
        ).read_text(encoding="utf-8")
    )
    names = set_payload[feature_set]
    if len(names) != 15 or len(set(names)) != 15 or not set(names).issubset(all_names):
        raise ValueError(f"Invalid feature set {dataset}/{repetition}/{model_name}/{feature_set}")

    split = internal[f"split_{repetition}"].astype(str).to_numpy()
    masks = {name: split == name for name in ("train", "validation", "test")}
    X_train = internal.loc[masks["train"], names].to_numpy(dtype=np.float32)
    y_train = internal.loc[masks["train"], "label"].to_numpy(dtype=np.int8)
    X_validation = internal.loc[masks["validation"], names].to_numpy(dtype=np.float32)
    y_validation = internal.loc[masks["validation"], "label"].to_numpy(dtype=np.int8)
    test = internal.loc[masks["test"]]
    X_test = test[names].to_numpy(dtype=np.float32)
    y_test = test["label"].to_numpy(dtype=np.int8)
    model_seed = MODEL_SEEDS[int(repetition[1:])]

    started = time.perf_counter()
    selection = select_model(
        model_name, X_train, y_train, X_validation, y_validation, model_seed, n_jobs
    )
    internal_started = time.perf_counter()
    internal_probabilities = selection.model.predict_proba(X_test)[:, 1]
    internal_inference = time.perf_counter() - internal_started
    internal_metrics = {
        "dataset": dataset,
        "scenario": "S3-domain",
        "scenario_key": "s3",
        "repetition": repetition,
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
        "inference_seconds": internal_inference,
        "inference_seconds_per_sample": internal_inference / len(test),
        **calculate_metrics(y_test, internal_probabilities, selection.threshold),
    }

    X_target = target[names].to_numpy(dtype=np.float32)
    y_target = target["label"].to_numpy(dtype=np.int8)
    external_started = time.perf_counter()
    external_probabilities = selection.model.predict_proba(X_target)[:, 1]
    external_inference = time.perf_counter() - external_started
    primary = (target[f"role_{repetition}"] == "target_external_primary").to_numpy()
    external_metrics = []
    for cohort, mask in (
        ("unfiltered", np.ones(len(target), dtype=bool)),
        ("primary_domain_filtered", primary),
    ):
        external_metrics.append({
            "source_dataset": dataset,
            "target_dataset": DATASETS[dataset]["target"],
            "scenario": "S4-external",
            "repetition": repetition,
            "model": model_name,
            "model_seed": model_seed,
            "feature_set": feature_set,
            "n_features": len(names),
            "feature_names_json": json.dumps(names),
            "cohort": cohort,
            "selected_candidate": selection.selected_candidate,
            "selected_params_json": json.dumps(selection.selected_params, sort_keys=True),
            "source_validation_threshold": selection.threshold,
            "inference_seconds_all_target": external_inference,
            "inference_seconds_per_sample": external_inference / len(target),
            **calculate_metrics(y_target[mask], external_probabilities[mask], selection.threshold),
        })

    internal_prediction_path = (
        ROOT / "predictions" / "internal" / dataset / "s3"
        / repetition / model_name / f"{feature_set}.parquet"
    )
    external_prediction_path = (
        ROOT / "predictions" / "external" / dataset
        / repetition / model_name / f"{feature_set}.parquet"
    )
    write_parquet(
        internal_prediction_path,
        prediction_frame(test, y_test, internal_probabilities, selection.threshold),
    )
    write_parquet(
        external_prediction_path,
        prediction_frame(
            target, y_target, external_probabilities, selection.threshold, primary
        ),
    )
    model_path = (
        ROOT / "models" / dataset / "s3" / repetition
        / model_name / f"{feature_set}.joblib"
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(selection.model, model_path, compress=3)
    pd.DataFrame(selection.tuning_rows).to_csv(run_dir / "tuning_results.csv", index=False)
    write_json(run_dir / "internal_metrics.json", internal_metrics)
    write_json(run_dir / "external_metrics.json", external_metrics)
    if selection.warnings:
        write_json(run_dir / "model_warnings.json", selection.warnings)
        for warning in selection.warnings:
            append_issue({
                "type": "baseline_model_warning",
                "dataset": dataset,
                "repetition": repetition,
                "model": model_name,
                "feature_set": feature_set,
                **warning,
            })
    write_json(complete, {
        "completed_utc": utc_now(),
        "internal_prediction": str(internal_prediction_path),
        "external_prediction": str(external_prediction_path),
        "model": str(model_path),
        "total_seconds": time.perf_counter() - started,
    })
    print(
        f"DONE train {dataset} {repetition} {model_name} {feature_set} "
        f"internal_f1={internal_metrics['macro_f1']:.4f} "
        f"external_f1={external_metrics[1]['macro_f1']:.4f}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    names = feature_names()
    for dataset in args.datasets:
        internal = load_internal(dataset, names)
        target = load_target(dataset, names)
        for repetition in args.repetitions:
            for model_name in args.models:
                for feature_set in args.feature_sets:
                    try:
                        run_one(
                            dataset, repetition, model_name, feature_set,
                            internal, target, names, args.n_jobs, args.force,
                        )
                    except Exception as exc:
                        append_issue({
                            "type": "baseline_training_failure",
                            "dataset": dataset,
                            "repetition": repetition,
                            "model": model_name,
                            "feature_set": feature_set,
                            "error": f"{type(exc).__name__}: {exc}",
                        })
                        raise


if __name__ == "__main__":
    main()
