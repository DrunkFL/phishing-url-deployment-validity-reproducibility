from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT.parent
PART2 = EXPERIMENT_ROOT / "part2_url_normalization_leakage_audit"
PART3 = EXPERIMENT_ROOT / "part3_s0_s4_data_splits"
PART4 = EXPERIMENT_ROOT / "part4_baseline_models"
sys.path.insert(0, str(PART4 / "scripts"))

from model_utils import calculate_metrics, select_model  # noqa: E402


DATASETS = ["iscx_url2016_binary", "phiusiil"]
SCENARIOS = {"s0": "s0_row", "s3": "s3_domain"}
MODELS = ["lr", "rf", "xgb"]
REPETITIONS = [f"r{i:02d}" for i in range(10)]
MODEL_SEEDS = [20261001 + i for i in range(10)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Part 10 deduplicated sensitivity retraining")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--repetitions", nargs="+", choices=REPETITIONS, default=REPETITIONS)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--n-jobs", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def feature_names() -> list[str]:
    frame = pd.read_csv(PART2 / "results" / "feature_dictionary.csv").sort_values("feature_order")
    names = frame["feature"].tolist()
    if len(names) != 35 or len(set(names)) != 35:
        raise ValueError("Expected 35 frozen features")
    return names


def load_data(dataset: str, scenario: str, names: list[str]) -> pd.DataFrame:
    features = pd.read_parquet(
        PART2 / "data" / f"{dataset}_deduplicated_features.parquet",
        columns=["source_row", "raw_url_sha256", "label", *names],
    )
    assignment = pd.read_parquet(
        PART3 / "data" / "assignments" / f"{dataset}_deduplicated_{SCENARIOS[scenario]}_assignments.parquet",
        columns=["sample_id", "source_row", "raw_url_sha256", "registrable_domain_sha256", *[f"split_{r}" for r in REPETITIONS]],
    )
    merged = assignment.merge(features, on=["source_row", "raw_url_sha256"], how="inner", validate="one_to_one")
    if len(merged) != len(features) or len(merged) != len(assignment):
        raise ValueError(f"Feature-assignment mismatch: {dataset}/{scenario}")
    return merged


def run_one(dataset: str, scenario: str, repetition: str, model_name: str, data: pd.DataFrame,
            names: list[str], n_jobs: int, force: bool) -> None:
    run_dir = ROOT / "runs" / "deduplicated" / dataset / scenario / repetition / model_name
    complete = run_dir / "complete.json"
    if complete.exists() and not force:
        print(f"SKIP {dataset} {scenario} {repetition} {model_name}", flush=True)
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    split = data[f"split_{repetition}"].astype(str).to_numpy()
    masks = {role: split == role for role in ("train", "validation", "test")}
    if any(mask.sum() == 0 for mask in masks.values()):
        raise ValueError(f"Empty split: {dataset}/{scenario}/{repetition}")
    X = data[names].to_numpy(dtype=np.float32)
    y = data["label"].to_numpy(dtype=np.int8)
    seed = MODEL_SEEDS[int(repetition[1:])]
    started = time.perf_counter()
    selected = select_model(
        model_name, X[masks["train"]], y[masks["train"]],
        X[masks["validation"]], y[masks["validation"]], seed, n_jobs,
    )
    test_started = time.perf_counter()
    probability = selected.model.predict_proba(X[masks["test"]])[:, 1]
    inference_seconds = time.perf_counter() - test_started
    metrics = {
        "dataset": dataset,
        "corpus_version": "deduplicated",
        "scenario": "S0-row" if scenario == "s0" else "S3-domain",
        "scenario_key": scenario,
        "repetition": repetition,
        "model": model_name,
        "model_seed": seed,
        "partition_seed": 20260902 + int(repetition[1:]),
        "feature_set": "f_all",
        "n_features": len(names),
        "selected_candidate": selected.selected_candidate,
        "selected_params_json": json.dumps(selected.selected_params, sort_keys=True),
        "validation_macro_f1": selected.validation_metrics["macro_f1"],
        "validation_roc_auc": selected.validation_metrics["roc_auc"],
        "tuning_seconds": selected.tuning_seconds,
        "inference_seconds": inference_seconds,
        "total_run_seconds": time.perf_counter() - started,
        **calculate_metrics(y[masks["test"]], probability, selected.threshold),
    }
    predictions = pd.DataFrame({
        "sample_id": data.loc[masks["test"], "sample_id"].to_numpy(),
        "source_row": data.loc[masks["test"], "source_row"].to_numpy(dtype=np.int64),
        "registrable_domain_sha256": data.loc[masks["test"], "registrable_domain_sha256"].to_numpy(),
        "label": y[masks["test"]],
        "probability_phishing": probability.astype(np.float32),
        "prediction": (probability >= selected.threshold).astype(np.int8),
    })
    prediction_path = ROOT / "predictions" / "deduplicated" / dataset / scenario / repetition / f"{model_name}.parquet"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(prediction_path, index=False, compression="zstd")
    pd.DataFrame(selected.tuning_rows).to_csv(run_dir / "tuning_results.csv", index=False)
    write_json(run_dir / "metrics.json", metrics)
    if selected.warnings:
        write_json(run_dir / "warnings.json", selected.warnings)
    write_json(complete, {"completed_utc": datetime.now(timezone.utc).isoformat(), "prediction": str(prediction_path)})
    print(f"DONE {dataset} {scenario} {repetition} {model_name} macro_f1={metrics['macro_f1']:.4f}", flush=True)


def aggregate() -> None:
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in ROOT.glob("runs/deduplicated/**/metrics.json")]
    metrics = pd.DataFrame(rows).sort_values(["dataset", "scenario_key", "repetition", "model"])
    results = ROOT / "results"
    results.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(results / "deduplicated_metrics.csv", index=False)
    summary = metrics.groupby(["dataset", "scenario", "model"], observed=True).agg(
        n_runs=("macro_f1", "size"), macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
        roc_auc_mean=("roc_auc", "mean"), pr_auc_mean=("pr_auc", "mean"), recall_mean=("recall", "mean"),
        fpr_mean=("fpr", "mean"), threshold_mean=("threshold", "mean"),
    ).reset_index()
    summary.to_csv(results / "deduplicated_performance_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    names = feature_names()
    for dataset in args.datasets:
        for scenario in args.scenarios:
            data = load_data(dataset, scenario, names)
            for repetition in args.repetitions:
                for model_name in args.models:
                    run_one(dataset, scenario, repetition, model_name, data, names, args.n_jobs, args.force)
    aggregate()
    print("Part 10 deduplicated retraining complete.", flush=True)


if __name__ == "__main__":
    main()
