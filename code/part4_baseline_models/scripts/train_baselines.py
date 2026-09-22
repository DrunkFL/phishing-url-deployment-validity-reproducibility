from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost

from model_utils import calculate_metrics, select_model


PART4_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = PART4_ROOT.parent
PART2_ROOT = EXPERIMENT_ROOT / "part2_url_normalization_leakage_audit"
PART3_ROOT = EXPERIMENT_ROOT / "part3_s0_s4_data_splits"
FEATURE_DICTIONARY = PART2_ROOT / "results" / "feature_dictionary.csv"
MODEL_SEEDS = [20261001 + index for index in range(10)]
REPETITIONS = [f"r{index:02d}" for index in range(10)]
SCENARIOS = {
    "s0": ("S0-row", "s0_row"),
    "s1": ("S1-url", "s1_url"),
    "s2": ("S2-host", "s2_host"),
    "s3": ("S3-domain", "s3_domain"),
}
DATASETS = {
    "phiusiil": {
        "display": "PhiUSIIL",
        "feature_file": "phiusiil_master_features.parquet",
        "assignment_prefix": "phiusiil_master",
        "s4_file": "s4_phiusiil_to_iscx_url2016_binary_master_assignments.parquet",
        "s4_target": "iscx_url2016_binary",
    },
    "iscx_url2016_binary": {
        "display": "ISCX-URL2016-binary",
        "feature_file": "iscx_url2016_binary_master_features.parquet",
        "assignment_prefix": "iscx_url2016_binary_master",
        "s4_file": "s4_iscx_url2016_binary_to_phiusiil_master_assignments.parquet",
        "s4_target": "phiusiil",
    },
}
MODEL_NAMES = ["lr", "rf", "xgb"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Part 4 baseline experiments")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--repetitions", nargs="+", choices=REPETITIONS, default=REPETITIONS)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=MODEL_NAMES)
    parser.add_argument("--n-jobs", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def append_runtime_issue(issue: dict[str, Any]) -> None:
    path = PART4_ROOT / "logs" / "runtime_issues.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time_utc": utc_now(), **issue}, sort_keys=True) + "\n")


def load_feature_names() -> list[str]:
    dictionary = pd.read_csv(FEATURE_DICTIONARY).sort_values("feature_order")
    names = dictionary["feature"].tolist()
    if len(names) != 35 or len(names) != len(set(names)):
        raise ValueError(f"Expected 35 unique features, found {len(names)}")
    return names


def load_feature_table(dataset_key: str, feature_names: list[str]) -> pd.DataFrame:
    path = PART2_ROOT / "data" / DATASETS[dataset_key]["feature_file"]
    columns = ["source_row", "raw_url_sha256", "label", *feature_names]
    frame = pd.read_parquet(path, columns=columns)
    if frame[["source_row", "raw_url_sha256"]].duplicated().any():
        raise ValueError(f"Non-unique feature identifier in {dataset_key}")
    values = frame[feature_names].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite feature value in {dataset_key}")
    if set(frame["label"].unique()) != {0, 1}:
        raise ValueError(f"Unexpected labels in {dataset_key}")
    return frame


def merge_assignment(dataset_key: str, scenario_key: str, features: pd.DataFrame) -> pd.DataFrame:
    _, file_token = SCENARIOS[scenario_key]
    filename = f"{DATASETS[dataset_key]['assignment_prefix']}_{file_token}_assignments.parquet"
    assignment_path = PART3_ROOT / "data" / "assignments" / filename
    columns = ["sample_id", "source_row", "raw_url_sha256", *[f"split_{rep}" for rep in REPETITIONS]]
    assignment = pd.read_parquet(assignment_path, columns=columns)
    merged = assignment.merge(
        features,
        on=["source_row", "raw_url_sha256"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(features) or len(merged) != len(assignment):
        raise ValueError(f"Assignment-feature mismatch for {dataset_key}/{scenario_key}")
    return merged


def predict_and_time(model: Any, X: np.ndarray) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    probabilities = model.predict_proba(X)[:, 1]
    elapsed = time.perf_counter() - start
    return probabilities, elapsed


def prediction_frame(
    sample_ids: np.ndarray,
    source_rows: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    primary_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    output = pd.DataFrame({
        "sample_id": sample_ids,
        "source_row": source_rows.astype(np.int64),
        "label": labels.astype(np.int8),
        "probability_phishing": probabilities.astype(np.float32),
        "prediction": (probabilities >= threshold).astype(np.int8),
    })
    if primary_mask is not None:
        output["included_in_primary"] = primary_mask.astype(bool)
    return output


def evaluate_s4(
    source_key: str,
    repetition: str,
    model_name: str,
    selection: Any,
    feature_tables: dict[str, pd.DataFrame],
    feature_names: list[str],
    output_dir: Path,
) -> list[dict[str, Any]]:
    target_key = DATASETS[source_key]["s4_target"]
    assignment_path = PART3_ROOT / "data" / "assignments" / DATASETS[source_key]["s4_file"]
    role_column = f"role_{repetition}"
    assignment = pd.read_parquet(
        assignment_path,
        columns=["sample_id", "origin", "source_row", "raw_url_sha256", "label", role_column],
    )
    target_assignment = assignment.loc[assignment["origin"] == "target"].copy()
    target_features = feature_tables[target_key]
    target = target_assignment.merge(
        target_features.drop(columns="label"),
        on=["source_row", "raw_url_sha256"],
        how="inner",
        validate="one_to_one",
    )
    if len(target) != len(target_features):
        raise ValueError(f"S4 target mismatch for {source_key}/{repetition}")

    X_target = target[feature_names].to_numpy(dtype=np.float32)
    y_target = target["label"].to_numpy(dtype=np.int8)
    probabilities, inference_seconds = predict_and_time(selection.model, X_target)
    primary_mask = (target[role_column] == "target_external_primary").to_numpy()
    prediction_path = output_dir / "s4_predictions.parquet"
    prediction_frame(
        target["sample_id"].to_numpy(),
        target["source_row"].to_numpy(),
        y_target,
        probabilities,
        selection.threshold,
        primary_mask,
    ).to_parquet(prediction_path, index=False, compression="zstd")

    rows = []
    for cohort, mask in (("unfiltered", np.ones(len(target), dtype=bool)), ("primary_domain_filtered", primary_mask)):
        metrics = calculate_metrics(y_target[mask], probabilities[mask], selection.threshold)
        rows.append({
            "source_dataset": source_key,
            "target_dataset": target_key,
            "corpus_version": "master",
            "scenario": "S4-external",
            "repetition": repetition,
            "model": model_name,
            "cohort": cohort,
            "selected_candidate": selection.selected_candidate,
            "selected_params_json": json.dumps(selection.selected_params, sort_keys=True),
            "model_seed": MODEL_SEEDS[int(repetition[1:])],
            "inference_seconds_all_target": inference_seconds,
            "inference_seconds_per_sample": inference_seconds / len(target),
            **metrics,
        })
    return rows


def run_one(
    dataset_key: str,
    scenario_key: str,
    repetition: str,
    model_name: str,
    merged: pd.DataFrame,
    feature_tables: dict[str, pd.DataFrame],
    feature_names: list[str],
    n_jobs: int,
    force: bool,
) -> None:
    scenario_name = SCENARIOS[scenario_key][0]
    run_dir = PART4_ROOT / "runs" / "internal" / dataset_key / scenario_key / repetition / model_name
    complete_path = run_dir / "complete.json"
    if complete_path.exists() and not force:
        print(f"SKIP {dataset_key} {scenario_key} {repetition} {model_name}", flush=True)
        return
    run_dir.mkdir(parents=True, exist_ok=True)

    split_column = f"split_{repetition}"
    split = merged[split_column].astype(str).to_numpy()
    masks = {name: split == name for name in ("train", "validation", "test")}
    if any(mask.sum() == 0 for mask in masks.values()):
        raise ValueError(f"Empty partition for {dataset_key}/{scenario_key}/{repetition}")

    X_all = merged[feature_names].to_numpy(dtype=np.float32)
    y_all = merged["label"].to_numpy(dtype=np.int8)
    model_seed = MODEL_SEEDS[int(repetition[1:])]
    started = time.perf_counter()
    selection = select_model(
        model_name,
        X_all[masks["train"]],
        y_all[masks["train"]],
        X_all[masks["validation"]],
        y_all[masks["validation"]],
        model_seed,
        n_jobs,
    )
    if selection.warnings:
        write_json(run_dir / "warnings.json", selection.warnings)
        for warning in selection.warnings:
            append_runtime_issue({
                "type": "model_warning",
                "dataset": dataset_key,
                "scenario": scenario_key,
                "repetition": repetition,
                "model": model_name,
                **warning,
            })
    failed_candidates = [row for row in selection.tuning_rows if row["status"] == "failed"]
    for failure in failed_candidates:
        append_runtime_issue({
            "type": "candidate_failure",
            "dataset": dataset_key,
            "scenario": scenario_key,
            "repetition": repetition,
            "model": model_name,
            **failure,
        })

    test_mask = masks["test"]
    test_probabilities, inference_seconds = predict_and_time(selection.model, X_all[test_mask])
    test_metrics = calculate_metrics(y_all[test_mask], test_probabilities, selection.threshold)
    metric_row = {
        "dataset": dataset_key,
        "dataset_display": DATASETS[dataset_key]["display"],
        "corpus_version": "master",
        "scenario": scenario_name,
        "scenario_key": scenario_key,
        "repetition": repetition,
        "partition_seed": 20260902 + int(repetition[1:]),
        "model": model_name,
        "model_seed": model_seed,
        "selected_candidate": selection.selected_candidate,
        "selected_params_json": json.dumps(selection.selected_params, sort_keys=True),
        "validation_macro_f1": selection.validation_metrics["macro_f1"],
        "validation_roc_auc": selection.validation_metrics["roc_auc"],
        "tuning_seconds": selection.tuning_seconds,
        "inference_seconds": inference_seconds,
        "inference_seconds_per_sample": inference_seconds / test_mask.sum(),
        "total_run_seconds": time.perf_counter() - started,
        **test_metrics,
    }
    write_json(run_dir / "internal_metrics.json", metric_row)
    pd.DataFrame(selection.tuning_rows).to_csv(run_dir / "tuning_results.csv", index=False)

    prediction_path = PART4_ROOT / "predictions" / "internal" / dataset_key / scenario_key / repetition
    prediction_path.mkdir(parents=True, exist_ok=True)
    prediction_frame(
        merged.loc[test_mask, "sample_id"].to_numpy(),
        merged.loc[test_mask, "source_row"].to_numpy(),
        y_all[test_mask],
        test_probabilities,
        selection.threshold,
    ).to_parquet(prediction_path / f"{model_name}.parquet", index=False, compression="zstd")

    if repetition == "r00":
        model_path = PART4_ROOT / "models" / dataset_key / scenario_key
        model_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(selection.model, model_path / f"{model_name}.joblib", compress=3)

    s4_rows = []
    if scenario_key == "s3":
        s4_output = PART4_ROOT / "predictions" / "s4" / dataset_key / repetition / model_name
        s4_output.mkdir(parents=True, exist_ok=True)
        s4_rows = evaluate_s4(
            dataset_key,
            repetition,
            model_name,
            selection,
            feature_tables,
            feature_names,
            s4_output,
        )
        write_json(run_dir / "s4_metrics.json", s4_rows)

    write_json(complete_path, {
        "completed_utc": utc_now(),
        "internal_metrics": str(run_dir / "internal_metrics.json"),
        "s4_evaluated": bool(s4_rows),
    })
    print(
        f"DONE {dataset_key} {scenario_key} {repetition} {model_name} "
        f"macro_f1={test_metrics['macro_f1']:.4f} auc={test_metrics['roc_auc']:.4f} "
        f"seconds={metric_row['total_run_seconds']:.1f}",
        flush=True,
    )


def bootstrap_summary(frame: pd.DataFrame, group_columns: list[str], metric_columns: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(20261001)
    rows = []
    for keys, group in frame.groupby(group_columns, sort=True, observed=True):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        base = dict(zip(group_columns, key_values))
        for metric in metric_columns:
            values = group[metric].to_numpy(dtype=float)
            samples = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
            rows.append({
                **base,
                "metric": metric,
                "n_repetitions": len(values),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "ci95_lower": float(np.quantile(samples, 0.025)),
                "ci95_upper": float(np.quantile(samples, 0.975)),
                "min": float(values.min()),
                "max": float(values.max()),
            })
    return pd.DataFrame(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_outputs() -> None:
    results_dir = PART4_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    internal_rows = [json.loads(path.read_text(encoding="utf-8")) for path in PART4_ROOT.glob("runs/internal/**/internal_metrics.json")]
    internal = pd.DataFrame(internal_rows).sort_values(["dataset", "scenario_key", "repetition", "model"])
    internal.to_csv(results_dir / "internal_metrics.csv", index=False)

    tuning_frames = []
    for path in PART4_ROOT.glob("runs/internal/**/tuning_results.csv"):
        relative = path.relative_to(PART4_ROOT / "runs" / "internal").parts
        frame = pd.read_csv(path)
        frame.insert(0, "model", relative[3])
        frame.insert(0, "repetition", relative[2])
        frame.insert(0, "scenario_key", relative[1])
        frame.insert(0, "dataset", relative[0])
        tuning_frames.append(frame)
    pd.concat(tuning_frames, ignore_index=True).to_csv(results_dir / "tuning_results.csv", index=False)

    s4_rows = []
    for path in PART4_ROOT.glob("runs/internal/**/s4_metrics.json"):
        s4_rows.extend(json.loads(path.read_text(encoding="utf-8")))
    metrics = ["accuracy", "balanced_accuracy", "precision", "recall", "macro_f1", "roc_auc", "pr_auc", "fpr"]
    if s4_rows:
        s4 = pd.DataFrame(s4_rows).sort_values(["source_dataset", "repetition", "model", "cohort"])
    else:
        s4 = pd.DataFrame(columns=[
            "source_dataset", "target_dataset", "corpus_version", "scenario",
            "repetition", "model", "cohort", *metrics,
        ])
    s4.to_csv(results_dir / "s4_metrics.csv", index=False)

    bootstrap_summary(internal, ["dataset", "scenario", "model"], metrics).to_csv(
        results_dir / "internal_performance_summary.csv", index=False
    )
    if s4.empty:
        pd.DataFrame(columns=[
            "source_dataset", "target_dataset", "cohort", "model", "metric",
            "n_repetitions", "mean", "std", "ci95_lower", "ci95_upper", "min", "max",
        ]).to_csv(results_dir / "s4_performance_summary.csv", index=False)
    else:
        bootstrap_summary(s4, ["source_dataset", "target_dataset", "cohort", "model"], metrics).to_csv(
            results_dir / "s4_performance_summary.csv", index=False
        )

    selected = (
        internal.groupby(["dataset", "scenario", "model", "selected_candidate"], observed=True)
        .size().rename("selection_count").reset_index()
    )
    selected.to_csv(results_dir / "selected_hyperparameter_counts.csv", index=False)

    manifest_paths = [
        results_dir / "internal_metrics.csv",
        results_dir / "tuning_results.csv",
        results_dir / "s4_metrics.csv",
        results_dir / "internal_performance_summary.csv",
        results_dir / "s4_performance_summary.csv",
        results_dir / "selected_hyperparameter_counts.csv",
    ]
    manifest = pd.DataFrame([
        {"path": str(path.relative_to(PART4_ROOT)), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in manifest_paths
    ])
    manifest.to_csv(results_dir / "result_manifest.csv", index=False)


def main() -> None:
    args = parse_args()
    for directory in ("logs", "models", "predictions", "results", "runs"):
        (PART4_ROOT / directory).mkdir(parents=True, exist_ok=True)
    feature_names = load_feature_names()
    feature_tables = {key: load_feature_table(key, feature_names) for key in DATASETS}
    write_json(PART4_ROOT / "results" / "run_metadata.json", {
        "started_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "sklearn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "model_seeds": MODEL_SEEDS,
        "n_jobs": args.n_jobs,
        "arguments": vars(args),
    })

    for dataset_key in args.datasets:
        for scenario_key in args.scenarios:
            merged = merge_assignment(dataset_key, scenario_key, feature_tables[dataset_key])
            for repetition in args.repetitions:
                for model_name in args.models:
                    try:
                        run_one(
                            dataset_key,
                            scenario_key,
                            repetition,
                            model_name,
                            merged,
                            feature_tables,
                            feature_names,
                            args.n_jobs,
                            args.force,
                        )
                    except Exception as exc:
                        append_runtime_issue({
                            "type": "run_failure",
                            "dataset": dataset_key,
                            "scenario": scenario_key,
                            "repetition": repetition,
                            "model": model_name,
                            "error": f"{type(exc).__name__}: {exc}",
                        })
                        raise
    aggregate_outputs()
    print("Part 4 baseline experiment completed.", flush=True)


if __name__ == "__main__":
    main()
