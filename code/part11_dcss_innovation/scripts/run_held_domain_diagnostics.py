from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT.parent
PART4 = EXPERIMENT_ROOT / "part4_baseline_models"
sys.path.insert(0, str(PART4 / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))

from main_comparison_utils import extended_metrics  # noqa: E402
from model_utils import build_model  # noqa: E402
from run_dcss_selection import (  # noqa: E402
    DATASETS,
    MODELS,
    REPETITIONS,
    feature_names,
    feature_table,
    load_outer_training_only,
    selected_parameter_map,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refit Stage 11C held-domain models for performance diagnostics")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--repetitions", nargs="+", choices=REPETITIONS, default=list(REPETITIONS))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--n-jobs", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--skip-aggregate", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def append_runtime_issue(payload: dict[str, Any]) -> None:
    path = ROOT / "logs" / "stage11e_runtime_issues.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time_utc": utc_now(), **payload}, sort_keys=True) + "\n")


def run_one(
    dataset: str,
    repetition: str,
    model_name: str,
    held_fold: int,
    outer_train: pd.DataFrame,
    names: list[str],
    selected: dict[str, Any],
    n_jobs: int,
    force: bool,
) -> None:
    run = ROOT / "runs" / "held_domain_diagnostics" / dataset / "s3" / repetition / model_name / f"h{held_fold:02d}"
    complete = run / "complete.json"
    if complete.is_file() and not force:
        print(f"SKIP {dataset}/{repetition}/{model_name}/h{held_fold:02d}", flush=True)
        return
    run.mkdir(parents=True, exist_ok=True)
    assignment_path = ROOT / "data" / "processed" / "dcss_fold_assignments" / dataset / f"{repetition}.parquet"
    assignments = pd.read_parquet(assignment_path, columns=["sample_id", "held_fold"])
    frame = outer_train.merge(assignments, on="sample_id", how="inner", validate="one_to_one")
    if len(frame) != len(outer_train):
        raise ValueError(f"Held-fold assignment mismatch: {dataset}/{repetition}")
    fit = frame.loc[frame["held_fold"] != held_fold]
    held = frame.loc[frame["held_fold"] == held_fold]
    if set(fit["registrable_domain_sha256"]) & set(held["registrable_domain_sha256"]):
        raise ValueError(f"Domain overlap: {dataset}/{repetition}/h{held_fold:02d}")

    model_seed = 20261401 + 10 * int(repetition[1:]) + held_fold
    model = build_model(model_name, selected["selected_params"], model_seed, n_jobs)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(fit[names].to_numpy(dtype=np.float32), fit["label"].to_numpy(dtype=np.int8))
    fit_seconds = time.perf_counter() - started
    classes = [int(value) for value in model.classes_]
    if classes != [0, 1]:
        raise ValueError(f"Unexpected classes {classes}: {dataset}/{repetition}/{model_name}/h{held_fold:02d}")
    inference_started = time.perf_counter()
    probabilities = model.predict_proba(held[names].to_numpy(dtype=np.float32))[:, 1]
    inference_seconds = time.perf_counter() - inference_started
    metrics = {
        "stage": "11E",
        "diagnostic": "held_domain_performance",
        "dataset": dataset,
        "scenario": "source_outer_train_held_domain_fold",
        "repetition": repetition,
        "model": model_name,
        "held_fold": held_fold,
        "model_seed": model_seed,
        "selected_candidate": selected["selected_candidate"],
        "selected_params_json": json.dumps(selected["selected_params"], sort_keys=True),
        "n_features": len(names),
        "fit_rows": len(fit),
        "held_rows": len(held),
        "fit_domains": int(fit["registrable_domain_sha256"].nunique()),
        "held_domains": int(held["registrable_domain_sha256"].nunique()),
        "model_classes_json": json.dumps(classes),
        "threshold_policy": "fixed_0.5_no_held_label_tuning",
        "fit_seconds": fit_seconds,
        "inference_seconds": inference_seconds,
        "warning_count": len(caught),
        **extended_metrics(
            held["label"].to_numpy(dtype=np.int8), probabilities, 0.5
        ),
    }
    prediction = held[["sample_id", "source_row", "registrable_domain_sha256", "label", "held_fold"]].reset_index(drop=True).copy()
    prediction["probability_phishing"] = probabilities.astype(np.float64)
    prediction["prediction"] = (probabilities >= 0.5).astype(np.int8)
    prediction_path = ROOT / "predictions" / "held_domain_diagnostics" / dataset / repetition / model_name / f"h{held_fold:02d}.parquet"
    write_parquet(prediction_path, prediction)
    write_json(run / "metrics.json", metrics)
    if caught:
        warning_rows = [
            {"category": warning.category.__name__, "message": str(warning.message)}
            for warning in caught
        ]
        write_json(run / "model_warnings.json", warning_rows)
        for warning in warning_rows:
            append_runtime_issue(
                {
                    "type": "held_domain_model_warning",
                    "dataset": dataset,
                    "repetition": repetition,
                    "model": model_name,
                    "held_fold": held_fold,
                    **warning,
                }
            )
    write_json(
        complete,
        {
            "completed_utc": utc_now(),
            "metrics": str(run / "metrics.json"),
            "prediction": str(prediction_path),
        },
    )
    print(f"DONE {dataset}/{repetition}/{model_name}/h{held_fold:02d} auc={metrics['roc_auc']:.4f}", flush=True)


def aggregate() -> int:
    rows = []
    base = ROOT / "runs" / "held_domain_diagnostics"
    for path in sorted(base.glob("*/s3/r??/*/h??/metrics.json")):
        if (path.parent / "complete.json").is_file():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    if rows:
        pd.DataFrame(rows).sort_values(
            ["dataset", "repetition", "model", "held_fold"]
        ).to_csv(ROOT / "results" / "stage11e_held_domain_metrics.csv", index=False)
    return len(rows)


def main() -> None:
    args = parse_args()
    if not args.aggregate_only:
        names = feature_names()
        parameter_map = selected_parameter_map()
        for dataset in args.datasets:
            features = feature_table(dataset, names)
            for repetition in args.repetitions:
                outer_train = load_outer_training_only(dataset, repetition, features)
                for model_name in args.models:
                    selected = parameter_map[(dataset, repetition, model_name)]
                    for held_fold in range(5):
                        try:
                            run_one(dataset, repetition, model_name, held_fold, outer_train, names, selected, args.n_jobs, args.force)
                        except Exception as error:
                            append_runtime_issue(
                                {
                                    "type": "held_domain_failure",
                                    "dataset": dataset,
                                    "repetition": repetition,
                                    "model": model_name,
                                    "held_fold": held_fold,
                                    "error": f"{type(error).__name__}: {error}",
                                }
                            )
                            raise
    if not args.skip_aggregate:
        print({"aggregated_held_domain_rows": aggregate()}, flush=True)


if __name__ == "__main__":
    main()
