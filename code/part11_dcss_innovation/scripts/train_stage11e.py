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
PART4 = EXPERIMENT_ROOT / "part4_baseline_models"
sys.path.insert(0, str(PART4 / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))

from ablation_utils import feature_list_hash  # noqa: E402
from main_comparison_utils import extended_metrics  # noqa: E402
from model_utils import select_model  # noqa: E402
from train_dcss_main_comparison import (  # noqa: E402
    DATASETS,
    MODELS,
    REPETITIONS,
    feature_names,
    load_internal,
    load_target,
    prediction_frame,
)


ABLATION_METHODS = ("no_direction", "no_rank_dispersion", "no_frequency")
RANDOM_SEEDS = tuple(range(20261501, 20261531))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage 11E ablation and random baselines")
    parser.add_argument("--families", nargs="+", choices=("ablation", "random"), default=["ablation", "random"])
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--repetitions", nargs="+", choices=REPETITIONS, default=list(REPETITIONS))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--ablation-methods", nargs="+", choices=ABLATION_METHODS, default=list(ABLATION_METHODS))
    parser.add_argument("--random-seeds", nargs="+", type=int, choices=RANDOM_SEEDS, default=list(RANDOM_SEEDS))
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


def ablation_features(dataset: str, repetition: str, model: str, method: str) -> list[str]:
    path = ROOT / "feature_sets" / "ablation" / dataset / "s3" / repetition / model / "feature_sets.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = [str(value) for value in payload[method]]
    if len(selected) != len(set(selected)) or len(selected) != 15:
        raise ValueError(f"Invalid ablation feature set: {dataset}/{repetition}/{model}/{method}")
    return selected


def random_features(seed: int) -> list[str]:
    path = ROOT / "feature_sets" / "random" / "random_feature_sets.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = [str(value) for value in payload["sets"][str(seed)]]
    if len(selected) != len(set(selected)) or len(selected) != 15:
        raise ValueError(f"Invalid random feature set: {seed}")
    if feature_list_hash(selected) != payload["sha256"][str(seed)]:
        raise ValueError(f"Random feature hash mismatch: {seed}")
    return selected


def output_paths(
    family: str, dataset: str, repetition: str, model: str, method: str
) -> dict[str, Path]:
    if family == "ablation":
        leaf = method
        run = ROOT / "runs" / "ablation" / dataset / "s3" / repetition / model / leaf
        return {
            "run": run,
            "model": ROOT / "models" / "ablation" / dataset / "s3" / repetition / model / f"{leaf}.joblib",
            "internal_prediction": ROOT / "predictions" / "ablation" / "internal" / dataset / "s3" / repetition / model / f"{leaf}.parquet",
            "external_prediction": ROOT / "predictions" / "ablation" / "external" / dataset / repetition / model / f"{leaf}.parquet",
        }
    run = ROOT / "runs" / "random_baseline" / dataset / "s3" / repetition / model / method
    return {"run": run}


def run_one(
    family: str,
    dataset: str,
    repetition: str,
    model_name: str,
    method: str,
    selected: list[str],
    internal: pd.DataFrame,
    target: pd.DataFrame,
    n_jobs: int,
    force: bool,
    random_seed: int | None = None,
) -> None:
    paths = output_paths(family, dataset, repetition, model_name, method)
    run = paths["run"]
    complete = run / "complete.json"
    if complete.is_file() and not force:
        print(f"SKIP {family}/{dataset}/{repetition}/{model_name}/{method}", flush=True)
        return
    run.mkdir(parents=True, exist_ok=True)

    split = internal[f"split_{repetition}"].astype(str).to_numpy()
    train = internal.loc[split == "train"]
    validation = internal.loc[split == "validation"]
    test = internal.loc[split == "test"]
    model_seed = 20261001 + int(repetition[1:])

    started = time.perf_counter()
    selection = select_model(
        model_name,
        train[selected].to_numpy(dtype=np.float32),
        train["label"].to_numpy(dtype=np.int8),
        validation[selected].to_numpy(dtype=np.float32),
        validation["label"].to_numpy(dtype=np.int8),
        model_seed,
        n_jobs,
    )
    classes = [int(value) for value in selection.model.classes_]
    if classes != [0, 1]:
        raise ValueError(f"Unexpected classes {classes}: {dataset}/{repetition}/{model_name}/{method}")

    internal_started = time.perf_counter()
    internal_probabilities = selection.model.predict_proba(test[selected].to_numpy(dtype=np.float32))[:, 1]
    internal_inference = time.perf_counter() - internal_started
    common = {
        "stage": "11E",
        "family": family,
        "dataset": dataset,
        "scenario": "S3-domain",
        "scenario_key": "s3",
        "repetition": repetition,
        "model": model_name,
        "model_seed": model_seed,
        "feature_set": method,
        "random_seed": random_seed,
        "n_features": len(selected),
        "feature_names_json": json.dumps(selected),
        "feature_set_sha256": feature_list_hash(selected),
        "selected_candidate": selection.selected_candidate,
        "selected_params_json": json.dumps(selection.selected_params, sort_keys=True),
        "validation_macro_f1": selection.validation_metrics["macro_f1"],
        "validation_roc_auc": selection.validation_metrics["roc_auc"],
        "tuning_seconds": selection.tuning_seconds,
        "inference_seconds": internal_inference,
        "inference_seconds_per_sample": internal_inference / len(test),
        "model_classes_json": json.dumps(classes),
    }
    internal_metrics = {
        **common,
        **extended_metrics(
            test["label"].to_numpy(dtype=np.int8),
            internal_probabilities,
            selection.threshold,
        ),
    }

    external_started = time.perf_counter()
    external_probabilities = selection.model.predict_proba(target[selected].to_numpy(dtype=np.float32))[:, 1]
    external_inference = time.perf_counter() - external_started
    primary = (target[f"role_{repetition}"] == "target_external_primary").to_numpy()
    external_metrics = []
    for cohort, mask in (("unfiltered", np.ones(len(target), dtype=bool)), ("primary_domain_filtered", primary)):
        external_metrics.append(
            {
                **common,
                "source_dataset": dataset,
                "target_dataset": DATASETS[dataset]["target"],
                "scenario": "S4-external",
                "cohort": cohort,
                "source_validation_threshold": selection.threshold,
                "inference_seconds_all_target": external_inference,
                "inference_seconds_per_sample": external_inference / len(target),
                **extended_metrics(
                    target.loc[mask, "label"].to_numpy(dtype=np.int8),
                    external_probabilities[mask],
                    selection.threshold,
                ),
            }
        )

    if family == "ablation":
        write_parquet(paths["internal_prediction"], prediction_frame(test, internal_probabilities, selection.threshold))
        write_parquet(paths["external_prediction"], prediction_frame(target, external_probabilities, selection.threshold, primary))
        paths["model"].parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(selection.model, paths["model"], compress=3)

    pd.DataFrame(selection.tuning_rows).to_csv(run / "tuning_results.csv", index=False)
    write_json(run / "internal_metrics.json", internal_metrics)
    write_json(run / "external_metrics.json", external_metrics)
    if selection.warnings:
        write_json(run / "model_warnings.json", selection.warnings)
        for warning in selection.warnings:
            append_runtime_issue(
                {
                    "type": "model_warning",
                    "family": family,
                    "dataset": dataset,
                    "repetition": repetition,
                    "model": model_name,
                    "feature_set": method,
                    **warning,
                }
            )
    completion = {
        "completed_utc": utc_now(),
        "family": family,
        "feature_set_sha256": feature_list_hash(selected),
        "metrics_only": family == "random",
        "total_seconds": time.perf_counter() - started,
    }
    if family == "ablation":
        completion.update({key: str(paths[key]) for key in ("model", "internal_prediction", "external_prediction")})
    write_json(complete, completion)
    print(
        f"DONE {family}/{dataset}/{repetition}/{model_name}/{method} "
        f"internal_f1={internal_metrics['macro_f1']:.4f} external_auc={external_metrics[1]['roc_auc']:.4f}",
        flush=True,
    )


def aggregate_family(family: str) -> tuple[int, int]:
    base = ROOT / "runs" / ("ablation" if family == "ablation" else "random_baseline")
    internal_rows = []
    external_rows = []
    for path in sorted(base.glob("*/s3/r??/*/*/internal_metrics.json")):
        if not (path.parent / "complete.json").is_file():
            continue
        internal_rows.append(json.loads(path.read_text(encoding="utf-8")))
        external_rows.extend(json.loads((path.parent / "external_metrics.json").read_text(encoding="utf-8")))
    prefix = "stage11e_ablation" if family == "ablation" else "stage11e_random"
    if internal_rows:
        pd.DataFrame(internal_rows).sort_values(
            ["dataset", "repetition", "model", "feature_set"]
        ).to_csv(ROOT / "results" / f"{prefix}_internal_metrics.csv", index=False)
        pd.DataFrame(external_rows).sort_values(
            ["source_dataset", "repetition", "model", "feature_set", "cohort"]
        ).to_csv(ROOT / "results" / f"{prefix}_external_metrics.csv", index=False)
    return len(internal_rows), len(external_rows)


def main() -> None:
    args = parse_args()
    if not args.aggregate_only:
        names = feature_names()
        universe = set(names)
        for dataset in args.datasets:
            internal = load_internal(dataset, names)
            target = load_target(dataset, names)
            for repetition in args.repetitions:
                for model_name in args.models:
                    if "ablation" in args.families:
                        for method in args.ablation_methods:
                            selected = ablation_features(dataset, repetition, model_name, method)
                            if not set(selected) <= universe:
                                raise ValueError(f"Unknown feature in {method}")
                            run_one("ablation", dataset, repetition, model_name, method, selected, internal, target, args.n_jobs, args.force)
                    if "random" in args.families:
                        for seed in args.random_seeds:
                            selected = random_features(seed)
                            method = f"random_{seed}"
                            run_one("random", dataset, repetition, model_name, method, selected, internal, target, args.n_jobs, args.force, seed)
    if not args.skip_aggregate:
        counts = {family: aggregate_family(family) for family in args.families}
        print({"aggregated": counts}, flush=True)


if __name__ == "__main__":
    main()
