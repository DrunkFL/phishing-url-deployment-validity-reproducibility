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
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT.parent
PART2 = EXPERIMENT_ROOT / "part2_url_normalization_leakage_audit"
PART3 = EXPERIMENT_ROOT / "part3_s0_s4_data_splits"
PART4 = EXPERIMENT_ROOT / "part4_baseline_models"
PART5 = EXPERIMENT_ROOT / "part5_shap_stability"
for path in (PART3 / "scripts", PART4 / "scripts", PART5 / "scripts"):
    sys.path.insert(0, str(path))

from model_utils import build_model  # noqa: E402
from shap_utils import deterministic_stratified_sample  # noqa: E402
from split_utils import assert_group_disjoint, grouped_stratified_assignment  # noqa: E402
from selection_utils import exact_top_k, rank_scores  # noqa: E402


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
REPETITIONS = [f"r{index:02d}" for index in range(10)]
INNER_SPLIT_SEED = 20261101
INNER_MODEL_SEED = 20261201
MI_SEEDS = [20261501 + index for index in range(10)]
PERMUTATION_SEEDS = [20261601 + index for index in range(10)]
TOP_K = 15
PERMUTATION_REPEATS = 5
PERMUTATION_COHORT_SIZE = 2000
FEATURE_DICTIONARY = PART2 / "results" / "feature_dictionary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select MI and permutation Top-15 features")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--repetitions", nargs="+", choices=REPETITIONS, default=REPETITIONS)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
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
            "sample_id", "source_row", "raw_url_sha256", "registrable_domain_sha256",
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
        raise ValueError(f"Assignment-feature mismatch for {dataset}")
    return merged


def selected_param_map() -> dict[tuple[str, str, str], dict[str, Any]]:
    frame = pd.read_csv(PART4 / "results" / "internal_metrics.csv")
    frame = frame.loc[frame["scenario_key"] == "s3"]
    result = {
        (row.dataset, row.repetition, row.model): json.loads(row.selected_params_json)
        for row in frame.itertuples(index=False)
    }
    if len(result) != 60:
        raise ValueError(f"Expected 60 S3 parameter rows, found {len(result)}")
    return result


def prepare_scope(
    dataset: str,
    repetition: str,
    internal: pd.DataFrame,
    names: list[str],
    force: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    outer_train = internal.loc[internal[f"split_{repetition}"] == "train"].copy()
    labels = outer_train["label"].to_numpy(dtype=np.int8)
    groups = outer_train["registrable_domain_sha256"].astype(str).to_numpy()
    assignment = grouped_stratified_assignment(labels, groups, INNER_SPLIT_SEED)
    assert_group_disjoint(assignment, groups)
    fit = outer_train.loc[assignment == "train"].copy()
    validation = outer_train.loc[assignment == "validation"].copy()
    cohort = deterministic_stratified_sample(
        validation,
        min(PERMUTATION_COHORT_SIZE, len(validation)),
        f"part8:{dataset}:s3:{repetition}:permutation_validation",
    )
    if set(fit["label"].unique()) != {0, 1} or set(cohort["label"].unique()) != {0, 1}:
        raise ValueError(f"Invalid i00 selection scope for {dataset}/{repetition}")

    scope_dir = ROOT / "selection_scope" / dataset / "s3" / repetition
    scope_dir.mkdir(parents=True, exist_ok=True)
    fit_ids = scope_dir / "i00_train_ids.parquet"
    cohort_ids = scope_dir / "i00_validation_cohort_ids.parquet"
    if force or not fit_ids.exists():
        fit[["sample_id", "source_row", "label"]].to_parquet(
            fit_ids, index=False, compression="zstd"
        )
    if force or not cohort_ids.exists():
        cohort[["sample_id", "source_row", "label"]].to_parquet(
            cohort_ids, index=False, compression="zstd"
        )

    mi_started = time.perf_counter()
    mi_scores = mutual_info_classif(
        fit[names].to_numpy(dtype=np.float32),
        fit["label"].to_numpy(dtype=np.int8),
        discrete_features=False,
        random_state=MI_SEEDS[int(repetition[1:])],
    )
    mi_seconds = time.perf_counter() - mi_started
    mi_ranks = rank_scores(names, mi_scores)
    mi_frame = pd.DataFrame({
        "feature": names,
        "mi_score": mi_scores,
        "mi_rank": [mi_ranks[name] for name in names],
        "mi_seconds": mi_seconds,
    })
    mi_frame.to_csv(scope_dir / "mi_scores.csv", index=False)
    return fit, validation, cohort, mi_scores


def run_model_selection(
    dataset: str,
    repetition: str,
    model_name: str,
    fit: pd.DataFrame,
    cohort: pd.DataFrame,
    names: list[str],
    mi_scores: np.ndarray,
    params: dict[str, Any],
    n_jobs: int,
    force: bool,
) -> None:
    run_dir = ROOT / "selection_runs" / dataset / "s3" / repetition / model_name
    complete = run_dir / "complete.json"
    set_path = (
        ROOT / "feature_sets" / dataset / "s3" / repetition
        / model_name / "traditional_feature_sets.json"
    )
    if complete.exists() and set_path.exists() and not force:
        print(f"SKIP selection {dataset} {repetition} {model_name}", flush=True)
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(model_name, params, INNER_MODEL_SEED, 1)
    fit_started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(
            fit[names].to_numpy(dtype=np.float32),
            fit["label"].to_numpy(dtype=np.int8),
        )
    fit_seconds = time.perf_counter() - fit_started
    permutation_started = time.perf_counter()
    permutation = permutation_importance(
        model,
        cohort[names].to_numpy(dtype=np.float32),
        cohort["label"].to_numpy(dtype=np.int8),
        scoring="roc_auc",
        n_repeats=PERMUTATION_REPEATS,
        random_state=PERMUTATION_SEEDS[int(repetition[1:])],
        n_jobs=n_jobs,
    )
    permutation_seconds = time.perf_counter() - permutation_started
    permutation_ranks = rank_scores(names, permutation.importances_mean)
    mi_ranks = rank_scores(names, mi_scores)
    scores = pd.DataFrame({
        "feature": names,
        "mi_score": mi_scores,
        "mi_rank": [mi_ranks[name] for name in names],
        "permutation_mean_auc_decrease": permutation.importances_mean,
        "permutation_std_auc_decrease": permutation.importances_std,
        "permutation_rank": [permutation_ranks[name] for name in names],
    })
    scores.to_csv(run_dir / "feature_scores.csv", index=False)

    feature_sets = {
        "dataset": dataset,
        "scenario": "s3",
        "outer_repetition": repetition,
        "model": model_name,
        "selection_scope": "outer_train_i00_only",
        "top_k": TOP_K,
        "f_mi": exact_top_k(names, mi_scores, TOP_K),
        "f_permutation": exact_top_k(names, permutation.importances_mean, TOP_K),
        "permutation_scoring": "roc_auc",
        "permutation_repeats": PERMUTATION_REPEATS,
        "permutation_cohort_size": len(cohort),
    }
    write_json(set_path, feature_sets)
    warning_rows = [
        {"category": item.category.__name__, "message": str(item.message)}
        for item in caught
    ]
    if warning_rows:
        write_json(run_dir / "warnings.json", warning_rows)
        for warning in warning_rows:
            append_issue({
                "type": "traditional_selection_warning",
                "dataset": dataset,
                "repetition": repetition,
                "model": model_name,
                **warning,
            })
    write_json(run_dir / "metadata.json", {
        "dataset": dataset,
        "scenario": "s3",
        "outer_repetition": repetition,
        "inner_repetition": "i00",
        "inner_split_seed": INNER_SPLIT_SEED,
        "inner_model_seed": INNER_MODEL_SEED,
        "mi_seed": MI_SEEDS[int(repetition[1:])],
        "permutation_seed": PERMUTATION_SEEDS[int(repetition[1:])],
        "fixed_outer_selected_params": params,
        "inner_fit_rows": len(fit),
        "permutation_cohort_rows": len(cohort),
        "model_fit_seconds": fit_seconds,
        "permutation_seconds": permutation_seconds,
        "warning_count": len(warning_rows),
        "completed_utc": utc_now(),
    })
    write_json(complete, {
        "completed_utc": utc_now(),
        "feature_sets": str(set_path),
        "scores": str(run_dir / "feature_scores.csv"),
    })
    print(
        f"DONE selection {dataset} {repetition} {model_name} "
        f"fit={fit_seconds:.1f}s permutation={permutation_seconds:.1f}s",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    names = feature_names()
    params = selected_param_map()
    for dataset in args.datasets:
        internal = load_internal(dataset, names)
        for repetition in args.repetitions:
            fit, _, cohort, mi_scores = prepare_scope(
                dataset, repetition, internal, names, args.force
            )
            for model_name in args.models:
                try:
                    run_model_selection(
                        dataset, repetition, model_name, fit, cohort, names,
                        mi_scores, params[(dataset, repetition, model_name)],
                        args.n_jobs, args.force,
                    )
                except Exception as exc:
                    append_issue({
                        "type": "traditional_selection_failure",
                        "dataset": dataset,
                        "repetition": repetition,
                        "model": model_name,
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    raise


if __name__ == "__main__":
    main()
