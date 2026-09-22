from __future__ import annotations

import argparse
import hashlib
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
PART2 = EXPERIMENT_ROOT / "part2_url_normalization_leakage_audit"
PART3 = EXPERIMENT_ROOT / "part3_s0_s4_data_splits"
PART4 = EXPERIMENT_ROOT / "part4_baseline_models"
for scripts_path in (PART3 / "scripts", PART4 / "scripts", EXPERIMENT_ROOT / "part5_shap_stability" / "scripts"):
    sys.path.insert(0, str(scripts_path))

from model_utils import build_model  # noqa: E402
from shap_utils import (  # noqa: E402
    build_explainer,
    deterministic_stratified_sample,
    explain,
    global_feature_summary,
)
from split_utils import assert_group_disjoint, grouped_stratified_assignment  # noqa: E402
from selection_utils import exact_top_k, selected_features, summarize_selection_runs  # noqa: E402


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
OUTER_REPETITIONS = [f"r{index:02d}" for index in range(10)]
INNER_REPETITIONS = [f"i{index:02d}" for index in range(10)]
INNER_SPLIT_SEEDS = [20261101 + index for index in range(10)]
INNER_MODEL_SEEDS = [20261201 + index for index in range(10)]
PRIMARY_K = 15
PRIMARY_FREQUENCY = 0.8
PRIMARY_DIRECTION = 0.8
BACKGROUND_SIZE = 200
EXPLANATION_SIZE = 200
FEATURE_DICTIONARY = PART2 / "results" / "feature_dictionary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute training-only SHAP feature sets")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--outer-repetitions", nargs="+", choices=OUTER_REPETITIONS,
                        default=OUTER_REPETITIONS)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--n-jobs", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
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
    feature_path = PART2 / "data" / DATASETS[dataset]["features"]
    features = pd.read_parquet(
        feature_path,
        columns=["source_row", "raw_url_sha256", "label", *names],
    )
    assignment_path = PART3 / "data" / "assignments" / DATASETS[dataset]["assignment"]
    assignment = pd.read_parquet(
        assignment_path,
        columns=[
            "sample_id", "source_row", "raw_url_sha256", "registrable_domain_sha256",
            *[f"split_{rep}" for rep in OUTER_REPETITIONS],
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
    result = {}
    for row in frame.itertuples(index=False):
        key = (row.dataset, row.repetition, row.model)
        if key in result:
            raise ValueError(f"Duplicate Part 4 parameter row: {key}")
        result[key] = json.loads(row.selected_params_json)
    if len(result) != 60:
        raise ValueError(f"Expected 60 S3 parameter rows, found {len(result)}")
    return result


def inner_assignments(outer_train: pd.DataFrame) -> dict[str, np.ndarray]:
    labels = outer_train["label"].to_numpy(dtype=np.int8)
    groups = outer_train["registrable_domain_sha256"].astype(str).to_numpy()
    result = {}
    for inner_rep, seed in zip(INNER_REPETITIONS, INNER_SPLIT_SEEDS):
        assigned = grouped_stratified_assignment(labels, groups, seed)
        assert_group_disjoint(assigned, groups)
        for split in ("train", "validation", "test"):
            subset = labels[assigned == split]
            if len(subset) == 0 or set(np.unique(subset)) != {0, 1}:
                raise ValueError(f"Invalid inner split {inner_rep}/{split}")
        result[inner_rep] = assigned
    return result


def save_ids(path: Path, frame: pd.DataFrame) -> None:
    frame[["sample_id", "source_row", "label"]].to_parquet(
        path, index=False, compression="zstd"
    )


def run_one(
    dataset: str,
    outer_rep: str,
    inner_rep: str,
    model_name: str,
    outer_train: pd.DataFrame,
    assignment: np.ndarray,
    names: list[str],
    params: dict[str, Any],
    n_jobs: int,
    force: bool,
) -> None:
    output = ROOT / "selection_runs" / dataset / "s3" / outer_rep / model_name / inner_rep
    complete = output / "complete.json"
    if complete.exists() and not force:
        print(f"SKIP selection {dataset} {outer_rep} {model_name} {inner_rep}", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    fit = outer_train.loc[assignment == "train"]
    validation = outer_train.loc[assignment == "validation"]
    sample_tag = f"part6:{dataset}:s3:{outer_rep}:{inner_rep}"
    background = deterministic_stratified_sample(
        fit, BACKGROUND_SIZE, f"{sample_tag}:background"
    )
    cohort = deterministic_stratified_sample(
        validation, EXPLANATION_SIZE, f"{sample_tag}:cohort"
    )

    X_fit = fit[names].to_numpy(dtype=np.float32)
    y_fit = fit["label"].to_numpy(dtype=np.int8)
    model_seed = INNER_MODEL_SEEDS[int(inner_rep[1:])]
    model = build_model(model_name, params, model_seed, n_jobs)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as fit_warnings:
        warnings.simplefilter("always")
        model.fit(X_fit, y_fit)
    fit_seconds = time.perf_counter() - started
    X_background = background[names].to_numpy(dtype=np.float32)
    bundle = build_explainer(model_name, model, X_background)
    X_cohort = cohort[names].to_numpy(dtype=np.float32)
    values, explain_seconds, explain_warnings = explain(bundle, X_cohort)
    summary = global_feature_summary(X_cohort, values, names)
    summary.to_csv(output / "global_importance.csv", index=False)
    save_ids(output / "background_ids.parquet", background)
    save_ids(output / "cohort_ids.parquet", cohort)

    warning_rows = [
        {"stage": "fit", "category": item.category.__name__, "message": str(item.message)}
        for item in fit_warnings
    ] + [{"stage": "explain", **item} for item in explain_warnings]
    if warning_rows:
        write_json(output / "warnings.json", warning_rows)
        for warning in warning_rows:
            append_issue({
                "type": "selection_warning", "dataset": dataset,
                "outer_repetition": outer_rep, "inner_repetition": inner_rep,
                "model": model_name, **warning,
            })
    metadata = {
        "purpose": "training_side_feature_selection",
        "dataset": dataset,
        "scenario": "s3",
        "outer_repetition": outer_rep,
        "inner_repetition": inner_rep,
        "inner_split_seed": INNER_SPLIT_SEEDS[int(inner_rep[1:])],
        "inner_model_seed": model_seed,
        "fixed_outer_selected_params": params,
        "outer_train_rows": int(len(outer_train)),
        "inner_fit_rows": int(len(fit)),
        "inner_validation_rows": int(len(validation)),
        "inner_unused_test_rows": int((assignment == "test").sum()),
        "background_size": int(len(background)),
        "cohort_size": int(len(cohort)),
        "fit_seconds": fit_seconds,
        "explainer_setup_seconds": bundle.setup_seconds,
        "explain_seconds": explain_seconds,
        "model_output_scale": bundle.model_output_scale,
        "perturbation": bundle.perturbation,
        "warning_count": len(warning_rows),
        "completed_utc": utc_now(),
    }
    write_json(output / "metadata.json", metadata)
    write_json(complete, {"completed_utc": utc_now(), "metadata": str(output / "metadata.json")})
    print(
        f"DONE selection {dataset} {outer_rep} {model_name} {inner_rep} "
        f"fit={fit_seconds:.1f}s shap={explain_seconds:.1f}s",
        flush=True,
    )


def aggregate_feature_sets() -> None:
    all_rows = []
    timing_rows = []
    for path in sorted(ROOT.glob("selection_runs/*/s3/*/*/*/global_importance.csv")):
        parts = path.relative_to(ROOT / "selection_runs").parts
        dataset, scenario, outer_rep, model_name, inner_rep = parts[:5]
        if not (path.parent / "complete.json").exists():
            continue
        frame = pd.read_csv(path)
        frame.insert(0, "run_id", inner_rep)
        frame.insert(0, "model", model_name)
        frame.insert(0, "outer_repetition", outer_rep)
        frame.insert(0, "scenario", scenario)
        frame.insert(0, "dataset", dataset)
        all_rows.append(frame)
        metadata = json.loads((path.parent / "metadata.json").read_text(encoding="utf-8"))
        timing_rows.append(metadata)
    if not all_rows:
        raise ValueError("No completed selection runs")
    all_summaries = pd.concat(all_rows, ignore_index=True)
    results = ROOT / "results"
    results.mkdir(parents=True, exist_ok=True)
    all_summaries.to_csv(results / "training_side_global_importance.csv", index=False)
    pd.DataFrame(timing_rows).to_csv(results / "selection_timing.csv", index=False)

    stability_rows = []
    feature_set_rows = []
    sensitivity_rows = []
    for keys, group in all_summaries.groupby(
        ["dataset", "scenario", "outer_repetition", "model"], sort=True
    ):
        if group["run_id"].nunique() != 10:
            continue
        primary = summarize_selection_runs(
            group, PRIMARY_K, PRIMARY_FREQUENCY, PRIMARY_DIRECTION, True
        )
        for column, value in zip(
            ["dataset", "scenario", "outer_repetition", "model"], keys
        ):
            primary.insert(len(primary.columns), column, value)
        stability_rows.append(primary)
        single_run = group.loc[group["run_id"] == "i00"]
        f_single = exact_top_k(single_run, PRIMARY_K)
        f_stable = selected_features(primary)
        if not f_stable:
            append_issue({
                "type": "empty_stable_set", "dataset": keys[0],
                "outer_repetition": keys[2], "model": keys[3],
            })
            continue
        payload = {
            "dataset": keys[0], "scenario": keys[1],
            "outer_repetition": keys[2], "model": keys[3],
            "selection_scope": "outer_train_only",
            "primary_k": PRIMARY_K,
            "frequency_threshold": PRIMARY_FREQUENCY,
            "direction_threshold": PRIMARY_DIRECTION,
            "f_single_source_run": "i00",
            "f_all": feature_names(),
            "f_single": f_single,
            "f_stable": f_stable,
        }
        set_path = ROOT / "feature_sets" / keys[0] / "s3" / keys[2] / keys[3] / "feature_sets.json"
        write_json(set_path, payload)
        feature_set_rows.append({
            **{name: payload[name] for name in (
                "dataset", "scenario", "outer_repetition", "model",
                "primary_k", "frequency_threshold", "direction_threshold",
            )},
            "n_f_all": len(payload["f_all"]),
            "n_f_single": len(f_single),
            "n_f_stable": len(f_stable),
            "f_single_json": json.dumps(f_single),
            "f_stable_json": json.dumps(f_stable),
        })
        for k in (10, 15, 20):
            for frequency in (0.6, 0.8, 1.0):
                for require_direction in (False, True):
                    sensitivity = summarize_selection_runs(
                        group, k, frequency, PRIMARY_DIRECTION, require_direction
                    )
                    selected = selected_features(sensitivity)
                    sensitivity_rows.append({
                        "dataset": keys[0], "scenario": keys[1],
                        "outer_repetition": keys[2], "model": keys[3],
                        "k": k, "frequency_threshold": frequency,
                        "direction_threshold": PRIMARY_DIRECTION,
                        "rule": "frequency_plus_direction" if require_direction else "frequency_only",
                        "n_features": len(selected),
                        "features_json": json.dumps(selected),
                    })
    stability = pd.concat(stability_rows, ignore_index=True)
    stability.to_csv(results / "training_side_feature_stability.csv", index=False)
    pd.DataFrame(feature_set_rows).sort_values(
        ["dataset", "outer_repetition", "model"]
    ).to_csv(results / "feature_set_summary.csv", index=False)
    pd.DataFrame(sensitivity_rows).sort_values(
        ["dataset", "outer_repetition", "model", "k", "frequency_threshold", "rule"]
    ).to_csv(results / "feature_selection_sensitivity.csv", index=False)
    print(f"selection_runs={all_summaries[['dataset','outer_repetition','model','run_id']].drop_duplicates().shape[0]}")
    print(f"feature_sets={len(feature_set_rows)}")


def main() -> None:
    args = parse_args()
    names = feature_names()
    if not args.aggregate_only:
        params = selected_param_map()
        for dataset in args.datasets:
            internal = load_internal(dataset, names)
            for outer_rep in args.outer_repetitions:
                outer_train = internal.loc[
                    internal[f"split_{outer_rep}"].astype(str) == "train"
                ].reset_index(drop=True)
                assignments = inner_assignments(outer_train)
                for model_name in args.models:
                    selected_params = params[(dataset, outer_rep, model_name)]
                    for inner_rep in INNER_REPETITIONS:
                        run_one(
                            dataset, outer_rep, inner_rep, model_name, outer_train,
                            assignments[inner_rep], names, selected_params,
                            args.n_jobs, args.force,
                        )
    aggregate_feature_sets()


if __name__ == "__main__":
    main()

