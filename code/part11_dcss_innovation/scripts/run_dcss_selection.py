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
PART2 = EXPERIMENT_ROOT / "part2_url_normalization_leakage_audit"
PART3 = EXPERIMENT_ROOT / "part3_s0_s4_data_splits"
PART4 = EXPERIMENT_ROOT / "part4_baseline_models"
PART5 = EXPERIMENT_ROOT / "part5_shap_stability"
for scripts_path in (PART4 / "scripts", PART5 / "scripts", ROOT / "scripts"):
    sys.path.insert(0, str(scripts_path))

from dcss_utils import (  # noqa: E402
    FEATURE_COUNTS,
    aggregate_dcss,
    deterministic_stratified_sample,
    domain_fold_assignment,
    fold_shap_summary,
    selected_features,
)
from model_utils import build_model  # noqa: E402
from shap_utils import build_explainer, explain  # noqa: E402


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
MODELS = ("lr", "rf", "xgb")
REPETITIONS = tuple(f"r{index:02d}" for index in range(10))
BACKGROUND_SIZE = 200
COHORT_SIZE = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run source-training-only DCSS selection")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--repetitions", nargs="+", choices=REPETITIONS, default=list(REPETITIONS))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--n-jobs", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def append_runtime_issue(payload: dict[str, Any]) -> None:
    path = ROOT / "logs" / "dcss_runtime_issues.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time_utc": utc_now(), **payload}, sort_keys=True) + "\n")


def feature_names() -> list[str]:
    dictionary = pd.read_csv(PART2 / "results" / "feature_dictionary.csv")
    names = dictionary.sort_values("feature_order")["feature"].tolist()
    if len(names) != 35 or len(set(names)) != 35:
        raise ValueError("Expected 35 frozen features")
    return names


def feature_table(dataset: str, names: list[str]) -> pd.DataFrame:
    frame = pd.read_parquet(
        PART2 / "data" / DATASETS[dataset]["features"],
        columns=["source_row", "raw_url_sha256", *names],
    )
    if frame[["source_row", "raw_url_sha256"]].duplicated().any():
        raise ValueError(f"Duplicate feature key: {dataset}")
    return frame


def load_outer_training_only(
    dataset: str, repetition: str, features: pd.DataFrame
) -> pd.DataFrame:
    role_column = f"split_{repetition}"
    assignment_path = PART3 / "data" / "assignments" / DATASETS[dataset]["assignment"]
    assignment = pd.read_parquet(
        assignment_path,
        columns=[
            "sample_id",
            "source_row",
            "raw_url_sha256",
            "registrable_domain_sha256",
            "label",
            role_column,
        ],
        filters=[(role_column, "==", "train")],
    )
    if len(assignment) == 0 or set(assignment[role_column].astype(str)) != {"train"}:
        raise ValueError(f"Filtered outer training read failed: {dataset}/{repetition}")
    merged = assignment.merge(
        features,
        on=["source_row", "raw_url_sha256"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(assignment):
        raise ValueError(f"Outer training feature mismatch: {dataset}/{repetition}")
    if set(merged["label"].unique()) != {0, 1}:
        raise ValueError(f"Outer training label set failure: {dataset}/{repetition}")
    return merged.sort_values("sample_id").reset_index(drop=True)


def selected_parameter_map() -> dict[tuple[str, str, str], dict[str, Any]]:
    metrics = pd.read_csv(PART4 / "results" / "internal_metrics.csv")
    metrics = metrics.loc[(metrics["scenario_key"] == "s3") & (metrics["corpus_version"] == "master")]
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in metrics.itertuples(index=False):
        key = (row.dataset, row.repetition, row.model)
        if key in result:
            raise ValueError(f"Duplicate selected parameter row: {key}")
        result[key] = {
            "selected_candidate": row.selected_candidate,
            "selected_params": json.loads(row.selected_params_json),
        }
    if len(result) != 60:
        raise ValueError(f"Expected 60 S3 selected parameter rows, found {len(result)}")
    return result


def fold_assignment_path(dataset: str, repetition: str) -> Path:
    return (
        ROOT
        / "data"
        / "processed"
        / "dcss_fold_assignments"
        / dataset
        / f"{repetition}.parquet"
    )


def prepare_folds(
    dataset: str, repetition: str, outer_train: pd.DataFrame, force: bool
) -> pd.DataFrame:
    path = fold_assignment_path(dataset, repetition)
    fold_seed = 20261301 + int(repetition[1:])
    if path.is_file() and not force:
        stored = pd.read_parquet(path)
        expected_ids = outer_train[["sample_id"]]
        if not expected_ids.equals(stored[["sample_id"]]):
            raise ValueError(f"Stored fold IDs do not match outer training: {dataset}/{repetition}")
        recomputed = domain_fold_assignment(outer_train, fold_seed)
        if not np.array_equal(stored["held_fold"].to_numpy(np.int8), recomputed):
            raise ValueError(f"Stored fold assignment is not deterministic: {dataset}/{repetition}")
        return stored

    folds = domain_fold_assignment(outer_train, fold_seed)
    output = outer_train[
        ["sample_id", "source_row", "registrable_domain_sha256", "label"]
    ].copy()
    output["held_fold"] = folds
    output["fold_seed"] = fold_seed
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(path, index=False, compression="zstd")
    return output


def save_sample_ids(path: Path, frame: pd.DataFrame) -> None:
    frame[
        ["sample_id", "source_row", "registrable_domain_sha256", "label"]
    ].to_parquet(path, index=False, compression="zstd")


def run_fold(
    dataset: str,
    repetition: str,
    model_name: str,
    held_fold: int,
    outer_train: pd.DataFrame,
    fold_assignments: pd.DataFrame,
    names: list[str],
    selected: dict[str, Any],
    n_jobs: int,
    force: bool,
) -> None:
    output = (
        ROOT
        / "runs"
        / "dcss_selection"
        / dataset
        / "s3"
        / repetition
        / model_name
        / f"h{held_fold:02d}"
    )
    complete = output / "complete.json"
    if complete.is_file() and not force:
        print(f"SKIP {dataset}/{repetition}/{model_name}/h{held_fold:02d}", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)

    working = outer_train.merge(
        fold_assignments[["sample_id", "held_fold"]],
        on="sample_id",
        validate="one_to_one",
    )
    fit = working.loc[working["held_fold"] != held_fold].copy()
    held = working.loc[working["held_fold"] == held_fold].copy()
    if set(fit["registrable_domain_sha256"]) & set(held["registrable_domain_sha256"]):
        raise ValueError(f"Domain leakage: {dataset}/{repetition}/h{held_fold:02d}")

    sample_tag = f"part11:dcss:{dataset}:{repetition}:h{held_fold:02d}"
    background, background_reduced = deterministic_stratified_sample(
        fit, BACKGROUND_SIZE, f"{sample_tag}:background"
    )
    cohort, cohort_reduced = deterministic_stratified_sample(
        held, COHORT_SIZE, f"{sample_tag}:cohort"
    )
    if background_reduced or cohort_reduced:
        append_runtime_issue(
            {
                "type": "reduced_shap_sample",
                "dataset": dataset,
                "repetition": repetition,
                "model": model_name,
                "held_fold": held_fold,
                "background_reduced": background_reduced,
                "cohort_reduced": cohort_reduced,
                "background_size": len(background),
                "cohort_size": len(cohort),
            }
        )

    X_fit = fit[names].to_numpy(dtype=np.float32)
    y_fit = fit["label"].to_numpy(dtype=np.int8)
    model_seed = 20261401 + 10 * int(repetition[1:]) + held_fold
    model = build_model(
        model_name, selected["selected_params"], model_seed, n_jobs
    )
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as fit_warnings:
        warnings.simplefilter("always")
        model.fit(X_fit, y_fit)
    fit_seconds = time.perf_counter() - started
    classes = [int(value) for value in model.classes_]
    if classes != [0, 1]:
        raise ValueError(f"Unexpected class order {classes}: {dataset}/{repetition}/{model_name}")

    X_background = background[names].to_numpy(dtype=np.float32)
    explainer = build_explainer(model_name, model, X_background)
    X_cohort = cohort[names].to_numpy(dtype=np.float32)
    shap_values, shap_seconds, shap_warnings = explain(explainer, X_cohort)
    summary = fold_shap_summary(names, X_cohort, shap_values)
    summary.insert(0, "held_fold", held_fold)
    summary.to_csv(output / "fold_shap_summary.csv", index=False)
    save_sample_ids(output / "background_ids.parquet", background)
    save_sample_ids(output / "cohort_ids.parquet", cohort)

    warning_rows = [
        {"stage": "fit", "category": item.category.__name__, "message": str(item.message)}
        for item in fit_warnings
    ] + [{"stage": "shap", **item} for item in shap_warnings]
    if warning_rows:
        write_json(output / "warnings.json", warning_rows)
        for warning in warning_rows:
            append_runtime_issue(
                {
                    "type": "dcss_warning",
                    "dataset": dataset,
                    "repetition": repetition,
                    "model": model_name,
                    "held_fold": held_fold,
                    **warning,
                }
            )

    metadata = {
        "stage": "11C",
        "selection_scope": "source_outer_train_only",
        "dataset": dataset,
        "scenario": "s3",
        "repetition": repetition,
        "model": model_name,
        "held_fold": held_fold,
        "fold_seed": 20261301 + int(repetition[1:]),
        "model_seed": model_seed,
        "selected_candidate": selected["selected_candidate"],
        "selected_params": selected["selected_params"],
        "model_classes": classes,
        "outer_train_rows": len(outer_train),
        "fit_rows": len(fit),
        "held_rows": len(held),
        "fit_domains": int(fit["registrable_domain_sha256"].nunique()),
        "held_domains": int(held["registrable_domain_sha256"].nunique()),
        "background_size": len(background),
        "cohort_size": len(cohort),
        "background_reduced": background_reduced,
        "cohort_reduced": cohort_reduced,
        "fit_seconds": fit_seconds,
        "explainer_setup_seconds": explainer.setup_seconds,
        "shap_seconds": shap_seconds,
        "model_output_scale": explainer.model_output_scale,
        "perturbation": explainer.perturbation,
        "warning_count": len(warning_rows),
        "completed_utc": utc_now(),
    }
    write_json(output / "metadata.json", metadata)
    write_json(complete, {"completed_utc": utc_now(), "metadata": str(output / "metadata.json")})
    print(
        f"DONE {dataset}/{repetition}/{model_name}/h{held_fold:02d} "
        f"fit={fit_seconds:.2f}s shap={shap_seconds:.2f}s",
        flush=True,
    )


def aggregate_results() -> None:
    fold_rows = []
    timing_rows = []
    for path in sorted((ROOT / "runs" / "dcss_selection").glob("*/s3/r??/*/h??/fold_shap_summary.csv")):
        if not (path.parent / "complete.json").is_file():
            continue
        relative = path.relative_to(ROOT / "runs" / "dcss_selection").parts
        dataset, scenario, repetition, model_name, held_name = relative[:5]
        frame = pd.read_csv(path)
        frame.insert(0, "model", model_name)
        frame.insert(0, "repetition", repetition)
        frame.insert(0, "scenario", scenario)
        frame.insert(0, "dataset", dataset)
        fold_rows.append(frame)
        metadata = json.loads((path.parent / "metadata.json").read_text(encoding="utf-8"))
        timing_rows.append(metadata)
    if not fold_rows:
        raise ValueError("No completed DCSS fold runs")

    all_folds = pd.concat(fold_rows, ignore_index=True)
    score_rows = []
    feature_set_rows = []
    for keys, group in all_folds.groupby(
        ["dataset", "scenario", "repetition", "model"], sort=True
    ):
        if group["held_fold"].nunique() != 5:
            continue
        scores = aggregate_dcss(group)
        for column, value in zip(["dataset", "scenario", "repetition", "model"], keys):
            scores.insert(0, column, value=value)
        score_rows.append(scores)
        payload = {
            "stage": "11C",
            "dataset": keys[0],
            "scenario": keys[1],
            "repetition": keys[2],
            "model": keys[3],
            "selection_scope": "source_outer_train_only",
            "fold_count": 5,
            "formula": "mean_importance * frequency * direction_consistency * (1-rank_dispersion)",
        }
        row = {name: payload[name] for name in ("dataset", "scenario", "repetition", "model")}
        for k in FEATURE_COUNTS:
            features = selected_features(scores, k)
            payload[f"f_dcss_{k}"] = features
            row[f"n_f_dcss_{k}"] = len(features)
            row[f"f_dcss_{k}_json"] = json.dumps(features)
        set_path = (
            ROOT
            / "feature_sets"
            / "dcss"
            / keys[0]
            / "s3"
            / keys[2]
            / keys[3]
            / "feature_sets.json"
        )
        write_json(set_path, payload)
        feature_set_rows.append(row)

    results = ROOT / "results"
    results.mkdir(parents=True, exist_ok=True)
    all_folds.sort_values(
        ["dataset", "repetition", "model", "held_fold", "rank"]
    ).to_csv(results / "dcss_fold_shap_summary.csv", index=False)
    pd.concat(score_rows, ignore_index=True).sort_values(
        ["dataset", "repetition", "model", "dcss_rank_15"]
    ).to_csv(results / "dcss_scores.csv", index=False)
    pd.DataFrame(feature_set_rows).sort_values(
        ["dataset", "repetition", "model"]
    ).to_csv(results / "dcss_feature_sets.csv", index=False)
    pd.DataFrame(timing_rows).sort_values(
        ["dataset", "repetition", "model", "held_fold"]
    ).to_csv(results / "dcss_timing.csv", index=False)
    print(
        f"AGGREGATED folds={len(timing_rows)} feature_sets={len(feature_set_rows)}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    names = feature_names()
    if not args.aggregate_only:
        parameter_map = selected_parameter_map()
        for dataset in args.datasets:
            features = feature_table(dataset, names)
            for repetition in args.repetitions:
                outer_train = load_outer_training_only(dataset, repetition, features)
                folds = prepare_folds(dataset, repetition, outer_train, args.force)
                for model_name in args.models:
                    selected = parameter_map[(dataset, repetition, model_name)]
                    for held_fold in range(5):
                        run_fold(
                            dataset,
                            repetition,
                            model_name,
                            held_fold,
                            outer_train,
                            folds,
                            names,
                            selected,
                            args.n_jobs,
                            args.force,
                        )
    aggregate_results()


if __name__ == "__main__":
    main()
