from __future__ import annotations

import argparse
import gc
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

from model_utils import build_model  # noqa: E402
from shap_utils import (  # noqa: E402
    build_explainer,
    deterministic_stratified_sample,
    explain,
    global_feature_summary,
    shap_value_frame,
)


DATASETS = {
    "phiusiil": {
        "feature_file": "phiusiil_master_features.parquet",
        "assignment_prefix": "phiusiil_master",
        "target": "iscx_url2016_binary",
        "s4_file": "s4_phiusiil_to_iscx_url2016_binary_master_assignments.parquet",
    },
    "iscx_url2016_binary": {
        "feature_file": "iscx_url2016_binary_master_features.parquet",
        "assignment_prefix": "iscx_url2016_binary_master",
        "target": "phiusiil",
        "s4_file": "s4_iscx_url2016_binary_to_phiusiil_master_assignments.parquet",
    },
}
SCENARIOS = {
    "s0": "s0_row",
    "s1": "s1_url",
    "s2": "s2_host",
    "s3": "s3_domain",
}
MODELS = ["lr", "rf", "xgb"]
REPETITIONS = [f"r{index:02d}" for index in range(10)]
MODEL_SEEDS = [20261001 + index for index in range(10)]
STABILITY_SIZE = 200
BACKGROUND_SIZE = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exact SHAP stability experiments")
    parser.add_argument("--estimands", nargs="+", choices=["seed", "partition"], default=["seed", "partition"])
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--runs", nargs="+", choices=REPETITIONS, default=REPETITIONS)
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
    dictionary = pd.read_csv(PART2 / "results" / "feature_dictionary.csv")
    names = dictionary.sort_values("feature_order")["feature"].tolist()
    if len(names) != 35 or len(names) != len(set(names)):
        raise ValueError("Expected 35 unique features")
    return names


def load_features(dataset: str, names: list[str]) -> pd.DataFrame:
    columns = ["source_row", "raw_url_sha256", "label", *names]
    frame = pd.read_parquet(PART2 / "data" / DATASETS[dataset]["feature_file"], columns=columns)
    if not np.isfinite(frame[names].to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite features in {dataset}")
    return frame


def load_internal(dataset: str, scenario: str, features: pd.DataFrame) -> pd.DataFrame:
    filename = f"{DATASETS[dataset]['assignment_prefix']}_{SCENARIOS[scenario]}_assignments.parquet"
    columns = [
        "sample_id", "source_row", "raw_url_sha256", "label",
        *[f"split_{repetition}" for repetition in REPETITIONS],
    ]
    assignment = pd.read_parquet(PART3 / "data" / "assignments" / filename, columns=columns)
    merged = assignment.merge(
        features.drop(columns="label"),
        on=["source_row", "raw_url_sha256"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(assignment) or set(merged["label"].unique()) != {0, 1}:
        raise ValueError(f"Internal merge failed for {dataset}/{scenario}")
    return merged


def load_external_common(
    source: str,
    feature_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    target_dataset = DATASETS[source]["target"]
    role_columns = [f"role_{repetition}" for repetition in REPETITIONS]
    columns = [
        "sample_id", "origin", "source_row", "raw_url_sha256", "label", *role_columns,
    ]
    assignment = pd.read_parquet(
        PART3 / "data" / "assignments" / DATASETS[source]["s4_file"],
        columns=columns,
    )
    target = assignment.loc[assignment["origin"] == "target"].copy()
    common_mask = np.logical_and.reduce([
        target[column].astype(str).to_numpy() == "target_external_primary" for column in role_columns
    ])
    target = target.loc[common_mask, ["sample_id", "source_row", "raw_url_sha256", "label"]]
    target_features = feature_tables[target_dataset]
    merged = target.merge(
        target_features.drop(columns="label"),
        on=["source_row", "raw_url_sha256"],
        how="inner",
        validate="one_to_one",
    )
    expected = 40656 if source == "phiusiil" else 233286
    if len(merged) != expected:
        raise ValueError(f"Unexpected S4 common cohort for {source}: {len(merged)}")
    return merged


def selected_params() -> dict[tuple[str, str, str], dict[str, Any]]:
    metrics = pd.read_csv(PART4 / "results" / "internal_metrics.csv")
    primary = metrics.loc[metrics["repetition"] == "r00"]
    output = {}
    for row in primary.itertuples():
        output[(row.dataset, row.scenario_key, row.model)] = json.loads(row.selected_params_json)
    if len(output) != 2 * 4 * 3:
        raise ValueError(f"Expected 24 fixed parameter sets, found {len(output)}")
    return output


def fit_or_load_model(
    dataset: str,
    scenario: str,
    model_name: str,
    model_seed: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict[str, Any],
    n_jobs: int,
    can_load_primary: bool,
) -> tuple[Any, float, str]:
    if can_load_primary:
        path = PART4 / "models" / dataset / scenario / f"{model_name}.joblib"
        start = time.perf_counter()
        return joblib.load(path), time.perf_counter() - start, "loaded_part4_r00"
    model = build_model(model_name, params, model_seed, n_jobs)
    start = time.perf_counter()
    model.fit(X_train, y_train)
    return model, time.perf_counter() - start, "trained_part5"


def save_explanation(
    output_dir: Path,
    cohort: pd.DataFrame,
    names: list[str],
    X: np.ndarray,
    values: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    global_feature_summary(X, values, names).to_csv(output_dir / "global_summary.csv", index=False)
    shap_value_frame(cohort, X, values, names).to_parquet(
        output_dir / "shap_values.parquet", index=False, compression="zstd"
    )
    write_json(output_dir / "metadata.json", metadata)


def run_one(
    estimand: str,
    dataset: str,
    scenario: str,
    model_name: str,
    run_id: str,
    internal: pd.DataFrame,
    external_common: pd.DataFrame | None,
    names: list[str],
    params: dict[str, Any],
    n_jobs: int,
    force: bool,
) -> None:
    output_dir = ROOT / "runs" / estimand / dataset / scenario / model_name / run_id
    complete_path = output_dir / "complete.json"
    if complete_path.exists() and not force:
        print(f"SKIP {estimand} {dataset} {scenario} {model_name} {run_id}", flush=True)
        return

    if estimand == "seed":
        partition_rep = "r00"
        model_seed = MODEL_SEEDS[int(run_id[1:])]
        sample_tag = f"seed:{dataset}:{scenario}"
        can_load_primary = run_id == "r00"
    elif estimand == "partition":
        partition_rep = run_id
        model_seed = MODEL_SEEDS[0]
        sample_tag = f"partition:{dataset}:{scenario}:{partition_rep}"
        can_load_primary = partition_rep == "r00"
    else:
        raise ValueError(estimand)

    split = internal[f"split_{partition_rep}"].astype(str)
    train = internal.loc[split == "train"]
    test = internal.loc[split == "test"]
    background = deterministic_stratified_sample(train, BACKGROUND_SIZE, f"{sample_tag}:background")
    cohort = deterministic_stratified_sample(test, STABILITY_SIZE, f"{sample_tag}:cohort")

    X_train = train[names].to_numpy(dtype=np.float32)
    y_train = train["label"].to_numpy(dtype=np.int8)
    model, fit_seconds, model_origin = fit_or_load_model(
        dataset,
        scenario,
        model_name,
        model_seed,
        X_train,
        y_train,
        params,
        n_jobs,
        can_load_primary,
    )
    X_background = background[names].to_numpy(dtype=np.float32)
    bundle = build_explainer(model_name, model, X_background)
    X_cohort = cohort[names].to_numpy(dtype=np.float32)
    values, explain_seconds, warning_rows = explain(bundle, X_cohort)
    metadata = {
        "purpose": "evaluation_only",
        "estimand": estimand,
        "dataset": dataset,
        "scenario": scenario,
        "model": model_name,
        "run_id": run_id,
        "partition_repetition": partition_rep,
        "model_seed": model_seed,
        "fixed_params": params,
        "model_origin": model_origin,
        "fit_seconds": fit_seconds,
        "cohort_size": len(cohort),
        "background_size": len(background),
        "explainer_setup_seconds": bundle.setup_seconds,
        "explain_seconds": explain_seconds,
        "seconds_per_sample": explain_seconds / len(cohort),
        "model_output_scale": bundle.model_output_scale,
        "perturbation": bundle.perturbation,
        "expected_value": bundle.expected_value,
        "shap_version": __import__("shap").__version__,
        "completed_utc": utc_now(),
    }
    save_explanation(output_dir / "internal", cohort, names, X_cohort, values, metadata)

    if warning_rows:
        write_json(output_dir / "warnings.json", warning_rows)
        for warning in warning_rows:
            append_issue({
                "type": "shap_warning", "estimand": estimand, "dataset": dataset,
                "scenario": scenario, "model": model_name, "run_id": run_id, **warning,
            })

    external_seconds = None
    if estimand == "partition" and scenario == "s3":
        if external_common is None:
            raise ValueError("Missing external common cohort")
        external_cohort = deterministic_stratified_sample(
            external_common,
            STABILITY_SIZE,
            f"source:{dataset}:external_common",
        )
        X_external = external_cohort[names].to_numpy(dtype=np.float32)
        external_values, external_seconds, external_warnings = explain(bundle, X_external)
        external_metadata = {
            **metadata,
            "cohort": "external_common",
            "target_dataset": DATASETS[dataset]["target"],
            "common_pool_size": len(external_common),
            "explain_seconds": external_seconds,
            "seconds_per_sample": external_seconds / len(external_cohort),
        }
        save_explanation(
            output_dir / "external_common",
            external_cohort,
            names,
            X_external,
            external_values,
            external_metadata,
        )
        if external_warnings:
            write_json(output_dir / "external_warnings.json", external_warnings)
            for warning in external_warnings:
                append_issue({
                    "type": "external_shap_warning", "dataset": dataset,
                    "scenario": scenario, "model": model_name, "run_id": run_id, **warning,
                })

    write_json(complete_path, {
        "completed_utc": utc_now(),
        "internal_explain_seconds": explain_seconds,
        "external_explain_seconds": external_seconds,
    })
    print(
        f"DONE {estimand} {dataset} {scenario} {model_name} {run_id} "
        f"fit={fit_seconds:.1f}s shap={explain_seconds:.1f}s"
        + (f" external={external_seconds:.1f}s" if external_seconds is not None else ""),
        flush=True,
    )
    del model, bundle, values
    gc.collect()


def main() -> None:
    args = parse_args()
    for directory in ("logs", "results", "runs"):
        (ROOT / directory).mkdir(parents=True, exist_ok=True)
    names = feature_names()
    feature_tables = {dataset: load_features(dataset, names) for dataset in DATASETS}
    params_by_run = selected_params()
    external_cache: dict[str, pd.DataFrame] = {}

    write_json(ROOT / "results" / "run_metadata.json", {
        "started_utc": utc_now(),
        "stability_size": STABILITY_SIZE,
        "background_size": BACKGROUND_SIZE,
        "model_seeds": MODEL_SEEDS,
        "feature_names": names,
        "arguments": vars(args),
    })

    for estimand in args.estimands:
        valid_scenarios = [scenario for scenario in args.scenarios if estimand == "partition" or scenario in {"s0", "s3"}]
        for dataset in args.datasets:
            for scenario in valid_scenarios:
                internal = load_internal(dataset, scenario, feature_tables[dataset])
                external_common = None
                if estimand == "partition" and scenario == "s3":
                    if dataset not in external_cache:
                        external_cache[dataset] = load_external_common(dataset, feature_tables)
                    external_common = external_cache[dataset]
                for model_name in args.models:
                    params = params_by_run[(dataset, scenario, model_name)]
                    for run_id in args.runs:
                        try:
                            run_one(
                                estimand, dataset, scenario, model_name, run_id,
                                internal, external_common, names, params, args.n_jobs, args.force,
                            )
                        except Exception as exc:
                            append_issue({
                                "type": "run_failure", "estimand": estimand,
                                "dataset": dataset, "scenario": scenario,
                                "model": model_name, "run_id": run_id,
                                "error": f"{type(exc).__name__}: {exc}",
                            })
                            raise

    print("Part 5 SHAP stability runs completed.", flush=True)


if __name__ == "__main__":
    main()
