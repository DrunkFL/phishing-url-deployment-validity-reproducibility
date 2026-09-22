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
sys.path.insert(0, str(ROOT / "scripts"))

from main_comparison_utils import extended_metrics  # noqa: E402
from model_utils import select_model  # noqa: E402


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
MODELS = ("lr", "rf", "xgb")
FEATURE_SETS = ("f_dcss_10", "f_dcss_15", "f_dcss_20")
REPETITIONS = tuple(f"r{index:02d}" for index in range(10))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage 11D F-DCSS models")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--repetitions", nargs="+", choices=REPETITIONS, default=list(REPETITIONS))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--feature-sets", nargs="+", choices=FEATURE_SETS, default=list(FEATURE_SETS))
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


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def append_runtime_issue(payload: dict[str, Any]) -> None:
    path = ROOT / "logs" / "main_comparison_runtime_issues.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time_utc": utc_now(), **payload}, sort_keys=True) + "\n")


def feature_names() -> list[str]:
    dictionary = pd.read_csv(PART2 / "results" / "feature_dictionary.csv")
    names = dictionary.sort_values("feature_order")["feature"].tolist()
    if len(names) != 35 or len(set(names)) != 35:
        raise ValueError("Expected 35 frozen features")
    return names


def load_internal(dataset: str, names: list[str]) -> pd.DataFrame:
    features = pd.read_parquet(
        PART2 / "data" / DATASETS[dataset]["features"],
        columns=["source_row", "raw_url_sha256", "label", *names],
    )
    assignment = pd.read_parquet(
        PART3 / "data" / "assignments" / DATASETS[dataset]["assignment"],
        columns=[
            "sample_id",
            "source_row",
            "raw_url_sha256",
            "registrable_domain_sha256",
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
        raise ValueError(f"Internal assignment-feature mismatch: {dataset}")
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
            "sample_id",
            "origin",
            "source_row",
            "raw_url_sha256",
            "registrable_domain_sha256",
            "label",
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
        raise ValueError(f"External assignment-feature mismatch: {source}")
    if not np.array_equal(target["label"], target["feature_label"]):
        raise ValueError(f"External label mismatch: {source}")
    return target.drop(columns="feature_label")


def selected_names(
    dataset: str, repetition: str, model: str, feature_set: str, all_names: list[str]
) -> list[str]:
    path = (
        ROOT
        / "feature_sets"
        / "dcss"
        / dataset
        / "s3"
        / repetition
        / model
        / "feature_sets.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    names = payload[feature_set]
    expected = int(feature_set.rsplit("_", 1)[1])
    if len(names) != expected or len(set(names)) != expected or not set(names) <= set(all_names):
        raise ValueError(f"Invalid {feature_set}: {dataset}/{repetition}/{model}")
    return names


def prediction_frame(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    primary: np.ndarray | None = None,
) -> pd.DataFrame:
    output = frame[
        ["sample_id", "source_row", "registrable_domain_sha256", "label"]
    ].reset_index(drop=True).copy()
    output["probability_phishing"] = probabilities.astype(np.float64)
    output["prediction"] = (probabilities >= threshold).astype(np.int8)
    if primary is not None:
        output["included_in_primary"] = primary.astype(bool)
    return output


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
    run = (
        ROOT
        / "runs"
        / "main_comparison"
        / dataset
        / "s3"
        / repetition
        / model_name
        / feature_set
    )
    complete = run / "complete.json"
    if complete.is_file() and not force:
        print(f"SKIP {dataset}/{repetition}/{model_name}/{feature_set}", flush=True)
        return
    run.mkdir(parents=True, exist_ok=True)
    names = selected_names(dataset, repetition, model_name, feature_set, all_names)

    split = internal[f"split_{repetition}"].astype(str).to_numpy()
    masks = {role: split == role for role in ("train", "validation", "test")}
    train = internal.loc[masks["train"]]
    validation = internal.loc[masks["validation"]]
    test = internal.loc[masks["test"]]
    model_seed = 20261001 + int(repetition[1:])

    started = time.perf_counter()
    selection = select_model(
        model_name,
        train[names].to_numpy(dtype=np.float32),
        train["label"].to_numpy(dtype=np.int8),
        validation[names].to_numpy(dtype=np.float32),
        validation["label"].to_numpy(dtype=np.int8),
        model_seed,
        n_jobs,
    )
    classes = [int(value) for value in selection.model.classes_]
    if classes != [0, 1]:
        raise ValueError(f"Unexpected class order {classes}: {dataset}/{repetition}/{model_name}")

    internal_started = time.perf_counter()
    internal_probabilities = selection.model.predict_proba(
        test[names].to_numpy(dtype=np.float32)
    )[:, 1]
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
        "model_classes_json": json.dumps(classes),
        **extended_metrics(
            test["label"].to_numpy(dtype=np.int8),
            internal_probabilities,
            selection.threshold,
        ),
    }

    target_started = time.perf_counter()
    external_probabilities = selection.model.predict_proba(
        target[names].to_numpy(dtype=np.float32)
    )[:, 1]
    external_inference = time.perf_counter() - target_started
    primary = (target[f"role_{repetition}"] == "target_external_primary").to_numpy()
    external_metrics = []
    for cohort, mask in (
        ("unfiltered", np.ones(len(target), dtype=bool)),
        ("primary_domain_filtered", primary),
    ):
        external_metrics.append(
            {
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
                "model_classes_json": json.dumps(classes),
                **extended_metrics(
                    target.loc[mask, "label"].to_numpy(dtype=np.int8),
                    external_probabilities[mask],
                    selection.threshold,
                ),
            }
        )

    internal_prediction = (
        ROOT
        / "predictions"
        / "main_comparison"
        / "internal"
        / dataset
        / "s3"
        / repetition
        / model_name
        / f"{feature_set}.parquet"
    )
    external_prediction = (
        ROOT
        / "predictions"
        / "main_comparison"
        / "external"
        / dataset
        / repetition
        / model_name
        / f"{feature_set}.parquet"
    )
    model_path = (
        ROOT
        / "models"
        / "main_comparison"
        / dataset
        / "s3"
        / repetition
        / model_name
        / f"{feature_set}.joblib"
    )
    write_parquet(
        internal_prediction,
        prediction_frame(test, internal_probabilities, selection.threshold),
    )
    write_parquet(
        external_prediction,
        prediction_frame(target, external_probabilities, selection.threshold, primary),
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(selection.model, model_path, compress=3)
    pd.DataFrame(selection.tuning_rows).to_csv(run / "tuning_results.csv", index=False)
    write_json(run / "internal_metrics.json", internal_metrics)
    write_json(run / "external_metrics.json", external_metrics)
    if selection.warnings:
        write_json(run / "model_warnings.json", selection.warnings)
        for warning in selection.warnings:
            append_runtime_issue(
                {
                    "type": "model_warning",
                    "dataset": dataset,
                    "repetition": repetition,
                    "model": model_name,
                    "feature_set": feature_set,
                    **warning,
                }
            )
    write_json(
        complete,
        {
            "completed_utc": utc_now(),
            "internal_prediction": str(internal_prediction),
            "external_prediction": str(external_prediction),
            "model": str(model_path),
            "total_seconds": time.perf_counter() - started,
        },
    )
    print(
        f"DONE {dataset}/{repetition}/{model_name}/{feature_set} "
        f"internal_f1={internal_metrics['macro_f1']:.4f} "
        f"external_auc={external_metrics[1]['roc_auc']:.4f}",
        flush=True,
    )


def aggregate_dcss_metrics() -> None:
    internal_rows = []
    external_rows = []
    for path in sorted(
        (ROOT / "runs" / "main_comparison").glob("*/s3/r??/*/f_dcss_*/internal_metrics.json")
    ):
        if not (path.parent / "complete.json").is_file():
            continue
        internal_rows.append(json.loads(path.read_text(encoding="utf-8")))
        external_rows.extend(
            json.loads((path.parent / "external_metrics.json").read_text(encoding="utf-8"))
        )
    if not internal_rows:
        raise ValueError("No completed Stage 11D runs")
    results = ROOT / "results"
    pd.DataFrame(internal_rows).sort_values(
        ["dataset", "repetition", "model", "feature_set"]
    ).to_csv(results / "dcss_main_internal_metrics.csv", index=False)
    pd.DataFrame(external_rows).sort_values(
        ["source_dataset", "repetition", "model", "feature_set", "cohort"]
    ).to_csv(results / "dcss_main_external_metrics.csv", index=False)
    print(
        f"AGGREGATED internal={len(internal_rows)} external={len(external_rows)}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if not args.aggregate_only:
        all_names = feature_names()
        for dataset in args.datasets:
            internal = load_internal(dataset, all_names)
            target = load_target(dataset, all_names)
            for repetition in args.repetitions:
                for model_name in args.models:
                    for feature_set in args.feature_sets:
                        try:
                            run_one(
                                dataset,
                                repetition,
                                model_name,
                                feature_set,
                                internal,
                                target,
                                all_names,
                                args.n_jobs,
                                args.force,
                            )
                        except Exception as error:
                            append_runtime_issue(
                                {
                                    "type": "main_comparison_failure",
                                    "dataset": dataset,
                                    "repetition": repetition,
                                    "model": model_name,
                                    "feature_set": feature_set,
                                    "error": f"{type(error).__name__}: {error}",
                                }
                            )
                            raise
    aggregate_dcss_metrics()


if __name__ == "__main__":
    main()
