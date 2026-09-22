from __future__ import annotations

import argparse
import json
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
PART6 = EXPERIMENT_ROOT / "part6_stable_feature_selection"
sys.path.insert(0, str(PART4 / "scripts"))

from model_utils import calculate_metrics  # noqa: E402


DATASETS = {
    "phiusiil": {
        "features": "phiusiil_master_features.parquet",
        "target": "iscx_url2016_binary",
        "s4": "s4_phiusiil_to_iscx_url2016_binary_master_assignments.parquet",
    },
    "iscx_url2016_binary": {
        "features": "iscx_url2016_binary_master_features.parquet",
        "target": "phiusiil",
        "s4": "s4_iscx_url2016_binary_to_phiusiil_master_assignments.parquet",
    },
}
MODELS = ["lr", "rf", "xgb"]
FEATURE_SETS = ["f_single", "f_stable"]
REPETITIONS = [f"r{index:02d}" for index in range(10)]
FEATURE_DICTIONARY = PART2 / "results" / "feature_dictionary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate frozen reduced models on S4 targets")
    parser.add_argument("--sources", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--repetitions", nargs="+", choices=REPETITIONS, default=REPETITIONS)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--feature-sets", nargs="+", choices=FEATURE_SETS, default=FEATURE_SETS)
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
        raise ValueError(f"S4 target-feature mismatch for source {source}")
    if not np.array_equal(
        target["label"].to_numpy(dtype=np.int8),
        target["feature_label"].to_numpy(dtype=np.int8),
    ):
        raise ValueError(f"S4 assignment-feature label mismatch for source {source}")
    if not np.isfinite(target[names].to_numpy(dtype=np.float64)).all():
        raise ValueError(f"Non-finite target feature for source {source}")
    return target.drop(columns="feature_label")


def write_prediction(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def run_one(
    source: str,
    repetition: str,
    model_name: str,
    feature_set: str,
    target: pd.DataFrame,
    all_names: list[str],
    force: bool,
) -> None:
    run_dir = ROOT / "runs" / source / repetition / model_name / feature_set
    complete = run_dir / "complete.json"
    prediction_path = ROOT / "predictions" / source / repetition / model_name / f"{feature_set}.parquet"
    if complete.exists() and prediction_path.exists() and not force:
        print(f"SKIP {source} {repetition} {model_name} {feature_set}", flush=True)
        return
    run_dir.mkdir(parents=True, exist_ok=True)

    internal_path = (
        PART6 / "runs" / source / "s3" / repetition / model_name
        / feature_set / "internal_metrics.json"
    )
    internal = json.loads(internal_path.read_text(encoding="utf-8"))
    names = json.loads(internal["feature_names_json"])
    if not names or len(names) != len(set(names)) or not set(names).issubset(all_names):
        raise ValueError(f"Invalid frozen feature set: {source}/{repetition}/{model_name}/{feature_set}")
    model_path = PART6 / "models" / source / "s3" / repetition / model_name / f"{feature_set}.joblib"
    model = joblib.load(model_path)
    if int(getattr(model, "n_features_in_", len(names))) != len(names):
        raise ValueError(f"Model-feature dimension mismatch: {model_path}")

    X = target[names].to_numpy(dtype=np.float32)
    y = target["label"].to_numpy(dtype=np.int8)
    started = time.perf_counter()
    probabilities = model.predict_proba(X)[:, 1]
    inference_seconds = time.perf_counter() - started
    if not np.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError(f"Invalid probabilities: {source}/{repetition}/{model_name}/{feature_set}")
    threshold = float(internal["threshold"])
    primary = (target[f"role_{repetition}"] == "target_external_primary").to_numpy()
    predictions = pd.DataFrame({
        "sample_id": target["sample_id"].to_numpy(),
        "source_row": target["source_row"].to_numpy(dtype=np.int64),
        "label": y,
        "probability_phishing": probabilities.astype(np.float64),
        "prediction": (probabilities >= threshold).astype(np.int8),
        "included_in_primary": primary.astype(bool),
    })
    write_prediction(prediction_path, predictions)

    rows = []
    for cohort, mask in (
        ("unfiltered", np.ones(len(target), dtype=bool)),
        ("primary_domain_filtered", primary),
    ):
        rows.append({
            "source_dataset": source,
            "target_dataset": DATASETS[source]["target"],
            "corpus_version": "master",
            "scenario": "S4-external",
            "repetition": repetition,
            "model": model_name,
            "feature_set": feature_set,
            "n_features": len(names),
            "feature_names_json": json.dumps(names),
            "cohort": cohort,
            "selected_candidate": internal["selected_candidate"],
            "selected_params_json": internal["selected_params_json"],
            "model_seed": int(internal["model_seed"]),
            "source_validation_threshold": threshold,
            "inference_seconds_all_target": inference_seconds,
            "inference_seconds_per_sample": inference_seconds / len(target),
            **calculate_metrics(y[mask], probabilities[mask], threshold),
        })
    write_json(run_dir / "external_metrics.json", rows)
    write_json(complete, {
        "completed_utc": utc_now(),
        "source_model": str(model_path),
        "source_internal_metrics": str(internal_path),
        "prediction": str(prediction_path),
        "external_metrics": str(run_dir / "external_metrics.json"),
    })
    print(
        f"DONE {source}->{DATASETS[source]['target']} {repetition} {model_name} "
        f"{feature_set} n={len(names)} primary_f1={rows[1]['macro_f1']:.4f}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    names = feature_names()
    for source in args.sources:
        target = load_target(source, names)
        for repetition in args.repetitions:
            for model_name in args.models:
                for feature_set in args.feature_sets:
                    try:
                        run_one(
                            source, repetition, model_name, feature_set,
                            target, names, args.force,
                        )
                    except Exception as exc:
                        append_issue({
                            "type": "external_transfer_failure",
                            "source": source,
                            "target": DATASETS[source]["target"],
                            "repetition": repetition,
                            "model": model_name,
                            "feature_set": feature_set,
                            "error": f"{type(exc).__name__}: {exc}",
                        })
                        raise


if __name__ == "__main__":
    main()
