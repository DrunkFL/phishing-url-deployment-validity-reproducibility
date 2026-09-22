from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from main_comparison_utils import extended_metrics


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT.parent
PART2 = EXPERIMENT_ROOT / "part2_url_normalization_leakage_audit"
PART6 = EXPERIMENT_ROOT / "part6_stable_feature_selection"
DATASETS = ("iscx_url2016_binary", "phiusiil")
MODELS = ("lr", "rf", "xgb")
REPETITIONS = tuple(f"r{i:02d}" for i in range(10))
METHODS = ("f_all", "f_stable", "f_dcss_15", "f_dcss_20")
COHORTS = ("unfiltered_deduplicated", "source_domain_filtered")
BATCH_SIZE = 20000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--repetitions", nargs="+", choices=REPETITIONS, default=list(REPETITIONS))
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def all_features() -> list[str]:
    frame = pd.read_csv(PART2 / "results" / "feature_dictionary.csv")
    return frame.sort_values("feature_order")["feature"].tolist()


def artifacts(dataset: str, repetition: str, model: str, method: str) -> tuple[Path, list[str], float]:
    if method == "f_all":
        model_path = (
            ROOT / "models" / "validity_repair" / "reconstructed_f_all"
            / dataset / "s3" / repetition / f"{model}.joblib"
        )
        meta = read_json(model_path.with_suffix(".json"))
        return model_path, all_features(), float(meta["threshold"])
    if method == "f_stable":
        run = PART6 / "runs" / dataset / "s3" / repetition / model / method
        metric = read_json(run / "internal_metrics.json")
        model_path = PART6 / "models" / dataset / "s3" / repetition / model / f"{method}.joblib"
        return model_path, json.loads(metric["feature_names_json"]), float(metric["threshold"])
    run = ROOT / "runs" / "main_comparison" / dataset / "s3" / repetition / model / method
    metric = read_json(run / "internal_metrics.json")
    model_path = ROOT / "models" / "main_comparison" / dataset / "s3" / repetition / model / f"{method}.joblib"
    return model_path, json.loads(metric["feature_names_json"]), float(metric["threshold"])


def predict_in_batches(estimator: object, values: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "n_jobs"):
        estimator.n_jobs = 1
    if hasattr(estimator, "named_steps"):
        classifier = estimator.named_steps.get("classifier")
        if classifier is not None and hasattr(classifier, "n_jobs"):
            classifier.n_jobs = 1
    chunks = []
    for start in range(0, len(values), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(values))
        chunks.append(np.asarray(estimator.predict_proba(values[start:stop])[:, 1], dtype=np.float64))
    return np.concatenate(chunks)


def run_one(target: pd.DataFrame, dataset: str, repetition: str, model: str) -> None:
    run = ROOT / "runs" / "stage11f" / dataset / repetition / model
    complete = run / "complete.json"
    if complete.exists():
        print(f"SKIP {dataset}/{repetition}/{model}", flush=True)
        return

    metric_rows = []
    retained_predictions: dict[str, np.ndarray] = {}
    class_orders = {}
    for method in METHODS:
        model_path, features, threshold = artifacts(dataset, repetition, model, method)
        estimator = joblib.load(model_path)
        classes = [int(value) for value in estimator.classes_]
        if classes != [0, 1]:
            raise ValueError(f"Invalid class order: {dataset}/{repetition}/{model}/{method}")
        probabilities = predict_in_batches(estimator, target.loc[:, features].to_numpy())
        if not np.isfinite(probabilities).all():
            raise ValueError("Non-finite target probabilities")
        class_orders[method] = classes
        if method in {"f_stable", "f_dcss_15"}:
            retained_predictions[method] = probabilities
        for cohort in COHORTS:
            mask = (
                np.ones(len(target), dtype=bool)
                if cohort == "unfiltered_deduplicated"
                else target["included_source_domain_filtered"].to_numpy(dtype=bool)
            )
            metric_rows.append(
                {
                    "source_dataset": dataset,
                    "target_dataset": "url_phish_mendeley_v1",
                    "repetition": repetition,
                    "model": model,
                    "feature_set": method,
                    "cohort": cohort,
                    "n_features": len(features),
                    "model_path": str(model_path),
                    "feature_names_json": json.dumps(features),
                    "model_classes_json": json.dumps(classes),
                    **extended_metrics(
                        target.loc[mask, "label"].to_numpy(), probabilities[mask], threshold
                    ),
                }
            )

    prediction = target[
        ["sample_id", "source_row", "registrable_domain_sha256", "label",
         "included_source_domain_filtered"]
    ].copy()
    prediction["probability_f_stable"] = retained_predictions["f_stable"]
    prediction["probability_f_dcss_15"] = retained_predictions["f_dcss_15"]
    prediction_path = ROOT / "predictions" / "stage11f" / dataset / repetition / f"{model}.parquet"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction.to_parquet(prediction_path, index=False, compression="zstd")
    write_json(run / "metrics.json", metric_rows)
    write_json(
        complete,
        {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "class_orders": class_orders,
            "metrics": str(run / "metrics.json"),
            "prediction": str(prediction_path),
        },
    )
    print(f"DONE {dataset}/{repetition}/{model}", flush=True)


def aggregate() -> None:
    rows = []
    for path in (ROOT / "runs" / "stage11f").glob("*/r??/*/metrics.json"):
        rows.extend(read_json(path))
    frame = pd.DataFrame(rows).sort_values(
        ["source_dataset", "repetition", "model", "feature_set", "cohort"]
    )
    (ROOT / "results").mkdir(exist_ok=True)
    frame.to_csv(ROOT / "results" / "stage11f_external_metrics.csv", index=False)


def main() -> None:
    args = parse_args()
    if not args.aggregate_only:
        target = pd.read_parquet(ROOT / "data" / "processed" / "stage11f_url_phish_v1.parquet")
        for dataset in args.datasets:
            for repetition in args.repetitions:
                for model in args.models:
                    run_one(target, dataset, repetition, model)
    aggregate()


if __name__ == "__main__":
    main()
