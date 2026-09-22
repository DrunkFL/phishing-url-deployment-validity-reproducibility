from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PART4 = ROOT.parent / "part4_baseline_models"
sys.path.insert(0, str(PART4 / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))

from ablation_utils import (  # noqa: E402
    add_ablation_scores,
    feature_list_hash,
    random_feature_sets,
    select_top_features,
)
from main_comparison_utils import extended_metrics  # noqa: E402
from model_utils import select_model  # noqa: E402
from train_dcss_main_comparison import (  # noqa: E402
    DATASETS,
    MODELS,
    REPETITIONS,
    feature_names,
    load_internal,
    load_target,
)
from train_stage11e import ABLATION_METHODS, RANDOM_SEEDS, random_features  # noqa: E402


METRIC_FIELDS = (
    "n_samples", "n_benign", "n_phishing", "threshold", "accuracy",
    "balanced_accuracy", "macro_f1", "precision", "recall", "fpr",
    "roc_auc", "inverted_roc_auc", "pr_auc", "brier_score", "log_loss",
    "ece_15", "tn", "fp", "fn", "tp",
)


def assert_close_metrics(actual: dict, expected: dict, context: str, tolerance: float = 1e-10) -> None:
    for field in METRIC_FIELDS:
        left = float(actual[field])
        right = float(expected[field])
        if not np.isclose(left, right, atol=tolerance, rtol=tolerance, equal_nan=True):
            raise ValueError(f"Metric mismatch {context}/{field}: {left} != {right}")


def validate_feature_sets(names: list[str]) -> dict[str, int]:
    scores = add_ablation_scores(pd.read_csv(ROOT / "results" / "dcss_scores.csv"))
    key_count = 0
    for keys, group in scores.groupby(["dataset", "repetition", "model"], sort=True):
        path = ROOT / "feature_sets" / "ablation" / keys[0] / "s3" / keys[1] / keys[2] / "feature_sets.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        for method in ("dcss_full", *ABLATION_METHODS):
            expected = select_top_features(group, method)
            if payload[method] != expected or payload[f"{method}_sha256"] != feature_list_hash(expected):
                raise ValueError(f"Ablation feature mismatch: {keys}/{method}")
            if len(expected) != len(set(expected)) or not set(expected) <= set(names):
                raise ValueError(f"Invalid ablation features: {keys}/{method}")
            key_count += 1
    expected_random = random_feature_sets(names, RANDOM_SEEDS)
    stored = json.loads((ROOT / "feature_sets" / "random" / "random_feature_sets.json").read_text(encoding="utf-8"))
    for seed, expected in expected_random.items():
        if stored["sets"][str(seed)] != expected or stored["sha256"][str(seed)] != feature_list_hash(expected):
            raise ValueError(f"Random feature mismatch: {seed}")
    return {"ablation_feature_keys": key_count, "random_feature_sets": len(expected_random)}


def validate_ablation_predictions(names: list[str]) -> dict[str, int]:
    completed = list((ROOT / "runs" / "ablation").glob("*/s3/r??/*/*/complete.json"))
    if len(completed) != 180:
        raise ValueError(f"Expected 180 ablation runs, found {len(completed)}")
    models = list((ROOT / "models" / "ablation").glob("*/s3/r??/*/*.joblib"))
    internal_paths = list((ROOT / "predictions" / "ablation" / "internal").glob("*/s3/r??/*/*.parquet"))
    external_paths = list((ROOT / "predictions" / "ablation" / "external").glob("*/r??/*/*.parquet"))
    if (len(models), len(internal_paths), len(external_paths)) != (180, 180, 180):
        raise ValueError("Ablation model/prediction artifact count failure")

    checked = 0
    for dataset in DATASETS:
        internal = load_internal(dataset, names)
        target = load_target(dataset, names)
        for repetition in REPETITIONS:
            test_ids = set(internal.loc[internal[f"split_{repetition}"] == "test", "sample_id"])
            target_ids = set(target["sample_id"])
            expected_primary = dict(zip(target["sample_id"], target[f"role_{repetition}"] == "target_external_primary"))
            for model_name in MODELS:
                for method in ABLATION_METHODS:
                    base = ROOT / "runs" / "ablation" / dataset / "s3" / repetition / model_name / method
                    internal_metrics = json.loads((base / "internal_metrics.json").read_text(encoding="utf-8"))
                    external_metrics = json.loads((base / "external_metrics.json").read_text(encoding="utf-8"))
                    internal_prediction = pd.read_parquet(ROOT / "predictions" / "ablation" / "internal" / dataset / "s3" / repetition / model_name / f"{method}.parquet")
                    external_prediction = pd.read_parquet(ROOT / "predictions" / "ablation" / "external" / dataset / repetition / model_name / f"{method}.parquet")
                    if internal_prediction["probability_phishing"].dtype != np.dtype("float64") or external_prediction["probability_phishing"].dtype != np.dtype("float64"):
                        raise ValueError(f"Non-float64 probability: {dataset}/{repetition}/{model_name}/{method}")
                    if set(internal_prediction["sample_id"]) != test_ids or set(external_prediction["sample_id"]) != target_ids:
                        raise ValueError(f"Prediction cohort mismatch: {dataset}/{repetition}/{model_name}/{method}")
                    observed_primary = dict(zip(external_prediction["sample_id"], external_prediction["included_in_primary"].astype(bool)))
                    if observed_primary != expected_primary:
                        raise ValueError(f"External primary mask mismatch: {dataset}/{repetition}/{model_name}/{method}")
                    recalculated = extended_metrics(internal_prediction["label"].to_numpy(np.int8), internal_prediction["probability_phishing"].to_numpy(float), float(internal_metrics["threshold"]))
                    assert_close_metrics(recalculated, internal_metrics, f"internal/{dataset}/{repetition}/{model_name}/{method}")
                    for record in external_metrics:
                        mask = np.ones(len(external_prediction), dtype=bool) if record["cohort"] == "unfiltered" else external_prediction["included_in_primary"].to_numpy(bool)
                        recalculated = extended_metrics(external_prediction.loc[mask, "label"].to_numpy(np.int8), external_prediction.loc[mask, "probability_phishing"].to_numpy(float), float(record["threshold"]))
                        assert_close_metrics(recalculated, record, f"external/{dataset}/{repetition}/{model_name}/{method}/{record['cohort']}")
                    estimator = joblib.load(ROOT / "models" / "ablation" / dataset / "s3" / repetition / model_name / f"{method}.joblib")
                    if [int(value) for value in estimator.classes_] != [0, 1] or int(estimator.n_features_in_) != 15:
                        raise ValueError(f"Ablation estimator mismatch: {dataset}/{repetition}/{model_name}/{method}")
                    checked += 1
    return {"completed_ablation_runs": len(completed), "validated_ablation_predictions": checked}


def validate_random_records() -> dict[str, int]:
    completed = list((ROOT / "runs" / "random_baseline").glob("*/s3/r??/*/random_*/complete.json"))
    if len(completed) != 1800:
        raise ValueError(f"Expected 1800 random runs, found {len(completed)}")
    seen = set()
    for path in completed:
        metrics = json.loads((path.parent / "internal_metrics.json").read_text(encoding="utf-8"))
        external = json.loads((path.parent / "external_metrics.json").read_text(encoding="utf-8"))
        key = (metrics["dataset"], metrics["repetition"], metrics["model"], int(metrics["random_seed"]))
        if key in seen:
            raise ValueError(f"Duplicate random key: {key}")
        seen.add(key)
        expected = random_features(key[3])
        if metrics["feature_set_sha256"] != feature_list_hash(expected) or json.loads(metrics["feature_names_json"]) != expected:
            raise ValueError(f"Random feature hash failure: {key}")
        if json.loads(metrics["model_classes_json"]) != [0, 1] or len(external) != 2:
            raise ValueError(f"Random metadata failure: {key}")
    return {"completed_random_runs": len(completed), "unique_random_keys": len(seen)}


def validate_sampled_random_refits(names: list[str]) -> int:
    checked = 0
    for dataset in DATASETS:
        internal = load_internal(dataset, names)
        target = load_target(dataset, names)
        for model_name in MODELS:
            for repetition, seed in (("r00", 20261501), ("r09", 20261530)):
                selected = random_features(seed)
                split = internal[f"split_{repetition}"].astype(str).to_numpy()
                train = internal.loc[split == "train"]
                validation = internal.loc[split == "validation"]
                test = internal.loc[split == "test"]
                model_seed = 20261001 + int(repetition[1:])
                fitted = select_model(model_name, train[selected].to_numpy(np.float32), train["label"].to_numpy(np.int8), validation[selected].to_numpy(np.float32), validation["label"].to_numpy(np.int8), model_seed, 2)
                base = ROOT / "runs" / "random_baseline" / dataset / "s3" / repetition / model_name / f"random_{seed}"
                stored_i = json.loads((base / "internal_metrics.json").read_text(encoding="utf-8"))
                stored_e = json.loads((base / "external_metrics.json").read_text(encoding="utf-8"))
                if fitted.selected_candidate != stored_i["selected_candidate"] or not np.isclose(fitted.threshold, stored_i["threshold"], atol=1e-12):
                    raise ValueError(f"Sampled random selection mismatch: {dataset}/{repetition}/{model_name}/{seed}")
                probabilities = fitted.model.predict_proba(test[selected].to_numpy(np.float32))[:, 1]
                assert_close_metrics(extended_metrics(test["label"].to_numpy(np.int8), probabilities, fitted.threshold), stored_i, f"random-refit-internal/{dataset}/{repetition}/{model_name}/{seed}", 1e-8)
                target_probabilities = fitted.model.predict_proba(target[selected].to_numpy(np.float32))[:, 1]
                primary = (target[f"role_{repetition}"] == "target_external_primary").to_numpy(bool)
                record = next(value for value in stored_e if value["cohort"] == "primary_domain_filtered")
                assert_close_metrics(extended_metrics(target.loc[primary, "label"].to_numpy(np.int8), target_probabilities[primary], fitted.threshold), record, f"random-refit-external/{dataset}/{repetition}/{model_name}/{seed}", 1e-8)
                checked += 1
    return checked


def validate_held_domain() -> dict[str, int]:
    completed = list((ROOT / "runs" / "held_domain_diagnostics").glob("*/s3/r??/*/h??/complete.json"))
    predictions = list((ROOT / "predictions" / "held_domain_diagnostics").glob("*/r??/*/h??.parquet"))
    if len(completed) != 300 or len(predictions) != 300:
        raise ValueError(f"Expected 300 held-domain runs/predictions, found {len(completed)}/{len(predictions)}")
    for path in predictions:
        prediction = pd.read_parquet(path)
        if prediction["probability_phishing"].dtype != np.dtype("float64"):
            raise ValueError(f"Held-domain probability dtype failure: {path}")
        dataset, repetition, model_name = path.parts[-4:-1]
        held_fold = int(path.stem[1:])
        run = ROOT / "runs" / "held_domain_diagnostics" / dataset / "s3" / repetition / model_name / f"h{held_fold:02d}"
        metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        recalculated = extended_metrics(prediction["label"].to_numpy(np.int8), prediction["probability_phishing"].to_numpy(float), 0.5)
        assert_close_metrics(recalculated, metrics, f"held/{dataset}/{repetition}/{model_name}/h{held_fold:02d}")
        if prediction["registrable_domain_sha256"].duplicated().all():
            raise ValueError(f"Invalid held-domain identifiers: {path}")
    return {"completed_held_domain_runs": len(completed), "validated_held_predictions": len(predictions)}


def validate_aggregate_tables() -> dict[str, int]:
    counts = {
        "ablation_internal_metric_rows": len(pd.read_csv(ROOT / "results" / "stage11e_ablation_internal_metrics.csv")),
        "ablation_external_metric_rows": len(pd.read_csv(ROOT / "results" / "stage11e_ablation_external_metrics.csv")),
        "random_internal_metric_rows": len(pd.read_csv(ROOT / "results" / "stage11e_random_internal_metrics.csv")),
        "random_external_metric_rows": len(pd.read_csv(ROOT / "results" / "stage11e_random_external_metrics.csv")),
        "held_domain_metric_rows": len(pd.read_csv(ROOT / "results" / "stage11e_held_domain_metrics.csv")),
    }
    expected = {
        "ablation_internal_metric_rows": 180,
        "ablation_external_metric_rows": 360,
        "random_internal_metric_rows": 1800,
        "random_external_metric_rows": 3600,
        "held_domain_metric_rows": 300,
    }
    if counts != expected:
        raise ValueError(f"Aggregate row count failure: {counts}")
    required_analysis = (
        "stage11e_analysis_summary.json", "stage11e_ablation_external_summary.csv",
        "stage11e_random_distribution_summary.csv", "stage11e_feature_membership.csv",
        "stage11e_held_domain_variance.csv", "stage11e_diagnostic_correlations.csv",
    )
    if any(not (ROOT / "results" / name).is_file() for name in required_analysis):
        raise ValueError("Missing Stage 11E analysis output")
    return counts


def main() -> None:
    if not (ROOT / "STAGE11E_ANALYSIS_LOCK.md").is_file():
        raise ValueError("Missing Stage 11E analysis lock")
    names = feature_names()
    details = {}
    details.update(validate_feature_sets(names))
    details.update(validate_ablation_predictions(names))
    details.update(validate_random_records())
    details["sampled_random_refits"] = validate_sampled_random_refits(names)
    details.update(validate_held_domain())
    details.update(validate_aggregate_tables())
    runtime_log = ROOT / "logs" / "stage11e_runtime_issues.jsonl"
    details["runtime_issue_records"] = len(runtime_log.read_text(encoding="utf-8").splitlines()) if runtime_log.is_file() else 0
    output = {
        "stage": "11E", "status": "PASS",
        "checks": {
            "analysis_lock": "PASS", "feature_set_reproduction": "PASS",
            "ablation_artifacts_and_metrics": "PASS", "random_run_completeness": "PASS",
            "sampled_random_determinism": "PASS", "held_domain_metrics": "PASS",
            "aggregate_and_analysis_outputs": "PASS",
        },
        **details,
    }
    (ROOT / "results" / "stage11e_validation.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
