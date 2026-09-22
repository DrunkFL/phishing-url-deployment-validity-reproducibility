from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT.parent
PART3 = EXPERIMENT_ROOT / "part3_s0_s4_data_splits"
PART4 = EXPERIMENT_ROOT / "part4_baseline_models"
RESULTS = ROOT / "results"
DATASETS = {
    "phiusiil": "phiusiil_master_s3_domain_assignments.parquet",
    "iscx_url2016_binary": "iscx_url2016_binary_master_s3_domain_assignments.parquet",
}


def validate_selection_scope() -> None:
    assignments = {}
    for dataset, filename in DATASETS.items():
        assignments[dataset] = pd.read_parquet(
            PART3 / "data" / "assignments" / filename,
            columns=["sample_id", *[f"split_r{i:02d}" for i in range(10)]],
        ).set_index("sample_id")
    for ids_path in ROOT.glob("selection_runs/*/s3/*/*/*/*_ids.parquet"):
        parts = ids_path.relative_to(ROOT / "selection_runs").parts
        dataset, _, outer_rep = parts[:3]
        ids = pd.read_parquet(ids_path, columns=["sample_id"])["sample_id"]
        roles = assignments[dataset].loc[ids, f"split_{outer_rep}"].astype(str)
        if not (roles == "train").all():
            raise AssertionError(f"Non-training sample in selection: {ids_path}")


def validate_predictions() -> None:
    for path in ROOT.glob("predictions/*/s3/*/*/*.parquet"):
        parts = path.relative_to(ROOT / "predictions").parts
        dataset, _, repetition, model = parts[:4]
        reduced = pd.read_parquet(path, columns=["sample_id", "label", "probability_phishing"])
        baseline_path = PART4 / "predictions" / "internal" / dataset / "s3" / repetition / f"{model}.parquet"
        baseline = pd.read_parquet(baseline_path, columns=["sample_id", "label"])
        if not reduced[["sample_id", "label"]].equals(baseline[["sample_id", "label"]]):
            raise AssertionError(f"Prediction cohort mismatch: {path}")
        if not reduced["probability_phishing"].between(0, 1).all():
            raise AssertionError(f"Invalid probabilities: {path}")


def main() -> None:
    selection_complete = list(ROOT.glob("selection_runs/*/s3/*/*/*/complete.json"))
    feature_set_files = list(ROOT.glob("feature_sets/*/s3/*/*/feature_sets.json"))
    reduced_complete = list(ROOT.glob("runs/*/s3/*/*/*/complete.json"))
    predictions = list(ROOT.glob("predictions/*/s3/*/*/*.parquet"))
    models = list(ROOT.glob("models/*/s3/*/*/*.joblib"))
    if len(selection_complete) != 600:
        raise AssertionError(f"Expected 600 selection runs, found {len(selection_complete)}")
    if len(feature_set_files) != 60:
        raise AssertionError(f"Expected 60 feature-set files, found {len(feature_set_files)}")
    if len(reduced_complete) != 120 or len(predictions) != 120 or len(models) != 120:
        raise AssertionError("Reduced model artifact counts do not equal 120")

    feature_sets = pd.read_csv(RESULTS / "feature_set_summary.csv")
    sensitivity = pd.read_csv(RESULTS / "feature_selection_sensitivity.csv")
    stability = pd.read_csv(RESULTS / "training_side_feature_stability.csv")
    metrics = pd.read_csv(RESULTS / "internal_metrics.csv")
    deltas = pd.read_csv(RESULTS / "paired_metric_deltas.csv")
    if len(feature_sets) != 60 or len(sensitivity) != 1080 or len(stability) != 2100:
        raise AssertionError("Selection summary row counts are incomplete")
    if len(metrics) != 180 or len(deltas) != 1080:
        raise AssertionError("Model comparison row counts are incomplete")
    if not (feature_sets["n_f_single"] == 15).all():
        raise AssertionError("F-Single must contain exactly 15 features")
    if not feature_sets["n_f_stable"].between(1, 35).all():
        raise AssertionError("F-Stable must be nonempty and within the frozen feature set")
    if not metrics["threshold"].between(0.05, 0.95).all():
        raise AssertionError("Threshold outside the prespecified grid")
    for metric in ("macro_f1", "recall", "fpr", "roc_auc", "pr_auc"):
        if not metrics[metric].between(0, 1).all():
            raise AssertionError(f"Invalid metric: {metric}")

    validate_selection_scope()
    validate_predictions()
    summary = {
        "status": "PASS",
        "selection_runs": len(selection_complete),
        "feature_set_files": len(feature_set_files),
        "reduced_model_runs": len(reduced_complete),
        "comparison_metric_rows": len(metrics),
        "paired_delta_rows": len(deltas),
        "selection_scope": "outer_train_only",
    }
    (RESULTS / "validation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

