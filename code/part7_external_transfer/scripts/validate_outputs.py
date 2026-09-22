from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT.parent
PART4 = EXPERIMENT_ROOT / "part4_baseline_models"
PART6 = EXPERIMENT_ROOT / "part6_stable_feature_selection"
sys.path.insert(0, str(PART4 / "scripts"))

from model_utils import calculate_metrics  # noqa: E402


METRICS = ["macro_f1", "roc_auc", "pr_auc", "recall", "fpr", "balanced_accuracy"]
FEATURE_SETS = ["f_all", "f_single", "f_stable"]


def main() -> None:
    external = pd.read_csv(ROOT / "results" / "external_metrics.csv")
    deltas = pd.read_csv(ROOT / "results" / "paired_external_deltas.csv")
    gaps = pd.read_csv(ROOT / "results" / "internal_to_external_gaps.csv")
    shift = pd.read_csv(ROOT / "results" / "probability_shift_by_class.csv")
    overlap = pd.read_csv(ROOT / "results" / "cross_source_feature_overlap.csv")

    assert len(external) == 2 * 10 * 3 * 3 * 2, len(external)
    assert len(deltas) == 2 * 10 * 3 * 2 * 3 * 6, len(deltas)
    assert len(gaps) == 2 * 10 * 3 * 3 * 6, len(gaps)
    assert len(shift) == 2 * 10 * 3 * 3 * 2, len(shift)
    assert len(overlap) == 10 * 3 * 2, len(overlap)
    assert set(external["feature_set"]) == set(FEATURE_SETS)
    assert set(external["cohort"]) == {"unfiltered", "primary_domain_filtered"}
    assert external.groupby(
        ["source_dataset", "repetition", "model", "feature_set", "cohort"]
    ).size().eq(1).all()
    assert np.isfinite(external[METRICS].to_numpy(dtype=float)).all()
    assert external[METRICS].apply(lambda column: column.between(0, 1).all()).all()

    complete_files = list(ROOT.glob("runs/*/r??/*/f_*/complete.json"))
    prediction_files = list(ROOT.glob("predictions/*/r??/*/f_*.parquet"))
    assert len(complete_files) == 120, len(complete_files)
    assert len(prediction_files) == 120, len(prediction_files)

    checked = 0
    for path in prediction_files:
        relative = path.relative_to(ROOT / "predictions").parts
        source, repetition, model = relative[:3]
        feature_set = path.stem
        baseline_path = (
            PART4 / "predictions" / "s4" / source / repetition
            / model / "s4_predictions.parquet"
        )
        expected = pd.read_parquet(
            baseline_path,
            columns=["sample_id", "source_row", "label", "included_in_primary"],
        )
        actual = pd.read_parquet(path)
        for column in ("sample_id", "source_row", "label", "included_in_primary"):
            assert np.array_equal(actual[column].to_numpy(), expected[column].to_numpy()), (
                path, column
            )
        probabilities = actual["probability_phishing"].to_numpy(dtype=float)
        assert np.isfinite(probabilities).all()
        assert ((probabilities >= 0) & (probabilities <= 1)).all()

        internal_path = (
            PART6 / "runs" / source / "s3" / repetition / model
            / feature_set / "internal_metrics.json"
        )
        internal = json.loads(internal_path.read_text(encoding="utf-8"))
        threshold = float(internal["threshold"])
        assert np.array_equal(
            actual["prediction"].to_numpy(dtype=np.int8),
            (probabilities >= threshold).astype(np.int8),
        )
        names = json.loads(internal["feature_names_json"])
        saved_model = joblib.load(
            PART6 / "models" / source / "s3" / repetition
            / model / f"{feature_set}.joblib"
        )
        assert int(saved_model.n_features_in_) == len(names)

        metric_rows = external.loc[
            (external["source_dataset"] == source)
            & (external["repetition"] == repetition)
            & (external["model"] == model)
            & (external["feature_set"] == feature_set)
        ]
        assert len(metric_rows) == 2
        masks = {
            "unfiltered": np.ones(len(actual), dtype=bool),
            "primary_domain_filtered": actual["included_in_primary"].to_numpy(dtype=bool),
        }
        labels = actual["label"].to_numpy(dtype=np.int8)
        for cohort, mask in masks.items():
            recomputed = calculate_metrics(labels[mask], probabilities[mask], threshold)
            row = metric_rows.loc[metric_rows["cohort"] == cohort].iloc[0]
            assert int(row["n_samples"]) == int(mask.sum())
            assert float(row["source_validation_threshold"]) == threshold
            assert int(row["n_features"]) == len(names)
            for metric in METRICS:
                assert np.isclose(float(row[metric]), float(recomputed[metric]), atol=1e-6), (
                    path, cohort, metric
                )
        checked += 1

    baseline = pd.read_csv(PART4 / "results" / "s4_metrics.csv")
    copied = external.loc[external["feature_set"] == "f_all"].copy()
    keys = ["source_dataset", "target_dataset", "repetition", "model", "cohort"]
    merged = copied.merge(baseline, on=keys, suffixes=("_part7", "_part4"), validate="one_to_one")
    assert len(merged) == 120
    for metric in METRICS:
        assert np.allclose(merged[f"{metric}_part7"], merged[f"{metric}_part4"])

    summary = {
        "status": "PASS",
        "reduced_external_model_runs": len(complete_files),
        "reduced_prediction_files": len(prediction_files),
        "prediction_files_fully_aligned_to_f_all": checked,
        "external_metric_rows": len(external),
        "paired_delta_rows": len(deltas),
        "transfer_gap_rows": len(gaps),
        "probability_shift_rows": len(shift),
        "cross_source_overlap_rows": len(overlap),
        "target_feedback_used_for_selection": False,
    }
    path = ROOT / "results" / "validation_summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
