from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PART3 = ROOT.parent / "part3_s0_s4_data_splits"
METRICS = ["accuracy", "balanced_accuracy", "precision", "recall", "macro_f1", "roc_auc", "pr_auc", "fpr"]
S4_ASSIGNMENTS = {
    "phiusiil": "s4_phiusiil_to_iscx_url2016_binary_master_assignments.parquet",
    "iscx_url2016_binary": "s4_iscx_url2016_binary_to_phiusiil_master_assignments.parquet",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    internal = pd.read_csv(ROOT / "results" / "internal_metrics.csv")
    s4 = pd.read_csv(ROOT / "results" / "s4_metrics.csv")
    tuning = pd.read_csv(ROOT / "results" / "tuning_results.csv")

    assert len(internal) == 2 * 4 * 10 * 3, len(internal)
    assert len(s4) == 2 * 10 * 3 * 2, len(s4)
    assert len(tuning) == len(internal) * 3, len(tuning)
    assert set(internal["corpus_version"]) == {"master"}
    assert internal.groupby(["dataset", "scenario", "model"]).size().eq(10).all()
    assert s4.groupby(["source_dataset", "target_dataset", "cohort", "model"]).size().eq(10).all()
    assert internal["threshold"].between(0.05, 0.95).all()
    assert s4["threshold"].between(0.05, 0.95).all()
    assert np.isfinite(internal[METRICS].to_numpy(dtype=float)).all()
    assert np.isfinite(s4[METRICS].to_numpy(dtype=float)).all()
    assert internal[METRICS].apply(lambda column: column.between(0, 1).all()).all()
    assert s4[METRICS].apply(lambda column: column.between(0, 1).all()).all()
    assert (tuning["status"] == "ok").all(), "At least one tuning candidate failed"

    prediction_files = list((ROOT / "predictions" / "internal").glob("**/*.parquet"))
    s4_prediction_files = list((ROOT / "predictions" / "s4").glob("**/s4_predictions.parquet"))
    model_files = list((ROOT / "models").glob("**/*.joblib"))
    assert len(prediction_files) == 240, len(prediction_files)
    assert len(s4_prediction_files) == 60, len(s4_prediction_files)
    assert len(model_files) == 24, len(model_files)

    metric_counts = {
        (row.dataset, row.scenario_key, row.repetition, row.model): int(row.n_samples)
        for row in internal.itertuples()
    }
    for path in prediction_files:
        dataset, scenario, repetition, filename = path.relative_to(ROOT / "predictions" / "internal").parts
        model = Path(filename).stem
        frame = pd.read_parquet(path)
        assert len(frame) == metric_counts[(dataset, scenario, repetition, model)]
        assert frame["probability_phishing"].between(0, 1).all()

    for path in s4_prediction_files:
        frame = pd.read_parquet(path)
        assert len(frame) > 0
        assert 0 < frame["included_in_primary"].sum() <= len(frame)
        assert frame["probability_phishing"].between(0, 1).all()

    for source_dataset, filename in S4_ASSIGNMENTS.items():
        assignment_path = PART3 / "data" / "assignments" / filename
        role_columns = [f"role_r{index:02d}" for index in range(10)]
        assignment = pd.read_parquet(assignment_path, columns=["origin", *role_columns])
        target = assignment.loc[assignment["origin"] == "target"]
        for index, role_column in enumerate(role_columns):
            repetition = f"r{index:02d}"
            expected_unfiltered = len(target)
            expected_primary = int((target[role_column] == "target_external_primary").sum())
            metric_rows = s4.loc[
                (s4["source_dataset"] == source_dataset)
                & (s4["repetition"] == repetition)
            ]
            assert set(metric_rows.loc[metric_rows["cohort"] == "unfiltered", "n_samples"]) == {expected_unfiltered}
            assert set(metric_rows.loc[metric_rows["cohort"] == "primary_domain_filtered", "n_samples"]) == {expected_primary}
            for model in ("lr", "rf", "xgb"):
                prediction_path = ROOT / "predictions" / "s4" / source_dataset / repetition / model / "s4_predictions.parquet"
                prediction = pd.read_parquet(prediction_path, columns=["included_in_primary"])
                assert len(prediction) == expected_unfiltered
                assert int(prediction["included_in_primary"].sum()) == expected_primary

    manifest = pd.read_csv(ROOT / "results" / "result_manifest.csv")
    for row in manifest.itertuples():
        path = ROOT / row.path
        assert path.stat().st_size == row.bytes
        assert sha256_file(path) == row.sha256

    report = {
        "internal_metric_rows": len(internal),
        "s4_metric_rows": len(s4),
        "tuning_rows": len(tuning),
        "internal_prediction_files": len(prediction_files),
        "s4_prediction_files": len(s4_prediction_files),
        "saved_primary_models": len(model_files),
        "s4_role_counts_match_part3": True,
        "status": "PASS",
    }
    (ROOT / "results" / "validation_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
