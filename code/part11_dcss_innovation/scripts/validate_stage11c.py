from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from dcss_utils import FEATURE_COUNTS, aggregate_dcss, selected_features


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT.parent
PART3 = EXPERIMENT_ROOT / "part3_s0_s4_data_splits"
RESULTS = ROOT / "results"
DATASETS = {
    "phiusiil": "phiusiil_master_s3_domain_assignments.parquet",
    "iscx_url2016_binary": "iscx_url2016_binary_master_s3_domain_assignments.parquet",
}
REPETITIONS = tuple(f"r{index:02d}" for index in range(10))
MODELS = ("lr", "rf", "xgb")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_fold_assignments() -> dict[tuple[str, str], pd.DataFrame]:
    assignments = {}
    for dataset, filename in DATASETS.items():
        master_path = PART3 / "data" / "assignments" / filename
        for repetition in REPETITIONS:
            role = f"split_{repetition}"
            master_train = pd.read_parquet(
                master_path,
                columns=["sample_id", role],
                filters=[(role, "==", "train")],
            ).sort_values("sample_id").reset_index(drop=True)
            path = (
                ROOT
                / "data"
                / "processed"
                / "dcss_fold_assignments"
                / dataset
                / f"{repetition}.parquet"
            )
            require(path.is_file(), f"Missing fold assignment: {path}")
            frame = pd.read_parquet(path)
            require(
                master_train["sample_id"].equals(frame["sample_id"]),
                f"Fold assignment escaped outer train: {dataset}/{repetition}",
            )
            require(set(frame["held_fold"].unique()) == set(range(5)), "Missing held fold")
            require(
                frame.groupby("registrable_domain_sha256")["held_fold"].nunique().max() == 1,
                f"Domain assigned to multiple folds: {dataset}/{repetition}",
            )
            require(
                int(frame["fold_seed"].iloc[0]) == 20261301 + int(repetition[1:]),
                f"Fold seed mismatch: {dataset}/{repetition}",
            )
            assignments[(dataset, repetition)] = frame
    return assignments


def validate_subruns(assignments: dict[tuple[str, str], pd.DataFrame]) -> None:
    completed = list((ROOT / "runs" / "dcss_selection").glob("*/s3/r??/*/h??/complete.json"))
    require(len(completed) == 300, f"Expected 300 completion markers, found {len(completed)}")
    for dataset in DATASETS:
        for repetition in REPETITIONS:
            assignment = assignments[(dataset, repetition)].set_index("sample_id")
            for model in MODELS:
                for held_fold in range(5):
                    run = (
                        ROOT
                        / "runs"
                        / "dcss_selection"
                        / dataset
                        / "s3"
                        / repetition
                        / model
                        / f"h{held_fold:02d}"
                    )
                    metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
                    require(metadata["selection_scope"] == "source_outer_train_only", "Scope failure")
                    require(metadata["model_classes"] == [0, 1], "Class-order failure")
                    require(metadata["fold_seed"] == 20261301 + int(repetition[1:]), "Fold seed failure")
                    require(
                        metadata["model_seed"] == 20261401 + 10 * int(repetition[1:]) + held_fold,
                        "Model seed failure",
                    )
                    require(metadata["background_size"] == 200, "Background size failure")
                    require(metadata["cohort_size"] == 200, "Cohort size failure")
                    require(not metadata["background_reduced"], "Unexpected reduced background")
                    require(not metadata["cohort_reduced"], "Unexpected reduced cohort")

                    background = pd.read_parquet(run / "background_ids.parquet")
                    cohort = pd.read_parquet(run / "cohort_ids.parquet")
                    require(background["sample_id"].nunique() == 200, "Duplicate background ID")
                    require(cohort["sample_id"].nunique() == 200, "Duplicate cohort ID")
                    require(set(background["label"].value_counts().to_dict()) == {0, 1}, "Bad background labels")
                    require((background["label"].value_counts().sort_index() == [100, 100]).all(), "Unbalanced background")
                    require((cohort["label"].value_counts().sort_index() == [100, 100]).all(), "Unbalanced cohort")
                    require(set(background["sample_id"]) <= set(assignment.index), "Background outside train")
                    require(set(cohort["sample_id"]) <= set(assignment.index), "Cohort outside train")
                    require(
                        set(assignment.loc[background["sample_id"], "held_fold"]) <= (set(range(5)) - {held_fold}),
                        "Background includes held fold",
                    )
                    require(
                        set(assignment.loc[cohort["sample_id"], "held_fold"]) == {held_fold},
                        "Cohort is not from held fold",
                    )
                    require(
                        not (set(background["registrable_domain_sha256"]) & set(cohort["registrable_domain_sha256"])),
                        "Background-cohort domain overlap",
                    )

                    summary = pd.read_csv(run / "fold_shap_summary.csv")
                    require(len(summary) == 35, "Fold summary feature count failure")
                    require(summary["feature"].nunique() == 35, "Duplicate fold feature")
                    require(summary["rank"].tolist() == list(range(1, 36)), "Non-unique ranks")
                    require(np.isclose(summary["normalized_importance"].sum(), 1.0), "Normalization failure")
                    require(np.isfinite(summary.select_dtypes(include="number")).all().all(), "Non-finite summary")


def validate_aggregates() -> None:
    folds = pd.read_csv(RESULTS / "dcss_fold_shap_summary.csv")
    scores = pd.read_csv(RESULTS / "dcss_scores.csv")
    sets = pd.read_csv(RESULTS / "dcss_feature_sets.csv")
    timing = pd.read_csv(RESULTS / "dcss_timing.csv")
    require(len(folds) == 10_500, "Expected 10,500 fold-feature rows")
    require(len(scores) == 2_100, "Expected 2,100 DCSS score rows")
    require(len(sets) == 60, "Expected 60 feature-set rows")
    require(len(timing) == 300, "Expected 300 timing rows")
    require(timing["warning_count"].sum() == 0, "Unexpected runtime warnings")

    score_index = ["dataset", "scenario", "repetition", "model", "feature"]
    for keys, group in folds.groupby(["dataset", "scenario", "repetition", "model"], sort=True):
        calculated = aggregate_dcss(group).sort_values("feature").reset_index(drop=True)
        expected = scores.loc[
            (scores["dataset"] == keys[0])
            & (scores["scenario"] == keys[1])
            & (scores["repetition"] == keys[2])
            & (scores["model"] == keys[3])
        ].sort_values("feature").reset_index(drop=True)
        require(len(expected) == 35, f"Missing aggregate score rows: {keys}")
        numeric = [
            "mean_normalized_importance",
            "mean_rank",
            "median_rank",
            "rank_dispersion",
            "direction_consistency",
            "top10_frequency",
            "dcss_10",
            "top15_frequency",
            "dcss_15",
            "top20_frequency",
            "dcss_20",
        ]
        require(
            np.allclose(calculated[numeric], expected[numeric], atol=1e-12, rtol=1e-12),
            f"DCSS formula reproduction failure: {keys}",
        )
        set_row = sets.loc[
            (sets["dataset"] == keys[0])
            & (sets["repetition"] == keys[2])
            & (sets["model"] == keys[3])
        ]
        require(len(set_row) == 1, f"Missing feature-set row: {keys}")
        for k in FEATURE_COUNTS:
            actual = selected_features(calculated, k)
            stored = json.loads(set_row.iloc[0][f"f_dcss_{k}_json"])
            require(actual == stored, f"Stored F-DCSS-{k} mismatch: {keys}")
            require(len(stored) == k and len(set(stored)) == k, "Exact-k failure")


def validate_label_boundary() -> None:
    source = (ROOT / "scripts" / "run_dcss_selection.py").read_text(encoding="utf-8")
    forbidden = ["part7_external_transfer", "target_dataset", "external_target"]
    require(not any(token in source for token in forbidden), "Runner references target/external data")
    require('filters=[(role_column, "==", "train")]' in source, "Missing train-only row filter")
    require('columns=["source_row", "raw_url_sha256", *names]' in source, "Feature read changed")


def validate_analysis_and_determinism() -> None:
    expected_rows = {
        "dcss_selection_frequency.csv": 304,
        "dcss_repeat_pairwise_jaccard.csv": 810,
        "dcss_repeat_jaccard_summary.csv": 18,
        "dcss_comparator_overlap.csv": 120,
        "dcss_component_summary.csv": 210,
        "dcss_timing_summary.csv": 6,
        "dcss_fold_balance.csv": 100,
    }
    for name, rows in expected_rows.items():
        path = RESULTS / name
        require(path.is_file(), f"Missing analysis output: {name}")
        require(len(pd.read_csv(path)) == rows, f"Unexpected row count: {name}")
    audit = json.loads(
        (RESULTS / "dcss_determinism_audit.json").read_text(encoding="utf-8")
    )
    require(audit["status"] == "PASS", "Determinism audit did not pass")
    require(all(audit["matches"].values()), "Aggregate hash mismatch")


def main() -> None:
    assignments = validate_fold_assignments()
    validate_subruns(assignments)
    validate_aggregates()
    validate_label_boundary()
    validate_analysis_and_determinism()
    report = {
        "stage": "11C",
        "status": "PASS",
        "checks": {
            "fold_assignments": "PASS",
            "domain_disjointness": "PASS",
            "source_outer_train_only": "PASS",
            "subrun_artifacts": "PASS",
            "class_order_and_seeds": "PASS",
            "shap_normalization_and_ranks": "PASS",
            "dcss_formula_reproduction": "PASS",
            "exact_feature_counts": "PASS",
            "analysis_outputs": "PASS",
            "byte_identical_aggregation": "PASS",
        },
        "fold_assignment_files": 20,
        "completed_subruns": 300,
        "fold_feature_rows": 10_500,
        "dcss_feature_rows": 2_100,
        "feature_set_keys": 60,
        "runtime_warnings": 0,
        "reduced_shap_samples": 0,
    }
    (RESULTS / "stage11c_validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
