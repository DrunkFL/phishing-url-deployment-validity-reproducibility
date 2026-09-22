from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    checks: dict[str, object] = {}

    assignment = json.loads(
        (RESULTS / "fixed_role_assignment_summary.json").read_text(encoding="utf-8")
    )
    require(assignment["status"] == "PASS", "Assignment build did not pass")
    require(assignment["assignment_files"] == 4, "Expected four fixed assignments")
    require(assignment["audit_rows"] == 40, "Expected 40 assignment audits")
    require(assignment["fixed_role_agreement_min"] == 1.0, "Role agreement below 100%")
    require(assignment["s3_domain_overlap_max"] == 0, "S3 domain overlap detected")
    checks["fixed_role_assignments"] = "PASS"

    metrics = pd.read_csv(RESULTS / "fixed_role_deduplicated_metrics.csv")
    metric_key = ["dataset", "scenario_key", "repetition", "model"]
    require(len(metrics) == 120, "Expected 120 fixed-role metrics")
    require(not metrics.duplicated(metric_key).any(), "Duplicate fixed-role metric key")
    require(set(metrics["model_classes_json"]) == {"[0, 1]"}, "Unexpected model classes")
    require(
        set(metrics["partition_source"]) == {"filtered_master_assignment"},
        "Unexpected partition source",
    )
    checks["fixed_role_metric_rows"] = len(metrics)

    run_root = ROOT / "runs" / "validity_repair" / "deduplicated_fixed_role"
    fixed_model_root = ROOT / "models" / "validity_repair" / "deduplicated_fixed_role"
    prediction_root = ROOT / "predictions" / "validity_repair" / "deduplicated_fixed_role"
    complete_files = list(run_root.glob("**/complete.json"))
    fixed_models = list(fixed_model_root.glob("**/*.joblib"))
    predictions = list(prediction_root.glob("**/*.parquet"))
    require(len(complete_files) == 120, "Expected 120 completion markers")
    require(len(fixed_models) == 120, "Expected 120 fixed-role models")
    require(len(predictions) == 120, "Expected 120 fixed-role predictions")
    for path in fixed_models:
        require(list(joblib.load(path).classes_) == [0, 1], f"Class order failure: {path}")
    checks["fixed_role_artifacts"] = "PASS"

    reconstruction = pd.read_csv(RESULTS / "f_all_reconstruction_audit.csv")
    require(len(reconstruction) == 60, "Expected 60 F-All reconstructions")
    require(reconstruction["candidate_match"].all(), "Candidate mismatch")
    require(reconstruction["threshold_difference"].max() <= 1e-12, "Threshold mismatch")
    require(
        reconstruction["max_probability_abs_difference"].max() <= 2e-6,
        "Reconstructed probability mismatch",
    )
    checks["f_all_reconstruction"] = "PASS"

    class_audit = pd.read_csv(RESULTS / "external_model_class_order_audit.csv")
    require(len(class_audit) == 300, "Expected 300 model class audits")
    require(class_audit["class_order_pass"].all(), "Model class order failure")
    require(set(class_audit["model_classes_json"]) == {"[0, 1]"}, "Unexpected class order")
    checks["class_order_audit"] = "PASS"

    orientation = pd.read_csv(RESULTS / "external_probability_orientation_audit.csv")
    require(len(orientation) == 600, "Expected 600 probability audit rows")
    require(
        orientation["threshold_prediction_mismatches"].sum() == 0,
        "Threshold prediction mismatch",
    )
    require(
        (
            orientation["max_reported_metric_abs_difference"]
            <= orientation["reported_metric_tolerance"]
        ).all(),
        "Metric reproduction tolerance exceeded",
    )
    checks["probability_orientation_audit"] = "PASS"

    probability_summary = json.loads(
        (RESULTS / "probability_orientation_summary.json").read_text(encoding="utf-8")
    )
    require(probability_summary["status"] == "PASS", "Probability audit did not pass")
    require(probability_summary["class_order_failures"] == 0, "Class failures reported")
    require(probability_summary["feature_direction_audit_rows"] == 700, "Expected 700 direction rows")
    checks["probability_summary"] = "PASS"

    for name, expected_rows in {
        "fixed_role_paired_effects_with_ci.csv": 48,
        "external_orientation_summary.csv": 30,
        "feature_direction_flip_summary.csv": 70,
    }.items():
        path = RESULTS / name
        require(path.is_file(), f"Missing analysis output: {name}")
        require(len(pd.read_csv(path)) == expected_rows, f"Unexpected rows: {name}")
    checks["analysis_outputs"] = "PASS"

    report = {
        "stage": "11B",
        "status": "PASS",
        "checks": checks,
        "fixed_role_model_runs": 120,
        "reconstructed_f_all_models": 60,
        "audited_model_configurations": 300,
        "audited_metric_rows": 600,
        "maximum_metric_reproduction_error": float(
            orientation["max_reported_metric_abs_difference"].max()
        ),
    }
    (RESULTS / "stage11b_validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
