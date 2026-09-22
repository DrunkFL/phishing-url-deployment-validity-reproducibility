from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def require(condition: bool, message: str, checks: list[str]) -> None:
    if not condition:
        raise AssertionError(message)
    checks.append(message)


def main() -> None:
    checks: list[str] = []
    dedup = pd.read_csv(RESULTS / "deduplicated_metrics.csv")
    require(len(dedup) == 120, "120 deduplicated S0/S3 model runs are present", checks)
    require(not dedup.duplicated(["dataset", "scenario_key", "repetition", "model"]).any(), "Deduplicated run keys are unique", checks)
    require(dedup[["macro_f1", "roc_auc", "pr_auc", "fpr"]].notna().all().all(), "Deduplicated core metrics are complete", checks)

    paired = pd.read_csv(RESULTS / "deduplicated_paired_deltas.csv")
    require(len(paired) == 120, "All deduplicated runs have matched master-corpus comparators", checks)

    domain = pd.read_csv(RESULTS / "domain_equal_metrics.csv")
    require(len(domain) == 600, "Domain-equal metrics cover 2 scopes x 300 configurations", checks)
    require(not domain.duplicated(["scope", "source_dataset", "repetition", "model", "feature_set"]).any(), "Domain-equal configuration keys are unique", checks)
    require(domain["n_domains"].gt(0).all(), "Every domain-equal cohort contains registrable domains", checks)
    require(domain[["domain_equal_macro_f1", "domain_equal_balanced_accuracy", "domain_equal_recall", "domain_equal_fpr"]].notna().all().all(), "Domain-equal metrics are finite", checks)

    calibration = pd.read_csv(RESULTS / "external_calibration_metrics.csv")
    require(len(calibration) == 300, "Calibration diagnostics cover all external configurations", checks)
    require(calibration[["brier_score", "log_loss", "ece_15", "oracle_target_macro_f1", "threshold_regret"]].notna().all().all(), "Calibration diagnostics are finite", checks)
    require((calibration["threshold_regret"] >= -1e-12).all(), "Oracle threshold regret is nonnegative", checks)

    bins = pd.read_csv(RESULTS / "external_reliability_bins.csv")
    require(len(bins) == 4500, "Reliability table contains 15 bins for each external configuration", checks)
    cluster = pd.read_csv(RESULTS / "representative_domain_cluster_bootstrap.csv")
    require(len(cluster) == 4, "Representative cluster bootstrap contains four prespecified configurations", checks)
    require((cluster["n_bootstrap"] == 2000).all(), "Each representative cluster interval uses 2,000 replicates", checks)

    expected_figures = [
        ROOT / "figures" / "external_reliability_diagrams.png",
        ROOT / "figures" / "external_reliability_diagrams.pdf",
        ROOT / "figures" / "external_threshold_regret.png",
        ROOT / "figures" / "external_threshold_regret.pdf",
    ]
    require(all(path.exists() and path.stat().st_size > 1000 for path in expected_figures), "Four nonempty calibration figure files are present", checks)
    require((RESULTS / "part10_robustness_report.md").exists(), "Part 10 report is present", checks)

    report = {"status": "PASS", "n_checks": len(checks), "checks": checks}
    (RESULTS / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
