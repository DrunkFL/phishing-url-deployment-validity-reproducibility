from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from stage11g_utils import benjamini_hochberg


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
B = 5000


def main() -> None:
    h2 = pd.read_csv(RESULTS / "stage11g_h2_cluster_bootstrap_summary.csv")
    h2_dist = pd.read_csv(RESULTS / "stage11g_h2_bootstrap_distribution.csv")
    h1 = pd.read_csv(RESULTS / "stage11g_h1_noninferiority_summary.csv")
    h1_dist = pd.read_csv(RESULTS / "stage11g_h1_bootstrap_distribution.csv")
    inputs = json.loads((RESULTS / "stage11g_h2_bootstrap_inputs.json").read_text(encoding="utf-8"))
    checks = {
        "h2_groups_and_replicates": len(h2) == 2 and len(h2_dist) == 2 * B,
        "h1_groups_and_replicates": len(h1) == 6 and len(h1_dist) == 6 * B,
        "finite_outputs": bool(
            np.isfinite(h2.select_dtypes(include=[np.number])).all().all()
            and np.isfinite(h1.select_dtypes(include=[np.number])).all().all()
        ),
        "bh_h2_reproduces": bool(
            np.allclose(h2["bh_adjusted_p"], benjamini_hochberg(h2["raw_p_two_sided"]))
        ),
        "bh_h1_reproduces": bool(
            np.allclose(h1["bh_adjusted_p"], benjamini_hochberg(h1["raw_p_one_sided"]))
        ),
    }
    maximum_distribution_error = 0.0
    for source_index, source in enumerate(("iscx_url2016_binary", "phiusiil")):
        saved = inputs[source]
        observed = np.asarray(saved["observed_repetition_deltas"], dtype=float)
        covariance = np.asarray(saved["covariance"], dtype=float)
        rng_cluster = np.random.default_rng(saved["cluster_seed"])
        perturbation = rng_cluster.multivariate_normal(
            np.zeros(len(observed)), covariance, size=B, check_valid="raise"
        )
        rng_outer = np.random.default_rng(saved["outer_seed"])
        indices = rng_outer.integers(0, len(observed), size=(B, len(observed)))
        expected = np.take_along_axis(observed[None, :] + perturbation, indices, axis=1).mean(axis=1)
        actual = h2_dist.loc[
            h2_dist["source_dataset"] == source
        ].sort_values("bootstrap_index")["cluster_outer_delta"].to_numpy()
        maximum_distribution_error = max(
            maximum_distribution_error, float(np.max(np.abs(expected - actual)))
        )
    checks["h2_deterministic_rerun"] = maximum_distribution_error <= 1e-14

    stage11d = pd.read_csv(RESULTS / "h2_external_auc_by_direction.csv")
    merged = h2.merge(stage11d, on=["source_dataset", "target_dataset"], validate="one_to_one")
    point_error = float(
        np.max(np.abs(merged["observed_mean_delta"] - merged["mean_delta_across_models"]))
    )
    checks["stage11d_point_estimates_reproduce"] = point_error <= 1e-12
    conclusion = json.loads((RESULTS / "stage11g_conclusion.json").read_text(encoding="utf-8"))
    checks["conclusion_condition_reproduces"] = conclusion["strong_contribution_condition_met"] == bool(
        conclusion["h1_two_of_three_each_dataset"]
        and conclusion["h2_positive_both_directions"]
        and conclusion["at_least_one_h2_interval_excludes_zero"]
    )
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "maximum_h2_distribution_rerun_error": maximum_distribution_error,
        "maximum_stage11d_point_estimate_error": point_error,
    }
    (RESULTS / "stage11g_validation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
