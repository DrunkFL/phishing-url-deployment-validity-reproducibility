from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

from stage11g_utils import auc_influence, benjamini_hochberg


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT.parent
PART7 = EXPERIMENT_ROOT / "part7_external_transfer"
RESULTS = ROOT / "results"
REPETITIONS = tuple(f"r{i:02d}" for i in range(10))
MODELS = ("lr", "rf", "xgb")
SOURCES = ("iscx_url2016_binary", "phiusiil")
B = 5000


def paired_external(source: str, repetition: str, model: str) -> pd.DataFrame:
    stable_path = PART7 / "predictions" / source / repetition / model / "f_stable.parquet"
    dcss_path = (
        ROOT / "predictions" / "main_comparison" / "external"
        / source / repetition / model / "f_dcss_15.parquet"
    )
    stable = pd.read_parquet(stable_path).rename(
        columns={"probability_phishing": "p_stable", "prediction": "prediction_stable"}
    )
    dcss = pd.read_parquet(dcss_path).rename(
        columns={"probability_phishing": "p_dcss", "prediction": "prediction_dcss"}
    )
    merged = dcss.merge(
        stable[["sample_id", "label", "p_stable", "prediction_stable", "included_in_primary"]],
        on="sample_id",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_stable"),
    )
    if len(merged) != len(dcss) or len(merged) != len(stable):
        raise ValueError("Stage 11G external prediction alignment failed")
    if not np.array_equal(merged["label"], merged["label_stable"]):
        raise ValueError("Stage 11G label mismatch")
    if not np.array_equal(
        merged["included_in_primary"], merged["included_in_primary_stable"]
    ):
        raise ValueError("Stage 11G primary-cohort mismatch")
    return merged.loc[merged["included_in_primary"]].reset_index(drop=True)


def nearest_psd(covariance: np.ndarray) -> tuple[np.ndarray, float]:
    symmetric = (covariance + covariance.T) / 2.0
    values, vectors = np.linalg.eigh(symmetric)
    minimum = float(values.min())
    clipped = np.clip(values, 0.0, None)
    return (vectors * clipped) @ vectors.T, minimum


def run_h2() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    pair_reference = pd.read_csv(RESULTS / "h2_external_auc_pairs.csv")
    summaries = []
    distributions = []
    saved_inputs = {}
    raw_p = []
    for source_index, source in enumerate(SOURCES):
        observed = []
        cluster_columns = []
        target_name = None
        cluster_count = None
        for repetition in REPETITIONS:
            model_deltas = []
            model_cluster = []
            for model in MODELS:
                frame = paired_external(source, repetition, model)
                auc_dcss, influence_dcss = auc_influence(frame["label"], frame["p_dcss"])
                auc_stable, influence_stable = auc_influence(frame["label"], frame["p_stable"])
                delta = auc_dcss - auc_stable
                expected = pair_reference.loc[
                    (pair_reference["source_dataset"] == source)
                    & (pair_reference["repetition"] == repetition)
                    & (pair_reference["model"] == model)
                ].iloc[0]
                if abs(delta - float(expected["delta_dcss15_minus_stable"])) > 1e-12:
                    raise ValueError("Stage 11G point estimate does not reproduce Stage 11D")
                target_name = str(expected["target_dataset"])
                model_deltas.append(delta)
                contribution = pd.Series(
                    influence_dcss - influence_stable,
                    index=frame["registrable_domain_sha256"],
                ).groupby(level=0, sort=True).sum()
                model_cluster.append(contribution)
            observed.append(float(np.mean(model_deltas)))
            cluster_average = pd.concat(model_cluster, axis=1).mean(axis=1)
            cluster_columns.append(cluster_average.rename(repetition))

        cluster_matrix = pd.concat(cluster_columns, axis=1).fillna(0.0)
        cluster_count = len(cluster_matrix)
        covariance, min_eigenvalue = nearest_psd(cluster_matrix.to_numpy().T @ cluster_matrix.to_numpy())
        observed_array = np.asarray(observed, dtype=np.float64)
        rng_cluster = np.random.default_rng(20261701 + source_index)
        perturbation = rng_cluster.multivariate_normal(
            np.zeros(len(REPETITIONS)), covariance, size=B, check_valid="raise"
        )
        rng_outer = np.random.default_rng(20261702 + source_index)
        outer_indices = rng_outer.integers(0, len(REPETITIONS), size=(B, len(REPETITIONS)))
        combined_values = observed_array[None, :] + perturbation
        combined = np.take_along_axis(combined_values, outer_indices, axis=1).mean(axis=1)
        outer_only = np.take_along_axis(
            np.broadcast_to(observed_array, (B, len(REPETITIONS))), outer_indices, axis=1
        ).mean(axis=1)
        tail_p = max(1.0 / (B + 1), 2.0 * min(float((combined <= 0).mean()), float((combined >= 0).mean())))
        raw_p.append(tail_p)
        summaries.append(
            {
                "source_dataset": source,
                "target_dataset": target_name,
                "n_outer_repetitions": len(REPETITIONS),
                "n_target_clusters": cluster_count,
                "observed_mean_delta": float(observed_array.mean()),
                "cluster_outer_ci_low": float(np.quantile(combined, 0.025)),
                "cluster_outer_ci_high": float(np.quantile(combined, 0.975)),
                "outer_only_ci_low": float(np.quantile(outer_only, 0.025)),
                "outer_only_ci_high": float(np.quantile(outer_only, 0.975)),
                "raw_p_two_sided": tail_p,
                "positive_repetitions": int((observed_array > 0).sum()),
                "minimum_covariance_eigenvalue_before_clip": min_eigenvalue,
            }
        )
        for index in range(B):
            distributions.append(
                {
                    "source_dataset": source,
                    "bootstrap_index": index,
                    "cluster_outer_delta": combined[index],
                    "outer_only_delta": outer_only[index],
                }
            )
        saved_inputs[source] = {
            "observed_repetition_deltas": observed,
            "covariance": covariance.tolist(),
            "cluster_seed": 20261701 + source_index,
            "outer_seed": 20261702 + source_index,
            "n_target_clusters": cluster_count,
        }

    summary = pd.DataFrame(summaries)
    summary["bh_adjusted_p"] = benjamini_hochberg(np.asarray(raw_p))
    summary["ci_excludes_zero"] = (
        (summary["cluster_outer_ci_low"] > 0) | (summary["cluster_outer_ci_high"] < 0)
    )
    return summary, pd.DataFrame(distributions), saved_inputs


def run_h1() -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = pd.read_csv(RESULTS / "h1_internal_noninferiority_pairs.csv")
    rows = []
    distributions = []
    p_values = []
    for group_index, ((dataset, model), group) in enumerate(
        pairs.groupby(["dataset", "model"], sort=True)
    ):
        ordered = group.set_index("repetition").loc[list(REPETITIONS)]
        delta = ordered["delta_dcss15_minus_all"].to_numpy(dtype=float)
        rng = np.random.default_rng(20261703 + group_index)
        indices = rng.integers(0, len(delta), size=(B, len(delta)))
        boot = delta[indices].mean(axis=1)
        test = ttest_1samp(delta, popmean=-0.01, alternative="greater")
        p_values.append(float(test.pvalue))
        standard_deviation = float(delta.std(ddof=1))
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "n_outer_repetitions": len(delta),
                "mean_delta": float(delta.mean()),
                "std_delta": standard_deviation,
                "standardized_effect_vs_margin": float((delta.mean() + 0.01) / standard_deviation)
                if standard_deviation > 0
                else np.inf,
                "bootstrap_ci_low": float(np.quantile(boot, 0.025)),
                "bootstrap_ci_high": float(np.quantile(boot, 0.975)),
                "noninferiority_margin": -0.01,
                "raw_p_one_sided": float(test.pvalue),
                "mean_within_margin": bool(delta.mean() >= -0.01),
            }
        )
        for index, value in enumerate(boot):
            distributions.append(
                {"dataset": dataset, "model": model, "bootstrap_index": index, "mean_delta": value}
            )
    summary = pd.DataFrame(rows)
    summary["bh_adjusted_p"] = benjamini_hochberg(np.asarray(p_values))
    summary["noninferiority_at_0_05"] = summary["bh_adjusted_p"] < 0.05
    return summary, pd.DataFrame(distributions)


def main() -> None:
    h2, h2_distribution, h2_inputs = run_h2()
    h1, h1_distribution = run_h1()
    h2.to_csv(RESULTS / "stage11g_h2_cluster_bootstrap_summary.csv", index=False)
    h2_distribution.to_csv(RESULTS / "stage11g_h2_bootstrap_distribution.csv", index=False)
    h1.to_csv(RESULTS / "stage11g_h1_noninferiority_summary.csv", index=False)
    h1_distribution.to_csv(RESULTS / "stage11g_h1_bootstrap_distribution.csv", index=False)
    (RESULTS / "stage11g_h2_bootstrap_inputs.json").write_text(
        json.dumps(h2_inputs, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    h1_counts = h1.groupby("dataset")["mean_within_margin"].sum()
    h1_condition = all(int(h1_counts.get(dataset, 0)) >= 2 for dataset in SOURCES)
    h2_positive = bool((h2["observed_mean_delta"] > 0).all())
    any_interval_excludes_zero = bool(h2["ci_excludes_zero"].any())
    final = {
        "bootstrap_replicates": B,
        "h1_two_of_three_each_dataset": h1_condition,
        "h2_positive_both_directions": h2_positive,
        "at_least_one_h2_interval_excludes_zero": any_interval_excludes_zero,
        "strong_contribution_condition_met": bool(
            h1_condition and h2_positive and any_interval_excludes_zero
        ),
        "h2_interval_method": "linearized_paired_auc_domain_gaussian_multiplier_plus_outer_bootstrap",
        "protocol_execution_addendum": "1.0.1",
    }
    (RESULTS / "stage11g_conclusion.json").write_text(
        json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(h2.to_string(index=False))
    print(h1.to_string(index=False))
    print(json.dumps(final, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
