from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def bootstrap_ci(values: np.ndarray, *key: str) -> tuple[float, float]:
    rng = np.random.default_rng(stable_seed(*key))
    draws = rng.choice(values, size=(20_000, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def summarize_fixed_role() -> pd.DataFrame:
    frame = pd.read_csv(RESULTS / "fixed_role_vs_master_and_legacy.csv")
    measures = {
        "macro_f1": "macro_f1_fixed_minus_master",
        "roc_auc": "roc_auc_fixed_minus_master",
        "pr_auc": "pr_auc_fixed_minus_master",
        "fpr": "fpr_fixed_minus_master",
    }
    rows: list[dict[str, object]] = []
    keys = ["dataset", "scenario_key", "model"]
    for group_key, group in frame.groupby(keys, sort=True):
        for metric, column in measures.items():
            values = group[column].to_numpy(float)
            low, high = bootstrap_ci(values, *map(str, group_key), metric)
            rows.append(
                {
                    **dict(zip(keys, group_key)),
                    "metric": metric,
                    "n_paired_runs": len(values),
                    "mean_fixed_minus_master": float(values.mean()),
                    "std": float(values.std(ddof=1)),
                    "median": float(np.median(values)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "bootstrap_95_ci_low": low,
                    "bootstrap_95_ci_high": high,
                    "ci_excludes_zero": bool(low > 0 or high < 0),
                }
            )
    return pd.DataFrame(rows)


def summarize_orientation() -> pd.DataFrame:
    frame = pd.read_csv(RESULTS / "external_probability_orientation_audit.csv")
    frame = frame.loc[frame["cohort"] == "primary_domain_filtered"]
    keys = ["source_dataset", "target_dataset", "model", "feature_set"]
    return (
        frame.groupby(keys, sort=True)
        .agg(
            n_runs=("roc_auc", "size"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            roc_auc_min=("roc_auc", "min"),
            roc_auc_max=("roc_auc", "max"),
            inverted_roc_auc_mean=("inverted_roc_auc", "mean"),
            macro_f1_mean=("macro_f1", "mean"),
            maximum_metric_reproduction_error=(
                "max_reported_metric_abs_difference",
                "max",
            ),
            maximum_persistence_quantization_error=(
                "max_persisted_probability_abs_difference",
                "max",
            ),
        )
        .reset_index()
    )


def summarize_feature_directions() -> pd.DataFrame:
    frame = pd.read_csv(RESULTS / "source_target_feature_direction_audit.csv")
    keys = ["source_dataset", "target_dataset", "feature"]
    summary = (
        frame.groupby(keys, sort=True)
        .agg(
            n_repetitions=("repetition", "size"),
            direction_flip_count=("nonzero_direction_flip", "sum"),
            source_rho_mean=("source_train_spearman_with_label", "mean"),
            target_rho_mean=("target_primary_spearman_with_label", "mean"),
        )
        .reset_index()
    )
    summary["mean_rho_difference"] = (
        summary["target_rho_mean"] - summary["source_rho_mean"]
    )
    summary["absolute_mean_rho_difference"] = summary["mean_rho_difference"].abs()
    return summary.sort_values(
        ["source_dataset", "direction_flip_count", "absolute_mean_rho_difference"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def main() -> None:
    fixed = summarize_fixed_role()
    orientation = summarize_orientation()
    directions = summarize_feature_directions()
    fixed.to_csv(RESULTS / "fixed_role_paired_effects_with_ci.csv", index=False)
    orientation.to_csv(RESULTS / "external_orientation_summary.csv", index=False)
    directions.to_csv(RESULTS / "feature_direction_flip_summary.csv", index=False)
    print(
        {
            "fixed_role_summary_rows": len(fixed),
            "orientation_summary_rows": len(orientation),
            "feature_direction_rows": len(directions),
            "fixed_role_cis_excluding_zero": int(fixed["ci_excludes_zero"].sum()),
        }
    )


if __name__ == "__main__":
    main()
