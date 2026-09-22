from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def exact_top_k(frame: pd.DataFrame, k: int) -> list[str]:
    if k <= 0 or k > frame["feature"].nunique():
        raise ValueError("k must be between 1 and the number of features")
    ordered = frame.sort_values(
        ["mean_abs_shap", "feature"], ascending=[False, True], kind="mergesort"
    )
    return ordered.head(k)["feature"].tolist()


def summarize_selection_runs(
    summaries: pd.DataFrame,
    k: int,
    frequency_threshold: float,
    direction_threshold: float,
    require_direction: bool = True,
) -> pd.DataFrame:
    required = {"run_id", "feature", "mean_abs_shap", "direction_sign", "rank"}
    missing = required - set(summaries.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    run_ids = sorted(summaries["run_id"].unique())
    if not run_ids:
        raise ValueError("No selection runs")
    features = sorted(summaries["feature"].unique())
    if len(summaries) != len(run_ids) * len(features):
        raise ValueError("Every run must contain every feature exactly once")

    top_sets = {
        run_id: set(exact_top_k(group, k))
        for run_id, group in summaries.groupby("run_id", sort=True)
    }
    rows = []
    for feature in features:
        group = summaries.loc[summaries["feature"] == feature]
        frequency = sum(feature in top_sets[run_id] for run_id in run_ids) / len(run_ids)
        eligible = group.loc[group["direction_sign"] != 0, "direction_sign"].astype(int)
        if eligible.empty:
            majority_sign = 0
            direction_consistency = np.nan
        else:
            counts = eligible.value_counts().sort_index()
            best_count = int(counts.max())
            majority_sign = int(min(counts[counts == best_count].index))
            direction_consistency = best_count / len(eligible)
        selected = frequency >= frequency_threshold
        if require_direction:
            selected = selected and bool(
                len(eligible) > 0 and direction_consistency >= direction_threshold
            )
        rows.append({
            "feature": feature,
            "n_runs": len(run_ids),
            "k": k,
            "top_k_count": int(round(frequency * len(run_ids))),
            "top_k_frequency": float(frequency),
            "mean_rank": float(group["rank"].mean()),
            "rank_std": float(group["rank"].std(ddof=1)),
            "mean_abs_shap": float(group["mean_abs_shap"].mean()),
            "direction_eligible_runs": int(len(eligible)),
            "direction_excluded_runs": int(len(group) - len(eligible)),
            "direction_majority_sign": majority_sign,
            "direction_consistency": float(direction_consistency),
            "frequency_threshold": float(frequency_threshold),
            "direction_threshold": float(direction_threshold),
            "require_direction": bool(require_direction),
            "selected": bool(selected),
        })
    return pd.DataFrame(rows).sort_values(
        ["selected", "top_k_frequency", "mean_rank", "feature"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)


def selected_features(summary: pd.DataFrame) -> list[str]:
    selected = summary.loc[summary["selected"]].sort_values(
        ["top_k_frequency", "mean_rank", "feature"],
        ascending=[False, True, True],
    )
    return selected["feature"].tolist()


def validate_feature_names(features: Iterable[str], allowed: Iterable[str]) -> list[str]:
    result = list(features)
    allowed_set = set(allowed)
    if not result or len(result) != len(set(result)):
        raise ValueError("Feature set must be nonempty and unique")
    if not set(result) <= allowed_set:
        raise ValueError("Feature set contains unknown features")
    return result

