from __future__ import annotations

import hashlib
import math
import warnings

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, spearmanr
from sklearn.model_selection import StratifiedGroupKFold


FEATURE_COUNTS = (10, 15, 20)


def deterministic_stratified_sample(
    frame: pd.DataFrame, total_size: int, tag: str
) -> tuple[pd.DataFrame, bool]:
    if len(frame) < total_size:
        return frame.sort_values("sample_id").reset_index(drop=True), True
    if total_size % 2:
        raise ValueError("total_size must be even")
    selected = []
    per_class = total_size // 2
    for label in (0, 1):
        group = frame.loc[frame["label"] == label].copy()
        if len(group) < per_class:
            raise ValueError(
                f"Eligible cohort has {len(group)} label={label} rows; {per_class} required"
            )
        group["_sample_score"] = group["sample_id"].map(
            lambda value: hashlib.sha256(f"{tag}:{value}".encode("utf-8")).hexdigest()
        )
        selected.append(group.sort_values(["_sample_score", "sample_id"]).head(per_class))
    output = pd.concat(selected, ignore_index=True)
    output = output.sort_values(["label", "_sample_score", "sample_id"])
    return output.drop(columns="_sample_score").reset_index(drop=True), False


def domain_fold_assignment(frame: pd.DataFrame, seed: int) -> np.ndarray:
    labels = frame["label"].to_numpy(dtype=np.int8)
    groups = frame["registrable_domain_sha256"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = np.full(len(frame), -1, dtype=np.int8)
    for held_fold, (_, held_indices) in enumerate(
        splitter.split(np.zeros(len(frame), dtype=np.int8), labels, groups)
    ):
        folds[held_indices] = held_fold
    if (folds < 0).any() or set(np.unique(folds)) != set(range(5)):
        raise ValueError("Incomplete five-fold assignment")
    group_folds = pd.DataFrame({"group": groups, "fold": folds}).groupby("group")["fold"].nunique()
    if int(group_folds.max()) != 1:
        raise ValueError("A registrable domain appears in more than one held fold")
    for held_fold in range(5):
        if set(np.unique(labels[folds == held_fold])) != {0, 1}:
            raise ValueError(f"Held fold h{held_fold:02d} lacks one class")
    return folds


def fold_shap_summary(
    feature_names: list[str], X: np.ndarray, shap_values: np.ndarray
) -> pd.DataFrame:
    if X.shape != shap_values.shape or X.shape[1] != len(feature_names):
        raise ValueError("Feature and SHAP matrices do not align")
    rows = []
    for index, feature in enumerate(feature_names):
        feature_values = X[:, index]
        contributions = shap_values[:, index]
        importance = float(np.mean(np.abs(contributions)))
        if np.ptp(feature_values) == 0 or np.ptp(contributions) == 0:
            correlation = 0.0
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConstantInputWarning)
                correlation = float(spearmanr(feature_values, contributions).statistic)
            if not math.isfinite(correlation) or correlation == 0:
                correlation = 0.0
        rows.append(
            {
                "feature": feature,
                "mean_abs_shap": importance,
                "direction_correlation": correlation,
                "direction": int(np.sign(correlation)),
            }
        )
    output = pd.DataFrame(rows)
    denominator = float(output["mean_abs_shap"].sum())
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("Non-positive within-fold SHAP importance denominator")
    output["normalized_importance"] = output["mean_abs_shap"] / denominator
    output = output.sort_values(
        ["normalized_importance", "feature"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    output["rank"] = np.arange(1, len(output) + 1, dtype=int)
    return output


def aggregate_dcss(fold_summaries: pd.DataFrame) -> pd.DataFrame:
    required = {"held_fold", "feature", "normalized_importance", "rank", "direction"}
    missing = required - set(fold_summaries)
    if missing:
        raise ValueError(f"Missing DCSS columns: {sorted(missing)}")
    held_folds = sorted(fold_summaries["held_fold"].unique())
    features = sorted(fold_summaries["feature"].unique())
    if held_folds != list(range(5)):
        raise ValueError(f"Expected held folds 0-4, found {held_folds}")
    if len(features) != 35:
        raise ValueError(f"Expected 35 features, found {len(features)}")
    counts = fold_summaries.groupby("feature")["held_fold"].nunique()
    if not (counts == 5).all():
        raise ValueError("Each feature must occur in all five held folds")

    rows = []
    for feature, group in fold_summaries.groupby("feature", sort=True):
        importance = group["normalized_importance"].to_numpy(float)
        ranks = group["rank"].to_numpy(float)
        directions = group["direction"].to_numpy(int)
        median_rank = float(np.median(ranks))
        rank_dispersion = float(np.median(np.abs(ranks - median_rank)) / 34.0)
        positive = int(np.count_nonzero(directions == 1))
        negative = int(np.count_nonzero(directions == -1))
        direction_consistency = max(positive, negative) / 5.0
        row: dict[str, object] = {
            "feature": feature,
            "mean_normalized_importance": float(importance.mean()),
            "mean_rank": float(ranks.mean()),
            "median_rank": median_rank,
            "rank_dispersion": rank_dispersion,
            "positive_direction_count": positive,
            "negative_direction_count": negative,
            "zero_direction_count": int(np.count_nonzero(directions == 0)),
            "direction_consistency": direction_consistency,
        }
        for k in FEATURE_COUNTS:
            frequency = float(np.mean(ranks <= k))
            row[f"top{k}_frequency"] = frequency
            row[f"dcss_{k}"] = (
                row["mean_normalized_importance"]
                * frequency
                * direction_consistency
                * (1.0 - rank_dispersion)
            )
        rows.append(row)
    output = pd.DataFrame(rows)
    for k in FEATURE_COUNTS:
        ordered = output.sort_values(
            [f"dcss_{k}", "mean_normalized_importance", "mean_rank", "feature"],
            ascending=[False, False, True, True],
            kind="mergesort",
        )
        selected = set(ordered.head(k)["feature"])
        output[f"selected_{k}"] = output["feature"].isin(selected)
        ranks = {feature: index + 1 for index, feature in enumerate(ordered["feature"])}
        output[f"dcss_rank_{k}"] = output["feature"].map(ranks).astype(int)
    return output.sort_values(["dcss_rank_15", "feature"]).reset_index(drop=True)


def selected_features(scores: pd.DataFrame, k: int) -> list[str]:
    selected = scores.loc[scores[f"selected_{k}"]].sort_values(
        [f"dcss_rank_{k}", "feature"]
    )["feature"].tolist()
    if len(selected) != k or len(set(selected)) != k:
        raise ValueError(f"F-DCSS-{k} does not contain exactly {k} unique features")
    return selected
