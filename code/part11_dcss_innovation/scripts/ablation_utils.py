from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

import numpy as np
import pandas as pd


ABLATIONS = (
    "dcss_full",
    "no_direction",
    "no_rank_dispersion",
    "no_frequency",
)


def add_ablation_scores(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "mean_normalized_importance",
        "mean_rank",
        "rank_dispersion",
        "direction_consistency",
        "top15_frequency",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing DCSS columns: {sorted(missing)}")
    output = frame.copy()
    importance = output["mean_normalized_importance"].astype(float)
    frequency = output["top15_frequency"].astype(float)
    direction = output["direction_consistency"].astype(float)
    rank_stability = 1.0 - output["rank_dispersion"].astype(float)
    output["dcss_full"] = importance * frequency * direction * rank_stability
    output["no_direction"] = importance * frequency * rank_stability
    output["no_rank_dispersion"] = importance * frequency * direction
    output["no_frequency"] = importance * direction * rank_stability
    return output


def select_top_features(frame: pd.DataFrame, score_column: str, k: int = 15) -> list[str]:
    if score_column not in ABLATIONS:
        raise ValueError(f"Unknown ablation: {score_column}")
    if len(frame) < k or frame["feature"].duplicated().any():
        raise ValueError("Feature rows must be unique and contain at least k rows")
    records = frame.to_dict("records")
    ranked = sorted(
        records,
        key=lambda row: (
            -float(row[score_column]),
            -float(row["mean_normalized_importance"]),
            float(row["mean_rank"]),
            str(row["feature"]).encode("utf-8"),
        ),
    )
    return [str(row["feature"]) for row in ranked[:k]]


def random_feature_sets(
    feature_names: list[str], seeds: Iterable[int], k: int = 15
) -> dict[int, list[str]]:
    if len(feature_names) != len(set(feature_names)) or len(feature_names) < k:
        raise ValueError("Feature universe must be unique and contain at least k names")
    order = {name: index for index, name in enumerate(feature_names)}
    output: dict[int, list[str]] = {}
    names = np.asarray(feature_names, dtype=object)
    for seed in seeds:
        selected = np.random.default_rng(int(seed)).choice(names, size=k, replace=False)
        output[int(seed)] = sorted((str(value) for value in selected), key=order.__getitem__)
    return output


def feature_list_hash(features: list[str]) -> str:
    payload = json.dumps(features, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
