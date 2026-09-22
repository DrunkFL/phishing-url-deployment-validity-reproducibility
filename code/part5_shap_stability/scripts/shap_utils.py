from __future__ import annotations

import hashlib
import time
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr


@dataclass
class ExplainerBundle:
    model_name: str
    explainer: Any
    transform: Any
    model_output_scale: str
    perturbation: str
    expected_value: float
    setup_seconds: float


def deterministic_stratified_sample(
    frame: pd.DataFrame,
    total_size: int,
    tag: str,
) -> pd.DataFrame:
    if total_size % 2:
        raise ValueError("total_size must be even")
    per_class = total_size // 2
    selected = []
    for label in (0, 1):
        group = frame.loc[frame["label"] == label].copy()
        if len(group) < per_class:
            raise ValueError(f"Not enough label={label} rows for {total_size} samples")
        group["_sample_score"] = group["sample_id"].map(
            lambda value: hashlib.sha256(f"{tag}:{value}".encode("utf-8")).hexdigest()
        )
        selected.append(group.sort_values("_sample_score").head(per_class))
    output = pd.concat(selected, ignore_index=True).sort_values(["label", "_sample_score"])
    return output.drop(columns="_sample_score").reset_index(drop=True)


def build_explainer(
    model_name: str,
    model: Any,
    X_background: np.ndarray,
) -> ExplainerBundle:
    start = time.perf_counter()
    if model_name == "lr":
        scaler = model.named_steps["scale"]
        classifier = model.named_steps["classifier"]
        transform = scaler.transform
        explainer = shap.LinearExplainer(classifier, transform(X_background))
        expected = float(np.asarray(explainer.expected_value).reshape(-1)[0])
        scale = "raw_log_odds"
        perturbation = "interventional_linear"
    elif model_name in {"rf", "xgb"}:
        transform = lambda values: values
        explainer = shap.TreeExplainer(
            model,
            feature_perturbation="tree_path_dependent",
            model_output="raw",
        )
        expected_values = np.asarray(explainer.expected_value).reshape(-1)
        expected = float(expected_values[1] if model_name == "rf" else expected_values[0])
        scale = "positive_class_probability" if model_name == "rf" else "raw_margin"
        perturbation = "tree_path_dependent_exact"
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return ExplainerBundle(
        model_name=model_name,
        explainer=explainer,
        transform=transform,
        model_output_scale=scale,
        perturbation=perturbation,
        expected_value=expected,
        setup_seconds=time.perf_counter() - start,
    )


def normalize_shap_values(model_name: str, values: Any) -> np.ndarray:
    array = np.asarray(values)
    if model_name == "rf":
        if array.ndim == 3 and array.shape[2] == 2:
            array = array[:, :, 1]
        elif isinstance(values, list) and len(values) == 2:
            array = np.asarray(values[1])
    if array.ndim != 2:
        raise ValueError(f"Unexpected {model_name} SHAP shape: {array.shape}")
    return array.astype(np.float64, copy=False)


def explain(bundle: ExplainerBundle, X: np.ndarray) -> tuple[np.ndarray, float, list[dict[str, str]]]:
    transformed = bundle.transform(X)
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if bundle.model_name == "lr":
            raw = bundle.explainer(transformed).values
        else:
            raw = bundle.explainer.shap_values(
                transformed,
                approximate=False,
                check_additivity=False,
            )
    elapsed = time.perf_counter() - start
    warning_rows = [
        {"category": item.category.__name__, "message": str(item.message)} for item in caught
    ]
    return normalize_shap_values(bundle.model_name, raw), elapsed, warning_rows


def global_feature_summary(
    X: np.ndarray,
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    if X.shape != shap_values.shape or X.shape[1] != len(feature_names):
        raise ValueError("Feature and SHAP matrices do not align")
    rows = []
    for index, feature in enumerate(feature_names):
        feature_values = X[:, index]
        contributions = shap_values[:, index]
        if np.ptp(feature_values) == 0 or np.ptp(contributions) == 0:
            direction_correlation = 0.0
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                direction_correlation = float(spearmanr(feature_values, contributions).statistic)
            if not np.isfinite(direction_correlation):
                direction_correlation = 0.0
        rows.append({
            "feature": feature,
            "mean_abs_shap": float(np.mean(np.abs(contributions))),
            "mean_shap": float(np.mean(contributions)),
            "direction_correlation": direction_correlation,
            "direction_sign": int(np.sign(direction_correlation)),
        })
    output = pd.DataFrame(rows)
    output["rank"] = output["mean_abs_shap"].rank(method="min", ascending=False).astype(int)
    return output.sort_values(["rank", "feature"]).reset_index(drop=True)


def shap_value_frame(
    cohort: pd.DataFrame,
    X: np.ndarray,
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    output = cohort[["sample_id", "source_row", "label"]].reset_index(drop=True).copy()
    for index, feature in enumerate(feature_names):
        output[f"value__{feature}"] = X[:, index].astype(np.float32)
        output[f"shap__{feature}"] = shap_values[:, index].astype(np.float32)
    return output


def ranking_comparison(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, float]:
    left = left.set_index("feature")
    right = right.set_index("feature")
    if set(left.index) != set(right.index):
        raise ValueError("Feature sets differ")
    metrics: dict[str, float] = {}
    for k in (10, 15, 20):
        left_set = set(left.nsmallest(k, "rank").index)
        right_set = set(right.nsmallest(k, "rank").index)
        metrics[f"top{k}_jaccard"] = len(left_set & right_set) / len(left_set | right_set)
    aligned = left.join(right, lsuffix="_left", rsuffix="_right")
    metrics["rank_spearman"] = float(
        spearmanr(aligned["rank_left"], aligned["rank_right"]).statistic
    )
    metrics["direction_agreement_all"] = float(
        (aligned["direction_sign_left"] == aligned["direction_sign_right"]).mean()
    )
    top_union = set(left.nsmallest(20, "rank").index) | set(right.nsmallest(20, "rank").index)
    metrics["direction_agreement_top20_union"] = float(
        (aligned.loc[list(top_union), "direction_sign_left"] == aligned.loc[list(top_union), "direction_sign_right"]).mean()
    )
    return metrics
