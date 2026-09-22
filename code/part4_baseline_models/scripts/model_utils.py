from __future__ import annotations

import json
import math
import time
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.01), 2)


@dataclass
class SelectionResult:
    model: Any
    model_name: str
    selected_candidate: str
    selected_params: dict[str, Any]
    threshold: float
    validation_metrics: dict[str, float | int]
    tuning_rows: list[dict[str, Any]]
    warnings: list[dict[str, str]]
    tuning_seconds: float


def candidate_grid(model_name: str, y_train: np.ndarray) -> list[tuple[str, dict[str, Any]]]:
    if model_name == "lr":
        return [
            ("lr_c01", {"C": 0.1, "class_weight": None}),
            ("lr_c1", {"C": 1.0, "class_weight": None}),
            ("lr_c1_balanced", {"C": 1.0, "class_weight": "balanced"}),
        ]
    if model_name == "rf":
        return [
            (
                "rf_depth16_leaf1",
                {"n_estimators": 200, "max_depth": 16, "min_samples_leaf": 1,
                 "max_features": "sqrt", "class_weight": None},
            ),
            (
                "rf_depth24_leaf2",
                {"n_estimators": 200, "max_depth": 24, "min_samples_leaf": 2,
                 "max_features": "sqrt", "class_weight": None},
            ),
            (
                "rf_unlimited_leaf5_balanced",
                {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 5,
                 "max_features": "sqrt", "class_weight": "balanced_subsample"},
            ),
        ]
    if model_name == "xgb":
        negative = int((y_train == 0).sum())
        positive = int((y_train == 1).sum())
        scale_pos_weight = negative / positive
        common = {"subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 1.0}
        return [
            ("xgb_d4_lr005", {**common, "n_estimators": 300, "max_depth": 4,
                               "learning_rate": 0.05, "scale_pos_weight": 1.0}),
            ("xgb_d6_lr005", {**common, "n_estimators": 250, "max_depth": 6,
                               "learning_rate": 0.05, "scale_pos_weight": 1.0}),
            ("xgb_d6_lr01_balanced", {**common, "n_estimators": 200, "max_depth": 6,
                                       "learning_rate": 0.10,
                                       "scale_pos_weight": scale_pos_weight}),
        ]
    raise ValueError(f"Unknown model: {model_name}")


def build_model(model_name: str, params: dict[str, Any], seed: int, n_jobs: int) -> Any:
    if model_name == "lr":
        classifier = LogisticRegression(
            solver="lbfgs",
            max_iter=2000,
            random_state=seed,
            C=params["C"],
            class_weight=params["class_weight"],
        )
        return Pipeline([("scale", StandardScaler()), ("classifier", classifier)])
    if model_name == "rf":
        return RandomForestClassifier(
            random_state=seed,
            n_jobs=n_jobs,
            **params,
        )
    if model_name == "xgb":
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=n_jobs,
            verbosity=0,
            **params,
        )
    raise ValueError(f"Unknown model: {model_name}")


def calculate_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    predictions = (probabilities >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if fp + tn else math.nan
    return {
        "n_samples": int(len(y_true)),
        "n_benign": int((y_true == 0).sum()),
        "n_phishing": int((y_true == 1).sum()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "fpr": float(fpr),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def select_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, dict[str, float | int]]:
    scored = []
    for threshold in THRESHOLDS:
        metrics = calculate_metrics(y_true, probabilities, float(threshold))
        scored.append((metrics["macro_f1"], -abs(float(threshold) - 0.5), -float(threshold), metrics))
    return float(max(scored, key=lambda row: row[:3])[3]["threshold"]), max(scored, key=lambda row: row[:3])[3]


def select_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    seed: int,
    n_jobs: int,
) -> SelectionResult:
    start_all = time.perf_counter()
    best_key = None
    best_model = None
    best_candidate = None
    best_params = None
    best_threshold = None
    best_metrics = None
    tuning_rows: list[dict[str, Any]] = []
    captured_warnings: list[dict[str, str]] = []

    for candidate_index, (candidate_name, params) in enumerate(candidate_grid(model_name, y_train)):
        model = build_model(model_name, params, seed, n_jobs)
        fit_start = time.perf_counter()
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model.fit(X_train, y_train)
                validation_probabilities = model.predict_proba(X_validation)[:, 1]
            threshold, metrics = select_threshold(y_validation, validation_probabilities)
            for warning in caught:
                captured_warnings.append({
                    "candidate": candidate_name,
                    "category": warning.category.__name__,
                    "message": str(warning.message),
                })
            fit_seconds = time.perf_counter() - fit_start
            selection_key = (metrics["macro_f1"], metrics["roc_auc"], -candidate_index)
            tuning_rows.append({
                "candidate": candidate_name,
                "candidate_index": candidate_index,
                "params_json": json.dumps(params, sort_keys=True),
                "status": "ok",
                "fit_seconds": fit_seconds,
                "threshold": threshold,
                **{f"validation_{key}": value for key, value in metrics.items()},
            })
            if best_key is None or selection_key > best_key:
                best_key = selection_key
                best_model = model
                best_candidate = candidate_name
                best_params = params
                best_threshold = threshold
                best_metrics = metrics
        except Exception as exc:
            tuning_rows.append({
                "candidate": candidate_name,
                "candidate_index": candidate_index,
                "params_json": json.dumps(params, sort_keys=True),
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "fit_seconds": time.perf_counter() - fit_start,
            })

    if best_model is None:
        raise RuntimeError(f"All {model_name} candidates failed")

    return SelectionResult(
        model=best_model,
        model_name=model_name,
        selected_candidate=str(best_candidate),
        selected_params=dict(best_params),
        threshold=float(best_threshold),
        validation_metrics=dict(best_metrics),
        tuning_rows=tuning_rows,
        warnings=captured_warnings,
        tuning_seconds=time.perf_counter() - start_all,
    )
