from __future__ import annotations

import math

import numpy as np
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


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, n_bins: int = 15
) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=float)
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    if len(labels) != len(probabilities) or len(labels) == 0:
        raise ValueError("Labels and probabilities must have equal nonzero length")
    if not np.isfinite(probabilities).all() or not (
        (probabilities >= 0) & (probabilities <= 1)
    ).all():
        raise ValueError("Probabilities must be finite and in [0, 1]")
    bin_indices = np.minimum((probabilities * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for bin_index in range(n_bins):
        mask = bin_indices == bin_index
        if not mask.any():
            continue
        ece += float(mask.mean()) * abs(
            float(probabilities[mask].mean()) - float(labels[mask].mean())
        )
    return float(ece)


def extended_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    clipped = np.clip(probabilities, 1e-15, 1 - 1e-15)
    return {
        "n_samples": int(len(labels)),
        "n_benign": int((labels == 0).sum()),
        "n_phishing": int((labels == 1).sum()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if fp + tn else math.nan,
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "inverted_roc_auc": float(roc_auc_score(labels, 1.0 - probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "brier_score": float(np.mean((probabilities - labels) ** 2)),
        "log_loss": float(
            -np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))
        ),
        "ece_15": expected_calibration_error(labels, probabilities, 15),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
