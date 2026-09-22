from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def auc_influence(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, np.ndarray]:
    y = np.asarray(y_true, dtype=np.int8)
    score = np.asarray(scores, dtype=np.float64)
    positives = score[y == 1]
    negatives = score[y == 0]
    if len(positives) == 0 or len(negatives) == 0:
        raise ValueError("AUC influence requires both classes")

    sorted_neg = np.sort(negatives)
    lower_neg = np.searchsorted(sorted_neg, positives, side="left")
    upper_neg = np.searchsorted(sorted_neg, positives, side="right")
    v10 = (lower_neg + 0.5 * (upper_neg - lower_neg)) / len(negatives)

    sorted_pos = np.sort(positives)
    lower_pos = np.searchsorted(sorted_pos, negatives, side="left")
    upper_pos = np.searchsorted(sorted_pos, negatives, side="right")
    v01 = (len(positives) - upper_pos + 0.5 * (upper_pos - lower_pos)) / len(positives)

    auc = float(v10.mean())
    influence = np.empty(len(y), dtype=np.float64)
    influence[y == 1] = (v10 - auc) / len(positives)
    influence[y == 0] = (v01 - auc) / len(negatives)
    if not np.isclose(auc, roc_auc_score(y, score), atol=1e-12):
        raise ValueError("AUC influence point estimate mismatch")
    if not np.isclose(influence.sum(), 0.0, atol=1e-10):
        raise ValueError("AUC influence does not center to zero")
    return auc, influence


def benjamini_hochberg(values: np.ndarray) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0.0, 1.0)
    return output
