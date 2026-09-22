from __future__ import annotations

import numpy as np


PROBABILITY_BINS = np.linspace(0.0, 1.0, 21)


def jensen_shannon_divergence(
    left: np.ndarray,
    right: np.ndarray,
    bins: np.ndarray = PROBABILITY_BINS,
) -> float:
    left_hist, _ = np.histogram(np.asarray(left, dtype=float), bins=bins)
    right_hist, _ = np.histogram(np.asarray(right, dtype=float), bins=bins)
    left_prob = left_hist.astype(float) + 1e-12
    right_prob = right_hist.astype(float) + 1e-12
    left_prob /= left_prob.sum()
    right_prob /= right_prob.sum()
    middle = 0.5 * (left_prob + right_prob)
    return float(
        0.5 * np.sum(left_prob * np.log2(left_prob / middle))
        + 0.5 * np.sum(right_prob * np.log2(right_prob / middle))
    )


def jaccard(left: list[str], right: list[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    return float(len(left_set & right_set) / len(union)) if union else 1.0
