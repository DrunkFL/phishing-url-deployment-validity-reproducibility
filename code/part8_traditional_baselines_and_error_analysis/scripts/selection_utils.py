from __future__ import annotations

import numpy as np


def exact_top_k(names: list[str], scores: np.ndarray, k: int) -> list[str]:
    if len(names) != len(scores):
        raise ValueError("Feature names and scores must have equal length")
    if not 0 < k <= len(names):
        raise ValueError("k must be between 1 and the number of features")
    if not np.isfinite(np.asarray(scores, dtype=float)).all():
        raise ValueError("Feature scores must be finite")
    ranked = sorted(zip(names, scores), key=lambda item: (-float(item[1]), item[0]))
    return [name for name, _ in ranked[:k]]


def rank_scores(names: list[str], scores: np.ndarray) -> dict[str, int]:
    ordered = exact_top_k(names, scores, len(names))
    return {name: index + 1 for index, name in enumerate(ordered)}
