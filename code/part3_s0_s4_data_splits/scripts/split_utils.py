from __future__ import annotations

import hashlib

import numpy as np
from sklearn.model_selection import train_test_split


SEEDS = tuple(range(20260902, 20260912))
SPLIT_NAMES = ("train", "validation", "test")
TARGET_PROPORTIONS = {"train": 0.70, "validation": 0.15, "test": 0.15}


def stable_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def row_stratified_assignment(labels: np.ndarray, seed: int) -> np.ndarray:
    indices = np.arange(len(labels))
    train, remainder = train_test_split(
        indices,
        test_size=0.30,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )
    validation, test = train_test_split(
        remainder,
        test_size=0.50,
        random_state=seed,
        shuffle=True,
        stratify=labels[remainder],
    )
    assignment = np.empty(len(labels), dtype="U10")
    assignment[train] = "train"
    assignment[validation] = "validation"
    assignment[test] = "test"
    return assignment


def grouped_stratified_assignment(
    labels: np.ndarray, groups: np.ndarray, seed: int
) -> np.ndarray:
    if len(labels) != len(groups):
        raise ValueError("labels and groups must have equal length")
    unique_groups, inverse_groups = np.unique(groups, return_inverse=True)
    if len(unique_groups) < 3:
        raise ValueError("at least three groups are required")

    benign = np.bincount(
        inverse_groups, weights=(labels == 0).astype(np.int64)
    ).astype(np.int64)
    phishing = np.bincount(
        inverse_groups, weights=(labels == 1).astype(np.int64)
    ).astype(np.int64)
    sizes = benign + phishing
    proportions = np.array([0.70, 0.15, 0.15], dtype=float)
    target_benign = proportions * benign.sum()
    target_phishing = proportions * phishing.sum()
    target_rows = proportions * sizes.sum()
    assigned_benign = np.zeros(3, dtype=np.int64)
    assigned_phishing = np.zeros(3, dtype=np.int64)
    assigned_rows = np.zeros(3, dtype=np.int64)
    group_assignments = np.full(len(unique_groups), -1, dtype=np.int8)

    rng = np.random.default_rng(seed)
    randomized = rng.permutation(len(unique_groups))
    order = randomized[np.argsort(-sizes[randomized], kind="stable")]

    def score(split: int, benign_rows: int, phishing_rows: int) -> float:
        new_benign = assigned_benign[split] + benign_rows
        new_phishing = assigned_phishing[split] + phishing_rows
        new_rows = assigned_rows[split] + benign_rows + phishing_rows
        return (
            ((new_benign - target_benign[split]) / target_benign[split]) ** 2
            + ((new_phishing - target_phishing[split]) / target_phishing[split]) ** 2
            + ((new_rows - target_rows[split]) / target_rows[split]) ** 2
        )

    def current_score(split: int) -> float:
        return (
            ((assigned_benign[split] - target_benign[split]) / target_benign[split])
            ** 2
            + (
                (assigned_phishing[split] - target_phishing[split])
                / target_phishing[split]
            )
            ** 2
            + ((assigned_rows[split] - target_rows[split]) / target_rows[split]) ** 2
        )

    for group_index in order:
        candidate_order = rng.permutation(3)
        deltas = [
            (
                score(split, int(benign[group_index]), int(phishing[group_index]))
                - current_score(split),
                int(split),
            )
            for split in candidate_order
        ]
        _, selected = min(deltas, key=lambda item: item[0])
        group_assignments[group_index] = selected
        assigned_benign[selected] += benign[group_index]
        assigned_phishing[selected] += phishing[group_index]
        assigned_rows[selected] += sizes[group_index]

    if np.any(group_assignments < 0):
        raise RuntimeError("some groups were not assigned")
    return np.asarray(SPLIT_NAMES)[group_assignments[inverse_groups]]


def assert_group_disjoint(assignments: np.ndarray, groups: np.ndarray) -> None:
    group_to_split: dict[str, str] = {}
    for split, group in zip(assignments, groups):
        previous = group_to_split.setdefault(str(group), str(split))
        if previous != split:
            raise AssertionError(f"group occurs in multiple splits: {group}")
