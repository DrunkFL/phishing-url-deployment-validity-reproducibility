from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT.parent
PART6 = EXPERIMENT_ROOT / "part6_stable_feature_selection"
RESULTS = ROOT / "results"
FEATURE_COUNTS = (10, 15, 20)


def jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right)


def main() -> None:
    feature_sets = pd.read_csv(RESULTS / "dcss_feature_sets.csv")
    scores = pd.read_csv(RESULTS / "dcss_scores.csv")
    timing = pd.read_csv(RESULTS / "dcss_timing.csv")

    fold_balance_rows = []
    for path in sorted(
        (ROOT / "data" / "processed" / "dcss_fold_assignments").glob("*/*.parquet")
    ):
        frame = pd.read_parquet(path)
        for held_fold, group in frame.groupby("held_fold", sort=True):
            fold_balance_rows.append(
                {
                    "dataset": path.parent.name,
                    "repetition": path.stem,
                    "held_fold": held_fold,
                    "rows": len(group),
                    "domains": group["registrable_domain_sha256"].nunique(),
                    "benign": int((group["label"] == 0).sum()),
                    "phishing": int((group["label"] == 1).sum()),
                    "phishing_prevalence": float(group["label"].mean()),
                }
            )

    selection_rows = []
    pair_rows = []
    overlap_rows = []
    for keys, group in feature_sets.groupby(["dataset", "model"], sort=True):
        for k in FEATURE_COUNTS:
            sets = {
                row.repetition: set(json.loads(getattr(row, f"f_dcss_{k}_json")))
                for row in group.itertuples(index=False)
            }
            all_features = sorted(set().union(*sets.values()))
            for feature in all_features:
                count = sum(feature in selected for selected in sets.values())
                selection_rows.append(
                    {
                        "dataset": keys[0],
                        "model": keys[1],
                        "k": k,
                        "feature": feature,
                        "selection_count": count,
                        "selection_frequency": count / len(sets),
                    }
                )
            for left, right in combinations(sorted(sets), 2):
                pair_rows.append(
                    {
                        "dataset": keys[0],
                        "model": keys[1],
                        "k": k,
                        "left_repetition": left,
                        "right_repetition": right,
                        "jaccard": jaccard(sets[left], sets[right]),
                    }
                )

    for row in feature_sets.itertuples(index=False):
        baseline_path = (
            PART6
            / "feature_sets"
            / row.dataset
            / "s3"
            / row.repetition
            / row.model
            / "feature_sets.json"
        )
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        dcss = set(json.loads(row.f_dcss_15_json))
        for comparator in ("f_single", "f_stable"):
            reference = set(baseline[comparator])
            overlap_rows.append(
                {
                    "dataset": row.dataset,
                    "repetition": row.repetition,
                    "model": row.model,
                    "comparator": comparator,
                    "n_dcss": len(dcss),
                    "n_comparator": len(reference),
                    "intersection": len(dcss & reference),
                    "jaccard": jaccard(dcss, reference),
                }
            )

    selection = pd.DataFrame(selection_rows).sort_values(
        ["dataset", "model", "k", "selection_count", "feature"],
        ascending=[True, True, True, False, True],
    )
    pairwise = pd.DataFrame(pair_rows)
    pair_summary = (
        pairwise.groupby(["dataset", "model", "k"], sort=True)["jaccard"]
        .agg(pair_count="size", mean="mean", std="std", min="min", max="max")
        .reset_index()
    )
    overlap = pd.DataFrame(overlap_rows)
    component = (
        scores.groupby(["dataset", "model", "feature"], sort=True)
        .agg(
            mean_normalized_importance=("mean_normalized_importance", "mean"),
            mean_rank=("mean_rank", "mean"),
            mean_rank_dispersion=("rank_dispersion", "mean"),
            mean_direction_consistency=("direction_consistency", "mean"),
            mean_top15_frequency=("top15_frequency", "mean"),
            mean_dcss_15=("dcss_15", "mean"),
            selected_15_count=("selected_15", "sum"),
        )
        .reset_index()
        .sort_values(
            ["dataset", "model", "selected_15_count", "mean_dcss_15", "feature"],
            ascending=[True, True, False, False, True],
        )
    )
    timing_summary = (
        timing.groupby(["dataset", "model"], sort=True)
        .agg(
            subruns=("held_fold", "size"),
            fit_seconds_sum=("fit_seconds", "sum"),
            shap_seconds_sum=("shap_seconds", "sum"),
            setup_seconds_sum=("explainer_setup_seconds", "sum"),
            fit_seconds_mean=("fit_seconds", "mean"),
            shap_seconds_mean=("shap_seconds", "mean"),
            warnings=("warning_count", "sum"),
        )
        .reset_index()
    )
    fold_balance = pd.DataFrame(fold_balance_rows)

    selection.to_csv(RESULTS / "dcss_selection_frequency.csv", index=False)
    pairwise.to_csv(RESULTS / "dcss_repeat_pairwise_jaccard.csv", index=False)
    pair_summary.to_csv(RESULTS / "dcss_repeat_jaccard_summary.csv", index=False)
    overlap.to_csv(RESULTS / "dcss_comparator_overlap.csv", index=False)
    component.to_csv(RESULTS / "dcss_component_summary.csv", index=False)
    timing_summary.to_csv(RESULTS / "dcss_timing_summary.csv", index=False)
    fold_balance.to_csv(RESULTS / "dcss_fold_balance.csv", index=False)
    print(
        {
            "selection_frequency_rows": len(selection),
            "pairwise_jaccard_rows": len(pairwise),
            "comparator_overlap_rows": len(overlap),
            "component_rows": len(component),
            "fold_balance_rows": len(fold_balance),
            "total_fit_seconds": float(timing["fit_seconds"].sum()),
            "total_shap_seconds": float(timing["shap_seconds"].sum()),
        }
    )


if __name__ == "__main__":
    main()
