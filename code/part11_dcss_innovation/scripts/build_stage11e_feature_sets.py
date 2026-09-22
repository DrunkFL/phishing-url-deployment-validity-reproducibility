from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ablation_utils import (
    ABLATIONS,
    add_ablation_scores,
    feature_list_hash,
    random_feature_sets,
    select_top_features,
)
from train_dcss_main_comparison import feature_names


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RANDOM_SEEDS = tuple(range(20261501, 20261531))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_ablations() -> pd.DataFrame:
    scores = add_ablation_scores(pd.read_csv(RESULTS / "dcss_scores.csv"))
    rows = []
    keys = ["dataset", "repetition", "model"]
    for key, group in scores.groupby(keys, sort=True):
        payload = {
            "dataset": key[0],
            "repetition": key[1],
            "model": key[2],
            "k": 15,
            "selection_scope": "source_outer_train_only",
            "stage": "11E",
        }
        for method in ABLATIONS:
            selected = select_top_features(group, method)
            payload[method] = selected
            payload[f"{method}_sha256"] = feature_list_hash(selected)
            lookup = group.set_index("feature")
            for rank, feature in enumerate(selected, start=1):
                source = lookup.loc[feature]
                rows.append(
                    {
                        "dataset": key[0],
                        "repetition": key[1],
                        "model": key[2],
                        "method": method,
                        "selection_rank": rank,
                        "feature": feature,
                        "score": float(source[method]),
                        "mean_normalized_importance": float(source["mean_normalized_importance"]),
                        "mean_rank": float(source["mean_rank"]),
                        "rank_dispersion": float(source["rank_dispersion"]),
                        "direction_consistency": float(source["direction_consistency"]),
                        "top15_frequency": float(source["top15_frequency"]),
                    }
                )
        existing = set(group.loc[group["selected_15"].astype(bool), "feature"])
        if set(payload["dcss_full"]) != existing:
            raise ValueError(f"Full DCSS reproduction failed: {key}")
        path = ROOT / "feature_sets" / "ablation" / key[0] / "s3" / key[1] / key[2] / "feature_sets.json"
        write_json(path, payload)
    return pd.DataFrame(rows).sort_values(keys + ["method", "selection_rank"])


def build_random() -> pd.DataFrame:
    names = feature_names()
    sets = random_feature_sets(names, RANDOM_SEEDS)
    payload = {
        "stage": "11E",
        "feature_universe": names,
        "k": 15,
        "generator": "numpy.default_rng(seed).choice(replace=False)",
        "sets": {str(seed): values for seed, values in sets.items()},
        "sha256": {str(seed): feature_list_hash(values) for seed, values in sets.items()},
    }
    write_json(ROOT / "feature_sets" / "random" / "random_feature_sets.json", payload)
    rows = []
    for seed, values in sets.items():
        for rank, feature in enumerate(values, start=1):
            rows.append(
                {
                    "random_seed": seed,
                    "feature_order": rank,
                    "feature": feature,
                    "feature_set_sha256": feature_list_hash(values),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    ablations = build_ablations()
    random_sets = build_random()
    ablations.to_csv(RESULTS / "stage11e_ablation_feature_sets.csv", index=False)
    random_sets.to_csv(RESULTS / "stage11e_random_feature_sets.csv", index=False)
    print({"ablation_rows": len(ablations), "random_rows": len(random_sets)})


if __name__ == "__main__":
    main()
