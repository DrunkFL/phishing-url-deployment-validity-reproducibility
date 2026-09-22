from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from run_shap_stability import (
    BACKGROUND_SIZE,
    DATASETS,
    PART2,
    PART3,
    PART4,
    ROOT,
    SCENARIOS,
    feature_names,
    load_features,
    load_internal,
    utc_now,
    write_json,
)
from shap_utils import (
    build_explainer,
    deterministic_stratified_sample,
    explain,
    global_feature_summary,
    ranking_comparison,
)


SIZES = [100, 500, 1000]


def main() -> None:
    names = feature_names()
    rows = []
    ranking_rows = []
    for dataset in DATASETS:
        features = load_features(dataset, names)
        for scenario in ("s0", "s3"):
            internal = load_internal(dataset, scenario, features)
            split = internal["split_r00"].astype(str)
            train = internal.loc[split == "train"]
            test = internal.loc[split == "test"]
            background = deterministic_stratified_sample(
                train, BACKGROUND_SIZE, f"benchmark:{dataset}:{scenario}:background"
            )
            cohorts = {
                size: deterministic_stratified_sample(
                    test, size, f"benchmark:{dataset}:{scenario}:cohort"
                )
                for size in SIZES
            }
            for model_name in ("lr", "rf", "xgb"):
                output_dir = ROOT / "benchmark" / dataset / scenario / model_name
                complete_path = output_dir / "complete.json"
                if complete_path.exists():
                    timing = pd.read_csv(output_dir / "timing.csv")
                    rows.extend(timing.to_dict("records"))
                    ranks = {
                        size: pd.read_csv(output_dir / f"global_summary_n{size}.csv")
                        for size in SIZES
                    }
                else:
                    model = joblib.load(PART4 / "models" / dataset / scenario / f"{model_name}.joblib")
                    X_background = background[names].to_numpy(dtype=np.float32)
                    bundle = build_explainer(model_name, model, X_background)
                    ranks = {}
                    timing_rows = []
                    for size in SIZES:
                        cohort = cohorts[size]
                        X = cohort[names].to_numpy(dtype=np.float32)
                        values, seconds, warning_rows = explain(bundle, X)
                        summary = global_feature_summary(X, values, names)
                        output_dir.mkdir(parents=True, exist_ok=True)
                        summary.to_csv(output_dir / f"global_summary_n{size}.csv", index=False)
                        ranks[size] = summary
                        timing_rows.append({
                            "dataset": dataset,
                            "scenario": scenario,
                            "model": model_name,
                            "n_samples": size,
                            "setup_seconds": bundle.setup_seconds,
                            "explain_seconds": seconds,
                            "seconds_per_sample": seconds / size,
                            "warning_count": len(warning_rows),
                        })
                    timing = pd.DataFrame(timing_rows)
                    timing.to_csv(output_dir / "timing.csv", index=False)
                    write_json(complete_path, {"completed_utc": utc_now()})
                    rows.extend(timing_rows)
                    print(f"DONE benchmark {dataset} {scenario} {model_name}", flush=True)

                for left_size, right_size in ((100, 500), (100, 1000), (500, 1000)):
                    ranking_rows.append({
                        "dataset": dataset,
                        "scenario": scenario,
                        "model": model_name,
                        "left_n": left_size,
                        "right_n": right_size,
                        **ranking_comparison(ranks[left_size], ranks[right_size]),
                    })

    results = ROOT / "results"
    results.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["dataset", "scenario", "model", "n_samples"]).to_csv(
        results / "shap_scaling_benchmark.csv", index=False
    )
    pd.DataFrame(ranking_rows).sort_values(
        ["dataset", "scenario", "model", "left_n", "right_n"]
    ).to_csv(results / "cohort_size_rank_convergence.csv", index=False)
    print("SHAP scaling benchmark completed.", flush=True)


if __name__ == "__main__":
    main()
