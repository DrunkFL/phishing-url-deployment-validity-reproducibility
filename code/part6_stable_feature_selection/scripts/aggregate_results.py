from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT.parent
PART2 = EXPERIMENT_ROOT / "part2_url_normalization_leakage_audit"
PART4 = EXPERIMENT_ROOT / "part4_baseline_models"
PART5 = EXPERIMENT_ROOT / "part5_shap_stability"
RESULTS = ROOT / "results"
METRICS = ["macro_f1", "recall", "fpr", "roc_auc", "pr_auc", "balanced_accuracy"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_reduced_metrics() -> pd.DataFrame:
    rows = []
    for path in sorted(ROOT.glob("runs/*/s3/*/*/*/internal_metrics.json")):
        if (path.parent / "complete.json").exists():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    return pd.DataFrame(rows)


def read_f_all() -> pd.DataFrame:
    frame = pd.read_csv(PART4 / "results" / "internal_metrics.csv")
    frame = frame.loc[frame["scenario_key"] == "s3"].copy()
    names = pd.read_csv(PART2 / "results" / "feature_dictionary.csv").sort_values(
        "feature_order"
    )["feature"].tolist()
    frame["feature_set"] = "f_all"
    frame["n_features"] = len(names)
    frame["feature_names_json"] = json.dumps(names)
    timing = pd.read_csv(PART5 / "results" / "shap_run_timing.csv")
    timing = timing.loc[
        (timing["estimand"] == "partition")
        & (timing["scenario"] == "s3")
        & (timing["cohort"] == "internal"),
        ["dataset", "model", "run_id", "setup_seconds", "explain_seconds", "cohort_size"],
    ].rename(columns={
        "run_id": "repetition", "setup_seconds": "shap_setup_seconds",
        "explain_seconds": "shap_explain_seconds", "cohort_size": "shap_cohort_size",
    })
    frame = frame.merge(timing, on=["dataset", "model", "repetition"], validate="one_to_one")
    frame["shap_background_size"] = 200
    frame["shap_seconds_per_sample"] = frame["shap_explain_seconds"] / frame["shap_cohort_size"]
    frame["model_output_scale"] = frame["model"].map({
        "lr": "raw_log_odds", "rf": "positive_class_probability", "xgb": "raw_margin"
    })
    return frame


def bootstrap_interval(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(samples, [0.025, 0.975]))


def performance_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in metrics.groupby(["dataset", "model", "feature_set"], sort=True):
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            seed = int(hashlib.sha256(("|".join(keys) + metric).encode()).hexdigest()[:8], 16)
            low, high = bootstrap_interval(values, seed)
            rows.append({
                "dataset": keys[0], "model": keys[1], "feature_set": keys[2],
                "metric": metric, "n_runs": len(values), "mean": values.mean(),
                "std": values.std(ddof=1), "median": np.median(values),
                "min": values.min(), "max": values.max(),
                "bootstrap_ci_low": low, "bootstrap_ci_high": high,
            })
    return pd.DataFrame(rows)


def paired_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in metrics.groupby(["dataset", "model", "repetition"], sort=True):
        by_set = group.set_index("feature_set")
        for left, right in (("f_single", "f_all"), ("f_stable", "f_all"), ("f_stable", "f_single")):
            for metric in METRICS:
                rows.append({
                    "dataset": keys[0], "model": keys[1], "repetition": keys[2],
                    "comparison": f"{left}_minus_{right}", "metric": metric,
                    "delta": float(by_set.loc[left, metric] - by_set.loc[right, metric]),
                })
    return pd.DataFrame(rows)


def efficiency_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "n_features", "tuning_seconds", "inference_seconds_per_sample",
        "shap_explain_seconds", "shap_seconds_per_sample",
    ]
    rows = []
    for keys, group in metrics.groupby(["dataset", "model", "feature_set"], sort=True):
        for metric in columns:
            values = group[metric].to_numpy(dtype=float)
            rows.append({
                "dataset": keys[0], "model": keys[1], "feature_set": keys[2],
                "measure": metric, "mean": values.mean(), "std": values.std(ddof=1),
                "min": values.min(), "max": values.max(),
            })
    return pd.DataFrame(rows)


def feature_set_agreement() -> pd.DataFrame:
    sets = pd.read_csv(RESULTS / "feature_set_summary.csv")
    rows = []
    for keys, group in sets.groupby(["dataset", "model"], sort=True):
        for feature_set, column in (("f_single", "f_single_json"), ("f_stable", "f_stable_json")):
            values = {
                row.outer_repetition: set(json.loads(getattr(row, column)))
                for row in group.itertuples(index=False)
            }
            for left, right in combinations(sorted(values), 2):
                union = values[left] | values[right]
                rows.append({
                    "dataset": keys[0], "model": keys[1], "feature_set": feature_set,
                    "left_repetition": left, "right_repetition": right,
                    "jaccard": len(values[left] & values[right]) / len(union),
                })
    return pd.DataFrame(rows)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    reduced = read_reduced_metrics()
    if len(reduced) != 120:
        raise ValueError(f"Expected 120 reduced metrics, found {len(reduced)}")
    all_metrics = pd.concat([read_f_all(), reduced], ignore_index=True, sort=False)
    all_metrics = all_metrics.sort_values(
        ["dataset", "repetition", "model", "feature_set"]
    ).reset_index(drop=True)
    outputs = {
        "internal_metrics.csv": all_metrics,
        "internal_performance_summary.csv": performance_summary(all_metrics),
        "paired_metric_deltas.csv": paired_deltas(all_metrics),
        "efficiency_summary.csv": efficiency_summary(all_metrics),
        "feature_set_outer_agreement.csv": feature_set_agreement(),
    }
    for name, frame in outputs.items():
        frame.to_csv(RESULTS / name, index=False)
    manifest_inputs = [RESULTS / name for name in outputs]
    manifest_inputs.extend([
        RESULTS / "feature_set_summary.csv",
        RESULTS / "training_side_feature_stability.csv",
        RESULTS / "feature_selection_sensitivity.csv",
        RESULTS / "selection_timing.csv",
    ])
    pd.DataFrame([{
        "path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    } for path in manifest_inputs]).to_csv(RESULTS / "result_manifest.csv", index=False)
    print(f"internal_metrics={len(all_metrics)}")
    print(f"paired_deltas={len(outputs['paired_metric_deltas.csv'])}")


if __name__ == "__main__":
    main()

