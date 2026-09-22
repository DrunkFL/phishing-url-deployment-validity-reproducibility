from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_shap_stability import ROOT
from shap_utils import ranking_comparison


RESULTS = ROOT / "results"


def read_global_summaries() -> pd.DataFrame:
    frames = []
    for metadata_path in ROOT.glob("runs/*/*/*/*/*/*/metadata.json"):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        summary = pd.read_csv(metadata_path.parent / "global_summary.csv")
        cohort = metadata_path.parent.name
        for key in (
            "estimand", "dataset", "scenario", "model", "run_id",
            "partition_repetition", "model_seed", "model_output_scale", "perturbation",
        ):
            summary[key] = metadata[key]
        summary["cohort"] = cohort
        summary["target_dataset"] = metadata.get("target_dataset", "")
        frames.append(summary)
    if not frames:
        raise ValueError("No SHAP summaries found")
    return pd.concat(frames, ignore_index=True)


def compare_run_frames(
    left: pd.DataFrame,
    right: pd.DataFrame,
    base: dict,
) -> dict:
    return {**base, **ranking_comparison(left, right)}


def pairwise_stability(all_summaries: pd.DataFrame) -> pd.DataFrame:
    rows = []
    seed = all_summaries.loc[
        (all_summaries["estimand"] == "seed") & (all_summaries["cohort"] == "internal")
    ]
    for keys, group in seed.groupby(["dataset", "scenario", "model"], observed=True):
        runs = {run: frame for run, frame in group.groupby("run_id", observed=True)}
        for left_id, right_id in itertools.combinations(sorted(runs), 2):
            rows.append(compare_run_frames(runs[left_id], runs[right_id], {
                "estimand": "seed", "dataset": keys[0], "scenario": keys[1],
                "model": keys[2], "left_run": left_id, "right_run": right_id,
                "comparison": f"{left_id}_vs_{right_id}",
            }))

    partition = all_summaries.loc[
        (all_summaries["estimand"] == "partition") & (all_summaries["cohort"] == "internal")
    ]
    for keys, group in partition.groupby(["dataset", "scenario", "model"], observed=True):
        runs = {run: frame for run, frame in group.groupby("run_id", observed=True)}
        for left_id, right_id in itertools.combinations(sorted(runs), 2):
            rows.append(compare_run_frames(runs[left_id], runs[right_id], {
                "estimand": "partition", "dataset": keys[0], "scenario": keys[1],
                "model": keys[2], "left_run": left_id, "right_run": right_id,
                "comparison": f"{left_id}_vs_{right_id}",
            }))

    for keys, group in partition.groupby(["dataset", "model", "run_id"], observed=True):
        scenarios = {scenario: frame for scenario, frame in group.groupby("scenario", observed=True)}
        for left_scenario, right_scenario in itertools.combinations(sorted(scenarios), 2):
            rows.append(compare_run_frames(scenarios[left_scenario], scenarios[right_scenario], {
                "estimand": "regime", "dataset": keys[0],
                "scenario": f"{left_scenario}_vs_{right_scenario}", "model": keys[1],
                "left_run": keys[2], "right_run": keys[2],
                "comparison": f"{left_scenario}_vs_{right_scenario}",
            }))

    source = all_summaries.loc[
        (all_summaries["estimand"] == "partition") & (all_summaries["scenario"] == "s3")
    ]
    for keys, group in source.groupby(["dataset", "model", "run_id"], observed=True):
        cohorts = {cohort: frame for cohort, frame in group.groupby("cohort", observed=True)}
        rows.append(compare_run_frames(cohorts["internal"], cohorts["external_common"], {
            "estimand": "source", "dataset": keys[0], "scenario": "s3_internal_vs_external",
            "model": keys[1], "left_run": keys[2], "right_run": keys[2],
            "comparison": "internal_vs_external_common",
        }))
    return pd.DataFrame(rows)


def stability_summary(pairwise: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "top10_jaccard", "top15_jaccard", "top20_jaccard", "rank_spearman",
        "direction_agreement_all", "direction_agreement_top20_union",
    ]
    rows = []
    for keys, group in pairwise.groupby(["estimand", "dataset", "scenario", "model"], observed=True):
        for metric in metric_columns:
            values = group[metric].to_numpy(dtype=float)
            rows.append({
                "estimand": keys[0], "dataset": keys[1], "scenario": keys[2], "model": keys[3],
                "metric": metric, "n_comparisons": len(values),
                "mean": values.mean(), "std": values.std(ddof=1) if len(values) > 1 else 0.0,
                "min": values.min(), "max": values.max(),
            })
    return pd.DataFrame(rows)


def feature_stability(all_summaries: pd.DataFrame) -> pd.DataFrame:
    eligible = all_summaries.loc[
        all_summaries["estimand"].isin(["seed", "partition"])
        & (all_summaries["cohort"] == "internal")
    ].copy()
    rows = []
    for keys, group in eligible.groupby(
        ["estimand", "dataset", "scenario", "model", "feature"], observed=True
    ):
        n_runs = group["run_id"].nunique()
        eligible_signs = group.loc[group["direction_sign"] != 0, "direction_sign"]
        signs = eligible_signs.value_counts()
        if signs.empty:
            majority_sign = 0
            direction_consistency = float("nan")
        else:
            majority_sign = int(signs.index[0])
            direction_consistency = float(signs.iloc[0] / len(eligible_signs))
        rows.append({
            "estimand": keys[0], "dataset": keys[1], "scenario": keys[2],
            "model": keys[3], "feature": keys[4], "n_runs": n_runs,
            "top10_count": int((group["rank"] <= 10).sum()),
            "top10_frequency": float((group["rank"] <= 10).mean()),
            "top15_count": int((group["rank"] <= 15).sum()),
            "top15_frequency": float((group["rank"] <= 15).mean()),
            "top20_count": int((group["rank"] <= 20).sum()),
            "top20_frequency": float((group["rank"] <= 20).mean()),
            "mean_rank": float(group["rank"].mean()),
            "rank_std": float(group["rank"].std(ddof=1)),
            "mean_abs_shap": float(group["mean_abs_shap"].mean()),
            "direction_eligible_runs": int(len(eligible_signs)),
            "direction_excluded_runs": int(len(group) - len(eligible_signs)),
            "direction_majority_sign": majority_sign,
            "direction_consistency": direction_consistency,
        })
    return pd.DataFrame(rows)


def timing_summary(all_summaries: pd.DataFrame) -> pd.DataFrame:
    rows = []
    seen = set()
    for metadata_path in ROOT.glob("runs/*/*/*/*/*/*/metadata.json"):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        key = (metadata["estimand"], metadata["dataset"], metadata["scenario"], metadata["model"], metadata["run_id"], metadata_path.parent.name)
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "estimand": key[0], "dataset": key[1], "scenario": key[2],
            "model": key[3], "run_id": key[4], "cohort": key[5],
            "cohort_size": metadata["cohort_size"],
            "fit_seconds": metadata["fit_seconds"],
            "setup_seconds": metadata["explainer_setup_seconds"],
            "explain_seconds": metadata["explain_seconds"],
            "seconds_per_sample": metadata["seconds_per_sample"],
        })
    return pd.DataFrame(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    all_summaries = read_global_summaries()
    pairwise = pairwise_stability(all_summaries)
    summary = stability_summary(pairwise)
    features = feature_stability(all_summaries)
    timing = timing_summary(all_summaries)
    outputs = {
        "all_global_feature_summaries.csv": all_summaries,
        "pairwise_stability.csv": pairwise,
        "stability_summary.csv": summary,
        "feature_stability.csv": features,
        "shap_run_timing.csv": timing,
    }
    for filename, frame in outputs.items():
        frame.to_csv(RESULTS / filename, index=False)
    manifest_paths = [RESULTS / filename for filename in outputs]
    for optional in ("shap_scaling_benchmark.csv", "cohort_size_rank_convergence.csv"):
        path = RESULTS / optional
        if path.exists():
            manifest_paths.append(path)
    pd.DataFrame([
        {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in manifest_paths
    ]).to_csv(RESULTS / "result_manifest.csv", index=False)
    print(f"Global feature rows: {len(all_summaries)}")
    print(f"Pairwise stability rows: {len(pairwise)}")


if __name__ == "__main__":
    main()
