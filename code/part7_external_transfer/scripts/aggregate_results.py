from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance, wilcoxon

from transfer_utils import jaccard, jensen_shannon_divergence


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT.parent
PART4 = EXPERIMENT_ROOT / "part4_baseline_models"
PART6 = EXPERIMENT_ROOT / "part6_stable_feature_selection"
RESULTS = ROOT / "results"
METRICS = ["macro_f1", "roc_auc", "pr_auc", "recall", "fpr", "balanced_accuracy"]
FEATURE_SETS = ["f_all", "f_single", "f_stable"]
SOURCES = ["iscx_url2016_binary", "phiusiil"]
MODELS = ["lr", "rf", "xgb"]
REPETITIONS = [f"r{index:02d}" for index in range(10)]


def stable_seed(parts: tuple[str, ...]) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def bootstrap_interval(values: np.ndarray, key: tuple[str, ...]) -> tuple[float, float]:
    rng = np.random.default_rng(stable_seed(key))
    draws = rng.choice(values, size=(20000, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def feature_names() -> list[str]:
    dictionary = pd.read_csv(
        EXPERIMENT_ROOT / "part2_url_normalization_leakage_audit"
        / "results" / "feature_dictionary.csv"
    ).sort_values("feature_order")
    return dictionary["feature"].tolist()


def collect_external_metrics() -> pd.DataFrame:
    all_names = feature_names()
    baseline = pd.read_csv(PART4 / "results" / "s4_metrics.csv")
    baseline["feature_set"] = "f_all"
    baseline["n_features"] = 35
    baseline["feature_names_json"] = json.dumps(all_names)
    baseline["source_validation_threshold"] = baseline["threshold"]

    reduced_rows = []
    for path in ROOT.glob("runs/*/r??/*/f_*/external_metrics.json"):
        reduced_rows.extend(json.loads(path.read_text(encoding="utf-8")))
    reduced = pd.DataFrame(reduced_rows)
    external = pd.concat([baseline, reduced], ignore_index=True, sort=False)
    return external.sort_values(
        ["source_dataset", "repetition", "model", "feature_set", "cohort"]
    ).reset_index(drop=True)


def performance_summary(external: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = ["source_dataset", "target_dataset", "cohort", "model", "feature_set"]
    for keys, group in external.groupby(group_columns, sort=True):
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            low, high = bootstrap_interval(values, tuple(map(str, keys)) + (metric,))
            rows.append({
                **dict(zip(group_columns, keys)),
                "metric": metric,
                "n_runs": len(values),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
                "median": float(np.median(values)),
                "min": float(values.min()),
                "max": float(values.max()),
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
            })
    return pd.DataFrame(rows)


def paired_feature_set_deltas(external: pd.DataFrame) -> pd.DataFrame:
    rows = []
    index_columns = [
        "source_dataset", "target_dataset", "repetition", "model", "cohort"
    ]
    for keys, group in external.groupby(index_columns, sort=True):
        by_set = group.set_index("feature_set")
        for left, right in (
            ("f_single", "f_all"),
            ("f_stable", "f_all"),
            ("f_stable", "f_single"),
        ):
            for metric in METRICS:
                rows.append({
                    **dict(zip(index_columns, keys)),
                    "comparison": f"{left}_minus_{right}",
                    "metric": metric,
                    "delta": float(by_set.loc[left, metric] - by_set.loc[right, metric]),
                })
    return pd.DataFrame(rows)


def paired_summary(deltas: pd.DataFrame) -> pd.DataFrame:
    rows = []
    columns = [
        "source_dataset", "target_dataset", "cohort", "model", "comparison", "metric"
    ]
    for keys, group in deltas.groupby(columns, sort=True):
        values = group["delta"].to_numpy(dtype=float)
        low, high = bootstrap_interval(values, tuple(map(str, keys)))
        nonzero = values[values != 0]
        p_value = (
            float(wilcoxon(values, zero_method="wilcox", alternative="two-sided").pvalue)
            if len(nonzero)
            else 1.0
        )
        rows.append({
            **dict(zip(columns, keys)),
            "n_pairs": len(values),
            "mean_delta": float(values.mean()),
            "median_delta": float(np.median(values)),
            "bootstrap_ci_low": low,
            "bootstrap_ci_high": high,
            "wilcoxon_p_unadjusted": p_value,
            "positive_pairs": int((values > 0).sum()),
            "negative_pairs": int((values < 0).sum()),
            "zero_pairs": int((values == 0).sum()),
        })
    return pd.DataFrame(rows)


def transfer_gaps(external: pd.DataFrame) -> pd.DataFrame:
    internal = pd.read_csv(PART6 / "results" / "internal_metrics.csv")
    primary = external.loc[external["cohort"] == "primary_domain_filtered"]
    rows = []
    for row in primary.itertuples(index=False):
        source = internal.loc[
            (internal["dataset"] == row.source_dataset)
            & (internal["repetition"] == row.repetition)
            & (internal["model"] == row.model)
            & (internal["feature_set"] == row.feature_set)
        ]
        if len(source) != 1:
            raise ValueError(
                f"Missing internal metric for {row.source_dataset}/{row.repetition}/"
                f"{row.model}/{row.feature_set}"
            )
        source_row = source.iloc[0]
        for metric in METRICS:
            rows.append({
                "source_dataset": row.source_dataset,
                "target_dataset": row.target_dataset,
                "repetition": row.repetition,
                "model": row.model,
                "feature_set": row.feature_set,
                "metric": metric,
                "internal_value": float(source_row[metric]),
                "external_value": float(getattr(row, metric)),
                "external_minus_internal": float(getattr(row, metric) - source_row[metric]),
            })
    return pd.DataFrame(rows)


def prediction_paths(
    source: str,
    repetition: str,
    model: str,
    feature_set: str,
) -> tuple[Path, Path]:
    if feature_set == "f_all":
        internal = (
            PART4 / "predictions" / "internal" / source / "s3"
            / repetition / f"{model}.parquet"
        )
        external = (
            PART4 / "predictions" / "s4" / source / repetition
            / model / "s4_predictions.parquet"
        )
    else:
        internal = (
            PART6 / "predictions" / source / "s3" / repetition
            / model / f"{feature_set}.parquet"
        )
        external = (
            ROOT / "predictions" / source / repetition
            / model / f"{feature_set}.parquet"
        )
    return internal, external


def probability_shift() -> pd.DataFrame:
    rows = []
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    for source in SOURCES:
        target = "phiusiil" if source == "iscx_url2016_binary" else "iscx_url2016_binary"
        for repetition in REPETITIONS:
            for model in MODELS:
                for feature_set in FEATURE_SETS:
                    internal_path, external_path = prediction_paths(
                        source, repetition, model, feature_set
                    )
                    internal = pd.read_parquet(
                        internal_path, columns=["label", "probability_phishing", "prediction"]
                    )
                    external = pd.read_parquet(
                        external_path,
                        columns=[
                            "label", "probability_phishing", "prediction",
                            "included_in_primary",
                        ],
                    )
                    external = external.loc[external["included_in_primary"]]
                    for label in (0, 1):
                        left_frame = internal.loc[internal["label"] == label]
                        right_frame = external.loc[external["label"] == label]
                        left = left_frame["probability_phishing"].to_numpy(dtype=float)
                        right = right_frame["probability_phishing"].to_numpy(dtype=float)
                        left_q = np.quantile(left, quantiles)
                        right_q = np.quantile(right, quantiles)
                        row = {
                            "source_dataset": source,
                            "target_dataset": target,
                            "repetition": repetition,
                            "model": model,
                            "feature_set": feature_set,
                            "label": label,
                            "internal_n": len(left),
                            "external_n": len(right),
                            "internal_mean_probability": float(left.mean()),
                            "external_mean_probability": float(right.mean()),
                            "mean_probability_shift": float(right.mean() - left.mean()),
                            "internal_predicted_positive_rate": float(left_frame["prediction"].mean()),
                            "external_predicted_positive_rate": float(right_frame["prediction"].mean()),
                            "wasserstein_distance": float(wasserstein_distance(left, right)),
                            "jensen_shannon_divergence": jensen_shannon_divergence(left, right),
                        }
                        for name, value in zip(("q05", "q25", "q50", "q75", "q95"), left_q):
                            row[f"internal_{name}"] = float(value)
                        for name, value in zip(("q05", "q25", "q50", "q75", "q95"), right_q):
                            row[f"external_{name}"] = float(value)
                        rows.append(row)
    return pd.DataFrame(rows)


def cross_source_feature_overlap() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for repetition in REPETITIONS:
        for model in MODELS:
            left = json.loads(
                (
                    PART6 / "feature_sets" / "iscx_url2016_binary" / "s3"
                    / repetition / model / "feature_sets.json"
                ).read_text(encoding="utf-8")
            )
            right = json.loads(
                (
                    PART6 / "feature_sets" / "phiusiil" / "s3"
                    / repetition / model / "feature_sets.json"
                ).read_text(encoding="utf-8")
            )
            for feature_set in ("f_single", "f_stable"):
                rows.append({
                    "repetition": repetition,
                    "model": model,
                    "feature_set": feature_set,
                    "iscx_n_features": len(left[feature_set]),
                    "phiusiil_n_features": len(right[feature_set]),
                    "intersection_size": len(set(left[feature_set]) & set(right[feature_set])),
                    "jaccard": jaccard(left[feature_set], right[feature_set]),
                })
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["model", "feature_set"])["jaccard"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
        .rename(columns={"count": "n_pairs"})
    )
    return detail, summary


def write_manifest(paths: list[Path]) -> None:
    rows = []
    for path in paths:
        rows.append({
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    pd.DataFrame(rows).to_csv(RESULTS / "result_manifest.csv", index=False)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    external = collect_external_metrics()
    summary = performance_summary(external)
    deltas = paired_feature_set_deltas(external)
    delta_summary = paired_summary(deltas)
    gaps = transfer_gaps(external)
    shift = probability_shift()
    overlap, overlap_summary = cross_source_feature_overlap()

    outputs = {
        "external_metrics.csv": external,
        "external_performance_summary.csv": summary,
        "paired_external_deltas.csv": deltas,
        "paired_external_summary.csv": delta_summary,
        "internal_to_external_gaps.csv": gaps,
        "probability_shift_by_class.csv": shift,
        "cross_source_feature_overlap.csv": overlap,
        "cross_source_feature_overlap_summary.csv": overlap_summary,
    }
    for name, frame in outputs.items():
        frame.to_csv(RESULTS / name, index=False)
    write_manifest([RESULTS / name for name in outputs])
    print(
        f"external_metrics={len(external)} paired_deltas={len(deltas)} "
        f"transfer_gaps={len(gaps)} probability_shift={len(shift)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
