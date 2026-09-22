from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT.parent
PART3 = EXPERIMENT_ROOT / "part3_s0_s4_data_splits"
PART4 = EXPERIMENT_ROOT / "part4_baseline_models"
PART6 = EXPERIMENT_ROOT / "part6_stable_feature_selection"
PART7 = EXPERIMENT_ROOT / "part7_external_transfer"
RESULTS = ROOT / "results"

SOURCES = ["iscx_url2016_binary", "phiusiil"]
MODELS = ["lr", "rf", "xgb"]
REPETITIONS = [f"r{index:02d}" for index in range(10)]
FEATURE_SETS = ["f_all", "f_single", "f_stable", "f_mi", "f_permutation"]
TRADITIONAL_SETS = ["f_mi", "f_permutation"]
METRICS = ["macro_f1", "roc_auc", "pr_auc", "recall", "fpr", "balanced_accuracy"]
COMPARISONS = [
    ("f_stable", "f_mi"),
    ("f_stable", "f_permutation"),
    ("f_mi", "f_all"),
    ("f_permutation", "f_all"),
]


def stable_seed(parts: tuple[str, ...]) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap_interval(values: np.ndarray, key: tuple[str, ...]) -> tuple[float, float]:
    rng = np.random.default_rng(stable_seed(key))
    draws = rng.choice(values, size=(20000, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def adjust_bh(values: pd.Series) -> pd.Series:
    array = values.to_numpy(dtype=float)
    order = np.argsort(array)
    ranked = array[order]
    adjusted = ranked * len(array) / np.arange(1, len(array) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1.0)
    return pd.Series(restored, index=values.index)


def collect_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    internal = pd.read_csv(PART6 / "results" / "internal_metrics.csv")
    internal_rows = []
    external_rows = []
    for path in sorted(ROOT.glob("runs/*/s3/r??/*/f_*/internal_metrics.json")):
        internal_rows.append(json.loads(path.read_text(encoding="utf-8")))
        external_rows.extend(
            json.loads((path.parent / "external_metrics.json").read_text(encoding="utf-8"))
        )
    internal = pd.concat([internal, pd.DataFrame(internal_rows)], ignore_index=True, sort=False)
    external = pd.concat(
        [pd.read_csv(PART7 / "results" / "external_metrics.csv"), pd.DataFrame(external_rows)],
        ignore_index=True,
        sort=False,
    )
    internal = internal.sort_values(
        ["dataset", "repetition", "model", "feature_set"]
    ).reset_index(drop=True)
    external = external.sort_values(
        ["source_dataset", "repetition", "model", "feature_set", "cohort"]
    ).reset_index(drop=True)
    if len(internal) != 300 or len(external) != 600:
        raise ValueError(f"Unexpected metric counts: internal={len(internal)}, external={len(external)}")
    return internal, external


def performance_summary(metrics: pd.DataFrame, scope: str) -> pd.DataFrame:
    group_columns = (
        ["dataset", "model", "feature_set"]
        if scope == "internal"
        else ["source_dataset", "target_dataset", "cohort", "model", "feature_set"]
    )
    rows = []
    for keys, group in metrics.groupby(group_columns, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            low, high = bootstrap_interval(values, (scope,) + tuple(map(str, keys)) + (metric,))
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


def paired_deltas(metrics: pd.DataFrame, scope: str) -> pd.DataFrame:
    index_columns = (
        ["dataset", "repetition", "model"]
        if scope == "internal"
        else ["source_dataset", "target_dataset", "repetition", "model", "cohort"]
    )
    rows = []
    for keys, group in metrics.groupby(index_columns, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        by_set = group.set_index("feature_set")
        for left, right in COMPARISONS:
            for metric in METRICS:
                rows.append({
                    **dict(zip(index_columns, keys)),
                    "comparison": f"{left}_minus_{right}",
                    "metric": metric,
                    "delta": float(by_set.loc[left, metric] - by_set.loc[right, metric]),
                })
    return pd.DataFrame(rows)


def paired_summary(deltas: pd.DataFrame, scope: str) -> pd.DataFrame:
    columns = (
        ["dataset", "model", "comparison", "metric"]
        if scope == "internal"
        else ["source_dataset", "target_dataset", "cohort", "model", "comparison", "metric"]
    )
    rows = []
    for keys, group in deltas.groupby(columns, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        values = group["delta"].to_numpy(dtype=float)
        low, high = bootstrap_interval(values, (scope, "paired") + tuple(map(str, keys)))
        p_value = (
            float(wilcoxon(values, zero_method="wilcox", alternative="two-sided").pvalue)
            if np.any(values != 0)
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
    frame = pd.DataFrame(rows)
    frame["wilcoxon_p_bh"] = adjust_bh(frame["wilcoxon_p_unadjusted"])
    return frame


def load_feature_sets(dataset: str, repetition: str, model: str) -> dict[str, set[str]]:
    shap_sets = json.loads(
        (
            PART6 / "feature_sets" / dataset / "s3" / repetition / model / "feature_sets.json"
        ).read_text(encoding="utf-8")
    )
    traditional = json.loads(
        (
            ROOT / "feature_sets" / dataset / "s3" / repetition / model
            / "traditional_feature_sets.json"
        ).read_text(encoding="utf-8")
    )
    return {
        "f_single": set(shap_sets["f_single"]),
        "f_stable": set(shap_sets["f_stable"]),
        "f_mi": set(traditional["f_mi"]),
        "f_permutation": set(traditional["f_permutation"]),
    }


def feature_agreement() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outer_rows = []
    overlap_rows = []
    stored: dict[tuple[str, str, str, str], set[str]] = {}
    for dataset in SOURCES:
        for repetition in REPETITIONS:
            for model in MODELS:
                sets = load_feature_sets(dataset, repetition, model)
                for feature_set, values in sets.items():
                    stored[(dataset, repetition, model, feature_set)] = values
                for traditional in TRADITIONAL_SETS:
                    left, right = sets["f_stable"], sets[traditional]
                    overlap_rows.append({
                        "dataset": dataset,
                        "repetition": repetition,
                        "model": model,
                        "comparison": f"f_stable_vs_{traditional}",
                        "intersection_size": len(left & right),
                        "union_size": len(left | right),
                        "jaccard": len(left & right) / len(left | right),
                    })
    for dataset in SOURCES:
        for model in MODELS:
            for feature_set in TRADITIONAL_SETS:
                for left_rep, right_rep in combinations(REPETITIONS, 2):
                    left = stored[(dataset, left_rep, model, feature_set)]
                    right = stored[(dataset, right_rep, model, feature_set)]
                    outer_rows.append({
                        "dataset": dataset,
                        "model": model,
                        "feature_set": feature_set,
                        "left_repetition": left_rep,
                        "right_repetition": right_rep,
                        "jaccard": len(left & right) / len(left | right),
                    })
    cross_rows = []
    for repetition in REPETITIONS:
        for model in MODELS:
            for feature_set in TRADITIONAL_SETS:
                left = stored[(SOURCES[0], repetition, model, feature_set)]
                right = stored[(SOURCES[1], repetition, model, feature_set)]
                cross_rows.append({
                    "repetition": repetition,
                    "model": model,
                    "feature_set": feature_set,
                    "intersection_size": len(left & right),
                    "union_size": len(left | right),
                    "jaccard": len(left & right) / len(left | right),
                })
    return pd.DataFrame(outer_rows), pd.DataFrame(cross_rows), pd.DataFrame(overlap_rows)


def prediction_path(source: str, repetition: str, model: str, feature_set: str) -> Path:
    if feature_set == "f_all":
        return PART4 / "predictions" / "s4" / source / repetition / model / "s4_predictions.parquet"
    if feature_set in {"f_single", "f_stable"}:
        return PART7 / "predictions" / source / repetition / model / f"{feature_set}.parquet"
    return ROOT / "predictions" / "external" / source / repetition / model / f"{feature_set}.parquet"


def target_domains(source: str) -> pd.DataFrame:
    filename = (
        "s4_iscx_url2016_binary_to_phiusiil_master_assignments.parquet"
        if source == "iscx_url2016_binary"
        else "s4_phiusiil_to_iscx_url2016_binary_master_assignments.parquet"
    )
    frame = pd.read_parquet(
        PART3 / "data" / "assignments" / filename,
        columns=["sample_id", "origin", "registrable_domain_sha256"],
    )
    frame = frame.loc[frame["origin"] == "target", ["sample_id", "registrable_domain_sha256"]]
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate target sample IDs for {source}")
    return frame


def error_analysis() -> tuple[pd.DataFrame, pd.DataFrame]:
    disagreement_rows = []
    concentration_rows = []
    for source in SOURCES:
        domains = target_domains(source)
        for repetition in REPETITIONS:
            for model in MODELS:
                predictions = {}
                for feature_set in FEATURE_SETS:
                    frame = pd.read_parquet(
                        prediction_path(source, repetition, model, feature_set),
                        columns=["sample_id", "label", "prediction", "included_in_primary"],
                    )
                    frame = frame.loc[frame["included_in_primary"]].reset_index(drop=True)
                    if frame["sample_id"].duplicated().any():
                        raise ValueError(f"Duplicate prediction IDs: {source}/{repetition}/{model}/{feature_set}")
                    predictions[feature_set] = frame
                reference = predictions["f_stable"][["sample_id", "label"]]
                for feature_set, frame in predictions.items():
                    if not reference.equals(frame[["sample_id", "label"]]):
                        raise ValueError(
                            f"Prediction alignment failed: {source}/{repetition}/{model}/{feature_set}"
                        )
                stable_correct = predictions["f_stable"]["prediction"].eq(reference["label"])
                for baseline in TRADITIONAL_SETS:
                    baseline_correct = predictions[baseline]["prediction"].eq(reference["label"])
                    for label_scope in ["all", "benign", "phishing"]:
                        mask = pd.Series(True, index=reference.index)
                        if label_scope == "benign":
                            mask = reference["label"].eq(0)
                        elif label_scope == "phishing":
                            mask = reference["label"].eq(1)
                        sc = stable_correct[mask]
                        bc = baseline_correct[mask]
                        corrected = int((sc & ~bc).sum())
                        worsened = int((~sc & bc).sum())
                        disagreement_rows.append({
                            "source_dataset": source,
                            "target_dataset": "phiusiil" if source == SOURCES[0] else SOURCES[0],
                            "repetition": repetition,
                            "model": model,
                            "comparison": f"f_stable_vs_{baseline}",
                            "label_scope": label_scope,
                            "n_samples": int(mask.sum()),
                            "both_correct": int((sc & bc).sum()),
                            "stable_correct_baseline_wrong": corrected,
                            "stable_wrong_baseline_correct": worsened,
                            "both_wrong": int((~sc & ~bc).sum()),
                            "stable_net_corrected": corrected - worsened,
                        })
                for feature_set, frame in predictions.items():
                    audit = frame.merge(domains, on="sample_id", how="left", validate="one_to_one")
                    if audit["registrable_domain_sha256"].isna().any():
                        raise ValueError(f"Missing target domains: {source}/{repetition}/{model}")
                    false_positives = audit.loc[(audit["label"] == 0) & (audit["prediction"] == 1)]
                    counts = false_positives["registrable_domain_sha256"].value_counts()
                    n_fp = int(len(false_positives))
                    shares = counts / n_fp if n_fp else counts.astype(float)
                    concentration_rows.append({
                        "source_dataset": source,
                        "target_dataset": "phiusiil" if source == SOURCES[0] else SOURCES[0],
                        "repetition": repetition,
                        "model": model,
                        "feature_set": feature_set,
                        "n_benign": int((audit["label"] == 0).sum()),
                        "n_false_positives": n_fp,
                        "n_false_positive_domains": int(len(counts)),
                        "top1_domain_share": float(shares.iloc[:1].sum()) if n_fp else 0.0,
                        "top5_domain_share": float(shares.iloc[:5].sum()) if n_fp else 0.0,
                        "top10_domain_share": float(shares.iloc[:10].sum()) if n_fp else 0.0,
                        "domain_hhi": float(np.square(shares.to_numpy()).sum()) if n_fp else 0.0,
                    })
    return pd.DataFrame(disagreement_rows), pd.DataFrame(concentration_rows)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    internal, external = collect_metrics()
    internal_deltas = paired_deltas(internal, "internal")
    external_deltas = paired_deltas(external, "external")
    outer, cross, stable_overlap = feature_agreement()
    disagreement, concentration = error_analysis()
    outputs = {
        "internal_metrics_all_five.csv": internal,
        "external_metrics_all_five.csv": external,
        "internal_performance_summary.csv": performance_summary(internal, "internal"),
        "external_performance_summary.csv": performance_summary(external, "external"),
        "internal_paired_deltas.csv": internal_deltas,
        "external_paired_deltas.csv": external_deltas,
        "internal_paired_summary.csv": paired_summary(internal_deltas, "internal"),
        "external_paired_summary.csv": paired_summary(external_deltas, "external"),
        "traditional_outer_agreement.csv": outer,
        "traditional_cross_source_overlap.csv": cross,
        "stable_vs_traditional_overlap.csv": stable_overlap,
        "external_error_disagreement.csv": disagreement,
        "false_positive_domain_concentration.csv": concentration,
    }
    for name, frame in outputs.items():
        frame.to_csv(RESULTS / name, index=False)
    manifest = pd.DataFrame([
        {
            "path": str((RESULTS / name).relative_to(ROOT)),
            "bytes": (RESULTS / name).stat().st_size,
            "sha256": sha256_file(RESULTS / name),
        }
        for name in outputs
    ])
    manifest.to_csv(RESULTS / "result_manifest.csv", index=False)
    print(f"internal_metrics={len(internal)} external_metrics={len(external)}")
    print(f"internal_deltas={len(internal_deltas)} external_deltas={len(external_deltas)}")
    print(f"outer_agreement={len(outer)} cross_source={len(cross)} overlap={len(stable_overlap)}")
    print(f"error_disagreement={len(disagreement)} fp_concentration={len(concentration)}")


if __name__ == "__main__":
    main()
