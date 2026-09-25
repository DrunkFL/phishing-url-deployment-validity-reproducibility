from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT.parent
PART3 = EXPERIMENT_ROOT / "part3_s0_s4_data_splits"
PART4 = EXPERIMENT_ROOT / "part4_baseline_models"
PART6 = EXPERIMENT_ROOT / "part6_stable_feature_selection"
PART7 = EXPERIMENT_ROOT / "part7_external_transfer"
PART8 = EXPERIMENT_ROOT / "part8_traditional_baselines_and_error_analysis"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

DATASETS = ["iscx_url2016_binary", "phiusiil"]
MODELS = ["lr", "rf", "xgb"]
FEATURE_SETS = ["f_all", "f_single", "f_stable", "f_mi", "f_permutation"]
REPETITIONS = [f"r{i:02d}" for i in range(10)]
TARGET = {"iscx_url2016_binary": "phiusiil", "phiusiil": "iscx_url2016_binary"}
DISPLAY = {
    "iscx_url2016_binary": "ISCX-URL2016", "phiusiil": "PhiUSIIL",
    "lr": "LR", "rf": "RF", "xgb": "XGBoost",
    "f_all": "F-All", "f_single": "F-Single", "f_stable": "F-Stable",
    "f_mi": "F-MI", "f_permutation": "F-Permutation",
}


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    formatted = frame.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(lambda value: f"{value:.{digits}f}")
    header = "| " + " | ".join(formatted.columns) + " |"
    separator = "|" + "|".join(["---"] * len(formatted.columns)) + "|"
    body = ["| " + " | ".join(map(str, row)) + " |" for row in formatted.to_numpy()]
    return "\n".join([header, separator, *body])


def prediction_path(scope: str, dataset: str, repetition: str, model: str, feature_set: str) -> Path:
    if scope == "internal":
        if feature_set == "f_all":
            return PART4 / "predictions" / "internal" / dataset / "s3" / repetition / f"{model}.parquet"
        if feature_set in {"f_single", "f_stable"}:
            return PART6 / "predictions" / dataset / "s3" / repetition / model / f"{feature_set}.parquet"
        return PART8 / "predictions" / "internal" / dataset / "s3" / repetition / model / f"{feature_set}.parquet"
    if feature_set == "f_all":
        return PART4 / "predictions" / "s4" / dataset / repetition / model / "s4_predictions.parquet"
    if feature_set in {"f_single", "f_stable"}:
        return PART7 / "predictions" / dataset / repetition / model / f"{feature_set}.parquet"
    return PART8 / "predictions" / "external" / dataset / repetition / model / f"{feature_set}.parquet"


def assignment_map(scope: str, dataset: str) -> pd.DataFrame:
    if scope == "internal":
        filename = f"{dataset}_master_s3_domain_assignments.parquet"
        return pd.read_parquet(
            PART3 / "data" / "assignments" / filename,
            columns=["sample_id", "registrable_domain_sha256"],
        )
    target = TARGET[dataset]
    filename = f"s4_{dataset}_to_{target}_master_assignments.parquet"
    frame = pd.read_parquet(
        PART3 / "data" / "assignments" / filename,
        columns=["sample_id", "origin", "registrable_domain_sha256"],
    )
    return frame.loc[frame["origin"] == "target", ["sample_id", "registrable_domain_sha256"]]


def weighted_confusion_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    sizes = frame.groupby("registrable_domain_sha256", observed=True)["sample_id"].transform("size")
    weights = 1.0 / sizes.to_numpy(dtype=float)
    y = frame["label"].to_numpy(dtype=np.int8)
    pred = frame["prediction"].to_numpy(dtype=np.int8)
    tn = weights[(y == 0) & (pred == 0)].sum()
    fp = weights[(y == 0) & (pred == 1)].sum()
    fn = weights[(y == 1) & (pred == 0)].sum()
    tp = weights[(y == 1) & (pred == 1)].sum()
    recall = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    fpr = fp / (fp + tn) if fp + tn else np.nan
    f1_positive = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    f1_negative = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    return {
        "n_rows": len(frame), "n_domains": frame["registrable_domain_sha256"].nunique(),
        "domain_equal_macro_f1": (f1_positive + f1_negative) / 2,
        "domain_equal_balanced_accuracy": (recall + specificity) / 2,
        "domain_equal_recall": recall, "domain_equal_fpr": fpr,
        "weighted_tn": tn, "weighted_fp": fp, "weighted_fn": fn, "weighted_tp": tp,
    }


def exact_oracle_threshold(y: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    order = np.argsort(-probability, kind="stable")
    p = probability[order]
    labels = y[order]
    tp = np.cumsum(labels == 1)
    fp = np.cumsum(labels == 0)
    total_positive = tp[-1]
    total_negative = fp[-1]
    boundaries = np.flatnonzero(np.r_[p[:-1] != p[1:], True])
    tp_b = tp[boundaries].astype(float)
    fp_b = fp[boundaries].astype(float)
    fn_b = total_positive - tp_b
    tn_b = total_negative - fp_b
    f1_pos = np.divide(2 * tp_b, 2 * tp_b + fp_b + fn_b, out=np.zeros_like(tp_b), where=(2 * tp_b + fp_b + fn_b) > 0)
    f1_neg = np.divide(2 * tn_b, 2 * tn_b + fp_b + fn_b, out=np.zeros_like(tn_b), where=(2 * tn_b + fp_b + fn_b) > 0)
    macro = (f1_pos + f1_neg) / 2
    no_positive_macro = total_negative / (2 * total_negative + total_positive)
    best_index = int(np.argmax(macro))
    if no_positive_macro > macro[best_index]:
        return float(np.nextafter(p[0], np.inf)), float(no_positive_macro)
    return float(p[boundaries[best_index]]), float(macro[best_index])


def ece_and_bins(y: np.ndarray, probability: np.ndarray, n_bins: int = 15) -> tuple[float, list[dict[str, float | int]]]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.minimum(np.searchsorted(edges, probability, side="right") - 1, n_bins - 1)
    rows = []
    ece = 0.0
    for index in range(n_bins):
        mask = bins == index
        if mask.any():
            mean_probability = float(probability[mask].mean())
            observed_rate = float(y[mask].mean())
            count = int(mask.sum())
            ece += count / len(y) * abs(mean_probability - observed_rate)
        else:
            mean_probability, observed_rate, count = np.nan, np.nan, 0
        rows.append({
            "bin": index + 1, "lower": edges[index], "upper": edges[index + 1], "n": count,
            "mean_probability": mean_probability, "observed_phishing_rate": observed_rate,
        })
    return float(ece), rows


def macro_f1(y: np.ndarray, prediction: np.ndarray) -> float:
    tn = ((y == 0) & (prediction == 0)).sum()
    fp = ((y == 0) & (prediction == 1)).sum()
    fn = ((y == 1) & (prediction == 0)).sum()
    tp = ((y == 1) & (prediction == 1)).sum()
    f1_pos = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    f1_neg = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    return float((f1_pos + f1_neg) / 2)


def bootstrap_mean(values: np.ndarray, seed: int, n_bootstrap: int = 20000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(n_bootstrap, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))


def deduplicated_analysis() -> None:
    dedup = pd.read_csv(RESULTS / "deduplicated_metrics.csv")
    master = pd.read_csv(PART4 / "results" / "internal_metrics.csv")
    master = master.loc[master["scenario_key"].isin(["s0", "s3"])]
    keys = ["dataset", "scenario_key", "repetition", "model"]
    paired = dedup.merge(master, on=keys, suffixes=("_deduplicated", "_master"), validate="one_to_one")
    for metric in ["macro_f1", "roc_auc", "pr_auc", "recall", "fpr"]:
        paired[f"delta_{metric}_deduplicated_minus_master"] = paired[f"{metric}_deduplicated"] - paired[f"{metric}_master"]
    paired.to_csv(RESULTS / "deduplicated_paired_deltas.csv", index=False)
    rows = []
    for keys_value, group in paired.groupby(["dataset", "scenario_key", "model"], observed=True, sort=True):
        for metric in ["macro_f1", "roc_auc", "pr_auc", "recall", "fpr"]:
            values = group[f"delta_{metric}_deduplicated_minus_master"].to_numpy(float)
            low, high = bootstrap_mean(values, 20261100 + len(rows))
            rows.append({
                "dataset": keys_value[0], "scenario_key": keys_value[1], "model": keys_value[2], "metric": metric,
                "n_pairs": len(values), "mean_delta": values.mean(), "std_delta": values.std(ddof=1),
                "ci95_lower": low, "ci95_upper": high,
            })
    pd.DataFrame(rows).to_csv(RESULTS / "deduplicated_paired_summary.csv", index=False)


def domain_equal_analysis() -> pd.DataFrame:
    rows = []
    maps = {(scope, dataset): assignment_map(scope, dataset) for scope in ("internal", "external") for dataset in DATASETS}
    for scope in ("internal", "external"):
        for dataset in DATASETS:
            domain_map = maps[(scope, dataset)]
            for repetition in REPETITIONS:
                for model in MODELS:
                    for feature_set in FEATURE_SETS:
                        frame = pd.read_parquet(prediction_path(scope, dataset, repetition, model, feature_set))
                        if scope == "external":
                            frame = frame.loc[frame["included_in_primary"]].copy()
                        frame = frame.merge(domain_map, on="sample_id", how="left", validate="one_to_one")
                        if frame["registrable_domain_sha256"].isna().any():
                            raise ValueError(f"Missing domain mapping: {scope}/{dataset}/{repetition}/{model}/{feature_set}")
                        rows.append({
                            "scope": scope, "source_dataset": dataset,
                            "target_dataset": TARGET[dataset] if scope == "external" else dataset,
                            "repetition": repetition, "model": model, "feature_set": feature_set,
                            "row_macro_f1": macro_f1(frame["label"].to_numpy(), frame["prediction"].to_numpy()),
                            **weighted_confusion_metrics(frame),
                        })
    output = pd.DataFrame(rows)
    output.to_csv(RESULTS / "domain_equal_metrics.csv", index=False)
    summary = output.groupby(["scope", "source_dataset", "target_dataset", "model", "feature_set"], observed=True).agg(
        n_runs=("domain_equal_macro_f1", "size"), row_macro_f1_mean=("row_macro_f1", "mean"),
        domain_equal_macro_f1_mean=("domain_equal_macro_f1", "mean"),
        domain_equal_macro_f1_std=("domain_equal_macro_f1", "std"),
        domain_equal_balanced_accuracy_mean=("domain_equal_balanced_accuracy", "mean"),
        domain_equal_recall_mean=("domain_equal_recall", "mean"), domain_equal_fpr_mean=("domain_equal_fpr", "mean"),
        n_domains_mean=("n_domains", "mean"),
    ).reset_index()
    summary["domain_equal_minus_row_macro_f1"] = summary["domain_equal_macro_f1_mean"] - summary["row_macro_f1_mean"]
    summary.to_csv(RESULTS / "domain_equal_summary.csv", index=False)
    return output


def cluster_bootstrap() -> None:
    rng = np.random.default_rng(20261120)
    rows = []
    for source in DATASETS:
        domain_map = assignment_map("external", source)
        for feature_set in ("f_all", "f_stable"):
            frame = pd.read_parquet(prediction_path("external", source, "r00", "xgb", feature_set))
            frame = frame.loc[frame["included_in_primary"]].merge(domain_map, on="sample_id", validate="one_to_one")
            y = frame["label"].to_numpy(np.int8)
            pred = frame["prediction"].to_numpy(np.int8)
            frame = frame.assign(
                tn=((y == 0) & (pred == 0)).astype(float), fp=((y == 0) & (pred == 1)).astype(float),
                fn=((y == 1) & (pred == 0)).astype(float), tp=((y == 1) & (pred == 1)).astype(float),
            )
            rates = frame.groupby("registrable_domain_sha256", observed=True)[["tn", "fp", "fn", "tp"]].mean().to_numpy()
            n_domains = len(rates)
            boot_metrics = []
            for start in range(0, 2000, 20):
                count = min(20, 2000 - start)
                index = rng.integers(0, n_domains, size=(count, n_domains))
                confusion = rates[index].mean(axis=1)
                tn, fp, fn, tp = confusion.T
                f1_pos = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) > 0)
                f1_neg = np.divide(2 * tn, 2 * tn + fp + fn, out=np.zeros_like(tn), where=(2 * tn + fp + fn) > 0)
                boot_metrics.extend((f1_pos + f1_neg) / 2)
            boot = np.asarray(boot_metrics)
            point = weighted_confusion_metrics(frame)
            rows.append({
                "source_dataset": source, "target_dataset": TARGET[source], "model": "xgb", "repetition": "r00",
                "feature_set": feature_set, "n_domains": n_domains, "n_bootstrap": len(boot),
                "domain_equal_macro_f1": point["domain_equal_macro_f1"],
                "ci95_lower": np.quantile(boot, 0.025), "ci95_upper": np.quantile(boot, 0.975),
            })
    pd.DataFrame(rows).to_csv(RESULTS / "representative_domain_cluster_bootstrap.csv", index=False)


def calibration_analysis() -> pd.DataFrame:
    metric_source = pd.read_csv(PART8 / "results" / "external_metrics_all_five.csv")
    metric_source = metric_source.loc[metric_source["cohort"] == "primary_domain_filtered"]
    thresholds = metric_source.set_index(["source_dataset", "repetition", "model", "feature_set"])["threshold"]
    rows, bin_rows = [], []
    for source in DATASETS:
        for repetition in REPETITIONS:
            for model in MODELS:
                for feature_set in FEATURE_SETS:
                    frame = pd.read_parquet(prediction_path("external", source, repetition, model, feature_set))
                    frame = frame.loc[frame["included_in_primary"]]
                    y = frame["label"].to_numpy(np.int8)
                    p = frame["probability_phishing"].to_numpy(float)
                    threshold = float(thresholds.loc[(source, repetition, model, feature_set)])
                    source_macro = macro_f1(y, p >= threshold)
                    oracle_threshold, oracle_macro = exact_oracle_threshold(y, p)
                    clipped = np.clip(p, 1e-15, 1 - 1e-15)
                    ece, bins = ece_and_bins(y, p)
                    base = {
                        "source_dataset": source, "target_dataset": TARGET[source], "repetition": repetition,
                        "model": model, "feature_set": feature_set, "n_samples": len(y),
                    }
                    rows.append({
                        **base, "source_threshold": threshold, "source_threshold_macro_f1": source_macro,
                        "oracle_target_threshold": oracle_threshold, "oracle_target_macro_f1": oracle_macro,
                        "threshold_regret": oracle_macro - source_macro,
                        "brier_score": np.mean((p - y) ** 2),
                        "log_loss": -np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)), "ece_15": ece,
                    })
                    bin_rows.extend([{**base, **row} for row in bins])
    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS / "external_calibration_metrics.csv", index=False)
    bins = pd.DataFrame(bin_rows)
    bins.to_csv(RESULTS / "external_reliability_bins.csv", index=False)
    summary = metrics.groupby(["source_dataset", "target_dataset", "model", "feature_set"], observed=True).agg(
        n_runs=("threshold_regret", "size"), brier_mean=("brier_score", "mean"), log_loss_mean=("log_loss", "mean"),
        ece_mean=("ece_15", "mean"), source_threshold_mean=("source_threshold", "mean"),
        oracle_threshold_mean=("oracle_target_threshold", "mean"), source_macro_f1_mean=("source_threshold_macro_f1", "mean"),
        oracle_macro_f1_mean=("oracle_target_macro_f1", "mean"), threshold_regret_mean=("threshold_regret", "mean"),
        threshold_regret_std=("threshold_regret", "std"),
    ).reset_index()
    summary.to_csv(RESULTS / "external_calibration_summary.csv", index=False)
    make_calibration_figures(metrics, bins)
    return metrics


def make_calibration_figures(metrics: pd.DataFrame, bins: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})
    colors = {"f_all": "#1f77b4", "f_single": "#ff7f0e", "f_stable": "#2ca02c", "f_mi": "#d62728", "f_permutation": "#9467bd"}
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True)
    for row, source in enumerate(DATASETS):
        for col, model in enumerate(MODELS):
            ax = axes[row, col]
            subset = bins.loc[(bins["source_dataset"] == source) & (bins["model"] == model)]
            for feature_set in FEATURE_SETS:
                curve = subset.loc[subset["feature_set"] == feature_set].groupby("bin", observed=True).agg(
                    predicted=("mean_probability", "mean"), observed=("observed_phishing_rate", "mean"), n=("n", "sum")
                ).reset_index()
                curve = curve.loc[curve["n"] > 0]
                ax.plot(curve["predicted"], curve["observed"], marker="o", markersize=3, linewidth=1.2,
                        color=colors[feature_set], label=DISPLAY[feature_set])
            ax.plot([0, 1], [0, 1], color="#555555", linestyle="--", linewidth=1)
            ax.set_title(f"{DISPLAY[source]} to {DISPLAY[TARGET[source]]}: {DISPLAY[model]}")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    for ax in axes[-1, :]: ax.set_xlabel("Mean predicted probability")
    for ax in axes[:, 0]: ax.set_ylabel("Observed phishing rate")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(FIGURES / "external_reliability_diagrams.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "external_reliability_diagrams.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.25), sharey=True)
    for ax, source in zip(axes, DATASETS):
        subset = metrics.loc[metrics["source_dataset"] == source]
        positions, values = [], []
        for model_index, model in enumerate(MODELS):
            for feature_index, feature_set in enumerate(FEATURE_SETS):
                positions.append(model_index * 6 + feature_index)
                values.append(subset.loc[(subset["model"] == model) & (subset["feature_set"] == feature_set), "threshold_regret"])
        bp = ax.boxplot(values, positions=positions, widths=0.7, patch_artist=True, showfliers=False)
        for patch, position in zip(bp["boxes"], positions): patch.set_facecolor(colors[FEATURE_SETS[position % 6]])
        ax.set_xticks([2, 8, 14]); ax.set_xticklabels([DISPLAY[model] for model in MODELS])
        ax.set_title(f"{DISPLAY[source]} to {DISPLAY[TARGET[source]]}", fontsize=9)
        ax.set_ylabel("Oracle minus source-threshold Macro-F1", fontsize=8)
        ax.tick_params(axis="both", labelsize=8)
    legend = [Patch(facecolor=colors[name], edgecolor="black", label=DISPLAY[name]) for name in FEATURE_SETS]
    fig.legend(handles=legend, loc="lower center", ncol=5, frameon=False, fontsize=7)
    fig.tight_layout(rect=[0, 0.14, 1, 1])
    fig.savefig(FIGURES / "external_threshold_regret.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIGURES / "external_threshold_regret.pdf", bbox_inches="tight")
    plt.close(fig)


def write_report() -> None:
    dedup = pd.read_csv(RESULTS / "deduplicated_paired_summary.csv")
    domain = pd.read_csv(RESULTS / "domain_equal_summary.csv")
    calibration = pd.read_csv(RESULTS / "external_calibration_summary.csv")
    cluster = pd.read_csv(RESULTS / "representative_domain_cluster_bootstrap.csv")
    d = dedup.loc[dedup["metric"] == "macro_f1"].copy()
    e = domain.loc[domain["scope"] == "external"].copy()
    text = [
        "# Part 10 Robustness Report", "", "## Fully deduplicated sensitivity", "",
        "Mean paired Macro-F1 changes (deduplicated minus master) are reported below. The comparison reuses matched dataset, regime, repetition, model, features, and selection rules.", "",
        markdown_table(d[["dataset", "scenario_key", "model", "mean_delta", "ci95_lower", "ci95_upper"]]),
        "", "## Registrable-domain-equal evaluation", "",
        "Each domain contributes total weight one. External results are:", "",
        markdown_table(e[["source_dataset", "target_dataset", "model", "feature_set", "row_macro_f1_mean", "domain_equal_macro_f1_mean", "domain_equal_minus_row_macro_f1"]]),
        "", "Representative domain-cluster bootstrap (XGBoost r00):", "",
        markdown_table(cluster),
        "", "## External calibration and threshold transfer", "",
        markdown_table(calibration[["source_dataset", "target_dataset", "model", "feature_set", "brier_mean", "ece_mean", "source_macro_f1_mean", "oracle_macro_f1_mean", "threshold_regret_mean"]]),
        "", "The oracle threshold is diagnostic only. It uses target labels after prediction and must not be interpreted as a deployable external result.", "",
    ]
    (RESULTS / "part10_robustness_report.md").write_text("\n".join(text), encoding="utf-8")


def manifest() -> None:
    rows = []
    for path in sorted([*RESULTS.glob("*"), *FIGURES.glob("*")]):
        if path.is_file() and path.name != "result_manifest.csv":
            rows.append({
                "path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
    pd.DataFrame(rows).to_csv(RESULTS / "result_manifest.csv", index=False)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    deduplicated_analysis()
    domain_equal_analysis()
    cluster_bootstrap()
    calibration_analysis()
    write_report()
    manifest()
    print("Part 10 robustness analyses complete.")


if __name__ == "__main__":
    main()
