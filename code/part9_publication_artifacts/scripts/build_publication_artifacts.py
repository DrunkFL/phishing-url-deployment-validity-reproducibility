from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT.parent
PART2 = EXPERIMENT_ROOT / "part2_url_normalization_leakage_audit"
PART4 = EXPERIMENT_ROOT / "part4_baseline_models"
PART5 = EXPERIMENT_ROOT / "part5_shap_stability"
PART6 = EXPERIMENT_ROOT / "part6_stable_feature_selection"
PART8 = EXPERIMENT_ROOT / "part8_traditional_baselines_and_error_analysis"
FIGURES = ROOT / "figures"
FIGURE_DATA = ROOT / "figure_data"
TABLES = ROOT / "tables"
RESULTS = ROOT / "results"

DATASET_LABELS = {
    "iscx_url2016_binary": "ISCX-URL2016",
    "phiusiil": "PhiUSIIL",
}
MODEL_LABELS = {"lr": "LR", "rf": "RF", "xgb": "XGBoost"}
METHOD_LABELS = {
    "f_all": "F-All",
    "f_single": "F-Single",
    "f_stable": "F-Stable",
    "f_mi": "F-MI",
    "f_permutation": "F-Permutation",
}
METHOD_ORDER = list(METHOD_LABELS)
MODEL_COLORS = {"lr": "#4C78A8", "rf": "#F58518", "xgb": "#54A24B"}
METHOD_COLORS = {
    "f_all": "#4C78A8",
    "f_single": "#F58518",
    "f_stable": "#54A24B",
    "f_mi": "#E45756",
    "f_permutation": "#B279A2",
}


def stable_seed(parts: tuple[str, ...]) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def bootstrap(values: np.ndarray, key: tuple[str, ...]) -> tuple[float, float]:
    rng = np.random.default_rng(stable_seed(key))
    draws = rng.choice(values, size=(20000, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def grouped_summary(frame: pd.DataFrame, groups: list[str], values: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(groups, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(groups, keys))
        for value in values:
            array = group[value].to_numpy(dtype=float)
            low, high = bootstrap(array, tuple(map(str, keys)) + (value,))
            row.update({
                f"{value}_mean": float(array.mean()),
                f"{value}_std": float(array.std(ddof=1)),
                f"{value}_ci_low": low,
                f"{value}_ci_high": high,
                f"{value}_n": len(array),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include=["float"]).columns:
        display[column] = display[column].map(lambda value: f"{value:.4f}")
    headers = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines) + "\n"


def export_table(name: str, frame: pd.DataFrame) -> None:
    frame.to_csv(TABLES / f"{name}.csv", index=False)
    (TABLES / f"{name}.md").write_text(markdown_table(frame), encoding="utf-8")


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def table_dataset_audit(audit: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "input_rows", "conflict_cleaned_master_rows", "benign_rows", "phishing_rows",
        "unique_normalized_urls", "unique_registrable_domains", "conflicting_rows",
    ]
    frame = audit.loc[audit["metric"].isin(metrics) & audit["dataset"].ne("cross_dataset")]
    frame = frame.pivot(index="dataset", columns="metric", values="value").reset_index()
    frame["dataset"] = frame["dataset"].replace({"ISCX-URL2016-binary": "ISCX-URL2016"})
    return frame[["dataset"] + metrics]


def make_tables(internal: pd.DataFrame, pairwise: pd.DataFrame, five_internal: pd.DataFrame,
                external: pd.DataFrame, fp: pd.DataFrame) -> None:
    audit = pd.read_csv(PART2 / "results" / "audit_summary.csv")
    export_table("table1_dataset_audit", table_dataset_audit(audit))

    table2 = internal.groupby(["dataset", "scenario", "model"], sort=True).agg(
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
        roc_auc_mean=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"),
        fpr_mean=("fpr", "mean"), fpr_std=("fpr", "std"), n_runs=("repetition", "size"),
    ).reset_index()
    table2["dataset"] = table2["dataset"].map(DATASET_LABELS)
    table2["model"] = table2["model"].map(MODEL_LABELS)
    export_table("table2_internal_regime_performance", table2)

    table3 = pairwise.groupby(["estimand", "dataset", "model"], sort=True).agg(
        top15_jaccard_mean=("top15_jaccard", "mean"),
        top15_jaccard_std=("top15_jaccard", "std"),
        rank_spearman_mean=("rank_spearman", "mean"),
        rank_spearman_std=("rank_spearman", "std"),
        n_pairs=("comparison", "size"),
    ).reset_index()
    table3["dataset"] = table3["dataset"].map(DATASET_LABELS)
    table3["model"] = table3["model"].map(MODEL_LABELS)
    export_table("table3_shap_stability", table3)

    table4 = five_internal.groupby(["dataset", "model", "feature_set"], sort=True).agg(
        n_features_mean=("n_features", "mean"), n_features_std=("n_features", "std"),
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
        roc_auc_mean=("roc_auc", "mean"), fpr_mean=("fpr", "mean"),
        n_runs=("repetition", "size"),
    ).reset_index()
    table4["dataset"] = table4["dataset"].map(DATASET_LABELS)
    table4["model"] = table4["model"].map(MODEL_LABELS)
    table4["feature_set"] = table4["feature_set"].map(METHOD_LABELS)
    export_table("table4_s3_feature_methods", table4)

    primary = external.loc[external["cohort"].eq("primary_domain_filtered")]
    table5 = primary.groupby(
        ["source_dataset", "target_dataset", "model", "feature_set"], sort=True
    ).agg(
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
        roc_auc_mean=("roc_auc", "mean"), pr_auc_mean=("pr_auc", "mean"),
        recall_mean=("recall", "mean"), fpr_mean=("fpr", "mean"),
        n_runs=("repetition", "size"),
    ).reset_index()
    table5["source_dataset"] = table5["source_dataset"].map(DATASET_LABELS)
    table5["target_dataset"] = table5["target_dataset"].map(DATASET_LABELS)
    table5["model"] = table5["model"].map(MODEL_LABELS)
    table5["feature_set"] = table5["feature_set"].map(METHOD_LABELS)
    export_table("table5_external_transfer", table5)

    paired = pd.read_csv(PART8 / "results" / "external_paired_summary.csv")
    table6 = paired.loc[
        paired["cohort"].eq("primary_domain_filtered") & paired["metric"].eq("macro_f1"),
        ["source_dataset", "target_dataset", "model", "comparison", "n_pairs",
         "mean_delta", "bootstrap_ci_low", "bootstrap_ci_high", "wilcoxon_p_bh",
         "positive_pairs", "negative_pairs", "zero_pairs"],
    ].copy()
    table6["source_dataset"] = table6["source_dataset"].map(DATASET_LABELS)
    table6["target_dataset"] = table6["target_dataset"].map(DATASET_LABELS)
    table6["model"] = table6["model"].map(MODEL_LABELS)
    export_table("table6_external_paired_macro_f1", table6)

    table7 = fp.groupby(["source_dataset", "target_dataset", "feature_set"], sort=True).agg(
        false_positives_mean=("n_false_positives", "mean"),
        false_positive_domains_mean=("n_false_positive_domains", "mean"),
        top10_domain_share_mean=("top10_domain_share", "mean"),
        domain_hhi_mean=("domain_hhi", "mean"),
        n_runs=("repetition", "size"),
    ).reset_index()
    table7["source_dataset"] = table7["source_dataset"].map(DATASET_LABELS)
    table7["target_dataset"] = table7["target_dataset"].map(DATASET_LABELS)
    table7["feature_set"] = table7["feature_set"].map(METHOD_LABELS)
    export_table("table7_false_positive_domains", table7)


def figure_internal_regimes(internal: pd.DataFrame) -> None:
    summary = grouped_summary(internal, ["dataset", "scenario_key", "model"], ["macro_f1"])
    summary.to_csv(FIGURE_DATA / "fig1_internal_regime_performance.csv", index=False)
    scenarios = ["s0", "s1", "s2", "s3"]
    labels = ["S0 Row", "S1 URL", "S2 Host", "S3 Domain"]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.15), sharey=True)
    for axis, dataset in zip(axes, DATASET_LABELS):
        subset = summary.loc[summary["dataset"].eq(dataset)]
        for model in MODEL_LABELS:
            values = subset.loc[subset["model"].eq(model)].set_index("scenario_key").loc[scenarios]
            means = values["macro_f1_mean"].to_numpy()
            errors = np.vstack([
                means - values["macro_f1_ci_low"].to_numpy(),
                values["macro_f1_ci_high"].to_numpy() - means,
            ])
            axis.errorbar(
                range(4), means, yerr=errors, color=MODEL_COLORS[model], marker="o",
                linewidth=1.7, markersize=4.5, capsize=2.5, label=MODEL_LABELS[model],
            )
        axis.set_title(DATASET_LABELS[dataset], fontsize=10, fontweight="bold")
        axis.set_xticks(range(4), labels, rotation=18, ha="right")
        axis.set_ylim(0.72, 1.005)
        axis.grid(axis="y", alpha=0.25, linewidth=0.6)
    axes[0].set_ylabel("Macro-F1")
    axes[1].legend(frameon=False, loc="lower left")
    fig.suptitle("Internal performance across deployment-conditioned splits", fontsize=11)
    fig.tight_layout()
    save_figure(fig, "fig1_internal_regime_performance")


def figure_shap_stability(pairwise: pd.DataFrame) -> None:
    order = ["seed", "partition", "regime", "source"]
    summary = pairwise.groupby(["estimand", "model"], sort=True).agg(
        top15_jaccard_mean=("top15_jaccard", "mean"),
        rank_spearman_mean=("rank_spearman", "mean"),
        n_pairs=("comparison", "size"),
    ).reset_index()
    summary.to_csv(FIGURE_DATA / "fig2_shap_stability_estimands.csv", index=False)
    long = pairwise.melt(
        id_vars=["estimand", "model"], value_vars=["top15_jaccard", "rank_spearman"],
        var_name="metric", value_name="agreement",
    )
    long["model_label"] = long["model"].map(MODEL_LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.35), sharey=True)
    for axis, metric, title in zip(
        axes, ["top15_jaccard", "rank_spearman"], ["Top-15 Jaccard", "Rank Spearman"],
    ):
        sns.boxplot(
            data=long.loc[long["metric"].eq(metric)], x="estimand", y="agreement",
            hue="model_label", order=order, hue_order=list(MODEL_LABELS.values()),
            palette=[MODEL_COLORS[key] for key in MODEL_LABELS], showfliers=False,
            linewidth=0.8, width=0.72, ax=axis,
        )
        axis.set_title(title, fontsize=10, fontweight="bold")
        axis.set_xlabel("")
        axis.set_xticks(range(4), ["Seed", "Partition", "Regime", "Source"])
        axis.set_ylim(0.15, 1.02)
        axis.grid(axis="y", alpha=0.25, linewidth=0.6)
        legend = axis.get_legend()
        if legend:
            legend.remove()
    axes[0].set_ylabel("Pairwise agreement")
    axes[1].set_ylabel("")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[1].legend(handles, labels, frameon=False, loc="lower left")
    fig.suptitle("SHAP agreement under controlled sources of variation", fontsize=11)
    fig.tight_layout()
    save_figure(fig, "fig2_shap_stability_estimands")


def figure_feature_reduction(five_internal: pd.DataFrame) -> None:
    reference = five_internal.loc[five_internal["feature_set"].eq("f_all"),
        ["dataset", "repetition", "model", "macro_f1"]].rename(columns={"macro_f1": "f_all"})
    delta = five_internal.merge(reference, on=["dataset", "repetition", "model"], validate="many_to_one")
    delta["macro_f1_delta"] = delta["macro_f1"] - delta["f_all"]
    count_summary = grouped_summary(five_internal, ["dataset", "feature_set"], ["n_features"])
    delta.to_csv(FIGURE_DATA / "fig3_feature_reduction_internal_delta.csv", index=False)
    count_summary.to_csv(FIGURE_DATA / "fig3_feature_count.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.45))
    x = np.arange(len(METHOD_ORDER))
    width = 0.36
    for offset, dataset in zip([-width / 2, width / 2], DATASET_LABELS):
        values = count_summary.loc[count_summary["dataset"].eq(dataset)].set_index("feature_set").loc[METHOD_ORDER]
        axes[0].bar(
            x + offset, values["n_features_mean"], width=width,
            label=DATASET_LABELS[dataset], alpha=0.9,
        )
    axes[0].set_title("Retained features", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Number of features")
    axes[0].set_xticks(x, [METHOD_LABELS[key] for key in METHOD_ORDER], rotation=28, ha="right")
    axes[0].grid(axis="y", alpha=0.25, linewidth=0.6)
    axes[0].legend(frameon=False, fontsize=8)
    sns.boxplot(
        data=delta.loc[~delta["feature_set"].eq("f_all")].assign(
            method=lambda value: value["feature_set"].map(METHOD_LABELS),
            dataset_label=lambda value: value["dataset"].map(DATASET_LABELS),
        ),
        x="method", y="macro_f1_delta", hue="dataset_label",
        order=[METHOD_LABELS[key] for key in METHOD_ORDER[1:]], showfliers=False,
        linewidth=0.8, width=0.72, ax=axes[1],
    )
    axes[1].axhline(0, color="#333333", linewidth=0.9, linestyle="--")
    axes[1].set_title("S3 performance relative to F-All", fontsize=10, fontweight="bold")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Macro-F1 difference")
    axes[1].tick_params(axis="x", rotation=28)
    axes[1].grid(axis="y", alpha=0.25, linewidth=0.6)
    legend = axes[1].get_legend()
    if legend:
        legend.remove()
    fig.tight_layout()
    save_figure(fig, "fig3_feature_reduction_internal_performance")


def figure_external_transfer(external: pd.DataFrame) -> None:
    primary = external.loc[external["cohort"].eq("primary_domain_filtered")].copy()
    primary["direction"] = primary["source_dataset"].map({
        "iscx_url2016_binary": "ISCX -> PhiUSIIL",
        "phiusiil": "PhiUSIIL -> ISCX",
    })
    summary = grouped_summary(primary, ["direction", "feature_set"], ["macro_f1", "fpr"])
    summary.to_csv(FIGURE_DATA / "fig4_external_transfer_performance.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.45))
    x = np.arange(len(METHOD_ORDER))
    offsets = [-0.11, 0.11]
    directions = ["ISCX -> PhiUSIIL", "PhiUSIIL -> ISCX"]
    colors = ["#4C78A8", "#E45756"]
    markers = ["o", "s"]
    for axis, metric, title, ylim in zip(
        axes, ["macro_f1", "fpr"], ["External Macro-F1", "External false-positive rate"],
        [(0.14, 0.32), (0.84, 1.01)],
    ):
        for offset, direction, color, marker in zip(offsets, directions, colors, markers):
            values = summary.loc[summary["direction"].eq(direction)].set_index("feature_set").loc[METHOD_ORDER]
            means = values[f"{metric}_mean"].to_numpy()
            errors = np.vstack([
                means - values[f"{metric}_ci_low"].to_numpy(),
                values[f"{metric}_ci_high"].to_numpy() - means,
            ])
            axis.errorbar(
                x + offset, means, yerr=errors, marker=marker, linestyle="none", color=color,
                markersize=5, capsize=2.5, label=direction,
            )
        axis.set_title(title, fontsize=10, fontweight="bold")
        axis.set_xticks(x, [METHOD_LABELS[key] for key in METHOD_ORDER], rotation=28, ha="right")
        axis.set_ylim(*ylim)
        axis.grid(axis="y", alpha=0.25, linewidth=0.6)
    axes[0].set_ylabel("Score")
    axes[1].set_ylabel("Rate")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    save_figure(fig, "fig4_external_transfer_performance")


def figure_stability_external(external: pd.DataFrame) -> None:
    shap_agreement = pd.read_csv(PART6 / "results" / "feature_set_outer_agreement.csv")
    traditional = pd.read_csv(PART8 / "results" / "traditional_outer_agreement.csv")
    agreement = pd.concat([shap_agreement, traditional], ignore_index=True)
    agreement = agreement.groupby(["dataset", "model", "feature_set"], sort=True)["jaccard"].mean().reset_index()
    primary = external.loc[external["cohort"].eq("primary_domain_filtered")]
    performance = primary.groupby(["source_dataset", "model", "feature_set"], sort=True)["macro_f1"].mean().reset_index()
    frame = agreement.merge(
        performance, left_on=["dataset", "model", "feature_set"],
        right_on=["source_dataset", "model", "feature_set"], validate="one_to_one",
    )
    frame = frame.loc[frame["feature_set"].ne("f_all")].copy()
    frame.to_csv(FIGURE_DATA / "fig5_stability_vs_external_performance.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.25), sharex=True)
    markers = {"lr": "o", "rf": "s", "xgb": "^"}
    for axis, dataset in zip(axes, DATASET_LABELS):
        subset = frame.loc[frame["dataset"].eq(dataset)]
        for method in METHOD_ORDER[1:]:
            for model in MODEL_LABELS:
                point = subset.loc[
                    subset["feature_set"].eq(method) & subset["model"].eq(model)
                ]
                axis.scatter(
                    point["jaccard"], point["macro_f1"], s=42,
                    marker=markers[model], color=METHOD_COLORS[method], edgecolor="white",
                    linewidth=0.5,
                )
        rho, p_value = spearmanr(subset["jaccard"], subset["macro_f1"])
        axis.text(0.03, 0.95, f"Spearman rho={rho:.2f}, p={p_value:.3f}",
                  transform=axis.transAxes, va="top", fontsize=8)
        axis.set_title(f"Trained on {DATASET_LABELS[dataset]}", fontsize=10, fontweight="bold")
        axis.grid(alpha=0.25, linewidth=0.6)
    axes[0].set_ylabel("External Macro-F1")
    for axis in axes:
        axis.set_xlabel("Outer-split feature-set Jaccard")
        axis.set_xlim(0.25, 1.03)
    method_handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=METHOD_COLORS[key],
                                 label=METHOD_LABELS[key], markersize=6) for key in METHOD_ORDER[1:]]
    model_handles = [plt.Line2D([0], [0], marker=markers[key], linestyle="", color="#555555",
                                label=MODEL_LABELS[key], markersize=6) for key in MODEL_LABELS]
    axes[1].legend(handles=method_handles + model_handles, frameon=False, fontsize=7,
                   loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    save_figure(fig, "fig5_stability_vs_external_performance")


def figure_false_positive_structure(fp: pd.DataFrame) -> None:
    frame = fp.copy()
    frame["direction"] = frame["source_dataset"].map({
        "iscx_url2016_binary": "ISCX -> PhiUSIIL",
        "phiusiil": "PhiUSIIL -> ISCX",
    })
    summary = grouped_summary(
        frame, ["direction", "feature_set"],
        ["n_false_positive_domains", "top10_domain_share"],
    )
    summary.to_csv(FIGURE_DATA / "fig6_false_positive_domain_structure.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.45))
    x = np.arange(len(METHOD_ORDER))
    directions = ["ISCX -> PhiUSIIL", "PhiUSIIL -> ISCX"]
    colors = ["#4C78A8", "#E45756"]
    markers = ["o", "s"]
    panels = [
        (
            "n_false_positive_domains",
            "Distinct domains among false positives",
            "Domain count (log scale)",
            (200, 2e5),
        ),
        (
            "top10_domain_share",
            "Concentration in top 10 domains",
            "Share of false positives (log scale)",
            (3e-3, 0.4),
        ),
    ]
    for axis, (metric, title, ylabel, ylim) in zip(axes, panels):
        for offset, direction, color, marker in zip(
            [-0.10, 0.10], directions, colors, markers,
        ):
            values = (
                summary.loc[summary["direction"].eq(direction)]
                .set_index("feature_set")
                .loc[METHOD_ORDER]
            )
            means = values[f"{metric}_mean"].to_numpy()
            errors = np.vstack([
                means - values[f"{metric}_ci_low"].to_numpy(),
                values[f"{metric}_ci_high"].to_numpy() - means,
            ])
            axis.errorbar(
                x + offset,
                means,
                yerr=errors,
                marker=marker,
                linestyle="none",
                color=color,
                markerfacecolor=color,
                markeredgecolor="black",
                markeredgewidth=0.5,
                markersize=6,
                capsize=2.5,
                linewidth=1,
                label=direction,
            )
        axis.set_yscale("log")
        axis.set_ylim(*ylim)
        if metric == "n_false_positive_domains":
            axis.set_yticks([300, 1e3, 1e4, 1e5])
            axis.set_yticklabels(["300", "1,000", "10,000", "100,000"])
        else:
            axis.set_yticks([0.005, 0.01, 0.1, 0.3])
            axis.set_yticklabels(["0.5%", "1%", "10%", "30%"])
        axis.set_title(title, fontsize=10, fontweight="bold")
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, [METHOD_LABELS[key] for key in METHOD_ORDER], rotation=28, ha="right")
        axis.grid(axis="y", which="both", alpha=0.25, linewidth=0.6)
    handles = [
        plt.Line2D(
            [0], [0], marker=marker, linestyle="none", color=color,
            markerfacecolor=color, markeredgecolor="black", markeredgewidth=0.5,
            label=direction, markersize=6,
        )
        for direction, color, marker in zip(directions, colors, markers)
    ]
    labels = directions
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    save_figure(fig, "fig6_false_positive_domain_structure")


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    for directory in [FIGURES, FIGURE_DATA, TABLES, RESULTS, ROOT / "manuscript", ROOT / "logs"]:
        directory.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=0.9)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "axes.spines.top": False,
        "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    internal = pd.read_csv(PART4 / "results" / "internal_metrics.csv")
    pairwise = pd.read_csv(PART5 / "results" / "pairwise_stability.csv")
    five_internal = pd.read_csv(PART8 / "results" / "internal_metrics_all_five.csv")
    external = pd.read_csv(PART8 / "results" / "external_metrics_all_five.csv")
    fp = pd.read_csv(PART8 / "results" / "false_positive_domain_concentration.csv")

    make_tables(internal, pairwise, five_internal, external, fp)
    figure_internal_regimes(internal)
    figure_shap_stability(pairwise)
    figure_feature_reduction(five_internal)
    figure_external_transfer(external)
    figure_stability_external(external)
    figure_false_positive_structure(fp)

    artifacts = sorted(
        [path for folder in [FIGURES, FIGURE_DATA, TABLES] for path in folder.iterdir() if path.is_file()]
    )
    manifest = pd.DataFrame([{
        "path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
        "sha256": hash_file(path),
    } for path in artifacts])
    manifest.to_csv(RESULTS / "publication_artifact_manifest.csv", index=False)
    print(f"figures={len(list(FIGURES.glob('*')))} figure_data={len(list(FIGURE_DATA.glob('*.csv')))}")
    print(f"tables={len(list(TABLES.glob('*')))} manifest_rows={len(manifest)}")


if __name__ == "__main__":
    main()
