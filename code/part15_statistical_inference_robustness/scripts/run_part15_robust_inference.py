from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t, ttest_1samp


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT.parent
PART11 = EXPERIMENT_ROOT / "part11_dcss_innovation"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
LOGS = ROOT / "logs"
REPETITIONS = tuple(f"r{i:02d}" for i in range(10))
RHO_GRID = (0.0, 0.10, 0.25, 0.50, 0.75, 0.90)
BOOTSTRAP_REPLICATES = 20000
H1_MARGIN = -0.01
H2_MARGIN = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Part 15 repeated-split inference audit")
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def benjamini_hochberg(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def effective_sample_size(n: int, rho: float) -> float:
    if n <= 0 or not 0.0 <= rho < 1.0:
        raise ValueError("Invalid n or rho")
    return n / (1.0 + (n - 1) * rho)


def correlated_standard_error(sd: float, n: int, rho: float) -> float:
    return sd / math.sqrt(effective_sample_size(n, rho))


def dependence_row(
    hypothesis: str,
    group_key: str,
    values: np.ndarray,
    reference: float,
    rho: float,
) -> dict[str, object]:
    n = len(values)
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    se = correlated_standard_error(sd, n, rho)
    statistic = (mean - reference) / se if se > 0 else math.inf
    p_value = float(t.sf(statistic, df=n - 1)) if math.isfinite(statistic) else 0.0
    critical = float(t.ppf(0.975, df=n - 1))
    return {
        "hypothesis": hypothesis,
        "group_key": group_key,
        "reference_margin": reference,
        "assumed_pairwise_correlation": rho,
        "effective_repetitions": effective_sample_size(n, rho),
        "mean_delta": mean,
        "correlation_adjusted_se": se,
        "ci_low": mean - critical * se,
        "ci_high": mean + critical * se,
        "raw_p_one_sided": p_value,
        "lower_bound_above_reference": bool(mean - critical * se > reference),
    }


def summarize_group(
    hypothesis: str,
    group_key: str,
    values: np.ndarray,
    reference: float,
    seed: int,
    bootstrap_replicates: int,
) -> tuple[dict[str, object], np.ndarray]:
    n = len(values)
    if n != 10:
        raise ValueError(f"Expected ten repetitions for {group_key}, found {n}")
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    sem = sd / math.sqrt(n)
    critical = float(t.ppf(0.975, df=n - 1))
    test = ttest_1samp(values, popmean=reference, alternative="greater")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(bootstrap_replicates, n))
    bootstrap = values[indices].mean(axis=1)
    return (
        {
            "hypothesis": hypothesis,
            "group_key": group_key,
            "n_repetitions": n,
            "reference_margin": reference,
            "mean_delta": mean,
            "median_delta": float(np.median(values)),
            "sd_delta": sd,
            "sem_naive": sem,
            "minimum_delta": float(values.min()),
            "maximum_delta": float(values.max()),
            "q25_delta": float(np.quantile(values, 0.25)),
            "q75_delta": float(np.quantile(values, 0.75)),
            "positive_repetitions": int((values > 0).sum()),
            "repetitions_above_reference": int((values > reference).sum()),
            "direction_consistency": float((values > reference).mean()),
            "standardized_effect_vs_reference": (mean - reference) / sd if sd > 0 else math.inf,
            "naive_t_ci_low": mean - critical * sem,
            "naive_t_ci_high": mean + critical * sem,
            "repetition_bootstrap_ci_low": float(np.quantile(bootstrap, 0.025)),
            "repetition_bootstrap_ci_high": float(np.quantile(bootstrap, 0.975)),
            "raw_p_one_sided_t": float(test.pvalue),
            "mean_above_reference": bool(mean > reference),
        },
        bootstrap,
    )


def load_paired_differences() -> tuple[pd.DataFrame, dict[str, np.ndarray], list[Path]]:
    h1_path = PART11 / "results" / "h1_internal_noninferiority_pairs.csv"
    h2_path = PART11 / "results" / "h2_external_auc_pairs.csv"
    h1 = pd.read_csv(h1_path)
    h2 = pd.read_csv(h2_path)
    rows: list[dict[str, object]] = []
    groups: dict[str, np.ndarray] = {}

    for (dataset, model), frame in h1.groupby(["dataset", "model"], sort=True):
        ordered = frame.set_index("repetition").loc[list(REPETITIONS)]
        group_key = f"{dataset}/{model}"
        values = ordered["delta_dcss15_minus_all"].to_numpy(dtype=float)
        groups[f"H1|{group_key}"] = values
        for repetition, value in zip(REPETITIONS, values):
            rows.append(
                {
                    "hypothesis": "H1_internal_noninferiority",
                    "group_key": group_key,
                    "repetition": repetition,
                    "delta": value,
                    "reference_margin": H1_MARGIN,
                    "endpoint": "S3 Macro-F1: F-DCSS-15 minus F-All",
                }
            )

    h2_counts = h2.groupby(["source_dataset", "target_dataset", "repetition"]).size()
    if not (h2_counts == 3).all():
        raise ValueError("H2 requires exactly three model-family deltas per repetition")
    h2_rep = (
        h2.groupby(["source_dataset", "target_dataset", "repetition"], observed=True)[
            "delta_dcss15_minus_stable"
        ]
        .mean()
        .reset_index(name="delta")
    )
    for (source, target), frame in h2_rep.groupby(["source_dataset", "target_dataset"], sort=True):
        ordered = frame.set_index("repetition").loc[list(REPETITIONS)]
        group_key = f"{source}->{target}"
        values = ordered["delta"].to_numpy(dtype=float)
        groups[f"H2|{group_key}"] = values
        for repetition, value in zip(REPETITIONS, values):
            rows.append(
                {
                    "hypothesis": "H2_external_improvement",
                    "group_key": group_key,
                    "repetition": repetition,
                    "delta": value,
                    "reference_margin": H2_MARGIN,
                    "endpoint": "External ROC-AUC: F-DCSS-15 minus F-Stable, averaged across models",
                }
            )
    return pd.DataFrame(rows), groups, [h1_path, h2_path]


def apply_family_adjustment(summary: pd.DataFrame) -> pd.DataFrame:
    output = summary.copy()
    output["bh_adjusted_p_auxiliary"] = np.nan
    for hypothesis, index in output.groupby("hypothesis").groups.items():
        values = output.loc[index, "raw_p_one_sided_t"].to_numpy(dtype=float)
        output.loc[index, "bh_adjusted_p_auxiliary"] = benjamini_hochberg(values)
    return output


def add_external_cluster_evidence(summary: pd.DataFrame) -> tuple[pd.DataFrame, Path]:
    path = PART11 / "results" / "stage11g_h2_cluster_bootstrap_summary.csv"
    cluster = pd.read_csv(path)
    output = summary.copy()
    output["domain_outer_ci_low"] = np.nan
    output["domain_outer_ci_high"] = np.nan
    output["domain_outer_bh_p_two_sided"] = np.nan
    output["domain_outer_ci_excludes_zero"] = False
    for row in cluster.itertuples(index=False):
        key = f"{row.source_dataset}->{row.target_dataset}"
        mask = (output["hypothesis"] == "H2_external_improvement") & (output["group_key"] == key)
        if mask.sum() != 1:
            raise ValueError(f"Missing H2 summary group: {key}")
        output.loc[mask, "domain_outer_ci_low"] = row.cluster_outer_ci_low
        output.loc[mask, "domain_outer_ci_high"] = row.cluster_outer_ci_high
        output.loc[mask, "domain_outer_bh_p_two_sided"] = row.bh_adjusted_p
        output.loc[mask, "domain_outer_ci_excludes_zero"] = bool(row.ci_excludes_zero)
    return output, path


def verdicts(summary: pd.DataFrame, dependence: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in summary.itertuples(index=False):
        dep = dependence.loc[
            (dependence["hypothesis"] == row.hypothesis)
            & (dependence["group_key"] == row.group_key)
        ].set_index("assumed_pairwise_correlation")
        rho25 = dep.loc[0.25]
        rho50 = dep.loc[0.50]
        if row.hypothesis == "H1_internal_noninferiority":
            if not row.mean_above_reference:
                verdict = "fails_mean_noninferiority_margin"
            elif row.repetitions_above_reference < 8:
                verdict = "mean_within_margin_but_direction_inconsistent"
            elif rho50.lower_bound_above_reference:
                verdict = "robust_through_rho_0.50"
            elif rho25.lower_bound_above_reference:
                verdict = "robust_through_rho_0.25_only"
            else:
                verdict = "mean_within_margin_but_dependence_sensitive"
        else:
            if row.mean_delta <= 0:
                verdict = "no_mean_external_improvement"
            elif not bool(row.domain_outer_ci_excludes_zero):
                verdict = "positive_mean_without_domain_outer_support"
            elif rho50.lower_bound_above_reference:
                verdict = "positive_domain_outer_and_rho_0.50_support"
            elif rho25.lower_bound_above_reference:
                verdict = "positive_domain_outer_and_rho_0.25_support"
            else:
                verdict = "positive_domain_outer_but_repetition_dependence_sensitive"
        rows.append(
            {
                "hypothesis": row.hypothesis,
                "group_key": row.group_key,
                "mean_delta": row.mean_delta,
                "reference_margin": row.reference_margin,
                "repetitions_above_reference": row.repetitions_above_reference,
                "rho_0_25_ci_low": rho25.ci_low,
                "rho_0_50_ci_low": rho50.ci_low,
                "verdict": verdict,
            }
        )
    return pd.DataFrame(rows)


def plot_results(pairs: pd.DataFrame, summary: pd.DataFrame, dependence: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    order = summary["group_key"].tolist()
    y = np.arange(len(order))

    fig, axis = plt.subplots(figsize=(9, 5))
    for index, row in summary.reset_index(drop=True).iterrows():
        axis.plot([row.naive_t_ci_low, row.naive_t_ci_high], [index, index], color="#4C78A8", linewidth=3)
        axis.scatter(row.mean_delta, index, color="#1f4e79", zorder=3)
        axis.scatter(row.reference_margin, index, marker="|", s=100, color="black", zorder=3)
    axis.set_yticks(y, order)
    axis.axvline(0.0, color="gray", linestyle="--", linewidth=0.8)
    axis.set_xlabel("Paired effect (point and naive 95% t interval); black tick = reference margin")
    axis.set_title("Part 15 paired effects before repeated-split correlation stress")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"fig15_1_effect_forest.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 5))
    rng = np.random.default_rng(20261530)
    for index, key in enumerate(order):
        values = pairs.loc[pairs["group_key"] == key, "delta"].to_numpy(dtype=float)
        jitter = rng.uniform(-0.12, 0.12, size=len(values))
        axis.scatter(values, np.full(len(values), index) + jitter, alpha=0.8)
        reference = float(pairs.loc[pairs["group_key"] == key, "reference_margin"].iloc[0])
        axis.scatter(reference, index, marker="|", s=120, color="black")
    axis.set_yticks(y, order)
    axis.axvline(0.0, color="gray", linestyle="--", linewidth=0.8)
    axis.set_xlabel("All ten repetition-level paired differences")
    axis.set_title("Repetition-level direction and dispersion")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"fig15_2_repetition_deltas.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)

    matrix = dependence.pivot(
        index="group_key", columns="assumed_pairwise_correlation", values="ci_low"
    ).loc[order]
    references = summary.set_index("group_key").loc[order, "reference_margin"].to_numpy()[:, None]
    relative = matrix.to_numpy() - references
    bound = max(abs(float(np.nanmin(relative))), abs(float(np.nanmax(relative))))
    fig, axis = plt.subplots(figsize=(9, 5))
    image = axis.imshow(relative, aspect="auto", cmap="RdBu_r", vmin=-bound, vmax=bound)
    axis.set_yticks(np.arange(len(order)), order)
    axis.set_xticks(np.arange(len(matrix.columns)), [f"{value:.2f}" for value in matrix.columns])
    axis.set_xlabel("Assumed pairwise correlation among repetitions")
    axis.set_title("Lower 95% bound minus hypothesis reference margin")
    fig.colorbar(image, ax=axis, label="Positive supports the prespecified direction")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"fig15_3_dependence_stress.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_reports(summary: pd.DataFrame, verdict: pd.DataFrame) -> None:
    h1 = summary.loc[summary["hypothesis"] == "H1_internal_noninferiority"]
    h2 = summary.loc[summary["hypothesis"] == "H2_external_improvement"]
    lines = [
        "# Part 15: Statistical Inference Robustness Audit",
        "",
        "Status: **COMPLETE**",
        "",
        "## Locked interpretation",
        "",
        "The ten repetitions are paired perturbations of fixed corpora, not ten independent datasets. "
        "Effect sizes, intervals, the ten disclosed differences, and directional consistency are primary. "
        "P-values are auxiliary. The earlier wording `paired t-test` is made precise as a one-sample, "
        "one-sided t-test applied to paired repetition-level differences.",
        "",
        "## H1 internal non-inferiority",
        "",
        "| Dataset/model | Mean delta | 10/10 count above -0.01 | Naive 95% CI | rho=0.25 lower | rho=0.50 lower | Verdict |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    verdict_index = verdict.set_index(["hypothesis", "group_key"])
    for row in h1.itertuples(index=False):
        v = verdict_index.loc[(row.hypothesis, row.group_key)]
        lines.append(
            f"| {row.group_key} | {row.mean_delta:+.5f} | {row.repetitions_above_reference}/10 | "
            f"[{row.naive_t_ci_low:+.5f}, {row.naive_t_ci_high:+.5f}] | "
            f"{v.rho_0_25_ci_low:+.5f} | {v.rho_0_50_ci_low:+.5f} | {v.verdict} |"
        )
    lines.extend(
        [
            "",
            "## H2 external improvement",
            "",
            "| Direction | Mean AUC delta | Positive repetitions | Domain-plus-outer 95% CI | rho=0.25 lower | rho=0.50 lower | Verdict |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in h2.itertuples(index=False):
        v = verdict_index.loc[(row.hypothesis, row.group_key)]
        lines.append(
            f"| {row.group_key} | {row.mean_delta:+.5f} | {row.positive_repetitions}/10 | "
            f"[{row.domain_outer_ci_low:+.5f}, {row.domain_outer_ci_high:+.5f}] | "
            f"{v.rho_0_25_ci_low:+.5f} | {v.rho_0_50_ci_low:+.5f} | {v.verdict} |"
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "The study-wide strong DCSS contribution condition remains false. ISCX LR and RF fail the "
            "H1 mean non-inferiority margin, and ISCX-to-PhiUSIIL has no positive mean H2 effect. "
            "The positive PhiUSIIL-to-ISCX effect remains a bounded, direction-specific result. "
            "It should be described as a prespecified comparison, not universal confirmatory evidence.",
            "",
        ]
    )
    (RESULTS / "PART15_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    zh = [
        "# 第15部分：统计推断稳健性审计",
        "",
        "状态：**已完成**",
        "",
        "## 统计定位",
        "",
        "10次重复是同一固定语料上的配对划分扰动，不是10个独立数据集。结论以效应量、区间、10个配对差和方向一致性为主，p值降为辅助证据。"
        "原“配对t检验”的精确实现是：对配对后的repetition-level差值执行单样本、单侧t检验。",
        "",
        "## 主要结论",
        "",
        "- ISCX的LR和RF平均差低于-0.01非劣效界值，H1明确失败。",
        "- ISCX→PhiUSIIL的H2平均AUC差为负，不支持双向改善。",
        "- PhiUSIIL→ISCX保留方向特定的正效果，但不能推广为普遍特征选择优势。",
        "- 研究级别的DCSS强贡献条件仍不成立，建议将`confirmatory hypothesis`改为`prespecified comparison`。",
        "",
        "## H1非劣效结果",
        "",
        "| 数据集/模型 | 平均差 | 高于-0.01的重复数 | rho=0.25下界 | rho=0.50下界 | 结论 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in h1.itertuples(index=False):
        v = verdict_index.loc[(row.hypothesis, row.group_key)]
        zh.append(
            f"| {row.group_key} | {row.mean_delta:+.5f} | {row.repetitions_above_reference}/10 | "
            f"{v.rho_0_25_ci_low:+.5f} | {v.rho_0_50_ci_low:+.5f} | {v.verdict} |"
        )
    zh.extend(
        [
            "",
            "## H2外部改善结果",
            "",
            "| 方向 | 平均AUC差 | 正向重复数 | 域+外层95%区间 | rho=0.50下界 | 结论 |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in h2.itertuples(index=False):
        v = verdict_index.loc[(row.hypothesis, row.group_key)]
        zh.append(
            f"| {row.group_key} | {row.mean_delta:+.5f} | {row.positive_repetitions}/10 | "
            f"[{row.domain_outer_ci_low:+.5f}, {row.domain_outer_ci_high:+.5f}] | "
            f"{v.rho_0_50_ci_low:+.5f} | {v.verdict} |"
        )
    zh.append("")
    (RESULTS / "PART15_REPORT_ZH.md").write_text("\n".join(zh), encoding="utf-8")

    implications = """# Manuscript implications from Part 15

1. Replace `paired t-test` with `one-sample, one-sided t-test on paired repetition-level differences`.
2. Replace `confirmatory hypothesis` with `prespecified comparison` unless the sentence explicitly refers to the locked protocol rather than strength of evidence.
3. State that the ten repetitions are partition perturbations of fixed corpora and are not independent datasets.
4. Make mean effect, interval, all ten paired differences, and direction consistency primary; retain adjusted p-values as auxiliary.
5. Keep both H2 directions together. The positive PhiUSIIL-to-ISCX result cannot support a bidirectional or universal DCSS claim.
6. Keep the low-FPR analysis framed as frozen-threshold transfer failure, not as a population or deployment guarantee.
"""
    (RESULTS / "manuscript_implications.md").write_text(implications, encoding="utf-8")


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    pairs, groups, input_paths = load_paired_differences()
    summaries = []
    bootstrap_rows = []
    dependence_rows = []
    for group_index, (compound_key, values) in enumerate(sorted(groups.items())):
        family, group_key = compound_key.split("|", 1)
        hypothesis = "H1_internal_noninferiority" if family == "H1" else "H2_external_improvement"
        reference = H1_MARGIN if family == "H1" else H2_MARGIN
        summary, bootstrap = summarize_group(
            hypothesis,
            group_key,
            values,
            reference,
            20261501 + group_index,
            args.bootstrap_replicates,
        )
        summaries.append(summary)
        for index, value in enumerate(bootstrap):
            bootstrap_rows.append(
                {
                    "hypothesis": hypothesis,
                    "group_key": group_key,
                    "bootstrap_index": index,
                    "mean_delta": value,
                }
            )
        for rho in RHO_GRID:
            dependence_rows.append(dependence_row(hypothesis, group_key, values, reference, rho))

    summary = apply_family_adjustment(pd.DataFrame(summaries))
    summary, cluster_path = add_external_cluster_evidence(summary)
    input_paths.append(cluster_path)
    dependence = pd.DataFrame(dependence_rows)
    verdict = verdicts(summary, dependence)

    pairs.to_csv(RESULTS / "paired_differences_long.csv", index=False)
    pairs.pivot(index=["hypothesis", "group_key", "reference_margin"], columns="repetition", values="delta").reset_index().to_csv(
        RESULTS / "paired_differences_wide.csv", index=False
    )
    summary.to_csv(RESULTS / "effect_size_and_auxiliary_tests.csv", index=False)
    dependence.to_csv(RESULTS / "repetition_dependence_sensitivity.csv", index=False)
    verdict.to_csv(RESULTS / "robustness_verdicts.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(RESULTS / "descriptive_repetition_bootstrap.csv", index=False)

    plot_results(pairs, summary, dependence)
    write_reports(summary, verdict)

    input_manifest = pd.DataFrame(
        [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(set(input_paths))
        ]
    )
    input_manifest.to_csv(RESULTS / "input_manifest.csv", index=False)

    issues = """# Part 15 Issues and Resolutions

## Resolved during execution

1. **The manuscript called the H1 procedure a paired t-test.** The implemented procedure is now named precisely: a one-sample, one-sided t-test on ten paired repetition-level differences against the -0.01 margin.
2. **Ten overlapping repetitions were easy to misread as independent datasets.** All ten differences are disclosed, and effect sizes, intervals, and direction consistency are primary. P-values are explicitly auxiliary.
3. **A repetition bootstrap alone does not remove dependence.** A transparent equicorrelation stress grid from rho=0 to 0.90 reports effective repetition count and widened intervals.
4. **The H2 interval is not a literal nested domain bootstrap.** The audit reuses the locked linearized domain-Gaussian-multiplier plus outer-bootstrap interval and keeps its approximation boundary visible.
5. **A positive result in one transfer direction could be selectively emphasized.** Both directions are kept in the same table and the study-wide strong-contribution condition remains false.

## Residual limitations

1. The equicorrelation grid is a stress analysis, not an estimate of the true correlation among repetitions.
2. With only ten repetitions, high-correlation effective sample sizes approach one; no p-value can substitute for new independent corpora.
3. The fixed two-corpus design supports corpus-conditional conclusions only.
4. The non-inferiority margin of 0.01 is a prespecified engineering tolerance, not an externally validated operational standard.
"""
    (LOGS / "issues_and_resolutions.md").write_text(issues, encoding="utf-8")

    conclusion = {
        "status": "PASS",
        "completed_utc": utc_now(),
        "elapsed_seconds": time.perf_counter() - started,
        "python": sys.version,
        "platform": platform.platform(),
        "bootstrap_replicates": args.bootstrap_replicates,
        "rho_grid": list(RHO_GRID),
        "paired_difference_rows": len(pairs),
        "effect_groups": len(summary),
        "dependence_rows": len(dependence),
        "strong_contribution_condition_met": False,
        "recommended_wording": "prespecified comparison",
        "low_fpr_framing": "frozen-threshold transfer failure diagnostic",
    }
    (RESULTS / "run_metadata.json").write_text(
        json.dumps(conclusion, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(conclusion, indent=2))


if __name__ == "__main__":
    main()
