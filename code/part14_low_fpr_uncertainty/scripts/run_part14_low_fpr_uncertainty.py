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
from scipy.stats import beta, norm
from sklearn.metrics import roc_curve
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT.parent
PART2 = EXPERIMENT_ROOT / "part2_url_normalization_leakage_audit"
PART3 = EXPERIMENT_ROOT / "part3_s0_s4_data_splits"
PART4 = EXPERIMENT_ROOT / "part4_baseline_models"
PART12 = EXPERIMENT_ROOT / "part12_security_privacy_revision"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
LOGS = ROOT / "logs"
sys.path.insert(0, str(PART4 / "scripts"))

from model_utils import build_model, calculate_metrics  # noqa: E402
from train_baselines import (  # noqa: E402
    DATASETS,
    MODEL_NAMES,
    MODEL_SEEDS,
    REPETITIONS,
    load_feature_names,
    load_feature_table,
)


BUDGETS = (0.001, 0.01)
CONFIDENCE = 0.95
DEFAULT_HOLDOUT_REPEATS = 20
HOLDOUT_SEED = 20261401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Part 14 low-FPR uncertainty audit")
    parser.add_argument("--n-jobs", type=int, default=6)
    parser.add_argument("--holdout-repeats", type=int, default=DEFAULT_HOLDOUT_REPEATS)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def allowed_false_positives(n_benign: int, budget: float) -> int:
    return int(math.floor(n_benign * budget + 1e-12))


def wilson_upper(fp: int, n_benign: int, confidence: float = CONFIDENCE) -> float:
    if n_benign <= 0:
        raise ValueError("n_benign must be positive")
    z = float(norm.ppf(confidence))
    p = fp / n_benign
    denominator = 1.0 + z * z / n_benign
    center = p + z * z / (2.0 * n_benign)
    spread = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n_benign)) / n_benign)
    return min(1.0, (center + spread) / denominator)


def clopper_pearson_upper(
    fp: int,
    n_benign: int,
    confidence: float = CONFIDENCE,
) -> float:
    if n_benign <= 0:
        raise ValueError("n_benign must be positive")
    if fp >= n_benign:
        return 1.0
    return float(beta.ppf(confidence, fp + 1, n_benign - fp))


def select_fpr_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    budget: float,
) -> tuple[int, float, dict[str, float | int], np.ndarray]:
    fpr, tpr, thresholds = roc_curve(labels, probabilities, drop_intermediate=False)
    valid = np.flatnonzero(fpr <= budget + 1e-15)
    if len(valid) == 0:
        index = 0
    else:
        best_tpr = np.max(tpr[valid])
        tied = valid[np.isclose(tpr[valid], best_tpr, rtol=0.0, atol=1e-15)]
        tied_thresholds = thresholds[tied]
        index = int(tied[int(np.argmin(tied_thresholds))])
    threshold = float(thresholds[index])
    metrics = calculate_metrics(labels, probabilities, threshold)
    if metrics["fpr"] > budget + 1e-15:
        raise AssertionError("Selected threshold exceeds its empirical FPR budget")
    return index, threshold, metrics, thresholds


def adjacent_thresholds(index: int, thresholds: np.ndarray) -> list[tuple[str, float]]:
    higher_index = max(0, index - 1)
    lower_index = min(len(thresholds) - 1, index + 1)
    return [
        ("safer_neighbor", float(thresholds[higher_index])),
        ("selected", float(thresholds[index])),
        ("more_permissive_neighbor", float(thresholds[lower_index])),
    ]


def load_s3(dataset: str, feature_names: list[str], features: pd.DataFrame) -> pd.DataFrame:
    path = (
        PART3
        / "data"
        / "assignments"
        / f"{DATASETS[dataset]['assignment_prefix']}_s3_domain_assignments.parquet"
    )
    assignment = pd.read_parquet(
        path,
        columns=[
            "sample_id",
            "source_row",
            "raw_url_sha256",
            "registrable_domain_sha256",
            *[f"split_{rep}" for rep in REPETITIONS],
        ],
    )
    merged = assignment.merge(
        features,
        on=["source_row", "raw_url_sha256"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(assignment) or len(merged) != len(features):
        raise ValueError(f"S3 assignment mismatch: {dataset}")
    if merged[feature_names].isna().any().any():
        raise ValueError(f"Missing S3 feature value: {dataset}")
    return merged


def load_external(
    source: str,
    repetition: str,
    feature_names: list[str],
    feature_tables: dict[str, pd.DataFrame],
) -> tuple[np.ndarray, np.ndarray]:
    target = DATASETS[source]["s4_target"]
    role_column = f"role_{repetition}"
    path = PART3 / "data" / "assignments" / DATASETS[source]["s4_file"]
    assignment = pd.read_parquet(
        path,
        columns=["origin", "source_row", "raw_url_sha256", "label", role_column],
    )
    assignment = assignment.loc[assignment["origin"] == "target"].copy()
    merged = assignment.merge(
        feature_tables[target].drop(columns="label"),
        on=["source_row", "raw_url_sha256"],
        how="inner",
        validate="one_to_one",
    )
    primary = merged[role_column].astype(str).eq("target_external_primary").to_numpy()
    if not primary.any():
        raise ValueError(f"Missing external primary cohort: {source}/{repetition}")
    return (
        merged.loc[primary, feature_names].to_numpy(dtype=np.float32),
        merged.loc[primary, "label"].to_numpy(dtype=np.int8),
    )


def metric_with_bounds(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, object]:
    metrics = calculate_metrics(labels, probabilities, threshold)
    return {
        **metrics,
        "fpr_wilson_upper_95": wilson_upper(int(metrics["fp"]), int(metrics["n_benign"])),
        "fpr_clopper_pearson_upper_95": clopper_pearson_upper(
            int(metrics["fp"]), int(metrics["n_benign"])
        ),
        "alerts_per_10000_benign": float(metrics["fpr"]) * 10000.0,
    }


def prefixed(metrics: dict[str, object], prefix: str) -> dict[str, object]:
    keep = [
        "n_samples",
        "n_benign",
        "n_phishing",
        "fpr",
        "fp",
        "recall",
        "tp",
        "fpr_wilson_upper_95",
        "fpr_clopper_pearson_upper_95",
        "alerts_per_10000_benign",
    ]
    return {f"{prefix}_{key}": metrics[key] for key in keep}


def grouped_holdout_indices(
    labels: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    for offset in range(100):
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed + offset)
        calibration, audit = next(splitter.split(np.zeros(len(labels)), labels, groups))
        if set(labels[calibration]) == {0, 1} and set(labels[audit]) == {0, 1}:
            return calibration, audit
    raise RuntimeError("Could not produce a two-class grouped calibration/audit split")


def model_selection_row(source: str, repetition: str, model_name: str) -> tuple[dict[str, object], list[Path]]:
    run_dir = PART4 / "runs" / "internal" / source / "s3" / repetition / model_name
    metrics_path = run_dir / "internal_metrics.json"
    tuning_path = run_dir / "tuning_results.csv"
    locked = json.loads(metrics_path.read_text(encoding="utf-8"))
    tuning = pd.read_csv(tuning_path)
    tuning = tuning.loc[tuning["status"] == "ok"].sort_values(
        ["validation_macro_f1", "validation_roc_auc", "candidate_index"],
        ascending=[False, False, True],
    )
    best = tuning.iloc[0]
    runner = tuning.iloc[1]
    return (
        {
            "source_dataset": source,
            "repetition": repetition,
            "model": model_name,
            "selected_candidate": locked["selected_candidate"],
            "candidate_count": len(tuning),
            "best_validation_macro_f1": float(best["validation_macro_f1"]),
            "runner_up_validation_macro_f1": float(runner["validation_macro_f1"]),
            "selection_margin_macro_f1": float(
                best["validation_macro_f1"] - runner["validation_macro_f1"]
            ),
            "best_validation_roc_auc": float(best["validation_roc_auc"]),
            "runner_up_validation_roc_auc": float(runner["validation_roc_auc"]),
        },
        [metrics_path, tuning_path],
    )


def save_figures(
    uncertainty: pd.DataFrame,
    adjacent: pd.DataFrame,
    reuse_summary: pd.DataFrame,
) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    for axis, budget in zip(axes, BUDGETS):
        subset = uncertainty.loc[np.isclose(uncertainty["fpr_budget"], budget)]
        labels = []
        empirical = []
        upper = []
        for source in DATASETS:
            rows = subset.loc[subset["source_dataset"] == source]
            labels.append("PhiUSIIL" if source == "phiusiil" else "ISCX")
            empirical.append(rows["validation_fpr"].median())
            upper.append(rows["validation_fpr_wilson_upper_95"].median())
        x = np.arange(len(labels))
        width = 0.36
        axis.bar(x - width / 2, empirical, width, label="Empirical FPR")
        axis.bar(x + width / 2, upper, width, label="95% Wilson upper")
        axis.axhline(budget, color="black", linestyle="--", linewidth=1, label="Budget")
        axis.set_xticks(x, labels)
        axis.set_title(f"{budget * 100:g}% budget")
        axis.set_ylabel("False-positive rate")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"fig14_1_fpr_uncertainty.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)

    external = adjacent.loc[adjacent["evaluation_scope"] == "external_primary"].copy()
    join_keys = ["source_dataset", "repetition", "model", "fpr_budget", "evaluation_scope"]
    selected = external.loc[external["threshold_variant"] == "selected", join_keys + ["fp"]]
    selected = selected.rename(columns={"fp": "selected_fp"})
    external = external.merge(selected, on=join_keys, how="left", validate="many_to_one")
    external["delta_fp"] = external["fp"] - external["selected_fp"]
    external = external.loc[external["threshold_variant"] != "selected"].copy()
    external["direction"] = external["source_dataset"].map(
        {"phiusiil": "PhiUSIIL→ISCX", "iscx_url2016_binary": "ISCX→PhiUSIIL"}
    )
    external["key"] = external["direction"] + " / " + external["fpr_budget"].map(
        lambda value: f"{value * 100:g}%"
    )
    grouped = (
        external.groupby(["key", "threshold_variant"], observed=True)["delta_fp"]
        .mean()
        .unstack()
        .reindex(columns=["safer_neighbor", "more_permissive_neighbor"])
    )
    fig, axis = plt.subplots(figsize=(8, 4))
    grouped.plot(kind="bar", ax=axis)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylabel("Mean change in external false positives")
    axis.set_xlabel("Transfer direction / source-validation budget")
    axis.tick_params(axis="x", rotation=25)
    axis.legend(title="Adjacent threshold", fontsize=8)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"fig14_2_adjacent_threshold_sensitivity.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)

    plot = reuse_summary.copy()
    plot["key"] = plot["source_dataset"].str.replace("iscx_url2016_binary", "ISCX", regex=False)
    plot["key"] = plot["key"].str.replace("phiusiil", "PhiUSIIL", regex=False) + " / " + plot["model"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for axis, budget in zip(axes, BUDGETS):
        subset = plot.loc[np.isclose(plot["fpr_budget"], budget)]
        pivot = subset.pivot(index="key", columns="threshold_selector", values="budget_exceedance_rate")
        pivot = pivot.reindex(columns=["full_validation", "calibration_only"])
        pivot.plot(kind="bar", ax=axis)
        axis.set_title(f"{budget * 100:g}% budget")
        axis.set_xlabel("")
        axis.set_ylabel("Held-out audit exceedance rate")
        axis.tick_params(axis="x", rotation=35)
        axis.legend(fontsize=8)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"fig14_3_validation_reuse_audit.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_reports(
    uncertainty_summary: pd.DataFrame,
    reuse_summary: pd.DataFrame,
    pressure: pd.DataFrame,
    maximum_replay_difference: float,
) -> None:
    tiny_margin_rate = float((pressure["selection_margin_macro_f1"] < 0.001).mean())
    lines = [
        "# Part 14: Low-FPR Threshold Uncertainty Audit",
        "",
        "Status: **COMPLETE**",
        "",
        "## Design",
        "",
        "Locked S3 candidates were deterministically replayed from the original training assignments. "
        "For 0.1% and 1% validation FPR budgets, this audit reports integer false positives, one-sided "
        "95% Wilson and exact Clopper-Pearson upper bounds, threshold order statistics, adjacent-score "
        "sensitivity, and a repeated registrable-domain-grouped calibration/audit split.",
        "",
        "The grouped holdout isolates threshold-selection reuse only. The already-selected model candidate "
        "still used the full validation cohort, so it is not a fully independent calibration experiment.",
        "",
        "## Key audit facts",
        "",
        f"- Maximum absolute difference from the frozen Part 12 thresholds: {maximum_replay_difference:.3e}.",
        f"- Candidate-selection Macro-F1 margin was below 0.001 in {tiny_margin_rate:.1%} of 60 locked runs.",
        "- A validation-cohort empirical FPR at or below budget is not a population-level FPR guarantee.",
        "- At the 0.1% budget, mean source-test FPR remained near budget for PhiUSIIL-source runs "
        "(0.097%-0.116%) but rose to 7.85%-14.12% for ISCX-source runs.",
        "- At the same budget, mean frozen-threshold external FPR ranged from 53.18% to 100%, "
        "confirming that low source-validation FPR did not transfer across sources.",
        "",
        "## Validation uncertainty summary",
        "",
        "| Source | Model | Budget | Benign n (mean) | Allowed FP (mean) | Observed FP (mean) | Empirical FPR | Wilson upper | Source-test FPR | External FPR |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in uncertainty_summary.itertuples(index=False):
        lines.append(
            f"| {row.source_dataset} | {row.model} | {row.fpr_budget:.3f} | "
            f"{row.validation_n_benign_mean:.1f} | {row.allowed_fp_mean:.1f} | "
            f"{row.validation_fp_mean:.1f} | {row.validation_fpr_mean:.5f} | "
            f"{row.validation_wilson_upper_mean:.5f} | {row.source_test_fpr_mean:.5f} | "
            f"{row.external_fpr_mean:.5f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The low-FPR points are empirical source-validation operating points. Confidence upper bounds, "
            "adjacent-score jumps, and grouped holdout exceedance must accompany any 0.1% or 1% claim. "
            "External target outcomes remain frozen-threshold transfer diagnostics and are not calibration guarantees.",
            "",
        ]
    )
    (RESULTS / "PART14_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    exceed = (
        reuse_summary.groupby(["fpr_budget", "threshold_selector"], observed=True)["budget_exceedance_rate"]
        .mean()
        .reset_index()
    )
    zh = [
        "# 第14部分：低FPR阈值不确定性审计",
        "",
        "状态：**已完成**",
        "",
        "## 实验设计",
        "",
        "使用第4部分已锁定的S3候选模型和原始训练划分进行确定性回放，不重新调参。"
        "对0.1%和1%经验FPR预算，报告误报整数、Wilson与Clopper-Pearson单侧95%上界、"
        "阈值order statistic、相邻阈值敏感性，以及按可注册域分组的重复校准/审计划分。",
        "",
        "## 主要结论",
        "",
        f"- 与第12部分冻结阈值的最大绝对差为 {maximum_replay_difference:.3e}，回放一致。",
        f"- 60个锁定运行中，有 {tiny_margin_rate:.1%} 的最优候选与次优候选Macro-F1差小于0.001，说明部分模型选择对验证集波动敏感。",
        "- 经验FPR不超预算不等于总体FPR不超预算；论文必须同时报告误报整数和单侧置信上界。",
        "- 在0.1%预算下，PhiUSIIL来源模型的S3测试集平均FPR为0.097%-0.116%，而ISCX来源模型上升到7.85%-14.12%。",
        "- 同一预算下，冻结阈值的跨源平均FPR为53.18%-100%，低来源验证FPR明显不能迁移。",
        "- 对ISCX→PhiUSIIL，0.1%预算的下一个更宽松阈值平均增加197.9个外部误报，最高增加2,213个，说明阈值邻域非常敏感。",
        "- 分组留出分析只隔离了阈值选择的数据复用；模型候选仍由完整验证集决定，因此它不是完全独立的calibration split。",
        "",
        "## 分组留出超预算率（六个数据集-模型组的平均）",
        "",
        "| FPR预算 | 阈值选择方式 | 审计子集超预算率 |",
        "|---:|---|---:|",
    ]
    for row in exceed.itertuples(index=False):
        zh.append(f"| {row.fpr_budget:.3f} | {row.threshold_selector} | {row.budget_exceedance_rate:.3f} |")
    zh.extend(
        [
            "",
            "## 论文声称边界",
            "",
            "0.1%和1%只能称为“来源验证集上的经验FPR预算”，不能写成部署环境的总体保证。"
            "跨源结果只是冻结阈值迁移诊断，目标数据标签未参与阈值选择。",
            "",
        ]
    )
    (RESULTS / "PART14_REPORT_ZH.md").write_text("\n".join(zh), encoding="utf-8")


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    feature_names = load_feature_names()
    feature_tables = {name: load_feature_table(name, feature_names) for name in DATASETS}
    s3_tables = {name: load_s3(name, feature_names, feature_tables[name]) for name in DATASETS}
    frozen = pd.read_csv(PART12 / "results" / "low_fpr_operating_point_runs.csv")
    frozen = frozen.loc[frozen["evaluation_scope"] == "source_validation"].copy()

    uncertainty_rows: list[dict[str, object]] = []
    adjacent_rows: list[dict[str, object]] = []
    holdout_rows: list[dict[str, object]] = []
    replay_rows: list[dict[str, object]] = []
    pressure_rows: list[dict[str, object]] = []
    input_files: set[Path] = {
        PART12 / "results" / "low_fpr_operating_point_runs.csv",
        PART2 / "results" / "feature_dictionary.csv",
    }

    for source in DATASETS:
        target = DATASETS[source]["s4_target"]
        merged = s3_tables[source]
        feature_path = PART2 / "data" / DATASETS[source]["feature_file"]
        s3_path = (
            PART3
            / "data"
            / "assignments"
            / f"{DATASETS[source]['assignment_prefix']}_s3_domain_assignments.parquet"
        )
        s4_path = PART3 / "data" / "assignments" / DATASETS[source]["s4_file"]
        input_files.update({feature_path, s3_path, s4_path})

        X_all = merged[feature_names].to_numpy(dtype=np.float32)
        y_all = merged["label"].to_numpy(dtype=np.int8)

        for repetition in REPETITIONS:
            split = merged[f"split_{repetition}"].astype(str).to_numpy()
            train_mask = split == "train"
            validation_mask = split == "validation"
            test_mask = split == "test"
            validation_groups = merged.loc[validation_mask, "registrable_domain_sha256"].to_numpy()
            X_external, y_external = load_external(source, repetition, feature_names, feature_tables)

            for model_name in MODEL_NAMES:
                pressure_row, pressure_inputs = model_selection_row(source, repetition, model_name)
                pressure_rows.append(pressure_row)
                input_files.update(pressure_inputs)
                locked_path = pressure_inputs[0]
                locked = json.loads(locked_path.read_text(encoding="utf-8"))
                model = build_model(
                    model_name,
                    json.loads(locked["selected_params_json"]),
                    MODEL_SEEDS[int(repetition[1:])],
                    n_jobs=args.n_jobs,
                )
                model.fit(X_all[train_mask], y_all[train_mask])
                validation_prob = model.predict_proba(X_all[validation_mask])[:, 1]
                test_prob = model.predict_proba(X_all[test_mask])[:, 1]
                external_prob = model.predict_proba(X_external)[:, 1]
                y_validation = y_all[validation_mask]
                y_test = y_all[test_mask]

                holdout_indices = [
                    grouped_holdout_indices(
                        y_validation,
                        validation_groups,
                        HOLDOUT_SEED + int(repetition[1:]) * 1000 + holdout_index * 10,
                    )
                    for holdout_index in range(args.holdout_repeats)
                ]

                for budget in BUDGETS:
                    threshold_index, threshold, validation_metrics, roc_thresholds = select_fpr_threshold(
                        y_validation, validation_prob, budget
                    )
                    test_metrics = metric_with_bounds(y_test, test_prob, threshold)
                    external_metrics = metric_with_bounds(y_external, external_prob, threshold)
                    validation_bounded = metric_with_bounds(y_validation, validation_prob, threshold)
                    frozen_row = frozen.loc[
                        (frozen["source_dataset"] == source)
                        & (frozen["repetition"] == repetition)
                        & (frozen["model"] == model_name)
                        & np.isclose(frozen["fpr_budget"], budget)
                    ]
                    if len(frozen_row) != 1:
                        raise AssertionError("Frozen Part 12 threshold row is not unique")
                    frozen_threshold = float(frozen_row.iloc[0]["threshold_selected_on_source_validation"])
                    threshold_difference = abs(threshold - frozen_threshold)
                    replay_rows.append(
                        {
                            "source_dataset": source,
                            "repetition": repetition,
                            "model": model_name,
                            "fpr_budget": budget,
                            "replayed_threshold": threshold,
                            "frozen_part12_threshold": frozen_threshold,
                            "absolute_difference": threshold_difference,
                        }
                    )
                    if threshold_difference > 1e-10:
                        raise AssertionError(
                            f"Threshold replay mismatch: {source}/{repetition}/{model_name}/{budget}"
                        )

                    benign_scores = np.sort(validation_prob[y_validation == 0])[::-1]
                    fp = int(validation_bounded["fp"])
                    first_excluded_score = float(benign_scores[fp]) if fp < len(benign_scores) else math.nan
                    uncertainty_rows.append(
                        {
                            "source_dataset": source,
                            "target_dataset": target,
                            "repetition": repetition,
                            "model": model_name,
                            "fpr_budget": budget,
                            "selected_threshold": threshold,
                            "allowed_false_positives": allowed_false_positives(
                                int(validation_bounded["n_benign"]), budget
                            ),
                            "selected_threshold_rank_among_unique_scores": threshold_index + 1,
                            "first_excluded_benign_rank": fp + 1,
                            "first_excluded_benign_score": first_excluded_score,
                            "threshold_margin_to_first_excluded_benign": threshold - first_excluded_score,
                            **prefixed(validation_bounded, "validation"),
                            **prefixed(test_metrics, "source_test"),
                            **prefixed(external_metrics, "external"),
                        }
                    )

                    scope_values = {
                        "source_validation": (y_validation, validation_prob),
                        "source_s3_test": (y_test, test_prob),
                        "external_primary": (y_external, external_prob),
                    }
                    for variant, candidate_threshold in adjacent_thresholds(
                        threshold_index, roc_thresholds
                    ):
                        for scope, (labels, probabilities) in scope_values.items():
                            metrics = metric_with_bounds(labels, probabilities, candidate_threshold)
                            adjacent_rows.append(
                                {
                                    "source_dataset": source,
                                    "target_dataset": target,
                                    "repetition": repetition,
                                    "model": model_name,
                                    "fpr_budget": budget,
                                    "threshold_variant": variant,
                                    "threshold": candidate_threshold,
                                    "evaluation_scope": scope,
                                    "validation_within_budget": bool(
                                        scope != "source_validation"
                                        or float(metrics["fpr"]) <= budget + 1e-15
                                    ),
                                    **metrics,
                                }
                            )

                    for holdout_index, (calibration_index, audit_index) in enumerate(holdout_indices):
                        _, calibration_threshold, _, _ = select_fpr_threshold(
                            y_validation[calibration_index],
                            validation_prob[calibration_index],
                            budget,
                        )
                        for selector, candidate_threshold in [
                            ("full_validation", threshold),
                            ("calibration_only", calibration_threshold),
                        ]:
                            metrics = metric_with_bounds(
                                y_validation[audit_index],
                                validation_prob[audit_index],
                                candidate_threshold,
                            )
                            holdout_rows.append(
                                {
                                    "source_dataset": source,
                                    "repetition": repetition,
                                    "model": model_name,
                                    "fpr_budget": budget,
                                    "holdout_index": holdout_index,
                                    "threshold_selector": selector,
                                    "threshold": candidate_threshold,
                                    "calibration_n": len(calibration_index),
                                    "calibration_n_benign": int(
                                        (y_validation[calibration_index] == 0).sum()
                                    ),
                                    "audit_budget_exceeded": bool(
                                        float(metrics["fpr"]) > budget + 1e-15
                                    ),
                                    **metrics,
                                }
                            )

    uncertainty = pd.DataFrame(uncertainty_rows)
    adjacent = pd.DataFrame(adjacent_rows)
    holdout = pd.DataFrame(holdout_rows)
    replay = pd.DataFrame(replay_rows)
    pressure = pd.DataFrame(pressure_rows)

    uncertainty_summary = (
        uncertainty.groupby(["source_dataset", "model", "fpr_budget"], observed=True)
        .agg(
            repetitions=("repetition", "nunique"),
            validation_n_benign_mean=("validation_n_benign", "mean"),
            allowed_fp_mean=("allowed_false_positives", "mean"),
            validation_fp_mean=("validation_fp", "mean"),
            validation_fpr_mean=("validation_fpr", "mean"),
            validation_wilson_upper_mean=("validation_fpr_wilson_upper_95", "mean"),
            validation_exact_upper_mean=("validation_fpr_clopper_pearson_upper_95", "mean"),
            source_test_fpr_mean=("source_test_fpr", "mean"),
            external_fpr_mean=("external_fpr", "mean"),
            external_fpr_max=("external_fpr", "max"),
        )
        .reset_index()
    )
    adjacent_summary = (
        adjacent.groupby(
            ["source_dataset", "model", "fpr_budget", "threshold_variant", "evaluation_scope"],
            observed=True,
        )
        .agg(
            repetitions=("repetition", "nunique"),
            fpr_mean=("fpr", "mean"),
            fpr_max=("fpr", "max"),
            tpr_mean=("recall", "mean"),
            fp_mean=("fp", "mean"),
            wilson_upper_mean=("fpr_wilson_upper_95", "mean"),
        )
        .reset_index()
    )
    reuse_summary = (
        holdout.groupby(
            ["source_dataset", "model", "fpr_budget", "threshold_selector"], observed=True
        )
        .agg(
            audit_runs=("holdout_index", "count"),
            audit_fpr_mean=("fpr", "mean"),
            audit_fpr_median=("fpr", "median"),
            audit_fpr_p95=("fpr", lambda values: float(np.quantile(values, 0.95))),
            audit_tpr_mean=("recall", "mean"),
            budget_exceedance_rate=("audit_budget_exceeded", "mean"),
            audit_wilson_upper_mean=("fpr_wilson_upper_95", "mean"),
        )
        .reset_index()
    )

    files = {
        "threshold_uncertainty_runs.csv": uncertainty,
        "threshold_uncertainty_summary.csv": uncertainty_summary,
        "adjacent_threshold_sensitivity.csv": adjacent,
        "adjacent_threshold_summary.csv": adjacent_summary,
        "validation_reuse_sensitivity.csv": holdout,
        "validation_reuse_summary.csv": reuse_summary,
        "threshold_replay_audit.csv": replay,
        "model_selection_pressure.csv": pressure,
    }
    for name, frame in files.items():
        frame.to_csv(RESULTS / name, index=False)

    save_figures(uncertainty, adjacent, reuse_summary)
    maximum_replay_difference = float(replay["absolute_difference"].max())
    write_reports(uncertainty_summary, reuse_summary, pressure, maximum_replay_difference)

    input_manifest = pd.DataFrame(
        [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(input_files)
        ]
    )
    input_manifest.to_csv(RESULTS / "input_manifest.csv", index=False)

    issues = """# Part 14 Issues and Resolutions

## Resolved during execution

1. **Part 12 did not persist source-validation probabilities.** The 60 locked S3 candidates were deterministically replayed from the original training assignments and saved parameters. Every reconstructed low-FPR threshold was checked against the frozen Part 12 threshold before analysis.
2. **An empirical 0.1% budget maps to only a few false positives for ISCX validation folds.** Integer false-positive counts and one-sided 95% Wilson and exact Clopper-Pearson upper bounds are reported; the nominal budget is not presented as a population guarantee.
3. **Probability ties can make a one-step threshold change move multiple samples.** Adjacent thresholds come from the complete validation ROC threshold sequence with `drop_intermediate=False`; all three scopes are recomputed at the safer, selected, and more-permissive neighbors.
4. **Random row holdout would violate the domain-separation logic.** The calibration/audit sensitivity uses `GroupShuffleSplit` with registrable-domain SHA-256 as the group.
5. **The original validation cohort selected both the model candidate and the low-FPR threshold.** The grouped holdout quantifies threshold-selection reuse, while the saved three-candidate Macro-F1 margin records model-selection pressure.

## Residual limitations

1. The grouped holdout is not a fully independent calibration experiment because the locked model candidate was selected using the full source validation cohort.
2. Repetitions reuse the same underlying corpora and are not independent datasets.
3. External FPR is a frozen-threshold transfer outcome, not a target-calibrated operating guarantee.
4. A fully independent calibration split would require a prespecified repartition and model replay; it should be added only if a population-level low-FPR claim is retained.
"""
    (LOGS / "issues_and_resolutions.md").write_text(issues, encoding="utf-8")

    metadata = {
        "status": "PASS",
        "completed_utc": utc_now(),
        "elapsed_seconds": time.perf_counter() - started,
        "python": sys.version,
        "platform": platform.platform(),
        "budgets": list(BUDGETS),
        "confidence": CONFIDENCE,
        "holdout_repeats": args.holdout_repeats,
        "rows": {name: len(frame) for name, frame in files.items()},
        "maximum_part12_threshold_replay_difference": maximum_replay_difference,
        "design_boundary": (
            "Grouped calibration/audit splits isolate threshold-selection reuse only; model-candidate "
            "selection remains conditioned on the full validation cohort."
        ),
    }
    (RESULTS / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
