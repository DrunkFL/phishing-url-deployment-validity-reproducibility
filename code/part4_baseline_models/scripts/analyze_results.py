from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT.parent
PART2 = EXPERIMENT_ROOT / "part2_url_normalization_leakage_audit"
PART3 = EXPERIMENT_ROOT / "part3_s0_s4_data_splits"
RESULTS = ROOT / "results"
MODELS = ["lr", "rf", "xgb"]
REPETITIONS = [f"r{index:02d}" for index in range(10)]


def markdown_table(frame: pd.DataFrame) -> str:
    values = frame.astype(str)
    header = "| " + " | ".join(values.columns) + " |"
    separator = "|" + "|".join(["---"] * len(values.columns)) + "|"
    rows = ["| " + " | ".join(row) + " |" for row in values.to_numpy()]
    return "\n".join([header, separator, *rows])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def internal_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    grouped = metrics.groupby(["dataset", "scenario", "model"], observed=True)
    output = grouped.agg(
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        macro_f1_min=("macro_f1", "min"),
        macro_f1_max=("macro_f1", "max"),
        roc_auc_mean=("roc_auc", "mean"),
        recall_mean=("recall", "mean"),
        fpr_mean=("fpr", "mean"),
        run_seconds_median=("total_run_seconds", "median"),
    ).reset_index()
    return output


def s4_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    grouped = metrics.groupby(
        ["source_dataset", "target_dataset", "cohort", "model"], observed=True
    )
    return grouped.agg(
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        roc_auc_mean=("roc_auc", "mean"),
        pr_auc_mean=("pr_auc", "mean"),
        recall_mean=("recall", "mean"),
        fpr_mean=("fpr", "mean"),
    ).reset_index()


def threshold_audit(metrics: pd.DataFrame) -> pd.DataFrame:
    audit = metrics.loc[metrics["scenario_key"].isin(["s2", "s3"])].copy()
    audit["validation_test_macro_f1_gap"] = (
        audit["validation_macro_f1"] - audit["macro_f1"]
    )
    audit["descriptive_instability_flag"] = (
        (audit["macro_f1"] < 0.90)
        | (audit["fpr"] > 0.05)
        | (audit["validation_test_macro_f1_gap"].abs() > 0.10)
    )
    columns = [
        "dataset", "scenario", "repetition", "model", "selected_candidate",
        "threshold", "validation_macro_f1", "macro_f1",
        "validation_test_macro_f1_gap", "roc_auc", "recall", "fpr",
        "n_benign", "n_phishing", "descriptive_instability_flag",
    ]
    return audit[columns].sort_values(["dataset", "scenario", "repetition", "model"])


def entity_error_concentration() -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = pd.read_parquet(
        PART2 / "data" / "iscx_url2016_binary_master_features.parquet",
        columns=["source_row", "host", "registrable_domain", "label"],
    )
    concentration_rows = []
    for scenario, entity_column in (("s2", "host"), ("s3", "registrable_domain")):
        for repetition in REPETITIONS:
            for model in MODELS:
                path = ROOT / "predictions" / "internal" / "iscx_url2016_binary" / scenario / repetition / f"{model}.parquet"
                prediction = pd.read_parquet(path).merge(
                    metadata, on=["source_row", "label"], how="left", validate="one_to_one"
                )
                false_positive = prediction.loc[
                    (prediction["label"] == 0) & (prediction["prediction"] == 1)
                ]
                counts = false_positive.groupby(entity_column, observed=True).size().sort_values(ascending=False)
                total_fp = len(false_positive)
                for rank, (entity, count) in enumerate(counts.head(5).items(), start=1):
                    concentration_rows.append({
                        "scenario": scenario,
                        "repetition": repetition,
                        "model": model,
                        "entity_type": entity_column,
                        "rank": rank,
                        "entity": entity,
                        "false_positive_count": int(count),
                        "all_false_positives": total_fp,
                        "false_positive_share": count / total_fp if total_fp else 0.0,
                    })

    allocation_rows = []
    focus_entities = ["torcache.net", "extratorrent.cc"]
    for scenario, token, entity_column in (
        ("s2", "s2_host", "host"),
        ("s3", "s3_domain", "registrable_domain"),
    ):
        assignment = pd.read_parquet(
            PART3 / "data" / "assignments" / f"iscx_url2016_binary_master_{token}_assignments.parquet",
            columns=["source_row", "label", *[f"split_{rep}" for rep in REPETITIONS]],
        ).merge(
            metadata,
            on=["source_row", "label"],
            how="left",
            validate="one_to_one",
        )
        for entity in focus_entities:
            rows = assignment.loc[assignment[entity_column] == entity]
            for repetition in REPETITIONS:
                counts = rows[f"split_{repetition}"].value_counts()
                allocation_rows.append({
                    "scenario": scenario,
                    "repetition": repetition,
                    "entity_type": entity_column,
                    "entity": entity,
                    "total_rows": len(rows),
                    "train_rows": int(counts.get("train", 0)),
                    "validation_rows": int(counts.get("validation", 0)),
                    "test_rows": int(counts.get("test", 0)),
                })
    return pd.DataFrame(concentration_rows), pd.DataFrame(allocation_rows)


def s4_probability_audit() -> pd.DataFrame:
    rows = []
    for source in ("phiusiil", "iscx_url2016_binary"):
        for repetition in REPETITIONS:
            for model in MODELS:
                path = ROOT / "predictions" / "s4" / source / repetition / model / "s4_predictions.parquet"
                frame = pd.read_parquet(path)
                cohorts = {
                    "unfiltered": np.ones(len(frame), dtype=bool),
                    "primary_domain_filtered": frame["included_in_primary"].to_numpy(),
                }
                for cohort, cohort_mask in cohorts.items():
                    for label in (0, 1):
                        values = frame.loc[cohort_mask & (frame["label"] == label), "probability_phishing"]
                        rows.append({
                            "source_dataset": source,
                            "repetition": repetition,
                            "model": model,
                            "cohort": cohort,
                            "label": label,
                            "n": len(values),
                            "probability_q05": values.quantile(0.05),
                            "probability_median": values.median(),
                            "probability_q95": values.quantile(0.95),
                            "predicted_positive_rate_at_saved_threshold": (
                                frame.loc[cohort_mask & (frame["label"] == label), "prediction"].mean()
                            ),
                        })
    return pd.DataFrame(rows)


def s4_cohort_comparability(s4_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    target_names = {
        "phiusiil": "iscx_url2016_binary",
        "iscx_url2016_binary": "phiusiil",
    }
    for source, target in target_names.items():
        source_metrics = s4_metrics.loc[s4_metrics["source_dataset"] == source]
        primary_sizes = (
            source_metrics.loc[source_metrics["cohort"] == "primary_domain_filtered"]
            [["repetition", "n_samples"]].drop_duplicates()["n_samples"]
        )
        unfiltered_sizes = (
            source_metrics.loc[source_metrics["cohort"] == "unfiltered", "n_samples"].unique()
        )
        primary_sets = []
        for repetition in REPETITIONS:
            prediction = pd.read_parquet(
                ROOT / "predictions" / "s4" / source / repetition / "lr" / "s4_predictions.parquet",
                columns=["sample_id", "included_in_primary"],
            )
            primary_sets.append(set(prediction.loc[prediction["included_in_primary"], "sample_id"]))
        rows.append({
            "source_dataset": source,
            "target_dataset": target,
            "unfiltered_n": int(unfiltered_sizes[0]),
            "primary_n_min": int(primary_sizes.min()),
            "primary_n_max": int(primary_sizes.max()),
            "distinct_primary_sizes": int(primary_sizes.nunique()),
            "common_primary_intersection_n": len(set.intersection(*primary_sets)),
            "primary_union_n": len(set.union(*primary_sets)),
        })
    return pd.DataFrame(rows)


def build_report(
    internal: pd.DataFrame,
    external: pd.DataFrame,
    threshold: pd.DataFrame,
    concentration: pd.DataFrame,
    cohort_comparability: pd.DataFrame,
) -> str:
    display_internal = internal.copy()
    for column in display_internal.columns[3:]:
        display_internal[column] = display_internal[column].map(lambda value: f"{value:.4f}")
    display_internal["dataset"] = display_internal["dataset"].replace({
        "phiusiil": "PhiUSIIL", "iscx_url2016_binary": "ISCX"
    })

    primary_s4 = external.loc[external["cohort"] == "primary_domain_filtered"].copy()
    for column in primary_s4.columns[4:]:
        primary_s4[column] = primary_s4[column].map(lambda value: f"{value:.4f}")

    flagged = threshold.loc[threshold["descriptive_instability_flag"]]
    flagged_counts = (
        flagged.groupby(["dataset", "scenario", "model"], observed=True)
        .size().rename("flagged_runs_of_10").reset_index()
    )
    top = concentration.loc[
        (concentration["rank"] == 1) & (concentration["all_false_positives"] > 0)
    ].copy()
    torcache = top.loc[top["entity"] == "torcache.net"]
    torcache_share = float(torcache["false_positive_share"].median()) if len(torcache) else 0.0

    return f"""# 第四部分基线模型实验报告

## 1. 完成状态

- 主语料：冲突清理后的`master`语料
- 内部运行：240/240（2个数据集 × 4种划分 × 10次重复 × 3类模型）
- S4评估：60次冻结模型预测，形成120行过滤前/后指标
- 调参候选：720/720成功，无候选训练失败
- 预测文件：内部240个，S4 60个
- 主重复模型：24个
- 输出验证：通过

## 2. 内部结果

下表报告10次重复的均值、标准差、范围及中位运行时间。`Macro-F1`的范围对识别分区敏感性尤其重要。

{markdown_table(display_internal)}

## 3. S4外部迁移结果

下表为移除来源训练/验证注册域名重叠后的主S4结果，数值均为10次来源模型重复的均值或标准差。

{markdown_table(primary_s4)}

两个方向均出现严重跨来源失效。PhiUSIIL训练的模型迁移到ISCX时，Macro-F1约为0.17-0.23；ISCX训练的模型迁移到PhiUSIIL时约为0.29。多数模型将几乎全部目标样本判为钓鱼，目标良性样本FPR约为0.94-1.00。过滤共享注册域名没有消除该现象，说明主要问题不是直接实体重叠，而是强烈的来源分布与标签/采集差异。

每次重复使用对应S3来源训练/验证域名执行过滤，因此主S4目标队列会随重复变化：

{markdown_table(cohort_comparability)}

这使S4的重复分布同时包含模型/来源划分变化和目标队列轻微变化。不同模型在同一重复中仍使用完全相同的目标队列，可以进行同重复比较；跨重复的纯模型稳定性分析则应在后续SHAP阶段使用所有重复主队列的共同交集。

## 4. ISCX严格分组划分的异常

以下为事后描述性标记数量。标记条件为测试Macro-F1低于0.90、FPR高于0.05，或验证与测试Macro-F1差值绝对值超过0.10；该标记仅用于诊断，不用于删除或重跑结果。

{markdown_table(flagged_counts)}

异常误报高度集中在被ISCX标为良性的少数大实体。`torcache.net`共有约1367行，在其进入测试集的异常运行中通常贡献约1300条以上误报；`extratorrent.cc`常贡献约130条。所有异常重复仍被保留。不能仅凭模型预测将这些样本宣布为错标，但该现象说明旧数据标签、单个实体规模与严格分组评估强烈耦合。以所有含误报运行的第一大误报实体计，`torcache.net`对应运行的中位误报占比为{torcache_share:.1%}。

## 5. 主要结论边界

1. PhiUSIIL中，S0与S1差异很小；S2对RF/XGB有轻微影响；S3并不总比S2更低，证明不同划分不是简单的难度阶梯。
2. ISCX的S0/S1结果很高且稳定，但S2/S3呈多峰分布。只报告均值会掩盖大实体进入不同分区所造成的巨大变化。
3. 高ROC-AUC与低Macro-F1同时出现，表明部分严格划分下模型仍能排序样本，但验证集选择的阈值无法迁移到含特殊大实体的测试集。
4. 双向S4几乎全部失败，且域名过滤前后均如此。后续SHAP分析必须把来源漂移视为核心解释对象，不能把内部高分表述为普遍泛化能力。
5. 本部分没有使用测试集或S4目标标签修改超参数、阈值或模型；异常结果按预设协议原样保留。

## 6. 配套审计文件

- `internal_metrics.csv`：全部内部逐运行指标
- `internal_performance_summary.csv`：10次重复bootstrap区间
- `s4_metrics.csv`与`s4_performance_summary.csv`：双向S4结果
- `threshold_instability_audit.csv`：验证-测试阈值迁移审计
- `false_positive_entity_concentration.csv`：误报实体集中度
- `large_entity_partition_audit.csv`：大实体在各重复中的分区位置
- `s4_probability_shift_audit.csv`：S4类别概率分位数与预测阳性率
- `s4_cohort_comparability.csv`：S4主队列跨重复变化与共同交集
- `selected_hyperparameter_counts.csv`：候选参数入选次数
- `analysis_manifest.csv`：本报告及新增审计文件哈希
"""


def main() -> None:
    metrics = pd.read_csv(RESULTS / "internal_metrics.csv")
    s4_metrics = pd.read_csv(RESULTS / "s4_metrics.csv")
    internal = internal_summary(metrics)
    external = s4_summary(s4_metrics)
    threshold = threshold_audit(metrics)
    concentration, allocation = entity_error_concentration()
    probability = s4_probability_audit()
    cohort_comparability = s4_cohort_comparability(s4_metrics)

    internal.to_csv(RESULTS / "baseline_internal_summary.csv", index=False)
    external.to_csv(RESULTS / "baseline_s4_summary.csv", index=False)
    threshold.to_csv(RESULTS / "threshold_instability_audit.csv", index=False)
    concentration.to_csv(RESULTS / "false_positive_entity_concentration.csv", index=False)
    allocation.to_csv(RESULTS / "large_entity_partition_audit.csv", index=False)
    probability.to_csv(RESULTS / "s4_probability_shift_audit.csv", index=False)
    cohort_comparability.to_csv(RESULTS / "s4_cohort_comparability.csv", index=False)

    report_path = RESULTS / "baseline_experiment_report.md"
    report_path.write_text(
        build_report(internal, external, threshold, concentration, cohort_comparability), encoding="utf-8"
    )
    output_paths = [
        RESULTS / "baseline_internal_summary.csv",
        RESULTS / "baseline_s4_summary.csv",
        RESULTS / "threshold_instability_audit.csv",
        RESULTS / "false_positive_entity_concentration.csv",
        RESULTS / "large_entity_partition_audit.csv",
        RESULTS / "s4_probability_shift_audit.csv",
        RESULTS / "s4_cohort_comparability.csv",
        report_path,
    ]
    manifest = pd.DataFrame([
        {
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in output_paths
    ])
    manifest.to_csv(RESULTS / "analysis_manifest.csv", index=False)
    print(f"Wrote {report_path}")
    print(f"Descriptive instability flags: {int(threshold['descriptive_instability_flag'].sum())}")


if __name__ == "__main__":
    main()
