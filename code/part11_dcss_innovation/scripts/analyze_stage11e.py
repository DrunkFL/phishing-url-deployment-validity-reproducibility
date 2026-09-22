from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
PART6 = ROOT.parent / "part6_stable_feature_selection"
RESULTS = ROOT / "results"
ABLATIONS = ("no_direction", "no_rank_dispersion", "no_frequency")
METRICS = ("macro_f1", "roc_auc", "pr_auc", "fpr", "brier_score", "log_loss", "ece_15")


def summarize(frame: pd.DataFrame, group_columns: list[str], metrics: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(group_columns, sort=True, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        base = dict(zip(group_columns, keys))
        for metric in metrics:
            values = group[metric].astype(float)
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "n": len(values),
                    "mean": values.mean(),
                    "std": values.std(ddof=1),
                    "median": values.median(),
                    "min": values.min(),
                    "max": values.max(),
                }
            )
    return pd.DataFrame(rows)


def load_full_and_ablations() -> tuple[pd.DataFrame, pd.DataFrame]:
    full_i = pd.read_csv(RESULTS / "dcss_main_internal_metrics.csv")
    full_i = full_i.loc[full_i["feature_set"] == "f_dcss_15"].copy()
    full_i["feature_set"] = "dcss_full"
    full_i["family"] = "ablation_reference"
    full_e = pd.read_csv(RESULTS / "dcss_main_external_metrics.csv")
    full_e = full_e.loc[full_e["feature_set"] == "f_dcss_15"].copy()
    full_e["feature_set"] = "dcss_full"
    full_e["family"] = "ablation_reference"
    ablation_i = pd.read_csv(RESULTS / "stage11e_ablation_internal_metrics.csv")
    ablation_e = pd.read_csv(RESULTS / "stage11e_ablation_external_metrics.csv")
    return pd.concat([full_i, ablation_i], ignore_index=True), pd.concat([full_e, ablation_e], ignore_index=True)


def ablation_analysis(internal: pd.DataFrame, external: pd.DataFrame) -> None:
    summarize(internal, ["dataset", "model", "feature_set"], METRICS).to_csv(
        RESULTS / "stage11e_ablation_internal_summary.csv", index=False
    )
    primary = external.loc[external["cohort"] == "primary_domain_filtered"].copy()
    summarize(primary, ["source_dataset", "target_dataset", "model", "feature_set"], METRICS).to_csv(
        RESULTS / "stage11e_ablation_external_summary.csv", index=False
    )
    full = primary.loc[primary["feature_set"] == "dcss_full", ["source_dataset", "target_dataset", "repetition", "model", *METRICS]]
    pairs = []
    for method in ABLATIONS:
        current = primary.loc[primary["feature_set"] == method, ["source_dataset", "target_dataset", "repetition", "model", *METRICS]]
        merged = current.merge(full, on=["source_dataset", "target_dataset", "repetition", "model"], suffixes=("_ablation", "_full"), validate="one_to_one")
        for row in merged.itertuples(index=False):
            record = {
                "source_dataset": row.source_dataset,
                "target_dataset": row.target_dataset,
                "repetition": row.repetition,
                "model": row.model,
                "ablation": method,
            }
            for metric in METRICS:
                record[f"{metric}_delta"] = float(getattr(row, f"{metric}_ablation") - getattr(row, f"{metric}_full"))
            pairs.append(record)
    paired = pd.DataFrame(pairs)
    paired.to_csv(RESULTS / "stage11e_ablation_paired_external.csv", index=False)
    summarize(
        paired.rename(columns={f"{metric}_delta": metric for metric in METRICS}),
        ["source_dataset", "target_dataset", "model", "ablation"],
        METRICS,
    ).to_csv(RESULTS / "stage11e_ablation_paired_external_summary.csv", index=False)


def random_analysis(internal: pd.DataFrame, external: pd.DataFrame) -> None:
    random_i = pd.read_csv(RESULTS / "stage11e_random_internal_metrics.csv")
    random_e = pd.read_csv(RESULTS / "stage11e_random_external_metrics.csv")
    random_e = random_e.loc[random_e["cohort"] == "primary_domain_filtered"].copy()
    full_i = internal.loc[internal["feature_set"] == "dcss_full"]
    full_e = external.loc[(external["feature_set"] == "dcss_full") & (external["cohort"] == "primary_domain_filtered")]

    seed_i = random_i.groupby(["dataset", "model", "random_seed"], sort=True)[["macro_f1", "roc_auc"]].mean().reset_index()
    seed_e = random_e.groupby(["source_dataset", "target_dataset", "model", "random_seed"], sort=True)[["macro_f1", "roc_auc", "ece_15", "fpr"]].mean().reset_index()
    seed_i.to_csv(RESULTS / "stage11e_random_internal_seed_means.csv", index=False)
    seed_e.to_csv(RESULTS / "stage11e_random_external_seed_means.csv", index=False)

    rows = []
    for keys, group in seed_i.groupby(["dataset", "model"], sort=True):
        reference = full_i.loc[(full_i["dataset"] == keys[0]) & (full_i["model"] == keys[1])]
        for metric in ("macro_f1", "roc_auc"):
            values = group[metric].to_numpy(float)
            full_value = float(reference[metric].mean())
            rows.append({"scope": "internal", "dataset": keys[0], "target_dataset": "", "model": keys[1], "metric": metric, "random_seeds": len(values), "random_mean": values.mean(), "random_std": values.std(ddof=1), "random_q05": np.quantile(values, 0.05), "random_median": np.median(values), "random_q95": np.quantile(values, 0.95), "dcss_full_mean": full_value, "fraction_random_ge_dcss": float(np.mean(values >= full_value))})
    for keys, group in seed_e.groupby(["source_dataset", "target_dataset", "model"], sort=True):
        reference = full_e.loc[(full_e["source_dataset"] == keys[0]) & (full_e["target_dataset"] == keys[1]) & (full_e["model"] == keys[2])]
        for metric in ("macro_f1", "roc_auc", "ece_15", "fpr"):
            values = group[metric].to_numpy(float)
            full_value = float(reference[metric].mean())
            favorable_random = values <= full_value if metric in {"ece_15", "fpr"} else values >= full_value
            rows.append({"scope": "external", "dataset": keys[0], "target_dataset": keys[1], "model": keys[2], "metric": metric, "random_seeds": len(values), "random_mean": values.mean(), "random_std": values.std(ddof=1), "random_q05": np.quantile(values, 0.05), "random_median": np.median(values), "random_q95": np.quantile(values, 0.95), "dcss_full_mean": full_value, "fraction_random_at_least_as_favorable_as_dcss": float(np.mean(favorable_random))})
    pd.DataFrame(rows).to_csv(RESULTS / "stage11e_random_distribution_summary.csv", index=False)


def feature_membership_analysis() -> pd.DataFrame:
    scores = pd.read_csv(RESULTS / "dcss_scores.csv")
    transition_rows = []
    key_rows = []
    for keys, group in scores.groupby(["dataset", "repetition", "model"], sort=True):
        stable_path = PART6 / "feature_sets" / keys[0] / "s3" / keys[1] / keys[2] / "feature_sets.json"
        stable = set(json.loads(stable_path.read_text(encoding="utf-8"))["f_stable"])
        dcss = set(group.loc[group["selected_15"].astype(bool), "feature"])
        local = []
        for row in group.itertuples(index=False):
            if row.feature in stable and row.feature in dcss:
                membership = "shared"
            elif row.feature in stable:
                membership = "stable_only"
            elif row.feature in dcss:
                membership = "dcss_only"
            else:
                membership = "neither"
            record = {
                "dataset": keys[0], "repetition": keys[1], "model": keys[2],
                "feature": row.feature, "membership": membership,
                "mean_normalized_importance": row.mean_normalized_importance,
                "mean_rank": row.mean_rank, "rank_dispersion": row.rank_dispersion,
                "direction_consistency": row.direction_consistency,
                "direction_conflict_rate": 1.0 - row.direction_consistency,
                "top15_frequency": row.top15_frequency, "dcss_15": row.dcss_15,
            }
            transition_rows.append(record)
            local.append(record)
        local_frame = pd.DataFrame(local)
        stable_only = local_frame.loc[local_frame["membership"] == "stable_only"]
        selected = local_frame.loc[local_frame["membership"].isin(["shared", "dcss_only"])]
        key_rows.append(
            {
                "dataset": keys[0], "repetition": keys[1], "model": keys[2],
                "n_stable": len(stable), "n_dcss": len(dcss), "n_shared": len(stable & dcss),
                "n_stable_only": len(stable - dcss), "n_dcss_only": len(dcss - stable),
                "stable_only_mean_rank_dispersion": stable_only["rank_dispersion"].mean(),
                "dcss_selected_mean_rank_dispersion": selected["rank_dispersion"].mean(),
                "rank_dispersion_delta_stable_only_minus_dcss": stable_only["rank_dispersion"].mean() - selected["rank_dispersion"].mean(),
            }
        )
    pd.DataFrame(transition_rows).to_csv(RESULTS / "stage11e_feature_membership.csv", index=False)
    key_frame = pd.DataFrame(key_rows)
    key_frame.to_csv(RESULTS / "stage11e_feature_membership_by_key.csv", index=False)
    return key_frame


def mean_pairwise_fold_jaccard(group: pd.DataFrame) -> float:
    sets = {fold: set(rows.nsmallest(15, "rank")["feature"]) for fold, rows in group.groupby("held_fold")}
    values = [len(sets[a] & sets[b]) / len(sets[a] | sets[b]) for a, b in combinations(sorted(sets), 2)]
    if len(values) != 10:
        raise ValueError("Expected five held-domain folds and ten pairs")
    return float(np.mean(values))


def mechanism_analysis(internal: pd.DataFrame, external: pd.DataFrame) -> None:
    held = pd.read_csv(RESULTS / "stage11e_held_domain_metrics.csv")
    held_summary = held.groupby(["dataset", "repetition", "model"], sort=True).agg(
        held_fold_count=("held_fold", "size"), held_auc_mean=("roc_auc", "mean"), held_auc_std=("roc_auc", "std"),
        held_macro_f1_mean=("macro_f1", "mean"), held_macro_f1_std=("macro_f1", "std"),
    ).reset_index()
    held_summary.to_csv(RESULTS / "stage11e_held_domain_variance.csv", index=False)

    fold_shap = pd.read_csv(RESULTS / "dcss_fold_shap_summary.csv")
    fold_jaccard = fold_shap.groupby(["dataset", "repetition", "model"], sort=True).apply(mean_pairwise_fold_jaccard, include_groups=False).rename("fold_top15_jaccard").reset_index()
    scores = pd.read_csv(RESULTS / "dcss_scores.csv")
    selected = scores.loc[scores["selected_15"].astype(bool)]
    score_summary = selected.groupby(["dataset", "repetition", "model"], sort=True).agg(
        selected_mean_rank_dispersion=("rank_dispersion", "mean"),
        selected_mean_direction_conflict=("direction_consistency", lambda values: float((1.0 - values).mean())),
        selected_mean_top15_frequency=("top15_frequency", "mean"),
    ).reset_index()
    diagnostics = held_summary.merge(fold_jaccard, on=["dataset", "repetition", "model"], validate="one_to_one").merge(score_summary, on=["dataset", "repetition", "model"], validate="one_to_one")

    full_i = internal.loc[internal["feature_set"] == "dcss_full", ["dataset", "repetition", "model", "roc_auc"]].rename(columns={"roc_auc": "internal_dcss_auc"})
    primary = external.loc[external["cohort"] == "primary_domain_filtered"]
    full_e = primary.loc[primary["feature_set"] == "dcss_full", ["source_dataset", "target_dataset", "repetition", "model", "roc_auc", "ece_15"]].rename(columns={"source_dataset": "dataset", "roc_auc": "external_dcss_auc", "ece_15": "external_dcss_ece"})
    stable_raw = pd.read_csv(RESULTS / "main_external_metrics_all.csv")
    stable = stable_raw.loc[(stable_raw["cohort"] == "primary_domain_filtered") & (stable_raw["feature_set"] == "f_stable"), ["source_dataset", "target_dataset", "repetition", "model", "roc_auc", "ece_15"]].rename(columns={"source_dataset": "dataset", "roc_auc": "external_stable_auc", "ece_15": "external_stable_ece"})
    diagnostics = diagnostics.merge(full_i, on=["dataset", "repetition", "model"], validate="one_to_one").merge(full_e, on=["dataset", "repetition", "model"], validate="one_to_one").merge(stable, on=["dataset", "target_dataset", "repetition", "model"], validate="one_to_one")
    diagnostics["external_auc_delta_dcss_minus_stable"] = diagnostics["external_dcss_auc"] - diagnostics["external_stable_auc"]
    diagnostics["external_ece_delta_dcss_minus_stable"] = diagnostics["external_dcss_ece"] - diagnostics["external_stable_ece"]
    diagnostics["dcss_internal_minus_external_auc"] = diagnostics["internal_dcss_auc"] - diagnostics["external_dcss_auc"]
    diagnostics.to_csv(RESULTS / "stage11e_source_diagnostics_and_external_outcomes.csv", index=False)

    predictors = ["held_auc_std", "held_macro_f1_std", "fold_top15_jaccard", "selected_mean_rank_dispersion", "selected_mean_direction_conflict", "selected_mean_top15_frequency", "external_ece_delta_dcss_minus_stable"]
    outcomes = ["external_auc_delta_dcss_minus_stable", "dcss_internal_minus_external_auc"]
    rows = []
    scopes = [("all_keys", diagnostics)] + [(f"{dataset}|{model}", group) for (dataset, model), group in diagnostics.groupby(["dataset", "model"], sort=True)]
    for scope, group in scopes:
        for predictor in predictors:
            for outcome in outcomes:
                valid = group[[predictor, outcome]].dropna()
                rho, p_value = spearmanr(valid[predictor], valid[outcome]) if len(valid) >= 3 else (np.nan, np.nan)
                rows.append({"scope": scope, "n_outer_keys": len(valid), "predictor": predictor, "outcome": outcome, "spearman_rho": rho, "nominal_p_value": p_value, "inference": "descriptive_dependent_outer_keys"})
    pd.DataFrame(rows).to_csv(RESULTS / "stage11e_diagnostic_correlations.csv", index=False)


def ablation_overlap() -> None:
    selected = pd.read_csv(RESULTS / "stage11e_ablation_feature_sets.csv")
    rows = []
    for keys, group in selected.groupby(["dataset", "repetition", "model"], sort=True):
        full = set(group.loc[group["method"] == "dcss_full", "feature"])
        for method in ABLATIONS:
            current = set(group.loc[group["method"] == method, "feature"])
            rows.append({"dataset": keys[0], "repetition": keys[1], "model": keys[2], "ablation": method, "intersection": len(full & current), "jaccard": len(full & current) / len(full | current), "features_added_json": json.dumps(sorted(current - full)), "features_removed_json": json.dumps(sorted(full - current))})
    pd.DataFrame(rows).to_csv(RESULTS / "stage11e_ablation_feature_overlap.csv", index=False)


def main() -> None:
    internal, external = load_full_and_ablations()
    ablation_analysis(internal, external)
    random_analysis(internal, external)
    feature_membership_analysis()
    mechanism_analysis(internal, external)
    ablation_overlap()
    summary = {
        "stage": "11E", "status": "ANALYZED",
        "ablation_internal_rows_including_full": len(internal),
        "ablation_external_rows_including_full": len(external),
        "random_internal_rows": len(pd.read_csv(RESULTS / "stage11e_random_internal_metrics.csv")),
        "random_external_rows": len(pd.read_csv(RESULTS / "stage11e_random_external_metrics.csv")),
        "held_domain_rows": len(pd.read_csv(RESULTS / "stage11e_held_domain_metrics.csv")),
    }
    (RESULTS / "stage11e_analysis_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
