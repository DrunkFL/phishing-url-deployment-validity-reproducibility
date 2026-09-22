from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import ks_2samp, spearmanr, wasserstein_distance
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT.parent
PART2 = EXPERIMENT_ROOT / "part2_url_normalization_leakage_audit"
PART5 = EXPERIMENT_ROOT / "part5_shap_stability"
PART11 = EXPERIMENT_ROOT / "part11_dcss_innovation"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
LOGS = ROOT / "logs"

DATASETS = ["phiusiil", "iscx_url2016_binary"]
DISPLAY = {"phiusiil": "PhiUSIIL", "iscx_url2016_binary": "ISCX-URL2016"}
RANDOM_SEEDS = [20261301, 20261302, 20261303, 20261304, 20261305]
DEADZONE = 0.02


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sign_with_deadzone(value: float, deadzone: float = DEADZONE) -> int:
    if pd.isna(value) or abs(value) < deadzone:
        return 0
    return 1 if value > 0 else -1


def symmetric_source_auc(y_source: np.ndarray, values: np.ndarray) -> float:
    auc = roc_auc_score(y_source, values)
    return float(max(auc, 1.0 - auc))


def feature_names() -> list[str]:
    dictionary = pd.read_csv(PART2 / "results" / "feature_dictionary.csv")
    names = dictionary.sort_values("feature_order")["feature"].tolist()
    if len(names) != 35 or len(set(names)) != 35:
        raise ValueError("Expected exactly 35 frozen URL features")
    return names


def load_features(dataset: str, names: list[str]) -> pd.DataFrame:
    path = PART2 / "data" / f"{dataset}_master_features.parquet"
    frame = pd.read_parquet(path, columns=["label", "registrable_domain", *names])
    if set(frame["label"].unique()) != {0, 1}:
        raise ValueError(f"Unexpected label coding for {dataset}")
    if frame[names].isna().any().any():
        raise ValueError(f"Missing frozen feature values for {dataset}")
    return frame


def build_orientation_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(PART11 / "results" / "external_probability_orientation_audit.csv")
    primary = source.loc[source["cohort"] == "primary_domain_filtered"].copy()
    expected = 2 * 10 * 3 * 5
    if len(primary) != expected:
        raise ValueError(f"Expected {expected} primary orientation rows, got {len(primary)}")

    primary["auc_identity_error"] = (
        primary["roc_auc"] + primary["inverted_roc_auc"] - 1.0
    ).abs()
    primary["ordinary_below_chance"] = primary["roc_auc"] < 0.5
    primary["strong_reversal"] = primary["roc_auc"] < 0.2
    primary.to_csv(RESULTS / "probability_orientation_audit.csv", index=False)

    summary = (
        primary.groupby(["source_dataset", "target_dataset", "model", "feature_set"], sort=True)
        .agg(
            repetitions=("repetition", "nunique"),
            auc_mean=("roc_auc", "mean"),
            auc_std=("roc_auc", "std"),
            auc_min=("roc_auc", "min"),
            auc_max=("roc_auc", "max"),
            inverted_auc_mean=("inverted_roc_auc", "mean"),
            below_chance_runs=("ordinary_below_chance", "sum"),
            strong_reversal_runs=("strong_reversal", "sum"),
            max_auc_identity_error=("auc_identity_error", "max"),
        )
        .reset_index()
    )
    summary.to_csv(RESULTS / "probability_orientation_summary.csv", index=False)
    return primary, summary


def build_feature_association_summary() -> pd.DataFrame:
    source = pd.read_csv(PART11 / "results" / "source_target_feature_direction_audit.csv")
    if len(source) != 700:
        raise ValueError(f"Expected 700 feature-direction rows, got {len(source)}")
    rows: list[dict[str, object]] = []
    for keys, group in source.groupby(["source_dataset", "target_dataset", "feature"], sort=True):
        source_rho = float(group["source_train_spearman_with_label"].mean())
        target_rho = float(group["target_primary_spearman_with_label"].mean())
        rows.append(
            {
                "source_dataset": keys[0],
                "target_dataset": keys[1],
                "feature": keys[2],
                "source_rho_mean": source_rho,
                "target_rho_mean": target_rho,
                "source_direction_deadzone": sign_with_deadzone(source_rho),
                "target_direction_deadzone": sign_with_deadzone(target_rho),
                "direction_flip_deadzone": (
                    sign_with_deadzone(source_rho) != 0
                    and sign_with_deadzone(target_rho) != 0
                    and sign_with_deadzone(source_rho) != sign_with_deadzone(target_rho)
                ),
                "absolute_rho_change": abs(target_rho - source_rho),
                "legacy_nonzero_flip_rate": float(group["nonzero_direction_flip"].mean()),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(RESULTS / "feature_label_association_summary.csv", index=False)
    return result


def build_class_conditional_drift(
    frames: dict[str, pd.DataFrame], names: list[str]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    left_name, right_name = DATASETS
    for label in [0, 1]:
        left = frames[left_name].loc[frames[left_name]["label"] == label]
        right = frames[right_name].loc[frames[right_name]["label"] == label]
        y_source = np.r_[np.zeros(len(left), dtype=np.int8), np.ones(len(right), dtype=np.int8)]
        for feature in names:
            x_left = left[feature].to_numpy(float)
            x_right = right[feature].to_numpy(float)
            values = np.r_[x_left, x_right]
            pooled_iqr = float(np.subtract(*np.percentile(values, [75, 25])))
            median_gap = float(np.median(x_right) - np.median(x_left))
            rho, _ = spearmanr(values, y_source)
            rows.append(
                {
                    "label": label,
                    "class_name": "benign" if label == 0 else "phishing",
                    "feature": feature,
                    "phiusiil_n": len(x_left),
                    "iscx_n": len(x_right),
                    "phiusiil_median": float(np.median(x_left)),
                    "iscx_median": float(np.median(x_right)),
                    "median_gap_iscx_minus_phiusiil": median_gap,
                    "pooled_iqr": pooled_iqr,
                    "robust_standardized_median_gap": median_gap / pooled_iqr if pooled_iqr > 0 else np.nan,
                    "ks_statistic": float(ks_2samp(x_left, x_right, method="asymp").statistic),
                    "wasserstein_distance": float(wasserstein_distance(x_left, x_right)),
                    "source_auc_symmetric": symmetric_source_auc(y_source, values),
                    "source_spearman": float(rho) if not pd.isna(rho) else np.nan,
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(RESULTS / "class_conditional_feature_drift.csv", index=False)
    return result


def grouped_source_discrimination(
    frames: dict[str, pd.DataFrame], names: list[str]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label in [0, 1]:
        pieces = []
        for source_id, dataset in enumerate(DATASETS):
            current = frames[dataset].loc[frames[dataset]["label"] == label, ["registrable_domain", *names]].copy()
            current["source_id"] = source_id
            pieces.append(current)
        data = pd.concat(pieces, ignore_index=True)
        groups = data["registrable_domain"].fillna("<missing>").astype(str).to_numpy()
        X = data[names]
        y = data["source_id"].to_numpy(np.int8)
        for split_index, seed in enumerate(RANDOM_SEEDS):
            splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
            train_idx, test_idx = next(splitter.split(X, y, groups))
            pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            C=1.0,
                            class_weight="balanced",
                            max_iter=2000,
                            random_state=seed,
                        ),
                    ),
                ]
            )
            pipeline.fit(X.iloc[train_idx], y[train_idx])
            probability = pipeline.predict_proba(X.iloc[test_idx])[:, 1]
            rows.append(
                {
                    "label": label,
                    "class_name": "benign" if label == 0 else "phishing",
                    "split": split_index,
                    "seed": seed,
                    "train_n": len(train_idx),
                    "test_n": len(test_idx),
                    "train_domains": len(set(groups[train_idx])),
                    "test_domains": len(set(groups[test_idx])),
                    "test_phiusiil_n": int((y[test_idx] == 0).sum()),
                    "test_iscx_n": int((y[test_idx] == 1).sum()),
                    "source_discrimination_auc": float(roc_auc_score(y[test_idx], probability)),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(RESULTS / "grouped_source_discrimination.csv", index=False)
    joblib.dump(
        {"method": "grouped logistic source discrimination", "seeds": RANDOM_SEEDS, "features": names},
        RESULTS / "source_discrimination_configuration.joblib",
    )
    return result


def build_shap_direction_audit(association: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    shap = pd.read_csv(PART5 / "results" / "all_global_feature_summaries.csv")
    external = shap.loc[(shap["scenario"] == "s3") & (shap["cohort"] == "external_common")].copy()
    if len(external) != 2100:
        raise ValueError(f"Expected 2100 external SHAP rows, got {len(external)}")
    external = external.rename(columns={"dataset": "source_dataset", "run_id": "repetition"})
    merged = external.merge(
        association,
        on=["source_dataset", "target_dataset", "feature"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if (merged["_merge"] != "both").any():
        raise ValueError("Missing target association after SHAP join")
    merged = merged.drop(columns="_merge")
    merged["source_direction_deadzone"] = merged["source_direction_deadzone"].fillna(0).astype(int)
    merged["target_direction_deadzone"] = merged["target_direction_deadzone"].fillna(0).astype(int)
    merged["shap_direction"] = merged["direction_sign"].fillna(0).astype(int)
    merged["shap_vs_source_label_conflict"] = (
        (merged["shap_direction"] != 0)
        & (merged["source_direction_deadzone"] != 0)
        & (merged["shap_direction"] != merged["source_direction_deadzone"])
    )
    merged["shap_vs_target_label_conflict"] = (
        (merged["shap_direction"] != 0)
        & (merged["target_direction_deadzone"] != 0)
        & (merged["shap_direction"] != merged["target_direction_deadzone"])
    )
    merged["top10_external"] = merged["rank"] <= 10
    merged.to_csv(RESULTS / "shap_direction_alignment.csv", index=False)

    summary_rows = []
    for keys, group in merged.groupby(["source_dataset", "target_dataset", "model"], sort=True):
        top = group.loc[group["top10_external"]]
        summary_rows.append(
            {
                "source_dataset": keys[0],
                "target_dataset": keys[1],
                "model": keys[2],
                "rows_all": len(group),
                "target_conflict_rate_all": float(group["shap_vs_target_label_conflict"].mean()),
                "target_conflict_rate_top10": float(top["shap_vs_target_label_conflict"].mean()),
                "source_conflict_rate_top10": float(top["shap_vs_source_label_conflict"].mean()),
                "mean_top10_direction_flip_rate": float(top["direction_flip_deadzone"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS / "shap_direction_alignment_summary.csv", index=False)
    return merged, summary


def make_figures(
    orientation: pd.DataFrame,
    drift: pd.DataFrame,
    shap_summary: pd.DataFrame,
) -> None:
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9})

    order = orientation.sort_values("auc_mean").reset_index(drop=True)
    labels = [f"{DISPLAY[r.source_dataset]}->{DISPLAY[r.target_dataset]} | {r.model.upper()} | {r.feature_set}" for r in order.itertuples()]
    fig, ax = plt.subplots(figsize=(8.4, 8.0))
    colors = np.where(order["auc_mean"] < 0.5, "#c94c4c", "#2b7a78")
    ax.barh(np.arange(len(order)), order["auc_mean"], color=colors)
    ax.axvline(0.5, color="black", linewidth=0.9, linestyle="--")
    ax.set_yticks(np.arange(len(order)), labels)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Ordinary target ROC-AUC (mean over 10 repetitions)")
    ax.set_title("Cross-source ranking orientation")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig13_1_orientation_auc.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig13_1_orientation_auc.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    top = drift.sort_values("ks_statistic", ascending=False).groupby("class_name", sort=False).head(12)
    top = top.sort_values(["class_name", "ks_statistic"]).reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2), sharex=True)
    for ax, class_name in zip(axes, ["benign", "phishing"]):
        current = top.loc[top["class_name"] == class_name]
        ax.barh(current["feature"], current["ks_statistic"], color="#3a6ea5" if class_name == "benign" else "#d17a22")
        ax.set_title(class_name.capitalize())
        ax.set_xlabel("Cross-source KS statistic")
        ax.set_xlim(0, 1)
    fig.suptitle("Largest class-conditional feature shifts")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig13_2_class_conditional_drift.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig13_2_class_conditional_drift.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    plot = shap_summary.copy()
    plot["key"] = plot.apply(lambda r: f"{DISPLAY[r.source_dataset]}->{DISPLAY[r.target_dataset]} | {r.model.upper()}", axis=1)
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    x = np.arange(len(plot))
    ax.bar(x - 0.18, plot["source_conflict_rate_top10"], width=0.36, label="SHAP vs source-label direction", color="#4c78a8")
    ax.bar(x + 0.18, plot["target_conflict_rate_top10"], width=0.36, label="SHAP vs target-label direction", color="#e45756")
    ax.set_xticks(x, plot["key"], rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Conflict rate among top-10 features")
    ax.legend(frameon=False)
    ax.set_title("Model explanation direction versus label association")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig13_3_shap_direction_conflict.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig13_3_shap_direction_conflict.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(
    orientation: pd.DataFrame,
    association: pd.DataFrame,
    drift: pd.DataFrame,
    discrimination: pd.DataFrame,
    shap_summary: pd.DataFrame,
    elapsed: float,
) -> None:
    reverse = orientation.loc[
        (orientation["source_dataset"] == "phiusiil")
        & (orientation["target_dataset"] == "iscx_url2016_binary")
        & (orientation["model"] == "xgb")
    ]
    flip_counts = association.groupby(["source_dataset", "target_dataset"])["direction_flip_deadzone"].sum()
    disc = discrimination.groupby("class_name")["source_discrimination_auc"].agg(["mean", "std", "min", "max"])
    top_drift = drift.sort_values("ks_statistic", ascending=False).groupby("class_name").head(5)
    top_lines = "\n".join(
        f"- {r.class_name}: `{r.feature}` (KS={r.ks_statistic:.3f}, symmetric source AUC={r.source_auc_symmetric:.3f})"
        for r in top_drift.itertuples()
    )
    shap_lines = "\n".join(
        f"- {DISPLAY[r.source_dataset]} to {DISPLAY[r.target_dataset]}, {r.model.upper()}: top-10 target conflict={r.target_conflict_rate_top10:.1%}, source conflict={r.source_conflict_rate_top10:.1%}."
        for r in shap_summary.itertuples()
    )
    report = f"""# Part 13 Reverse-Ranking Mechanism Audit

Status: **COMPLETE**  
Runtime: {elapsed:.1f} seconds  
Analysis role: post-hoc diagnostic; target labels are never used as a deployable correction.

## Questions and answers

### 1. Was the reverse AUC caused by a label or probability-column error?

No evidence supports that explanation. The inherited Stage 11B audit checked 300 fitted-model configurations: all model class orders were `[0, 1]`, stored threshold decisions matched the phishing-probability column, and the maximum reconstruction probability difference was below `3e-8`. In this independent Part 13 extraction, the largest numerical error in the identity `AUC(p) + AUC(1-p) = 1` was {orientation['max_auc_identity_error'].max():.3g}.

For PhiUSIIL-to-ISCX XGBoost, ordinary mean AUC spans {reverse['auc_mean'].min():.4f}-{reverse['auc_mean'].max():.4f} across the five feature methods, whereas diagnostic `AUC(1-p)` spans {reverse['inverted_auc_mean'].min():.4f}-{reverse['inverted_auc_mean'].max():.4f}. This is genuine source-conditioned ranking reversal, not a corrected model result.

### 2. Do feature-label associations change across sources?

Yes. With a conservative Spearman dead-zone of |rho| < {DEADZONE:.2f}, {int(flip_counts.loc[('phiusiil', 'iscx_url2016_binary')])}/35 feature directions reverse for PhiUSIIL-to-ISCX and {int(flip_counts.loc[('iscx_url2016_binary', 'phiusiil')])}/35 reverse in the opposite direction. The result is descriptive because both datasets are observational collections with different sampling processes.

### 3. Are the two sources distinguishable even within the same class?

Yes, strongly. A logistic source classifier evaluated with registrable-domain-grouped splits achieved mean AUC {disc.loc['benign', 'mean']:.4f} (range {disc.loc['benign', 'min']:.4f}-{disc.loc['benign', 'max']:.4f}) among benign URLs and {disc.loc['phishing', 'mean']:.4f} (range {disc.loc['phishing', 'min']:.4f}-{disc.loc['phishing', 'max']:.4f}) among phishing URLs. This supports collection-source shift within each class and cannot be explained only by different class prevalence.

Largest class-conditional shifts:

{top_lines}

### 4. Do learned explanation directions align with target-label associations?

Not consistently. The top-10 external SHAP features show the following direction-conflict rates:

{shap_lines}

This comparison separates two ideas: SHAP direction describes how a fitted source model uses a feature on an explanation cohort, while label association describes how the feature orders target labels. Their disagreement is evidence of mechanism mismatch, not causal proof.

## Defensible mechanism statement

The below-chance external AUC is best described as **source-conditioned ranking reversal associated with class-conditional covariate and association shift**. The evidence rules out label coding and probability-column mistakes, demonstrates that both benign and phishing samples are readily distinguishable by collection source under grouped evaluation, and shows that many feature-label directions reverse between corpora. These analyses do not identify a single causal factor, and reversing scores with target labels is diagnostic only.

## Limits

- Dataset source is confounded with collection period, curation rules, URL activity status, and sampling policy.
- Univariate KS and Spearman statistics do not capture all feature interactions.
- Grouped source discrimination reduces registrable-domain leakage but cannot remove higher-level infrastructure or temporal dependence.
- The SHAP analysis reuses frozen explanation cohorts from Part 5; it does not create a new target-tuned model.
- Statistical p-values were intentionally omitted for the very large samples; effect sizes and repeated grouped splits are more informative here.

## Output map

- `probability_orientation_audit.csv`: 300 configuration-level ordinary and inverted AUC records.
- `feature_label_association_summary.csv`: 70 source-target-feature association records.
- `class_conditional_feature_drift.csv`: 70 class-feature cross-source drift records.
- `grouped_source_discrimination.csv`: 10 grouped source-classifier evaluations.
- `shap_direction_alignment.csv`: 2,100 feature/model/repetition explanation-alignment records.
- `input_manifest.csv`: hashes of every direct input.
- `issues_and_resolutions.md`: problems, decisions, and residual limitations.
"""
    (RESULTS / "PART13_REPORT.md").write_text(report, encoding="utf-8")


def write_issues() -> None:
    text = """# Part 13 Issues and Resolutions

## Resolved during execution

1. **Float32 probability ties changed forensic AUC recomputation.** Earlier prediction files stored probabilities as float32, while metrics were computed from float64 values. Part 13 uses the validated Stage 11B regenerated audit table and checks the AUC complement identity; it does not reinterpret small Parquet recomputation differences as substantive findings.
2. **Zero-valued or near-zero associations can create unstable sign flips.** A fixed `|Spearman rho| < 0.02` dead-zone is used for the primary sign-flip count. The legacy strict nonzero flip rate is retained as a sensitivity column.
3. **Ordinary random splits could leak domains into source discrimination.** All source-classifier evaluations use `GroupShuffleSplit` with registrable domain as the group.
4. **Source classification could be driven by class prevalence.** Separate models are fitted inside benign and phishing strata, so prevalence cannot explain the source AUC.
5. **SHAP sign and label association answer different questions.** The output names the comparison `direction alignment`, not SHAP correctness, and reports source-label and target-label conflict separately.
6. **The first complete run stopped on a false missing-join alarm.** All 2,100 SHAP rows matched, but constant features legitimately had undefined Spearman correlations. The validation was changed to inspect the merge indicator directly; undefined directions are retained as neutral `0` and excluded from conflicts.

## Residual limitations

1. Source identity remains a bundle of collection-time, curation, and sampling effects; this audit does not isolate one causal driver.
2. The source-discrimination model uses all 35 lexical features and is diagnostic. It is not a phishing detector and is not proposed for deployment.
3. Target labels are used only after model fitting for diagnosis. Reversed-score AUC must never be presented as an achievable zero-target-feedback operating result.
4. Existing Part 5 SHAP cohorts are reused so the analysis remains tied to frozen experiments; no new target-optimized explanations are produced.
"""
    (LOGS / "issues_and_resolutions.md").write_text(text, encoding="utf-8")


def main() -> None:
    started = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    names = feature_names()
    direct_inputs = [
        PART2 / "results" / "feature_dictionary.csv",
        PART2 / "data" / "phiusiil_master_features.parquet",
        PART2 / "data" / "iscx_url2016_binary_master_features.parquet",
        PART5 / "results" / "all_global_feature_summaries.csv",
        PART11 / "results" / "external_probability_orientation_audit.csv",
        PART11 / "results" / "source_target_feature_direction_audit.csv",
        PART11 / "results" / "external_model_class_order_audit.csv",
        PART11 / "results" / "probability_orientation_summary.json",
    ]
    for path in direct_inputs:
        if not path.exists():
            raise FileNotFoundError(path)

    frames = {dataset: load_features(dataset, names) for dataset in DATASETS}
    _, orientation = build_orientation_audit()
    association = build_feature_association_summary()
    drift = build_class_conditional_drift(frames, names)
    discrimination = grouped_source_discrimination(frames, names)
    _, shap_summary = build_shap_direction_audit(association)
    make_figures(orientation, drift, shap_summary)
    elapsed = time.time() - started
    write_report(orientation, association, drift, discrimination, shap_summary, elapsed)
    write_issues()

    manifest = pd.DataFrame(
        [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in direct_inputs]
    )
    manifest.to_csv(RESULTS / "input_manifest.csv", index=False)
    metadata = {
        "status": "PASS",
        "elapsed_seconds": elapsed,
        "python": sys.version,
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "features": len(names),
        "random_seeds": RANDOM_SEEDS,
        "association_deadzone": DEADZONE,
        "output_counts": {
            "orientation_summary_rows": len(orientation),
            "association_rows": len(association),
            "drift_rows": len(drift),
            "source_discrimination_rows": len(discrimination),
            "shap_summary_rows": len(shap_summary),
        },
    }
    (RESULTS / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
