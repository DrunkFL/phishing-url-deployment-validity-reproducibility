from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
ORIGINAL = ROOT / "code"
EXTERNAL = ROOT / "code" / "part17_phreshphish_external_validation"
PART_ROOT = Path(__file__).resolve().parents[1]
OUT = PART_ROOT / "results"
FIGURE_OUT = PART_ROOT / "figures"

from figure_utils import SVG, axis, render_png, sx, sy  # noqa: E402

ORIGINAL_MAIN = (
    ORIGINAL
    / "part11_dcss_innovation"
    / "results"
    / "main_external_metrics_all.csv"
)
ORIGINAL_LOW_FPR = (
    ORIGINAL
    / "part12_security_privacy_revision"
    / "results"
    / "low_fpr_operating_point_runs.csv"
)
E1_METRICS = EXTERNAL / "results" / "full_split" / "external_metrics.csv"
E3_METRICS = (
    EXTERNAL
    / "results"
    / "reverse_transfer"
    / "server_full"
    / "external_metrics.csv"
)
E2_COMPARISONS = (
    EXTERNAL / "results" / "shap_external" / "comparison_metrics.csv"
)
E2_ASSOCIATIONS = (
    EXTERNAL
    / "results"
    / "shap_external"
    / "stability_performance_association.csv"
)

DATASET_LABEL = {
    "iscx": "ISCX-URL2016",
    "phiusiil": "PhiUSIIL",
    "phreshphish": "PhreshPhish",
}
MODEL_LABEL = {"lr": "LR", "rf": "RF", "xgb": "XGBoost"}
MODEL_MARKER = {"lr": "circle", "rf": "square", "xgb": "triangle"}
MODEL_COLOR = {"lr": "#2878B5", "rf": "#F2A104", "xgb": "#6F4E9C"}
SOURCE_COLOR = {"iscx": "#2878B5", "phiusiil": "#C23B75"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_rows(frame: pd.DataFrame, expected: int, label: str) -> None:
    if len(frame) != expected:
        raise ValueError(f"{label}: expected {expected} rows, found {len(frame)}")


def normalize_dataset(value: str) -> str:
    mapping = {
        "iscx_url2016_binary": "iscx",
        "iscx_master": "iscx",
        "phiusiil": "phiusiil",
        "phiusiil_master": "phiusiil",
        "phreshphish": "phreshphish",
        "phreshphish_train": "phreshphish",
    }
    try:
        return mapping[value]
    except KeyError as exc:
        raise ValueError(f"Unexpected dataset identifier: {value}") from exc


def validate_design(frame: pd.DataFrame, expected_rows: int, label: str) -> None:
    require_rows(frame, expected_rows, label)
    counts = frame.groupby(["source", "target", "model"]).size()
    if set(counts.tolist()) != {10}:
        raise ValueError(f"{label}: each direction/model must contain 10 repetitions")
    directions = set(zip(frame["source"], frame["target"]))
    if len(directions) != expected_rows // 30:
        raise ValueError(f"{label}: unexpected direction count {len(directions)}")


def load_macro_transfer() -> pd.DataFrame:
    original = pd.read_csv(ORIGINAL_MAIN)
    original = original[
        (original["feature_set"] == "f_all")
        & (original["cohort"] == "primary_domain_filtered")
    ].copy()
    require_rows(original, 60, "original S4 macro-threshold input")
    original["source"] = original["source_dataset"].map(normalize_dataset)
    original["target"] = original["target_dataset"].map(normalize_dataset)
    original["evidence_state"] = "user_authoritative"
    original["evidence_role"] = "original two-source primary analysis"
    original["alerts_per_10000_benign"] = original["fpr"] * 10000

    e1 = pd.read_csv(E1_METRICS)
    e1 = e1[
        (e1["cohort"] == "primary_domain_filtered")
        & (e1["threshold_kind"] == "macro_f1")
        & (e1["weighting"] == "row")
    ].copy()
    require_rows(e1, 60, "E1 macro-threshold input")
    e1["source"] = e1["source_dataset"].map(normalize_dataset)
    e1["target"] = e1["target_dataset"].map(normalize_dataset)
    e1["evidence_state"] = "verified"
    e1["evidence_role"] = "independent third-source evaluation"

    e3 = pd.read_csv(E3_METRICS)
    e3 = e3[
        e3["target_dataset"].isin(["iscx_master", "phiusiil_master"])
        & (e3["cohort"] == "primary_domain_filtered")
        & (e3["threshold_kind"] == "macro_f1")
        & (e3["weighting"] == "row")
    ].copy()
    require_rows(e3, 60, "E3 macro-threshold input")
    e3["source"] = e3["source_dataset"].map(normalize_dataset)
    e3["target"] = e3["target_dataset"].map(normalize_dataset)
    e3["evidence_state"] = "verified"
    e3["evidence_role"] = "secondary reciprocal-transfer diagnostic"

    columns = [
        "source",
        "target",
        "repetition",
        "model",
        "roc_auc",
        "macro_f1",
        "fpr",
        "alerts_per_10000_benign",
        "n_samples",
        "n_benign",
        "n_phishing",
        "evidence_state",
        "evidence_role",
    ]
    combined = pd.concat(
        [original[columns], e1[columns], e3[columns]], ignore_index=True
    )
    validate_design(combined, 180, "six-direction macro-threshold design")
    return combined


def load_low_fpr_transfer() -> pd.DataFrame:
    original = pd.read_csv(ORIGINAL_LOW_FPR)
    original = original[original["evaluation_scope"] == "external_primary"].copy()
    require_rows(original, 120, "original S4 low-FPR input")
    original["source"] = original["source_dataset"].map(normalize_dataset)
    original["target"] = original["target_dataset"].map(normalize_dataset)
    original["budget"] = original["fpr_budget"]
    original["evidence_state"] = "user_authoritative"
    original["evidence_role"] = "original two-source primary analysis"

    e1 = pd.read_csv(E1_METRICS)
    e1 = e1[
        (e1["cohort"] == "primary_domain_filtered")
        & (e1["threshold_kind"].isin(["fpr_0.001", "fpr_0.01"]))
        & (e1["weighting"] == "row")
    ].copy()
    require_rows(e1, 120, "E1 low-FPR input")
    e1["source"] = e1["source_dataset"].map(normalize_dataset)
    e1["target"] = e1["target_dataset"].map(normalize_dataset)
    e1["budget"] = e1["source_validation_fpr_budget"]
    e1["evidence_state"] = "verified"
    e1["evidence_role"] = "independent third-source evaluation"

    e3 = pd.read_csv(E3_METRICS)
    e3 = e3[
        e3["target_dataset"].isin(["iscx_master", "phiusiil_master"])
        & (e3["cohort"] == "primary_domain_filtered")
        & (e3["threshold_kind"].isin(["fpr_0.001", "fpr_0.01"]))
        & (e3["weighting"] == "row")
    ].copy()
    require_rows(e3, 120, "E3 low-FPR input")
    e3["source"] = e3["source_dataset"].map(normalize_dataset)
    e3["target"] = e3["target_dataset"].map(normalize_dataset)
    e3["budget"] = e3["source_validation_fpr_budget"]
    e3["evidence_state"] = "verified"
    e3["evidence_role"] = "secondary reciprocal-transfer diagnostic"

    columns = [
        "source",
        "target",
        "repetition",
        "model",
        "budget",
        "fpr",
        "alerts_per_10000_benign",
        "n_samples",
        "n_benign",
        "n_phishing",
        "evidence_state",
        "evidence_role",
    ]
    combined = pd.concat(
        [original[columns], e1[columns], e3[columns]], ignore_index=True
    )
    require_rows(combined, 360, "six-direction low-FPR design")
    counts = combined.groupby(["source", "target", "model", "budget"]).size()
    if set(counts.tolist()) != {10} or len(counts) != 36:
        raise ValueError("Low-FPR design must contain 36 cells with 10 repetitions each")
    combined["nominal_budget_exceedance"] = combined["fpr"] / combined["budget"]
    combined["exceeds_nominal_budget"] = combined["fpr"] > combined["budget"]
    return combined


def build_summaries(macro: pd.DataFrame, low_fpr: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_summary = (
        macro.groupby(
            ["source", "target", "model", "evidence_state", "evidence_role"],
            as_index=False,
        )
        .agg(
            n_repetitions=("repetition", "size"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_sd=("roc_auc", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_sd=("macro_f1", "std"),
            fpr_mean=("fpr", "mean"),
            fpr_sd=("fpr", "std"),
            alerts_per_10000_benign_mean=("alerts_per_10000_benign", "mean"),
            n_samples_min=("n_samples", "min"),
            n_samples_max=("n_samples", "max"),
        )
        .sort_values(["source", "target", "model"])
    )

    budget_model = (
        low_fpr.groupby(
            ["source", "target", "model", "budget", "evidence_state", "evidence_role"],
            as_index=False,
        )
        .agg(
            n_repetitions=("repetition", "size"),
            target_fpr_mean=("fpr", "mean"),
            target_fpr_sd=("fpr", "std"),
            budget_exceedance_mean=("nominal_budget_exceedance", "mean"),
            budget_exceedance_min=("nominal_budget_exceedance", "min"),
            budget_exceedance_max=("nominal_budget_exceedance", "max"),
            proportion_runs_exceeding_budget=("exceeds_nominal_budget", "mean"),
            alerts_per_10000_benign_mean=("alerts_per_10000_benign", "mean"),
        )
        .sort_values(["source", "target", "budget", "model"])
    )

    budget_direction = (
        budget_model.groupby(["source", "target", "budget"], as_index=False)
        .agg(
            model_family_exceedance_min=("budget_exceedance_mean", "min"),
            model_family_exceedance_max=("budget_exceedance_mean", "max"),
            model_family_exceedance_mean=("budget_exceedance_mean", "mean"),
            target_fpr_model_mean_min=("target_fpr_mean", "min"),
            target_fpr_model_mean_max=("target_fpr_mean", "max"),
        )
        .sort_values(["source", "target", "budget"])
    )
    return model_summary, budget_model, budget_direction


def load_explanation_summary() -> tuple[pd.DataFrame, pd.DataFrame, float]:
    comparisons = pd.read_csv(E2_COMPARISONS)
    require_rows(comparisons, 60, "E2 explanation-performance comparisons")
    comparisons["source"] = comparisons["source_dataset"].map(normalize_dataset)
    summary = (
        comparisons.groupby(["source", "model"], as_index=False)
        .agg(
            n_repetitions=("repetition", "size"),
            top15_jaccard_mean=("top15_jaccard", "mean"),
            rank_spearman_mean=("rank_spearman", "mean"),
            direction_agreement_mean=("direction_agreement_top20_union", "mean"),
            roc_auc_loss_mean=("roc_auc_loss", "mean"),
            macro_f1_loss_mean=("macro_f1_loss", "mean"),
            target_fpr_mean=("target_fpr", "mean"),
        )
        .sort_values(["source", "model"])
    )
    associations = pd.read_csv(E2_ASSOCIATIONS)
    rho_row = associations[
        (associations["scope"] == "all")
        & (associations["stability_metric"] == "top15_jaccard")
        & (associations["loss_metric"] == "roc_auc_loss")
    ]
    if len(rho_row) != 1:
        raise ValueError("Missing unique pooled Top-15-Jaccard/ROC-AUC-loss association")
    rho = float(rho_row.iloc[0]["spearman_rho_descriptive"])
    return comparisons, summary, rho


def direction_label(source: str, target: str) -> str:
    return f"{DATASET_LABEL[source]} → {DATASET_LABEL[target]}"


def draw_figure(
    model_summary: pd.DataFrame,
    budget_direction: pd.DataFrame,
    comparisons: pd.DataFrame,
    rho: float,
) -> None:
    directions = [
        ("iscx", "phiusiil"),
        ("phiusiil", "iscx"),
        ("iscx", "phreshphish"),
        ("phreshphish", "iscx"),
        ("phiusiil", "phreshphish"),
        ("phreshphish", "phiusiil"),
    ]
    y_labels = [direction_label(s, t) for s, t in directions]
    svg = SVG(width=2400, height=1600)
    svg.text(
        80,
        72,
        "Directional transfer and disagreement among deployment-validity claims",
        42,
        700,
    )
    svg.text(
        80,
        116,
        "Three sources, all six directed transfers; source-selected rules remain frozen on each target",
        25,
        fill="#4B5563",
    )

    # Panel A: all six directional edges, grouped into reciprocal pairs.
    svg.text(80, 184, "A  Complete directed transfer landscape", 29, 700)
    ax_x0, ax_y0, ax_w, ax_h = 500, 220, 1800, 520
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        x = sx(float(tick), ax_x0, ax_w, 0.0, 1.0)
        svg.line(x, ax_y0, x, ax_y0 + ax_h, "#E2E8F0", 2)
        svg.text(x, ax_y0 + ax_h + 40, f"{tick:.1f}", 21, anchor="middle", fill="#4B5563")
    svg.line(ax_x0, ax_y0 + ax_h, ax_x0 + ax_w, ax_y0 + ax_h, "#495057", 2)
    chance_x = sx(0.5, ax_x0, ax_w, 0.0, 1.0)
    svg.line(chance_x, ax_y0, chance_x, ax_y0 + ax_h, "#68788A", 3, "10,8")
    svg.text(chance_x + 10, ax_y0 + 26, "chance", 19, fill="#68788A")

    row_y = [270 + 80 * i for i in range(len(directions))]
    offsets = {"lr": -18, "rf": 0, "xgb": 18}
    for y, (source, target) in zip(row_y, directions):
        svg.text(470, y + 7, direction_label(source, target), 22, anchor="end", fill="#202124")
        subset = model_summary[
            (model_summary["source"] == source)
            & (model_summary["target"] == target)
        ]
        if len(subset) != 3:
            raise ValueError(f"Missing model summary for {source}->{target}")
        for row in subset.itertuples(index=False):
            cy = y + offsets[row.model]
            mean_x = sx(row.roc_auc_mean, ax_x0, ax_w, 0.0, 1.0)
            lo_x = sx(max(0.0, row.roc_auc_mean - row.roc_auc_sd), ax_x0, ax_w, 0.0, 1.0)
            hi_x = sx(min(1.0, row.roc_auc_mean + row.roc_auc_sd), ax_x0, ax_w, 0.0, 1.0)
            svg.line(lo_x, cy, hi_x, cy, MODEL_COLOR[row.model], 3)
            svg.line(lo_x, cy - 7, lo_x, cy + 7, MODEL_COLOR[row.model], 2)
            svg.line(hi_x, cy - 7, hi_x, cy + 7, MODEL_COLOR[row.model], 2)
            svg.marker(mean_x, cy, MODEL_MARKER[row.model], MODEL_COLOR[row.model], 9)
    svg.line(ax_x0, 388, ax_x0 + ax_w, 388, "#CBD5E1", 2)
    svg.line(ax_x0, 548, ax_x0 + ax_w, 548, "#CBD5E1", 2)
    svg.text(ax_x0 + ax_w / 2, 800, "Target ROC-AUC (mean ± SD across 10 repetitions)", 23, 600, anchor="middle")
    legend_x = 1720
    for i, model in enumerate(["lr", "rf", "xgb"]):
        x = legend_x + i * 190
        svg.marker(x, 190, MODEL_MARKER[model], MODEL_COLOR[model], 9)
        svg.text(x + 20, 197, MODEL_LABEL[model], 21)

    # Panel B: nominal-budget exceedance. Cell text is the range of model-family means.
    svg.text(80, 890, "B  Target-FPR exceedance of nominal source budget", 29, 700)
    heat_x0, heat_y0, cell_w, cell_h = 500, 945, 235, 68
    svg.text(heat_x0 + cell_w / 2, 930, "0.1% budget", 21, 600, anchor="middle")
    svg.text(heat_x0 + cell_w * 1.5, 930, "1% budget", 21, 600, anchor="middle")
    budgets = [0.001, 0.01]
    for i, (source, target) in enumerate(directions):
        y = heat_y0 + i * cell_h
        svg.text(470, y + 42, direction_label(source, target), 20, anchor="end", fill="#202124")
        for j, budget in enumerate(budgets):
            row = budget_direction[
                (budget_direction["source"] == source)
                & (budget_direction["target"] == target)
                & np.isclose(budget_direction["budget"], budget)
            ]
            if len(row) != 1:
                raise ValueError(f"Missing budget summary for {source}->{target}, {budget}")
            record = row.iloc[0]
            log_value = float(np.log10(record["model_family_exceedance_mean"]))
            lo = record["model_family_exceedance_min"]
            hi = record["model_family_exceedance_max"]
            if hi < 10:
                label = f"{lo:.2f}–{hi:.2f}×"
            elif lo < 100:
                label = f"{lo:.1f}–{hi:.1f}×"
            else:
                label = f"{lo:.0f}–{hi:.0f}×"
            if log_value < 0:
                fill, text_color = "#2C7BB6", "#FFFFFF"
            elif log_value < 1:
                fill, text_color = "#ABD9E9", "#111827"
            elif log_value < 2:
                fill, text_color = "#FFFFBF", "#111827"
            elif log_value < 3:
                fill, text_color = "#FDAE61", "#111827"
            else:
                fill, text_color = "#D7191C", "#FFFFFF"
            x = heat_x0 + j * cell_w
            svg.rect(x, y, cell_w - 4, cell_h - 4, fill, "#FFFFFF", 2)
            svg.text(x + (cell_w - 4) / 2, y + 41, label, 20, 700, anchor="middle", fill=text_color)

    legend_items = [
        ("#2C7BB6", "<1×"),
        ("#ABD9E9", "1–10×"),
        ("#FFFFBF", "10–100×"),
        ("#FDAE61", "100–1000×"),
        ("#D7191C", "≥1000×"),
    ]
    svg.text(1035, 980, "Target FPR / nominal budget", 20, 600)
    for i, (fill, label) in enumerate(legend_items):
        y = 1005 + i * 52
        svg.rect(1035, y, 40, 30, fill, "#FFFFFF", 1)
        svg.text(1090, y + 23, label, 19, fill="#374151")
    svg.text(1035, 1305, "Cell text: range across", 18, fill="#4B5563")
    svg.text(1035, 1332, "three model-family means", 18, fill="#4B5563")

    # Panel C: one focused association view, avoiding the dense six-panel candidate.
    svg.text(1360, 890, "C  Explanation agreement does not certify transfer", 29, 700)
    cx0, cy0, cw, ch = 1470, 950, 820, 470
    highlight_x = sx(0.75, cx0, cw, 0.4, 1.0)
    highlight_y = sy(0.70, cy0, ch, 0.1, 0.7)
    highlight_bottom = sy(0.25, cy0, ch, 0.1, 0.7)
    svg.rect(highlight_x, highlight_y, cx0 + cw - highlight_x, highlight_bottom - highlight_y, "#FCE8EC", "none", opacity=0.55)
    axis(
        svg,
        cx0,
        cy0,
        cw,
        ch,
        0.4,
        1.0,
        0.1,
        0.7,
        [0.4, 0.6, 0.8, 1.0],
        [0.1, 0.3, 0.5, 0.7],
        xfmt=lambda value: f"{value:.1f}",
    )
    for row in comparisons.itertuples(index=False):
        x = sx(row.top15_jaccard, cx0, cw, 0.4, 1.0)
        y = sy(row.roc_auc_loss, cy0, ch, 0.1, 0.7)
        svg.marker(x, y, MODEL_MARKER[row.model], SOURCE_COLOR[row.source], 7, opacity=0.72)
    svg.text(cx0 + cw - 15, cy0 + 28, "high agreement with large loss", 18, 700, anchor="end", fill="#8B1E3F")
    svg.text(cx0 + 18, cy0 + 38, f"Spearman ρ = {rho:.3f}", 20, 700, fill="#374151")
    svg.text(cx0 + 18, cy0 + 65, "n = 60 dependent comparisons; descriptive", 17, fill="#4B5563")
    svg.text(cx0 + cw / 2, cy0 + ch + 52, "Source-to-PhreshPhish Top-15 SHAP Jaccard", 21, 600, anchor="middle")
    svg.text(1368, cy0 + ch / 2, "ROC-AUC loss", 21, 600, anchor="middle")

    legend_y = 1510
    svg.circle(1500, legend_y - 8, 8, SOURCE_COLOR["iscx"])
    svg.text(1518, legend_y, "ISCX source", 18)
    svg.circle(1665, legend_y - 8, 8, SOURCE_COLOR["phiusiil"])
    svg.text(1683, legend_y, "PhiUSIIL source", 18)
    for i, model in enumerate(["lr", "rf", "xgb"]):
        x = 1890 + i * 130
        svg.marker(x, legend_y - 8, MODEL_MARKER[model], "#6B7280", 7)
        svg.text(x + 17, legend_y, MODEL_LABEL[model], 18)

    svg.text(
        80,
        1570,
        "Panel B uses nominal budgets, not achieved source FPRs. E3 supplies reciprocal directionality only and is not another independent source.",
        19,
        fill="#4B5563",
    )
    FIGURE_OUT.mkdir(parents=True, exist_ok=True)
    out_svg = FIGURE_OUT / "Figure_10_directional_validity_synthesis.svg"
    svg.save(out_svg)
    render_png(out_svg, FIGURE_OUT / "Figure_10_directional_validity_synthesis.png", 2400, 1600)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = [
        ORIGINAL_MAIN,
        ORIGINAL_LOW_FPR,
        E1_METRICS,
        E3_METRICS,
        E2_COMPARISONS,
        E2_ASSOCIATIONS,
    ]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)

    macro = load_macro_transfer()
    low_fpr = load_low_fpr_transfer()
    model_summary, budget_model, budget_direction = build_summaries(macro, low_fpr)
    comparisons, explanation_summary, rho = load_explanation_summary()

    macro.to_csv(OUT / "directed_transfer_run_level.csv", index=False)
    model_summary.to_csv(OUT / "directed_transfer_model_summary.csv", index=False)
    low_fpr.to_csv(OUT / "threshold_budget_run_level.csv", index=False)
    budget_model.to_csv(OUT / "threshold_budget_model_summary.csv", index=False)
    budget_direction.to_csv(OUT / "threshold_budget_direction_summary.csv", index=False)
    explanation_summary.to_csv(OUT / "explanation_performance_summary.csv", index=False)
    draw_figure(model_summary, budget_direction, comparisons, rho)

    manifest = {
        "purpose": "Derived synthesis of frozen results; no model fitting or threshold reselection",
        "input_files": [
            {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in inputs
        ],
        "design_counts": {
            "macro_threshold_rows": len(macro),
            "low_fpr_rows": len(low_fpr),
            "directed_edges": 6,
            "models": 3,
            "repetitions_per_edge_model": 10,
            "explanation_comparisons": len(comparisons),
        },
        "evidence_boundaries": [
            "Original ISCX/PhiUSIIL edges remain user-authoritative.",
            "E1 and E2 are primary verified additions.",
            "E3 is a secondary reciprocal-transfer diagnostic, not another independent source.",
            "Budget exceedance is relative to the nominal source-validation budget, not the achieved source FPR.",
            "Explanation-performance associations are descriptive and non-causal.",
        ],
    }
    (OUT / "INPUT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
