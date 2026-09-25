from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageStat


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT.parent
PART4 = EXPERIMENT_ROOT / "part4_baseline_models"
PART8 = EXPERIMENT_ROOT / "part8_traditional_baselines_and_error_analysis"
FIGURES = ROOT / "figures"
FIGURE_DATA = ROOT / "figure_data"
TABLES = ROOT / "tables"
RESULTS = ROOT / "results"


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_files() -> None:
    pngs = sorted(FIGURES.glob("*.png"))
    pdfs = sorted(FIGURES.glob("*.pdf"))
    if len(pngs) != 6 or len(pdfs) != 6:
        raise AssertionError(f"Expected 6 PNG and 6 PDF figures, found {len(pngs)} and {len(pdfs)}")
    if {path.stem for path in pngs} != {path.stem for path in pdfs}:
        raise AssertionError("PNG/PDF figure names do not match")
    for path in pngs:
        with Image.open(path) as image:
            if image.width < 1800 or image.height < 700:
                raise AssertionError(f"Figure resolution too small: {path} {image.size}")
            grayscale = image.convert("L")
            if ImageStat.Stat(grayscale).stddev[0] < 10:
                raise AssertionError(f"Figure appears blank: {path}")
            extrema = grayscale.getextrema()
            if extrema[0] > 100 or extrema[1] < 240:
                raise AssertionError(f"Unexpected figure tonal range: {path} {extrema}")
    for path in pdfs:
        if path.stat().st_size < 5000 or path.read_bytes()[:4] != b"%PDF":
            raise AssertionError(f"Invalid PDF figure: {path}")


def validate_tables() -> None:
    expected = {
        "table1_dataset_audit.csv": 2,
        "table2_internal_regime_performance.csv": 24,
        "table3_shap_stability.csv": 24,
        "table4_s3_feature_methods.csv": 30,
        "table5_external_transfer.csv": 30,
        "table6_external_paired_macro_f1.csv": 24,
        "table7_false_positive_domains.csv": 10,
    }
    for filename, rows in expected.items():
        frame = pd.read_csv(TABLES / filename)
        if len(frame) != rows:
            raise AssertionError(f"{filename}: expected {rows} rows, found {len(frame)}")
        markdown = TABLES / filename.replace(".csv", ".md")
        if not markdown.exists() or markdown.stat().st_size < 100:
            raise AssertionError(f"Missing Markdown table: {markdown}")


def validate_figure_data() -> None:
    expected = {
        "fig1_internal_regime_performance.csv": 24,
        "fig2_shap_stability_estimands.csv": 12,
        "fig3_feature_count.csv": 10,
        "fig3_feature_reduction_internal_delta.csv": 300,
        "fig4_external_transfer_performance.csv": 10,
        "fig5_stability_vs_external_performance.csv": 24,
        "fig6_false_positive_domain_structure.csv": 10,
    }
    for filename, rows in expected.items():
        found = len(pd.read_csv(FIGURE_DATA / filename))
        if found != rows:
            raise AssertionError(f"{filename}: expected {rows}, found {found}")

    source = pd.read_csv(PART4 / "results" / "internal_metrics.csv")
    source_mean = source.groupby(["dataset", "scenario_key", "model"])["macro_f1"].mean().sort_index()
    figure = pd.read_csv(FIGURE_DATA / "fig1_internal_regime_performance.csv")
    figure_mean = figure.set_index(["dataset", "scenario_key", "model"])["macro_f1_mean"].sort_index()
    if not np.allclose(source_mean.to_numpy(), figure_mean.to_numpy(), atol=1e-15, rtol=1e-15):
        raise AssertionError("Figure 1 values do not reproduce source metrics")

    external = pd.read_csv(PART8 / "results" / "external_metrics_all_five.csv")
    external = external.loc[external["cohort"].eq("primary_domain_filtered")].copy()
    external["direction"] = external["source_dataset"].map({
        "iscx_url2016_binary": "ISCX -> PhiUSIIL",
        "phiusiil": "PhiUSIIL -> ISCX",
    })
    source_external = external.groupby(["direction", "feature_set"])[["macro_f1", "fpr"]].mean().sort_index()
    figure_external = pd.read_csv(FIGURE_DATA / "fig4_external_transfer_performance.csv")
    figure_external = figure_external.set_index(["direction", "feature_set"])[
        ["macro_f1_mean", "fpr_mean"]
    ].sort_index()
    if not np.allclose(source_external.to_numpy(), figure_external.to_numpy(), atol=1e-15, rtol=1e-15):
        raise AssertionError("Figure 4 values do not reproduce source metrics")


def validate_manifest() -> None:
    manifest = pd.read_csv(RESULTS / "publication_artifact_manifest.csv")
    expected_files = [
        path for folder in [FIGURES, FIGURE_DATA, TABLES]
        for path in folder.iterdir() if path.is_file()
    ]
    if len(manifest) != len(expected_files):
        raise AssertionError("Manifest file count mismatch")
    for row in manifest.itertuples(index=False):
        path = ROOT / row.path
        if path.stat().st_size != row.bytes or hash_file(path) != row.sha256:
            raise AssertionError(f"Manifest mismatch: {path}")


def validate_manuscript() -> None:
    required = {
        "results_and_discussion_en.md": ["0.1089", "0.2908", "0.1937", "rho was 0.19"],
        "results_and_discussion_zh.md": ["0.1089", "0.2908", "0.1937", "Spearman rho"],
        "figure_captions_en.md": ["Figure 1", "Figure 6", "bootstrap 95%"],
    }
    for filename, markers in required.items():
        path = ROOT / "manuscript" / filename
        text = path.read_text(encoding="utf-8")
        if any(marker not in text for marker in markers):
            raise AssertionError(f"Missing required manuscript result in {filename}")
        if any(marker in text for marker in ["TBD", "TODO", "[insert", "待填写"]):
            raise AssertionError(f"Unresolved placeholder in {filename}")


def main() -> None:
    validate_files()
    validate_tables()
    validate_figure_data()
    validate_manifest()
    validate_manuscript()
    report = {
        "status": "PASS",
        "model_training_performed": False,
        "target_feedback_used": False,
        "png_figures": 6,
        "pdf_figures": 6,
        "figure_data_files": 7,
        "tables_csv": 7,
        "tables_markdown": 7,
        "visual_nonblank_checks": "PASS",
        "source_value_reproduction": "PASS",
        "manuscript_placeholder_check": "PASS",
        "manifest_integrity": "PASS",
    }
    (RESULTS / "validation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
