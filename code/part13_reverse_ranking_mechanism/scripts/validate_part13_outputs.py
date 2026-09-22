from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    metadata = json.loads((RESULTS / "run_metadata.json").read_text(encoding="utf-8"))
    require(metadata["status"] == "PASS", "Run metadata is not PASS")

    orientation = pd.read_csv(RESULTS / "probability_orientation_audit.csv")
    association = pd.read_csv(RESULTS / "feature_label_association_summary.csv")
    drift = pd.read_csv(RESULTS / "class_conditional_feature_drift.csv")
    discrimination = pd.read_csv(RESULTS / "grouped_source_discrimination.csv")
    shap = pd.read_csv(RESULTS / "shap_direction_alignment.csv")
    manifest = pd.read_csv(RESULTS / "input_manifest.csv")

    require(len(orientation) == 300, "Orientation audit must contain 300 configurations")
    require(orientation["auc_identity_error"].max() < 1e-4, "AUC complement identity failed")
    require(len(association) == 70, "Association summary must contain 70 rows")
    require(association["feature"].nunique() == 35, "Association summary must cover 35 features")
    require(len(drift) == 70, "Drift table must contain two classes x 35 features")
    require(set(drift["label"]) == {0, 1}, "Drift table must cover both labels")
    require(drift["ks_statistic"].between(0, 1).all(), "Invalid KS statistic")
    require(drift["source_auc_symmetric"].between(0.5, 1).all(), "Invalid symmetric source AUC")
    require(len(discrimination) == 10, "Expected two classes x five grouped splits")
    require(discrimination["source_discrimination_auc"].between(0.5, 1).all(), "Invalid source AUC")
    require((discrimination[["test_phiusiil_n", "test_iscx_n"]] > 0).all().all(), "A test split lacks a source class")
    require(len(shap) == 2100, "SHAP audit must contain 2,100 rows")
    require(shap["feature"].nunique() == 35, "SHAP audit must cover 35 features")

    for row in manifest.itertuples(index=False):
        path = Path(row.path)
        require(path.exists(), f"Missing input: {path}")
        require(sha256(path) == row.sha256, f"Input hash changed: {path}")

    expected_figures = [
        "fig13_1_orientation_auc",
        "fig13_2_class_conditional_drift",
        "fig13_3_shap_direction_conflict",
    ]
    for stem in expected_figures:
        for suffix in [".png", ".pdf"]:
            path = FIGURES / f"{stem}{suffix}"
            require(path.exists() and path.stat().st_size > 1000, f"Missing or empty figure: {path}")

    report = (RESULTS / "PART13_REPORT.md").read_text(encoding="utf-8")
    require("Status: **COMPLETE**" in report, "Completion marker missing from report")
    require("source-conditioned ranking reversal" in report, "Mechanism conclusion missing")

    output_files = [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and "logs" not in path.parts
        and path.name not in {"output_manifest.csv", "validation_summary.json"}
    ]
    output_manifest = pd.DataFrame(
        [
            {
                "relative_path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(output_files)
        ]
    )
    validation = {
        "status": "PASS",
        "checks": 19,
        "orientation_rows": len(orientation),
        "association_rows": len(association),
        "drift_rows": len(drift),
        "source_discrimination_rows": len(discrimination),
        "shap_rows": len(shap),
        "maximum_auc_identity_error": float(orientation["auc_identity_error"].max()),
        "manifested_files": len(output_manifest),
    }
    (RESULTS / "validation_summary.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    output_manifest.to_csv(RESULTS / "output_manifest.csv", index=False)
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
