from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
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
    uncertainty = pd.read_csv(RESULTS / "threshold_uncertainty_runs.csv")
    adjacent = pd.read_csv(RESULTS / "adjacent_threshold_sensitivity.csv")
    holdout = pd.read_csv(RESULTS / "validation_reuse_sensitivity.csv")
    replay = pd.read_csv(RESULTS / "threshold_replay_audit.csv")
    pressure = pd.read_csv(RESULTS / "model_selection_pressure.csv")

    require(len(uncertainty) == 120, "Expected 120 uncertainty rows")
    require(len(adjacent) == 1080, "Expected 1,080 adjacent-threshold rows")
    require(len(holdout) == 4800, "Expected 4,800 validation-reuse rows")
    require(len(replay) == 120, "Expected 120 replay rows")
    require(len(pressure) == 60, "Expected 60 model-selection rows")
    require(replay["absolute_difference"].max() <= 1e-10, "Part 12 threshold replay mismatch")
    require((uncertainty["validation_fp"] <= uncertainty["allowed_false_positives"]).all(), "Integer FP budget violated")
    require((uncertainty["validation_fpr_wilson_upper_95"] >= uncertainty["validation_fpr"]).all(), "Wilson upper bound below estimate")
    require((uncertainty["validation_fpr_clopper_pearson_upper_95"] >= uncertainty["validation_fpr"]).all(), "Exact upper bound below estimate")
    require(set(adjacent["threshold_variant"]) == {"safer_neighbor", "selected", "more_permissive_neighbor"}, "Missing threshold variant")
    require(set(holdout["threshold_selector"]) == {"full_validation", "calibration_only"}, "Missing reuse selector")
    require(np.isfinite(pressure["selection_margin_macro_f1"]).all(), "Non-finite selection margin")

    manifest = pd.read_csv(RESULTS / "input_manifest.csv")
    for row in manifest.itertuples(index=False):
        path = Path(row.path)
        require(path.exists(), f"Missing input: {path}")
        require(sha256(path) == row.sha256, f"Input hash changed: {path}")

    for stem in [
        "fig14_1_fpr_uncertainty",
        "fig14_2_adjacent_threshold_sensitivity",
        "fig14_3_validation_reuse_audit",
    ]:
        for suffix in [".png", ".pdf"]:
            path = FIGURES / f"{stem}{suffix}"
            require(path.exists() and path.stat().st_size > 1000, f"Missing figure: {path}")

    report = (RESULTS / "PART14_REPORT.md").read_text(encoding="utf-8")
    require("Status: **COMPLETE**" in report, "Completion marker missing")
    require("not a fully independent calibration experiment" in report, "Design boundary missing")

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
    output_manifest.to_csv(RESULTS / "output_manifest.csv", index=False)
    summary = {
        "status": "PASS",
        "checks": 18,
        "uncertainty_rows": len(uncertainty),
        "adjacent_rows": len(adjacent),
        "holdout_rows": len(holdout),
        "replay_rows": len(replay),
        "model_selection_rows": len(pressure),
        "maximum_threshold_replay_difference": float(replay["absolute_difference"].max()),
        "manifested_files": len(output_manifest),
    }
    (RESULTS / "validation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
