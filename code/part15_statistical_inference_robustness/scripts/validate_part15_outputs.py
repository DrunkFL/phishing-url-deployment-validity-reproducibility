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
    pairs = pd.read_csv(RESULTS / "paired_differences_long.csv")
    wide = pd.read_csv(RESULTS / "paired_differences_wide.csv")
    summary = pd.read_csv(RESULTS / "effect_size_and_auxiliary_tests.csv")
    dependence = pd.read_csv(RESULTS / "repetition_dependence_sensitivity.csv")
    verdict = pd.read_csv(RESULTS / "robustness_verdicts.csv")
    bootstrap = pd.read_csv(RESULTS / "descriptive_repetition_bootstrap.csv")

    require(metadata["status"] == "PASS", "Run metadata is not PASS")
    require(len(pairs) == 80, "Expected 80 disclosed paired differences")
    require(len(wide) == 8, "Expected eight wide paired-difference groups")
    require(len(summary) == 8, "Expected eight effect groups")
    require(len(dependence) == 48, "Expected eight groups x six rho values")
    require(len(verdict) == 8, "Expected eight robustness verdicts")
    require(len(bootstrap) == 160000, "Unexpected descriptive bootstrap row count")
    require((pairs.groupby(["hypothesis", "group_key"]).size() == 10).all(), "A group lacks ten repetitions")
    require(set(dependence["assumed_pairwise_correlation"].round(2)) == {0.0, 0.1, 0.25, 0.5, 0.75, 0.9}, "Rho grid mismatch")
    for _, group in dependence.groupby(["hypothesis", "group_key"]):
        ordered = group.sort_values("assumed_pairwise_correlation")
        require(np.all(np.diff(ordered["effective_repetitions"]) <= 1e-12), "Effective n is not monotone")
        require(np.all(np.diff(ordered["correlation_adjusted_se"]) >= -1e-12), "SE is not monotone")
    require(not bool(metadata["strong_contribution_condition_met"]), "Strong condition must remain false")
    require(metadata["low_fpr_framing"] == "frozen-threshold transfer failure diagnostic", "Low-FPR framing changed")

    manifest = pd.read_csv(RESULTS / "input_manifest.csv")
    for row in manifest.itertuples(index=False):
        path = Path(row.path)
        require(path.exists(), f"Missing input: {path}")
        require(sha256(path) == row.sha256, f"Input hash changed: {path}")

    for stem in ["fig15_1_effect_forest", "fig15_2_repetition_deltas", "fig15_3_dependence_stress"]:
        for suffix in [".png", ".pdf"]:
            path = FIGURES / f"{stem}{suffix}"
            require(path.exists() and path.stat().st_size > 1000, f"Missing figure: {path}")

    report = (RESULTS / "PART15_REPORT.md").read_text(encoding="utf-8")
    require("Status: **COMPLETE**" in report, "Completion marker missing")
    require("prespecified comparison" in report, "Wording recommendation missing")

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
            {"relative_path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(output_files)
        ]
    )
    output_manifest.to_csv(RESULTS / "output_manifest.csv", index=False)
    validation = {
        "status": "PASS",
        "checks": 17,
        "paired_difference_rows": len(pairs),
        "effect_groups": len(summary),
        "dependence_rows": len(dependence),
        "bootstrap_rows": len(bootstrap),
        "manifested_files": len(output_manifest),
    }
    (RESULTS / "validation_summary.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
