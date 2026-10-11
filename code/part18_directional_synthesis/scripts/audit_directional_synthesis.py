from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
HERE = SCRIPT_DIR.parent / "results"
ROOT = SCRIPT_DIR.parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(name: str) -> list[dict[str, str]]:
    with (HERE / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def close(actual: float, expected: float, tolerance: float = 5e-7) -> bool:
    return math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance)


failures: list[str] = []
checks: list[str] = []


def require(condition: bool, message: str) -> None:
    if condition:
        checks.append(message)
    else:
        failures.append(message)


manifest = json.loads((HERE / "INPUT_MANIFEST.json").read_text(encoding="utf-8"))
for item in manifest["input_files"]:
    path = ROOT / item["path"]
    require(path.is_file(), f"Manifest input exists: {item['path']}")
    if path.is_file():
        require(sha256(path) == item["sha256"], f"Manifest hash matches: {item['path']}")

macro = read_csv("directed_transfer_run_level.csv")
macro_summary = read_csv("directed_transfer_model_summary.csv")
low_fpr = read_csv("threshold_budget_run_level.csv")
low_fpr_summary = read_csv("threshold_budget_model_summary.csv")
direction_summary = read_csv("threshold_budget_direction_summary.csv")
explanation = read_csv("explanation_performance_summary.csv")

require(len(macro) == 180, "Macro-threshold synthesis has 180 run-level rows")
require(len(macro_summary) == 18, "Macro-threshold synthesis has 18 direction-model cells")
require(len(low_fpr) == 360, "Low-FPR synthesis has 360 run-level rows")
require(len(low_fpr_summary) == 36, "Low-FPR synthesis has 36 direction-model-budget cells")
require(len(direction_summary) == 12, "Low-FPR synthesis has 12 direction-budget cells")
require(len(explanation) == 6, "Explanation synthesis has six source-model cells")

macro_keys = Counter((r["source"], r["target"], r["model"]) for r in macro)
low_keys = Counter((r["source"], r["target"], r["model"], r["budget"]) for r in low_fpr)
directions = {(r["source"], r["target"]) for r in macro}
require(len(directions) == 6, "All six directed source-target edges are present")
require(set(macro_keys.values()) == {10} and len(macro_keys) == 18,
        "Every directed edge-model cell has ten repetitions")
require(set(low_keys.values()) == {10} and len(low_keys) == 36,
        "Every directed edge-model-budget cell has ten repetitions")

expected_auc = {
    ("iscx", "phiusiil", "lr"): 0.399180,
    ("iscx", "phiusiil", "rf"): 0.388170,
    ("iscx", "phiusiil", "xgb"): 0.412130,
    ("phiusiil", "iscx", "lr"): 0.550620,
    ("phiusiil", "iscx", "rf"): 0.452980,
    ("phiusiil", "iscx", "xgb"): 0.076370,
    ("iscx", "phreshphish", "lr"): 0.639670,
    ("iscx", "phreshphish", "rf"): 0.559990,
    ("iscx", "phreshphish", "xgb"): 0.553210,
    ("phiusiil", "phreshphish", "lr"): 0.371250,
    ("phiusiil", "phreshphish", "rf"): 0.413780,
    ("phiusiil", "phreshphish", "xgb"): 0.346470,
    ("phreshphish", "iscx", "lr"): 0.521380,
    ("phreshphish", "iscx", "rf"): 0.563660,
    ("phreshphish", "iscx", "xgb"): 0.535260,
    ("phreshphish", "phiusiil", "lr"): 0.729550,
    ("phreshphish", "phiusiil", "rf"): 0.884550,
    ("phreshphish", "phiusiil", "xgb"): 0.805560,
}
for row in macro_summary:
    key = (row["source"], row["target"], row["model"])
    require(key in expected_auc, f"Known direction-model cell: {key}")
    if key in expected_auc:
        require(close(float(row["roc_auc_mean"]), expected_auc[key], 5e-6),
                f"Independent expected ROC-AUC matches for {key}")

expected_budget_ranges = {
    ("iscx", "phiusiil", "0.001"): (531.85, 1000.00),
    ("iscx", "phiusiil", "0.01"): (90.68, 100.00),
    ("phiusiil", "iscx", "0.001"): (929.12, 1000.00),
    ("phiusiil", "iscx", "0.01"): (94.07, 100.00),
    ("iscx", "phreshphish", "0.001"): (402.51, 965.84),
    ("iscx", "phreshphish", "0.01"): (45.77, 96.96),
    ("phiusiil", "phreshphish", "0.001"): (872.29, 952.69),
    ("phiusiil", "phreshphish", "0.01"): (90.78, 95.35),
    ("phreshphish", "iscx", "0.001"): (272.93, 301.22),
    ("phreshphish", "iscx", "0.01"): (28.85, 32.19),
    ("phreshphish", "phiusiil", "0.001"): (0.047, 0.637),
    ("phreshphish", "phiusiil", "0.01"): (0.284, 2.251),
}
for row in direction_summary:
    key = (row["source"], row["target"], row["budget"])
    require(key in expected_budget_ranges, f"Known direction-budget cell: {key}")
    if key in expected_budget_ranges:
        expected_min, expected_max = expected_budget_ranges[key]
        require(close(float(row["model_family_exceedance_min"]), expected_min, 0.006),
                f"Budget-ratio minimum matches for {key}")
        require(close(float(row["model_family_exceedance_max"]), expected_max, 0.006),
                f"Budget-ratio maximum matches for {key}")

source_states = defaultdict(set)
for row in macro:
    source_states[(row["source"], row["target"])].add(row["evidence_state"])
require(source_states[("iscx", "phiusiil")] == {"user_authoritative"}
        and source_states[("phiusiil", "iscx")] == {"user_authoritative"},
        "Original two-source edges remain user-authoritative")
require(all(source_states[edge] == {"verified"} for edge in directions
            if "phreshphish" in edge),
        "E1 and E3 edges retain verified status")

association_path = ROOT / "code" / "part17_phreshphish_external_validation" / "results" / "shap_external" / "stability_performance_association.csv"
with association_path.open(encoding="utf-8-sig", newline="") as handle:
    association_rows = list(csv.DictReader(handle))
rho_rows = [r for r in association_rows if r.get("stability_metric") == "top15_jaccard" and r.get("loss_metric") == "roc_auc_loss" and r.get("scope") == "all"]
require(len(rho_rows) == 1, "E2 association source has one pooled Top-15-Jaccard/ROC-AUC-loss row")
if rho_rows:
    rho_field = next((name for name in ("spearman_rho_descriptive", "spearman_rho", "rho", "correlation") if name in rho_rows[0]), None)
    require(rho_field is not None, "E2 association row exposes a correlation field")
    if rho_field:
        require(close(float(rho_rows[0][rho_field]), 0.140, 0.0006),
                "Pooled descriptive Spearman rho rounds to 0.140")

status = "PASS" if not failures else "FAIL"
lines = [
    "# Directional Synthesis Audit",
    "",
    f"**Status:** {status}",
    "",
    "This audit independently checks the derived synthesis outputs against frozen input hashes, design cardinalities, prespecified direction-model means, threshold-budget ranges, evidence-state boundaries, and the E2 association source. It does not fit models or reselect thresholds.",
    "",
    "## Passed checks",
    "",
]
lines.extend(f"- {item}" for item in checks)
if failures:
    lines.extend(["", "## Failed checks", ""])
    lines.extend(f"- {item}" for item in failures)
lines.extend([
    "",
    "## Interpretation boundary",
    "",
    "- The six-edge graph is complete only for the three evaluated public sources.",
    "- The two original ISCX/PhiUSIIL directions remain user-authoritative because they were not regenerated from raw experiment artifacts in this revision.",
    "- PhreshPhish-to-ISCX/PhiUSIIL is a secondary reciprocal diagnostic, not another independent external validation.",
    "- Budget exceedance uses the nominal source-validation FPR budget as the denominator; it is not a ratio to an estimated population FPR.",
    "- The SHAP-performance association is descriptive and non-causal.",
    "",
])
(HERE / "AUDIT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

if failures:
    raise SystemExit("Directional synthesis audit failed; see AUDIT_REPORT.md")
print(f"PASS: {len(checks)} checks")
