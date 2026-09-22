from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FILES = (
    "dcss_fold_shap_summary.csv",
    "dcss_scores.csv",
    "dcss_feature_sets.csv",
    "dcss_timing.csv",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    before = {name: sha256_file(RESULTS / name) for name in FILES}
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_dcss_selection.py"), "--aggregate-only"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    after = {name: sha256_file(RESULTS / name) for name in FILES}
    matches = {name: before[name] == after[name] for name in FILES}
    report = {
        "stage": "11C",
        "status": "PASS" if all(matches.values()) else "FAIL",
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "command_stdout": completed.stdout.strip(),
        "hashes_before": before,
        "hashes_after": after,
        "matches": matches,
    }
    (RESULTS / "dcss_determinism_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if report["status"] != "PASS":
        raise AssertionError("DCSS aggregate hashes changed on deterministic rerun")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
