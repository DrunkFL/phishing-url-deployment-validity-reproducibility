from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PART11 = EXPERIMENT_ROOT / "part11_dcss_innovation"
PAPER_ROOT = Path(__file__).resolve().parents[1] / "manuscript"
MANIFEST = PART11 / "data" / "manifests" / "input_manifest.csv"
ENVIRONMENT = PART11 / "data" / "manifests" / "environment_snapshot.json"
PROTOCOL = PART11 / "PROTOCOL_LOCK.md"
VALIDATION = PART11 / "results" / "stage11a_validation.json"
PROTOCOL_HASH = PART11 / "data" / "manifests" / "protocol_lock.sha256"

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_PROTOCOL_TERMS = [
    "Protocol version: 1.0.0",
    "Status: locked before any Part 11 model training",
    "F-DCSS-15",
    "External target ROC-AUC",
    "StratifiedGroupKFold",
    "DCSS_j(k)",
    "NoDirection",
    "NoRankDispersion",
    "NoFrequency",
]

REQUIRED_INPUT_SUFFIXES = [
    "part2_url_normalization_leakage_audit/data/phiusiil_master_features.parquet",
    "part2_url_normalization_leakage_audit/data/iscx_url2016_binary_master_features.parquet",
    "part3_s0_s4_data_splits/data/assignments/phiusiil_master_s3_domain_assignments.parquet",
    "part3_s0_s4_data_splits/data/assignments/iscx_url2016_binary_master_s3_domain_assignments.parquet",
    "part4_baseline_models/scripts/model_utils.py",
    "part5_shap_stability/scripts/shap_utils.py",
    "part6_stable_feature_selection/scripts/run_training_side_selection.py",
    "part7_external_transfer/scripts/run_reduced_external_transfer.py",
    "part10_robustness_extensions/results/deduplicated_metrics.csv",
    "paper/phishing_url_paper_en_draft.md",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(relative_path: str) -> Path:
    if relative_path.startswith("paper/"):
        return PAPER_ROOT / relative_path.removeprefix("paper/")
    return EXPERIMENT_ROOT / Path(relative_path)


def require(condition: bool, message: str, checks: list[dict[str, object]]) -> None:
    checks.append({"check": message, "passed": bool(condition)})
    if not condition:
        raise AssertionError(message)


def main() -> None:
    checks: list[dict[str, object]] = []
    require(MANIFEST.is_file(), "Input manifest exists", checks)
    require(ENVIRONMENT.is_file(), "Environment snapshot exists", checks)
    require(PROTOCOL.is_file(), "Protocol lock exists", checks)

    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    paths = [row["relative_path"] for row in rows]
    require(len(rows) > 0, "Manifest is nonempty", checks)
    require(len(paths) == len(set(paths)), "Manifest paths are unique", checks)
    require(
        all(SHA256_PATTERN.fullmatch(row["sha256"]) for row in rows),
        "Every manifest SHA-256 has 64 lowercase hexadecimal characters",
        checks,
    )
    require(
        all(any(path.endswith(suffix) for path in paths) for suffix in REQUIRED_INPUT_SUFFIXES),
        "All required foundational inputs are represented",
        checks,
    )

    total_bytes = 0
    mismatches: list[dict[str, object]] = []
    for index, row in enumerate(rows, start=1):
        path = resolve(row["relative_path"])
        if not path.is_file():
            mismatches.append({"path": row["relative_path"], "reason": "missing"})
            continue
        size = path.stat().st_size
        total_bytes += size
        if size != int(row["size_bytes"]):
            mismatches.append(
                {
                    "path": row["relative_path"],
                    "reason": "size",
                    "expected": int(row["size_bytes"]),
                    "actual": size,
                }
            )
            continue
        digest = sha256_file(path)
        if digest != row["sha256"]:
            mismatches.append(
                {
                    "path": row["relative_path"],
                    "reason": "sha256",
                    "expected": row["sha256"],
                    "actual": digest,
                }
            )
        if index % 250 == 0 or index == len(rows):
            print(f"verified={index}/{len(rows)}", flush=True)

    require(not mismatches, "Every frozen input exists and matches its SHA-256", checks)

    protocol_text = PROTOCOL.read_text(encoding="utf-8")
    require(
        all(term in protocol_text for term in REQUIRED_PROTOCOL_TERMS),
        "Protocol contains all locked design declarations",
        checks,
    )

    env = json.loads(ENVIRONMENT.read_text(encoding="utf-8"))
    require(
        Path(env["python_executable"]).resolve()
        == (EXPERIMENT_ROOT / ".venv" / "Scripts" / "python.exe").resolve(),
        "Environment snapshot points to the shared experiment virtual environment",
        checks,
    )

    protocol_digest = sha256_file(PROTOCOL)
    PROTOCOL_HASH.write_text(
        f"{protocol_digest}  PROTOCOL_LOCK.md\n", encoding="ascii"
    )
    role_counts = Counter(row["artifact_role"] for row in rows)
    output = {
        "stage": "11A",
        "status": "PASS",
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_file_count": len(rows),
        "manifest_total_bytes": total_bytes,
        "manifest_sha256": sha256_file(MANIFEST),
        "environment_snapshot_sha256": sha256_file(ENVIRONMENT),
        "protocol_sha256": protocol_digest,
        "protocol_version": "1.0.0",
        "mismatch_count": len(mismatches),
        "artifact_role_counts": dict(sorted(role_counts.items())),
        "checks": checks,
    }
    VALIDATION.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
