from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PART11 = EXPERIMENT_ROOT / "part11_dcss_innovation"
PAPER_ROOT = Path(__file__).resolve().parents[1] / "manuscript"
MANIFEST_DIR = PART11 / "data" / "manifests"
RESULTS_DIR = PART11 / "results"

PARTS = [
    "part2_url_normalization_leakage_audit",
    "part3_s0_s4_data_splits",
    "part4_baseline_models",
    "part5_shap_stability",
    "part6_stable_feature_selection",
    "part7_external_transfer",
    "part8_traditional_baselines_and_error_analysis",
    "part9_publication_artifacts",
    "part10_robustness_extensions",
]

EXCLUDED_DIRS = {".pytest_cache", "__pycache__", "logs"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".tmp", ".temp"}
PACKAGES = [
    "numpy",
    "pandas",
    "pyarrow",
    "scikit-learn",
    "scipy",
    "shap",
    "xgboost",
    "joblib",
    "tldextract",
    "matplotlib",
    "seaborn",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def include_file(path: Path) -> bool:
    if any(part in EXCLUDED_DIRS for part in path.parts):
        return False
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    if path.name.startswith("~$"):
        return False
    return path.is_file()


def artifact_role(relative_path: str) -> str:
    parts = Path(relative_path).parts
    lowered = {part.lower() for part in parts}
    if "assignments" in lowered:
        return "assignment"
    if "data" in lowered or "resources" in lowered:
        return "data"
    if "models" in lowered:
        return "model"
    if "predictions" in lowered:
        return "prediction"
    if "feature_sets" in lowered or "selection_runs" in lowered or "selection_scope" in lowered:
        return "feature_selection"
    if "results" in lowered or "tables" in lowered or "figure_data" in lowered:
        return "result"
    if "figures" in lowered or "manuscript" in lowered:
        return "publication_artifact"
    if "scripts" in lowered or "tests" in lowered:
        return "source"
    if "runs" in lowered:
        return "run_record"
    return "documentation"


def collect_files() -> list[tuple[str, Path]]:
    collected: list[tuple[str, Path]] = []
    for part_name in PARTS:
        root = EXPERIMENT_ROOT / part_name
        if not root.is_dir():
            raise FileNotFoundError(f"Missing experiment part: {root}")
        for path in root.rglob("*"):
            if include_file(path):
                collected.append((path.relative_to(EXPERIMENT_ROOT).as_posix(), path))

    manuscript = PAPER_ROOT / "phishing_url_paper_en_draft.md"
    if not manuscript.is_file():
        raise FileNotFoundError(f"Missing manuscript: {manuscript}")
    collected.append((f"paper/{manuscript.name}", manuscript))
    return sorted(collected, key=lambda item: item[0].encode("utf-8"))


def environment_snapshot() -> dict[str, object]:
    versions: dict[str, str] = {}
    for package in PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "NOT_INSTALLED"
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "packages": versions,
    }


def main() -> None:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    files = collect_files()
    rows: list[dict[str, object]] = []
    total_bytes = 0

    for index, (relative_path, path) in enumerate(files, start=1):
        stat = path.stat()
        digest = sha256_file(path)
        total_bytes += stat.st_size
        rows.append(
            {
                "relative_path": relative_path,
                "artifact_role": artifact_role(relative_path),
                "size_bytes": stat.st_size,
                "modified_utc": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(),
                "sha256": digest,
            }
        )
        if index % 250 == 0 or index == len(files):
            print(f"hashed={index}/{len(files)}", flush=True)

    manifest_path = MANIFEST_DIR / "input_manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    env_path = MANIFEST_DIR / "environment_snapshot.json"
    env_path.write_text(
        json.dumps(environment_snapshot(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    role_counts: dict[str, int] = {}
    for row in rows:
        role = str(row["artifact_role"])
        role_counts[role] = role_counts.get(role, 0) + 1

    validation = {
        "stage": "11A",
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_file_count": len(rows),
        "manifest_total_bytes": total_bytes,
        "manifest_sha256": sha256_file(manifest_path),
        "environment_snapshot_sha256": sha256_file(env_path),
        "parts_present": PARTS,
        "manuscript_present": True,
        "excluded_directories": sorted(EXCLUDED_DIRS),
        "excluded_suffixes": sorted(EXCLUDED_SUFFIXES),
        "artifact_role_counts": dict(sorted(role_counts.items())),
    }
    validation_path = RESULTS_DIR / "stage11a_validation.json"
    validation_path.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(validation, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
