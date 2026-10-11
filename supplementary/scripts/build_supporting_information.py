from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]
PROJECT = PACKAGE.parent


@dataclass(frozen=True)
class Binding:
    label: str
    destination: str
    source: str
    evidence_class: str
    acceptance_check: str


BINDINGS = (
    Binding(
        "Supplementary Data S1",
        "supplementary_feature_dictionary.csv",
        "supplementary/supplementary_feature_dictionary.csv",
        "frozen method artifact",
        "35 unique features in frozen order",
    ),
    Binding(
        "Supplementary Data S2",
        "supplementary_model_candidates.json",
        "supplementary/supplementary_model_candidates.json",
        "frozen method artifact",
        "nine candidates: three LR, three RF, three XGBoost",
    ),
    Binding(
        "Supplementary Table S4",
        "tables/table_s4_s3_feature_methods.csv",
        "supplementary/tables/table_s4_s3_feature_methods.csv",
        "archived result summary",
        "30 rows; means, sample SDs, AUC, FPR, run counts, and feature counts reconciled",
    ),
    Binding(
        "Supplementary Table S5",
        "tables/table_s5_external_transfer.csv",
        "supplementary/tables/table_s5_external_transfer.csv",
        "deterministic result summary",
        "30 groups of ten primary-domain-filtered runs; means and sample SDs reproduced",
    ),
    Binding(
        "Supplementary Figure S1",
        "figures/Figure_S1.pdf",
        "supplementary/figures/Figure_S1.pdf",
        "archived result figure",
        "byte-copy of the accepted Part 10 reliability diagram",
    ),
    Binding(
        "Supplementary Section S6",
        "supplementary_section_s6.md",
        "supplementary/supplementary_section_s6.md",
        "secondary sensitivity report",
        "values reconciled to the frozen Stage 11F report",
    ),
    Binding(
        "Frozen PSL resource",
        "resources/public_suffix_list_tldextract_5.1.3.dat",
        "resources/public_suffix_list_tldextract_5.1.3.dat",
        "frozen preprocessing resource",
        "SHA-256 and private-suffix setting recorded",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def validate_package() -> None:
    features = PACKAGE / "supplementary_feature_dictionary.csv"
    with features.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    feature_key = "feature" if "feature" in rows[0] else "feature_name"
    names = [row[feature_key] for row in rows]
    if len(names) != 35 or len(set(names)) != 35:
        raise AssertionError("Supplementary Data S1 must contain 35 unique features")

    candidates = json.loads((PACKAGE / "supplementary_model_candidates.json").read_text(encoding="utf-8"))
    if any(len(candidates.get(family, [])) != 3 for family in ("lr", "rf", "xgb")):
        raise AssertionError("Supplementary Data S2 must contain three candidates per model family")

    if count_csv_rows(PACKAGE / "tables/table_s4_s3_feature_methods.csv") != 30:
        raise AssertionError("Supplementary Table S4 must contain 30 rows")
    if count_csv_rows(PACKAGE / "tables/table_s5_external_transfer.csv") != 30:
        raise AssertionError("Supplementary Table S5 must contain 30 rows")

    psl = PACKAGE / "resources/public_suffix_list_tldextract_5.1.3.dat"
    expected_psl = "91aeeed5dfc84a53265d95cfbb0e7bb52df8ed75a24a27568817cd69578a2f80"
    if sha256(psl) != expected_psl:
        raise AssertionError("Frozen PSL hash mismatch")

    required_authored = (
        "README.md",
        "supplementary_shap_protocol.md",
        "resources/PSL_PROVENANCE.md",
        "SUPPLEMENT_EVIDENCE_INDEX.md",
    )
    for relative in required_authored:
        if not (PACKAGE / relative).is_file():
            raise FileNotFoundError(relative)


def write_binding_matrix() -> None:
    output = PACKAGE / "SUPPLEMENT_BINDING_MATRIX.csv"
    fieldnames = (
        "manuscript_label",
        "package_file",
        "source_file",
        "source_sha256",
        "package_sha256",
        "evidence_class",
        "acceptance_check",
    )
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in BINDINGS:
            source = PROJECT / item.source
            destination = PACKAGE / item.destination
            writer.writerow(
                {
                    "manuscript_label": item.label,
                    "package_file": item.destination,
                    "source_file": item.source,
                    "source_sha256": sha256(source),
                    "package_sha256": sha256(destination),
                    "evidence_class": item.evidence_class,
                    "acceptance_check": item.acceptance_check,
                }
            )


def write_manifest() -> None:
    output = PACKAGE / "SUPPLEMENT_PACKAGE_MANIFEST.csv"
    files = sorted(
        path
        for path in PACKAGE.rglob("*")
        if path.is_file() and path != output and "__pycache__" not in path.parts
    )
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("package_path", "bytes", "sha256")
        )
        writer.writeheader()
        for path in files:
            writer.writerow(
                {
                    "package_path": path.relative_to(PACKAGE).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )


def main() -> None:
    validate_package()
    write_binding_matrix()
    write_manifest()
    print("PASS: supporting-information package validated and manifests refreshed")


if __name__ == "__main__":
    main()
