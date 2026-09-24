# Phishing URL Deployment Validity: Reproducibility Materials

This repository-ready package accompanies the manuscript on entity-aligned evaluation, explanation stability, low-false-positive-rate threshold transfer failure, and bounded deployment-conditioned stability selection (DCSS) for phishing URL detection.

## Repository contents

- `manuscript/`: the current manuscript PDF, Markdown source, generated LaTeX, build script, and the seven figures used by the paper.
- `code/`: exact experiment code snapshots for Parts 2-15. Each part keeps executable files under `scripts/` and compact result tables under `results/`.
- `configs/`: model candidates, outer seeds, and SHAP configuration.
- `features/`: the 35 feature definitions and extraction metadata.
- `splits/`: hashed S3 and S4 audit assignments; no plaintext URLs, hosts, or registrable domains.
- `protocols/`: frozen split, SHAP, DCSS, and ablation protocols.
- `data_metadata/`: data acquisition, provenance, source hashes, and redistribution notes.
- `environment/`: direct and fully locked Python dependencies.
- `issues/`: experiment problems, resolutions, and retained limitations.
- `resources/`: the frozen Public Suffix List snapshot.
- `supplementary/`: submission-ready Supporting Information tables, protocol files, feature definitions, and Figure S1.
- `tools/validate_repository.py`: integrity, size, local-path, and plaintext-identifier checks.
- `REPRODUCE.md`: integrity, smoke-test, representative-rerun, and full-rerun guidance.
- `RELEASE_VALIDATION.md`: the pre-upload integrity, entry-point, and representative-training checks.
- `ARTIFACT_README.md` and `ARTIFACT_MANIFEST.csv`: the original Part 16 artifact description and source-traceability manifest.
- `MANIFEST.csv`: SHA-256 and byte-size manifest for this assembled repository.

## Data boundary

Raw URL lists and original dataset archives are intentionally excluded. PhiUSIIL and URL-Phish v1 are available from their cited official sources under their stated terms. The ISCX-URL2016 preserved copy used in the experiments is identified by source, commit, filename, byte size, and SHA-256, but it is not redistributed because its redistribution terms and byte identity with the official archive were not established. See `DATA_AVAILABILITY.md` and `data_metadata/source_notes.md`.

The included Parquet files contain sanitized numerical features, labels, hashed identifiers, or frozen role assignments. They do not contain plaintext URLs, hosts, or registrable domains.

## Quick integrity check

The repository validator uses only the Python standard library for file integrity and uses `pyarrow` when available for Parquet schema checks:

```powershell
python tools/validate_repository.py
```

Expected result: `PASS` with all listed files matching `MANIFEST.csv`, no file at or above GitHub's 100 MB hard limit, no local user path, and no banned plaintext identifier column in the public Parquet schemas when `pyarrow` is available. To make the manifest portable across operating systems, text-file byte lengths and SHA-256 values are calculated after canonicalizing line endings to LF; binary files are hashed byte for byte.

After installing the frozen environment, also run the entry-point smoke test:

```powershell
.venv\Scripts\python tools\validate_repository.py --smoke
```

## Environment

The experiments used Python 3.9.13. To reconstruct the frozen environment on Windows:

```powershell
py -3.9 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r environment\requirements.lock.txt
```

The lock file records the original Windows environment. Rebuilding on a different operating system may require resolving platform-specific wheels while keeping the declared direct versions in `environment/requirements.txt`.

## Reproduction route

1. Read `data_metadata/source_notes.md` and obtain the source datasets under their respective terms.
2. Verify source filenames, sizes, and hashes against `data_metadata/original_input_manifest.csv` where the files are available.
3. Use the scripts under `code/part2_url_normalization_leakage_audit/scripts/` for normalization and feature extraction, or begin with the sanitized feature tables already present under that part's `data/` directory.
4. Follow `protocols/split_protocol.md` and the frozen assignments under `code/part3_s0_s4_data_splits/data/assignments/`.
5. Follow `REPRODUCE.md`. Run experiment parts in numerical order from Parts 4 through 15, using each part's `scripts/` directory, the frozen configurations, and the retained reference tables and validation summaries.
6. Run `python tools/validate_repository.py` after any public-release change and regenerate `MANIFEST.csv` with `python tools/generate_manifest.py`.

The package records the completed experiment, but it does not claim that the source-validation low-FPR budget is a population guarantee. The low-FPR analysis is a threshold-transfer failure diagnostic.

## Manuscript build

From `manuscript/`, run:

```powershell
python build_submission.py
```

This regenerates `main.tex` from `main.md`. A LaTeX engine is then required to compile the PDF. The current compiled manuscript is `manuscript/paper.pdf`.

## Licensing

Author-created code is released under the MIT License in `LICENSE-CODE`.
Author-created documentation, protocols, and aggregate results identified in
`LICENSE_SCOPE.md` are released under CC BY 4.0 through
`LICENSE-DOCUMENTATION`. Source datasets, record-level dataset derivatives,
the manuscript, and third-party files are not covered by those grants. The
frozen Public Suffix List retains its embedded MPL-2.0 notice.

## Versioned release

The manuscript-associated public snapshot is tagged `v1.0.0`. Run the checks
in `PUBLIC_UPLOAD_CHECKLIST.md` and `RELEASE_VALIDATION.md` before any later
release. The CRediT statement still needs to be completed after the author
contributions are confirmed.
