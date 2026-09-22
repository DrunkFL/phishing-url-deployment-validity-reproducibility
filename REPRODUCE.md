# Reproduction Guide

This repository supports three different levels of verification. They should not be treated as interchangeable.

## 1. Repository integrity

After creating the Python 3.9 environment described in `README.md`, run:

```powershell
.venv\Scripts\python tools\validate_repository.py
```

This checks the release manifest, file sizes and hashes, Python syntax, local-path leakage, and the schemas of the public Parquet files. It does not retrain models.

## 2. Entry-point smoke test

```powershell
.venv\Scripts\python tools\validate_repository.py --smoke
```

This additionally imports representative entry points from Parts 4, 11, 14, and 15 and runs each command's `--help` path. It detects broken package imports and directory-layout errors without changing experiment results.

## 3. Representative deterministic rerun

For a low-cost end-to-end check of the baseline machinery, use one dataset, split, repetition, and model:

```powershell
.venv\Scripts\python code\part4_baseline_models\scripts\train_baselines.py `
  --datasets phiusiil `
  --scenarios s3 `
  --repetitions r00 `
  --models lr `
  --n-jobs 1 `
  --force
```

Use the frozen source hashes, assignments, feature definitions, random seeds, and model grids supplied in this repository. Compare regenerated aggregate tables with the compact reference tables retained under each part's `results/` directory. Run the repository validator again after regeneration.

The reference Part 4 results were produced on the frozen Windows environment with the numerical libraries' default thread settings. Do not impose `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, or `OPENBLAS_NUM_THREADS` when checking exact reference values. A single-thread BLAS sensitivity run completed successfully but produced small logistic-regression differences (approximately 0.00009 in Macro-F1 and 0.00015 in ROC-AUC for the representative PhiUSIIL/S3/r00 run). This is numerical-environment sensitivity, not a different split or model selection. Exact equality outside the frozen platform is therefore not guaranteed; conclusions should be checked against reported tolerances and effect sizes.

## Full experiment

The full experiment proceeds from Parts 2 through 15. Scripts are located under `code/part*/scripts/`, while each part reads and writes data relative to its own part directory. Run Parts 4-15 in numerical order after preparing the feature tables and frozen assignments from Parts 2-3. Within a part, use the numbered stage scripts and then the corresponding `validate_*.py` script.

The complete run is computationally and storage intensive. The public package retains compact result tables and validation summaries, but intentionally excludes fitted model binaries, prediction matrices, temporary bootstrap tables, and raw URL strings. A complete regeneration therefore recreates these intermediates and can require several gigabytes of disk space. Part 11 is the dominant cost.

The retained validation records include:

- `code/part11_dcss_innovation/results/dcss_determinism_audit.json`, which records identical hashes before and after deterministic DCSS aggregation;
- `code/part11_dcss_innovation/results/stage11a_validation.json` through `stage11g_validation.json`;
- `code/part14_low_fpr_uncertainty/results/validation_summary.json`, including the maximum threshold replay difference;
- `code/part15_statistical_inference_robustness/results/validation_summary.json`.

The low-FPR experiment is a threshold-transfer failure diagnostic. It does not establish a population-wide false-positive-rate guarantee.
