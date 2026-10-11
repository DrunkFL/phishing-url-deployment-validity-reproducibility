# PhreshPhish Third-Source Validation

This directory contains the zero-target-feedback external test of the existing ISCX-URL2016 and PhiUSIIL source models on PhreshPhish v1.0.1. It reuses the frozen feature code and source artifacts in the adjacent `code/part2_*` through `code/part5_*` directories.

## What is implemented

1. `src/build_manifest.py` pins the Hugging Face commit and records all 77 Parquet shard sizes and LFS SHA-256 identifiers.
2. `src/prepare_dataset.py` reads only `sha256`, `url`, `label`, and `date` through HTTP byte ranges. It does not download or parse HTML. Raw URLs are not written to the derived feature table.
3. `src/evaluate_frozen.py` reconstructs each saved source-side model from its frozen candidate, source split, and seed; verifies the source-test replay; then evaluates the untouched target at ordinary and low-FPR thresholds.
4. `tests/` checks the frozen feature contract and threshold/calibration helpers.

Remote Range GETs use a bounded retry policy for connection/read timeouts,
408/429, and temporary 5xx responses. Permanent HTTP errors still fail
immediately, and every returned range is checked against the requested byte
interval. The data manifest and the hashes retained in the run metadata record
the public input revision and the derived files used in the reported runs.

## Local smoke test

Run from this directory in the repository environment:

```powershell
python src\build_manifest.py
python -m unittest discover -s tests -v
python src\prepare_dataset.py --split test --limit 200 --force
python src\evaluate_frozen.py data\features\phreshphish_test_200_row_smoke.parquet --sources phiusiil --repetitions r00 --models lr --allow-smoke --force
```

The smoke metrics prove only that data access, frozen preprocessing, model replay, prediction, and result serialization work together. They are not scientific results.

## Full server run

Use Python 3.9 with the dependencies listed in the repository-level reproduction guide. The scripts resolve the frozen source artifacts from adjacent `code/part*` directories. Fetch the public PhreshPhish shards directly; raw URLs and large derived feature tables are intentionally not redistributed here.

```bash
python src/build_manifest.py
python -m unittest discover -s tests -v
python src/prepare_dataset.py --split test
python src/evaluate_frozen.py data/features/phreshphish_test.parquet --n-jobs 8
```

Recommended resources are 8–16 physical CPU cores, 32 GiB RAM minimum (64 GiB preferred), and at least 100 GiB free storage. A GPU is optional; the locked models do not require one.

## Interpretation boundary

Do not copy smoke output into the paper. A full test result becomes evidence only after row-count, label, overlap, source-replay, metric, uncertainty, SHAP, and benchmark-view checks are complete. See `EXPERIMENT_PROTOCOL.md`.

## Source-versus-target SHAP analysis

The SHAP analysis explains the exact 60 frozen S3 models used in the full external validation. It compares deterministic, class-balanced source-test and PhreshPhish cohorts without feeding target explanations back into training or feature selection. The locked design is in `SHAP_EXTERNAL_PROTOCOL.md`.

Smoke test:

```powershell
python src\run_shap_external.py --scope smoke --sources phiusiil --repetitions r00 --models lr --n-jobs 4 --output-root results\shap_external_smoke --force
```

Full run:

```powershell
python src\run_shap_external.py --scope full --n-jobs 4
```

## Reciprocal-transfer diagnostic

`REVERSE_TRANSFER_PROTOCOL.md` locks the secondary diagnostic that trains on the
PhreshPhish training split and evaluates ISCX/PhiUSIIL master and deduplicated
views. This analysis reuses the original domain-grouped split, model grids, and
seeds. It is not labeled as independent external validation because both target
corpora were already used in the original study.

Pipeline smoke test (not manuscript evidence):

```powershell
python src\run_reverse_transfer.py data\features\phreshphish_test_200_row_smoke.parquet --allow-smoke --target-limit 200 --repetitions r00 --models lr --targets iscx_master --output-root results\reverse_transfer_smoke --n-jobs 4 --force
```

Full run after the verified training features are downloaded:

```powershell
python src\run_reverse_transfer.py data\features\server_train\phreshphish_train.parquet --n-jobs 8
```
