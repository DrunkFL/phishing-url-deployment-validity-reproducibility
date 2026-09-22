# Anonymous Reproducibility Artifact: Phishing URL Deployment Validity

This package supports audit and reconstruction of the paper's entity-aligned evaluation, explanation-stability analysis, frozen-threshold transfer-failure diagnostic, and bounded DCSS comparison.

## What is included

- exact code snapshots for Parts 2-15;
- 35 feature definitions and boundary behavior;
- the frozen PSL snapshot and environment lock;
- all model candidate grids, seeds, split protocols, and SHAP rules;
- hashed S3/S4 assignment records (843,408 rows across four files);
- repetition-level metrics, paired differences, and statistical inputs under the original `code/part*/results` layout;
- experiment issue logs and source/provenance notes.

## What is intentionally excluded

- raw URL strings and original dataset files;
- trained model binaries and prediction files containing unnecessary sample-level values;
- author identities, affiliations, repository URLs, and DOI placeholders;
- any claim that the source-validation low-FPR budget is a population guarantee.

## Reproduction order

1. Create the Python environment from `environment/requirements.lock.txt`.
2. Acquire source data using `data_metadata/source_notes.md`; verify local hashes against the source manifest where licensing permits.
3. Use the sanitized derived feature tables under `code/part2_url_normalization_leakage_audit/data`; rerun normalization only when licensed source data are locally available.
4. Use the original-name frozen assignments under `code/part3_s0_s4_data_splits/data/assignments`; the `splits/` folder is the compact audit view with entity hashes.
5. Follow `protocols/` and `configs/` for model selection, SHAP, DCSS, and statistical analysis.
6. Compare regenerated outputs with `code/part*/results/` and `MANIFEST.csv`.

## Integrity

Every release file is listed in `MANIFEST.csv` with SHA-256 and byte length. The PSL SHA-256 is `91aeeed5dfc84a53265d95cfbb0e7bb52df8ed75a24a27568817cd69578a2f80`.
