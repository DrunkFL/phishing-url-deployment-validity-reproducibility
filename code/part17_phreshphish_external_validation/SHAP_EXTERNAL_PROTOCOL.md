# Locked source-versus-PhreshPhish SHAP protocol

Status: repaired full run completed and audited  
Target paper: *Journal of Information Security and Applications*

## Question

For each frozen S3 model already evaluated on PhreshPhish, does the explanation ranking and feature direction remain stable between its source test distribution and an untouched third-source cohort? The analysis is post hoc and cannot change model selection, thresholds, features, or the primary performance estimates.

## Frozen inputs

- The exact 35 Part 2 URL features and frozen PSL snapshot.
- The two source datasets: ISCX-URL2016 binary and PhiUSIIL.
- Ten S3 domain-grouped source partitions per dataset.
- The selected candidate, parameters, seed, and threshold stored for each source/repetition/model run.
- The full PhreshPhish test revision `eabec4b7a66324b79cc8a0ad856d1731dc26fe1a`.
- The already generated domain-overlap masks and third-source predictions.

## Cohorts

- Source cohort: 200 source-test observations per repetition, balanced to 100 benign and 100 phishing URLs by deterministic SHA-256 ordering.
- LR background: 200 source-training observations per repetition, balanced in the same way.
- Target cohort: one fixed 200-row PhreshPhish cohort per source direction, balanced to 100 observations per class.
- Each target cohort is selected only from the intersection of all ten source-specific primary domain-filtered masks. It is fixed across repetitions and model families within the source direction.
- Target labels are used only after prediction to balance the explanation cohort.

## Exact explainers

- LR: `shap.LinearExplainer` on standardized inputs, raw log-odds scale.
- RF: exact `shap.TreeExplainer`, tree-path-dependent positive-class output.
- XGBoost: exact `shap.TreeExplainer`, tree-path-dependent raw-margin output.

SHAP magnitude is never compared across model families because their output scales differ.

## Prespecified comparisons

- Top-10, Top-15, and Top-20 Jaccard overlap.
- Spearman correlation across all 35 feature ranks.
- Direction-sign agreement across all features and across the Top-20 union.
- Top-15-union direction conflicts, requiring nonzero and opposite source/target direction signs.
- Descriptive association between explanation-stability metrics and the observed source-to-target ROC-AUC and Macro-F1 losses.

Association statistics are descriptive because the ten repetitions share the same target test set. Inferential bootstrap and multiple-comparison decisions remain deferred.

## Validity rules

- Every model must reproduce its stored source ROC-AUC and Macro-F1 within `2e-5` before SHAP is accepted.
- The archived Windows numerical path is fixed at 12 BLAS/OpenMP threads before
  NumPy/scikit-learn load. Detected thread pools and effective environment values
  are recorded in run metadata.
- Every reconstructed model must match its archived Part 4 source-test sample
  order and labels, stay within `1e-7` maximum absolute probability difference
  from the archived float32 vector, and produce zero classification changes at
  the frozen threshold. Any mismatch stops the run before SHAP calculation.
- The target feature hash and pinned dataset revision must match the completed external-validation run.
- The target cohort hash must be identical across all repetitions and models within a source direction.
- A full run requires 60 source/repetition/model combinations, 120 SHAP value files, and 120 global summaries.
- Every SHAP cohort must contain exactly 200 rows and all comparison metrics must be finite.
- Warnings, missing files, duplicate run keys, or unexpected cohort changes remain visible in validation output.

## Interpretation boundary

Ranking instability and direction conflicts can identify features associated with transport failure, but they do not establish a causal mechanism. Results may support a bounded claim about explanation transportability only after comparison with the external performance losses and validation of all artifacts.

## Completion record

The repaired full run completed all 60 comparisons and passed the complete
integrity audit. It produced 120 SHAP value files and 120 global summaries with
200 rows per cohort, no missing comparison values, no warnings, and exact frozen-
threshold replay. The result and its interpretation boundaries are recorded in
`results/shap_external/SHAP_EXTERNAL_REPORT.md`.
