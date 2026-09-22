# Prespecified Split Protocol

- Master partition seed list: `20260902` through `20260911` inclusive.
- Primary reporting repetition: `r00`, seed `20260902`.
- Target proportions for S0-S3: train 0.70, validation 0.15, test 0.15.
- S0 uses two-stage stratified row sampling.
- S1-S3 use weighted greedy group stratification. Complete groups are seed-shuffled, processed from largest to smallest, and assigned to the split that minimizes the incremental squared relative error in benign rows, phishing rows, and total rows.
- The same conflict-cleaned master observations are used in S0-S3. Fully deduplicated observations are split separately for sensitivity analysis.
- No repetition is removed because its achieved proportions or class balance looks unfavorable. Infeasible runs must be retained and reported as failures.
- S4 source assignments reuse S3. Target labels are stored for later evaluation but must not be used for model fitting, tuning, threshold selection, or feature selection.
