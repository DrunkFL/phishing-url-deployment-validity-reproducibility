# DCSS Protocol Lock

Protocol version: 1.0.0

Lock date: 2026-09-07 (Asia/Shanghai)

Status: locked before any Part 11 model training or target-result inspection

## 1. Research question

Can source-training-only, registrable-domain-conditioned SHAP stability identify
features that appear stable under random resampling but are sensitive to domain
composition, and can the resulting compact feature set retain internal
domain-disjoint performance while improving cross-source ranking?

## 2. Confirmatory hypotheses

- H1: F-DCSS-15 is internally non-inferior to F-All on S3 Macro-F1, using a
  non-inferiority margin of 0.01.
- H2: F-DCSS-15 has a positive paired external ROC-AUC difference relative to
  F-Stable in both source-to-target directions when averaged across the three
  prespecified model families.
- The method contribution is considered strongly supported only when H1 is met
  for at least two of three model families in each dataset, H2 has a positive
  mean in both directions, and the two-level 95% interval excludes zero in at
  least one direction.
- Failure to meet these conditions does not trigger formula or endpoint changes.
  The paper will instead report a diagnostic result about the limits of
  domain-conditioned explanation stability.

## 3. Data and label semantics

- Datasets: conflict-cleaned PhiUSIIL and binary ISCX-URL2016.
- Features: the existing 35 deterministic offline lexical and structural URL
  features extracted by Part 2.
- Label convention: 0 is benign and 1 is phishing.
- Entity unit: registrable domain (`eTLD+1`) derived with the frozen Public
  Suffix List and existing Part 2 normalization pipeline.
- No active URL is visited. Only saved strings, features, labels, hashes, and
  entity identifiers are processed.

## 4. Validity repair before DCSS

Part 11B must complete before DCSS training:

1. Derive the deduplicated corpus by retaining one representative per
   normalized URL from the master corpus.
2. Copy each retained observation's master train/validation/test role for every
   repetition; do not regenerate a partition.
3. Require 100% role agreement for every retained observation.
4. Refit F-All for S0 and S3 across 2 datasets, 3 models, and 10 repetitions
   (120 runs).
5. Require model classes to equal `[0, 1]` before reading probability column 1.
6. Recompute ROC-AUC, PR-AUC, and a diagnostic inverted-score ROC-AUC (`1-p`).
   The inverted score is never a deployable prediction or a replacement result.

## 5. Fixed outer design

- Primary internal regime: S3 registrable-domain-disjoint.
- External regime: the existing two zero-target-feedback S4 directions.
- Outer repetitions: `r00` through `r09`.
- Outer model seeds: 20261001 through 20261010.
- Outer train/validation/test assignments: frozen Part 3 master S3 assignments.
- Source-validation decision thresholds: 0.05 through 0.95 inclusive in steps
  of 0.01, selected by Macro-F1.
- The external target labels cannot influence model choice, feature selection,
  calibration, or the deployed threshold.

## 6. Fixed model families and candidate grids

The candidate order is part of the protocol and breaks exact validation ties.

### Logistic regression

1. `lr_c01`: C=0.1, class_weight=None.
2. `lr_c1`: C=1.0, class_weight=None.
3. `lr_c1_balanced`: C=1.0, class_weight=balanced.

Other fixed settings: lbfgs solver, max_iter=2000, StandardScaler.

### Random forest

1. `rf_depth16_leaf1`: 200 trees, max_depth=16, min_samples_leaf=1,
   max_features=sqrt, class_weight=None.
2. `rf_depth24_leaf2`: 200 trees, max_depth=24, min_samples_leaf=2,
   max_features=sqrt, class_weight=None.
3. `rf_unlimited_leaf5_balanced`: 200 trees, max_depth=None,
   min_samples_leaf=5, max_features=sqrt,
   class_weight=balanced_subsample.

### XGBoost

All candidates use binary logistic objective, log-loss evaluation, histogram
trees, subsample=0.9, colsample_bytree=0.9, and reg_lambda=1.0.

1. `xgb_d4_lr005`: 300 trees, max_depth=4, learning_rate=0.05,
   scale_pos_weight=1.
2. `xgb_d6_lr005`: 250 trees, max_depth=6, learning_rate=0.05,
   scale_pos_weight=1.
3. `xgb_d6_lr01_balanced`: 200 trees, max_depth=6, learning_rate=0.10,
   scale_pos_weight=n_benign/n_phishing in the fitting scope.

## 7. DCSS inner design

- DCSS uses only the outer training subset. The outer validation subset may
  determine the already-declared model candidate and source threshold, but is
  not explained or used to calculate DCSS.
- For each dataset, outer repetition, and model, create five folds with
  `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=fold_seed)`.
- Group is `registrable_domain_sha256`; stratification target is the binary
  label.
- Fold seed for outer repetition index `o` is `20261301 + o`.
- Each domain appears in exactly one held-domain fold.
- Each of five subruns fits on four domain folds and explains a deterministic,
  class-stratified sample from the held-domain fold.
- Model seed for outer index `o` and held fold `h` is
  `20261401 + 10*o + h`, where h is 0 through 4.
- SHAP background size is 200, sampled deterministically and class-stratified
  from the four fitting folds.
- SHAP explanation cohort size is 200, sampled deterministically and
  class-stratified from the held-domain fold.
- A smaller cohort is allowed only if the eligible fold has fewer than 200 rows;
  this must be logged and the entire eligible fold is then used.
- Logistic regression uses LinearExplainer. Random forest and XGBoost use
  TreeExplainer, inheriting the existing Part 5 implementation.

## 8. Exact DCSS definition

Let H=5 domain folds and p=35 features. For feature j and held fold h:

- `I_jh` is mean absolute SHAP value over the held-domain explanation cohort.
- `N_jh = I_jh / sum_l(I_lh)` is within-fold normalized importance. If the
  denominator is zero, the subrun fails validation rather than substituting a
  value.
- `r_jh` is the descending rank of `N_jh`. Ties are broken by feature name in
  ascending bytewise order.
- `d_jh` is the sign of Spearman correlation between the observed feature and
  its SHAP value in the held-domain cohort.
- A non-finite correlation or exact zero is direction 0. Direction 0 remains in
  the denominator and does not count toward either nonzero modal direction.

For selection size k in {10, 15, 20}:

`mean_importance_j = mean_h(N_jh)`

`frequency_j(k) = count_h(r_jh <= k) / H`

`direction_consistency_j = max(count_h(d_jh=+1), count_h(d_jh=-1)) / H`

`rank_dispersion_j = median_h(abs(r_jh - median_h(r_jh))) / (p - 1)`

`DCSS_j(k) = mean_importance_j * frequency_j(k) * direction_consistency_j * (1 - rank_dispersion_j)`

F-DCSS-k contains exactly k features with the largest DCSS score. Score ties are
broken by larger mean importance, then smaller mean rank, then ascending feature
name. The confirmatory method is F-DCSS-15. F-DCSS-10 and F-DCSS-20 are
sensitivity analyses only.

## 9. Comparators

- F-All: all 35 features.
- F-Single: existing prespecified single-run SHAP Top-15.
- F-Stable: existing repeated random-inner-split stability set.
- F-MI: existing mutual-information Top-15.
- F-Permutation: existing model-specific permutation-importance Top-15.
- F-Random-15: 30 same-size random feature sets using seeds 20261501 through
  20261530. Random results are summarized as a distribution and are not searched
  for the best seed.
- F-DCSS-15: proposed confirmatory method.

## 10. Outcomes

### Primary outcome

- External target ROC-AUC, because it is threshold independent and directly
  exposes the observed cross-source ordering reversal.

### Secondary outcomes

- External PR-AUC, Macro-F1, balanced accuracy, recall, FPR, Brier score,
  log loss, and 15-bin equal-width ECE.
- Internal S3 Macro-F1, ROC-AUC, PR-AUC, recall, and FPR.
- Number of retained features, training time, inference time, and SHAP time.

The source-validation threshold is the only deployable threshold. Target-oracle
thresholds and inverted-score AUC are diagnostics and cannot support an
operational claim.

## 11. Confirmatory comparisons and multiplicity

- Confirmatory family: F-DCSS-15 versus F-Stable for external ROC-AUC in the two
  source-to-target directions.
- Internal non-inferiority family: F-DCSS-15 versus F-All for S3 Macro-F1,
  separately by dataset and model.
- Secondary exploratory family: F-DCSS-15 versus F-All, F-Single, F-MI,
  F-Permutation, and F-Random-15 for secondary outcomes.
- Benjamini-Hochberg correction is applied only within each explicitly named
  family. Raw p-values, adjusted p-values, paired effect sizes, and win/tie/loss
  counts are retained.

## 12. Uncertainty analysis

- The primary robustness interval uses a two-level bootstrap: sample the ten
  outer repetitions with replacement, then sample target `eTLD+1` clusters with
  replacement inside each selected repetition.
- The number of bootstrap replicates is 5,000. Increasing this number may check
  numerical convergence but cannot be described as increasing experimental
  sample size.
- Results are interpreted as finite-corpus, fixed-protocol uncertainty and are
  not population-level guarantees for future phishing campaigns.
- Pairwise SHAP similarities sharing fitted models are descriptive unless an
  explicitly valid cluster unit is available.

## 13. Ablations

- DCSS-Full: complete locked formula.
- NoDirection: replace direction consistency by 1.
- NoRankDispersion: replace `(1-rank_dispersion)` by 1.
- NoFrequency: replace Top-k frequency by 1.
- Every ablation selects exactly 15 features and uses identical outer
  assignments, model families, and target cohorts.

## 14. Third-source decision rule

- A third external cohort must have reproducible provenance, license or research
  terms, raw complete URLs, explicit label semantics, collection time, and file
  hashes.
- It is target-only for the confirmatory DCSS method and cannot alter the locked
  score or feature sets.
- A cohort pairing phishing full URLs with benign bare domains is exploratory
  only because class and URL structure would be confounded.
- If a defensible benign complete-URL cohort is unavailable, third-source
  testing is omitted and documented rather than replaced by a weak construction.

## 15. Protocol deviations

Any deviation requires all of the following before the affected target results
are inspected:

1. a new semantic protocol version;
2. a dated reason in `issues/ISSUE_LOG.md`;
3. a list of affected runs and claims;
4. preservation of the original protocol and results;
5. explicit labeling as confirmatory, sensitivity, or post hoc.
