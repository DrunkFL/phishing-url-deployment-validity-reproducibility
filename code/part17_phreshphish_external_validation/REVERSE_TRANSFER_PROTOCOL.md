# Reciprocal Transfer Diagnostic Protocol

## Purpose and claim boundary

This experiment trains the same three baseline families on the pinned PhreshPhish
training split and evaluates them on ISCX-URL2016-binary and PhiUSIIL. It tests
whether the severe forward-transfer loss is direction-specific or reflects a
broader dataset shift.

This is a **reciprocal-transfer diagnostic**, not a second independent external
validation. ISCX and PhiUSIIL were already used in the original study, so their
labels cannot provide a new untouched confirmation. They remain completely
unavailable to PhreshPhish model fitting, candidate selection, feature selection,
and threshold selection.

## Locked inputs

- PhreshPhish release: `v1.0.1`.
- Pinned revision: `eabec4b7a66324b79cc8a0ad856d1731dc26fe1a`.
- Training split only: 498,255 rows expected.
- Feature contract: the same ordered 35 lexical/structural URL features used in
  the original experiment.
- Forbidden PhreshPhish columns remain unread: `html`, `target`, `lang`, and
  `lang_score`.
- Target views:
  - ISCX-URL2016-binary conflict-cleaned master and fully deduplicated tables;
  - PhiUSIIL conflict-cleaned master and fully deduplicated tables.

## Source splitting and model selection

- Repetitions: `r00`-`r09`.
- Partition seeds: `20260902`-`20260911`.
- Model seeds: `20261001`-`20261010`.
- Each repetition uses the original weighted greedy domain-grouped split with
  approximate proportions 70% train, 15% validation, and 15% internal test.
- A registrable domain must occur in exactly one source partition.
- Candidate grids, preprocessing, and Macro-F1 threshold selection are imported
  unchanged from original Part 4.
- Models: logistic regression, random forest, and XGBoost.
- Low-FPR thresholds are selected from source validation only at budgets 0.1%
  and 1%. No target observation affects any model or threshold choice.

## Target cohorts and overlap control

For every target view and repetition, retain both:

1. `unfiltered`: all target rows;
2. `primary_domain_filtered`: target rows whose registrable-domain hash does not
   occur in the corresponding PhreshPhish train or validation partitions.

Normalized-URL, host, and registrable-domain overlap counts are recorded. The
primary cohort is defined only by domain overlap, matching the original S4 rule.
The source internal-test partition is excluded from the overlap filter because it
is not used to fit or select the model.

## Outcomes

For each source repetition/model, evaluate the held-out PhreshPhish domain test.
For each target view, cohort, threshold, and both row-equal and domain-equal
weighting, record:

- ROC-AUC and PR-AUC;
- Macro-F1, balanced accuracy, precision, recall, and FPR;
- alerts per 10,000 benign URLs;
- Brier score, log loss, and 15-bin expected calibration error;
- confusion-matrix counts and cohort composition.

Predictions, tuning results, split assignments, warnings, file hashes, and run
metadata are retained. Smoke runs validate plumbing only and are not manuscript
evidence.

## Prespecified interpretation

- Strong internal performance with poor performance in both transfer directions
  supports dataset-specific decision rules and transportability limits.
- AUC below 0.5 is reported as ranking inversion, not repaired by target-label
  flipping.
- Directional asymmetry is reported directly rather than averaged away.
- Master/deduplicated disagreement is treated as a sensitivity result.
- Negative findings are retained and used to narrow claims; target outcomes are
  never used to revise the frozen feature set within this experiment.
