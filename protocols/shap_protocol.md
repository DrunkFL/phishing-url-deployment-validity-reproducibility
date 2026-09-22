# Prespecified SHAP Stability Protocol

## Fixed inputs

- Corpus: conflict-cleaned `master`
- Features: the 35 frozen Part 2 offline URL features
- Models: LR, RF, and XGBoost
- Hyperparameters: the selected Part 4 r00 configuration for each dataset,
  scenario, and model family
- Stability cohort: 200 observations, 100 per class
- LR background: 200 training observations, 100 per class
- Sampling: deterministic SHA-256 ordering of sample IDs under a declared tag

## Explainers

- LR: `shap.LinearExplainer` on training-standardized features; raw log-odds scale
- RF: exact `shap.TreeExplainer`, `tree_path_dependent`, positive-class output
- XGB: exact `shap.TreeExplainer`, `tree_path_dependent`, raw-margin output

SHAP magnitudes are compared only within one model family. Cross-model analyses
use rankings and set overlap because output scales differ.

## Stability estimands

### Seed

S0 and S3 use the fixed r00 partition, background, and explanation cohort. Ten
model seeds (`20261001-20261010`) are used while hyperparameters remain fixed.
LR with `lbfgs` is deterministic and therefore acts as a zero-seed-variation
control.

### Partition

S0-S3 use Part 3 r00-r09 assignments. The model seed is fixed at `20261001`, and
the r00-selected hyperparameters remain fixed. Each test partition contributes a
deterministic class-balanced cohort of equal size. Background and cohort are
shared across model families within a dataset/scenario/repetition.

### Regime

Within each dataset, model, and repetition, rankings are compared for every pair
among S0-S3. These comparisons describe regime sensitivity; they do not prove
that one regime is universally more realistic or difficult.

### Source

Each S3 partition model explains its internal cohort and a target cohort selected
from the intersection of all ten Part 3 primary S4 target sets. The external
cohort is fixed across repetitions and models within one transfer direction.
Target labels are used only to balance the post hoc explanation cohort.

## Metrics

- Top-10, Top-15, and Top-20 Jaccard similarity
- Spearman correlation of all 35 feature ranks
- Top-k selection count and frequency per feature
- sign agreement of feature-value/SHAP-value Spearman correlations
- exact SHAP setup and computation time

Pairwise values are dependent. Part 5 reports distributions; run-aware bootstrap
inference and multiple-comparison decisions remain in the later statistics part.

## Scaling benchmark

Saved Part 4 r00 models for S0 and S3 are explained on nested class-balanced
cohorts of 100, 500, and 1,000 observations. Exact SHAP is used at every size.
The benchmark measures computation after explainer setup and records rank
agreement with the 1,000-observation reference.
