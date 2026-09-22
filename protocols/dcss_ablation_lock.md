# Stage 11E Ablation and Mechanism Analysis Lock

Lock time: 2026-09-08 10:35 Asia/Shanghai  
Status: locked before Stage 11E ablation or random-baseline model training

## 1. Purpose

Stage 11E tests which locked DCSS components affect feature membership and
predictive behavior. It does not redefine the failed Stage 11D confirmatory
result and does not use target labels to alter the score, feature count, model,
or source-validation threshold.

## 2. Formula ablations

Every method selects exactly 15 of the same 35 frozen URL features. The
existing tie breakers remain: larger mean normalized importance, smaller mean
rank, then ascending bytewise feature name.

- `dcss_full`: `I * F * D * (1-R)`; reuse Stage 11D F-DCSS-15 results.
- `no_direction`: `I * F * (1-R)`.
- `no_rank_dispersion`: `I * F * D`.
- `no_frequency`: `I * D * (1-R)`.

Here `I` is mean normalized importance, `F` is Top-15 fold frequency, `D` is
direction consistency, and `R` is normalized fold-rank MAD. No component is
reweighted.

## 3. Random baseline

F-Random-15 uses exactly the 30 protocol seeds 20261501 through 20261530.
Each seed draws 15 features without replacement from the frozen 35-feature
universe using NumPy `default_rng`. A seed defines one global feature set that
is reused across both datasets, all outer repetitions, and all models. All 30
sets are retained; no best seed is selected.

The complete random design is 30 sets x 2 datasets x 10 repetitions x 3 models
= 1,800 fitted runs. Formula ablations add 3 x 2 x 10 x 3 = 180 fitted runs.

## 4. Training and storage

- Outer S3 assignments, model seeds, candidate grids, source-validation
  tuning, threshold grid, and external cohorts are identical to Stage 11D.
- Formula-ablation runs save fitted models and float64 internal/external
  predictions for independent metric reproduction.
- Random runs save the frozen feature set, tuning rows, full internal and two-
  cohort external metric records, warnings, and completion metadata.
- Random fitted models and per-sample predictions are intentionally not saved.
  Stage 11D files show that doing so would require roughly 18 GB, while only
  about 10 GB was free at lock time. The random family is exploratory and is
  validated by deterministic feature hashes, complete run keys, class-order
  assertions, and sampled deterministic refits.

## 5. Mechanism diagnostics

The following diagnostics are fixed before inspecting Stage 11E performance:

1. Compare F-Stable-only, shared, and F-DCSS-15-only feature membership using
   rank dispersion, direction consistency, Top-15 frequency, and importance.
   Feature rows are nested; inference will not treat them as independent.
2. Refit the 300 Stage 11C held-domain models with their frozen candidates and
   evaluate the complete held fold. ROC-AUC is threshold independent;
   held-fold Macro-F1 uses a fixed 0.5 threshold because no held-fold label may
   select a threshold. Summarize per outer key mean and standard deviation.
3. Compute the mean of all ten pairwise Jaccard similarities among the five
   held-fold SHAP Top-15 sets for each outer key.
4. Define direction-conflict rate as `1 - direction_consistency`, averaged over
   the 15 F-DCSS features for each outer key.
5. Relate source-only diagnostics to (a) F-DCSS-15 minus F-Stable external
   ROC-AUC and (b) internal-minus-external F-DCSS-15 ROC-AUC. Spearman
   correlations over 60 outer keys are descriptive because keys share data.
   Dataset-model strata with n=10 are also reported. Formal uncertainty is
   deferred to Stage 11G.
6. Relate F-DCSS-15 minus F-Stable ECE-15 change to the paired external
   ROC-AUC change. Lower ECE is favorable; higher ROC-AUC is favorable.

## 6. Interpretation rules

- DCSS-Full remains the primary method even if an ablation or F-DCSS-20 is
  numerically better.
- An ablation benefit shows that the removed component was harmful in that
  setting; it does not justify retrospectively changing the method.
- Random results are summarized by mean, quantiles, and the fraction exceeding
  DCSS-Full. They are not searched for a favorable seed.
- Correlations are exploratory diagnostics, not causal effects or independent
  sample evidence.
- Stage 11D's failed strong contribution condition remains unchanged.

## 7. Acceptance criteria

- Exactly 15 unique valid features per ablation key and random seed.
- Exactly 180 completed new ablation runs and 1,800 random runs.
- Identical outer roles, source-only tuning, and external cohorts.
- Model classes equal `[0, 1]` for every run.
- Formula-ablation metrics reproduce from float64 saved predictions.
- Every problem, interruption, warning, or scope decision is recorded in
  `issues/ISSUE_LOG.md`.
