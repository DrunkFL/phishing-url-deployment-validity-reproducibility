# Stage 11D Analysis Lock

Lock date: 2026-09-08 (Asia/Shanghai)

Status: locked before any F-DCSS final model was trained or any F-DCSS internal
or external performance was inspected

## Scope

Stage 11D trains and evaluates F-DCSS-10, F-DCSS-15, and F-DCSS-20 for two
source datasets, ten outer repetitions, and three model families: 180 final
model runs. F-DCSS-15 is confirmatory; F-DCSS-10 and F-DCSS-20 are sensitivity
analyses.

The five frozen deterministic comparators are F-All, F-Single, F-Stable, F-MI,
and F-Permutation. Their existing predictions and metrics are reused because
they already use the same master S3/S4 assignments, candidate grids, model
seeds, and source-validation threshold rule.

## Random baseline boundary

The overall protocol names F-Random-15 as a comparator, while the staged plan
also explicitly assigns 30 random feature sets to Stage 11E. Before inspecting
F-DCSS performance, this ambiguity is resolved as follows:

- Stage 11D performs the confirmatory F-DCSS-15 comparisons and the deterministic
  baseline comparisons.
- Stage 11E trains all 30 predeclared random sets together with the DCSS
  component ablations and reports their distribution.
- F-Random-15 is exploratory and is not part of H1 or H2. Stage 11D will not
  claim superiority over random selection.

## Training and threshold rule

Each F-DCSS feature set independently applies the frozen three-candidate model
grid. Candidate and decision threshold are selected only on the source outer
training and validation subsets. The threshold grid is 0.05-0.95 in steps of
0.01 and maximizes source-validation Macro-F1 with the frozen tie breakers.
Model seeds are 20261001-20261010. Model classes must equal `[0, 1]`.

## Evaluation

- Internal evaluation: frozen source S3 outer test subset.
- External evaluation: frozen S4 target, with the primary-domain-filtered cohort
  as primary and the unfiltered cohort as secondary.
- Primary endpoint: ordinary external ROC-AUC.
- Secondary endpoints: PR-AUC, Macro-F1, balanced accuracy, recall, FPR, Brier
  score, log loss, 15-bin equal-width ECE, training/tuning time, inference time,
  and feature count.
- Probabilities are stored as float64 to preserve ranking-metric reproducibility.
- The inverted-score AUC remains diagnostic only and cannot replace the primary
  endpoint.

## Predeclared decisions

- H1 descriptive Stage 11D check: for each dataset/model, report the paired mean
  F-DCSS-15 minus F-All internal Macro-F1 and whether it is at least -0.01.
- H2 descriptive Stage 11D check: average the paired F-DCSS-15 minus F-Stable
  primary external ROC-AUC across the three model families within each outer
  repetition, then report the ten-repetition mean for each direction.
- Formal two-level cluster-bootstrap intervals and multiplicity-controlled tests
  remain Stage 11G outputs.
- No feature, formula, k, model family, cohort, or endpoint may be replaced after
  target performance is observed.
