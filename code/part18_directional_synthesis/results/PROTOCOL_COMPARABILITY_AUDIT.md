# Protocol Comparability Audit for the Directed Transfer Synthesis

**Decision: conditionally comparable for a bounded directional synthesis.** The six directed source-to-target edges can be displayed together because they use the same 35 offline URL features, the same three model families (LR, RF, and XGBoost), ten repetitions per direction-model cell, source-only model/threshold selection, primary registrable-domain-overlap exclusion, and row-weighted target metrics. ROC-AUC is compared as a threshold-independent ranking measure. Target FPR is compared only at a source-selected threshold of the declared kind.

## Comparability conditions

- The graph contains the six directed edges among the three evaluated sources: ISCX-URL2016, PhiUSIIL, and PhreshPhish v1.0.1.
- ISCX/PhiUSIIL-to-PhreshPhish is the independent third-source evaluation (E1); PhreshPhish-to-ISCX/PhiUSIIL is the secondary reciprocal diagnostic (E3).
- Model families and feature definitions are aligned, but training-set sizes, source distributions, and selected hyperparameters need not be identical. The synthesis therefore describes direction-dependent transfer; it is not a controlled causal comparison of data-collection mechanisms.
- Ordinary-threshold FPR uses each source model's frozen source-validation Macro-F1 threshold.
- Low-FPR budget ratios divide target FPR by the nominal 0.1% or 1% source-validation budget. They do not divide by the realized source-validation or source-test FPR.
- The SHAP-performance panel covers only the 60 E2 source-to-PhreshPhish comparisons. It does not claim six-edge explanation coverage.

## Evidence-state boundary

- The two original ISCX/PhiUSIIL directions are carried forward as `user_authoritative`.
- E1 and E2 are verified primary additions.
- E3 is verified but remains a secondary reciprocal-transfer diagnostic.

## Prohibited interpretations

- Do not describe the graph as temporal, prospective, or production validation.
- Do not describe PhreshPhish as a universally better training source.
- Do not infer that collection mechanisms caused any transfer asymmetry.
- Do not treat SHAP agreement as evidence of predictive or operating-point portability.
- Do not generalize the six-edge pattern beyond the three evaluated fixed public sources.
