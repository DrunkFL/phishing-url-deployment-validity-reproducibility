# Part 14 Issues and Resolutions

## Resolved during execution

1. **Part 12 did not persist source-validation probabilities.** The 60 locked S3 candidates were deterministically replayed from the original training assignments and saved parameters. Every reconstructed low-FPR threshold was checked against the frozen Part 12 threshold before analysis.
2. **An empirical 0.1% budget maps to only a few false positives for ISCX validation folds.** Integer false-positive counts and one-sided 95% Wilson and exact Clopper-Pearson upper bounds are reported; the nominal budget is not presented as a population guarantee.
3. **Probability ties can make a one-step threshold change move multiple samples.** Adjacent thresholds come from the complete validation ROC threshold sequence with `drop_intermediate=False`; all three scopes are recomputed at the safer, selected, and more-permissive neighbors.
4. **Random row holdout would violate the domain-separation logic.** The calibration/audit sensitivity uses `GroupShuffleSplit` with registrable-domain SHA-256 as the group.
5. **The original validation cohort selected both the model candidate and the low-FPR threshold.** The grouped holdout quantifies threshold-selection reuse, while the saved three-candidate Macro-F1 margin records model-selection pressure.

## Residual limitations

1. The grouped holdout is not a fully independent calibration experiment because the locked model candidate was selected using the full source validation cohort.
2. Repetitions reuse the same underlying corpora and are not independent datasets.
3. External FPR is a frozen-threshold transfer outcome, not a target-calibrated operating guarantee.
4. A fully independent calibration split would require a prespecified repartition and model replay; it should be added only if a population-level low-FPR claim is retained.
