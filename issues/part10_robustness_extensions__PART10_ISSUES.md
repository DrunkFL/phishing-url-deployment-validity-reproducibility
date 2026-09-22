# Part 10 Issue Log

## Design issues identified before execution

1. Earlier notes used obsolete Part 2 and Part 3 directory names. Scripts use the verified on-disk paths `part2_url_normalization_leakage_audit` and `part3_s0_s4_data_splits`.
2. Row-level external metrics can be dominated by domains containing many URLs. A domain-equal analysis is therefore reported beside, not in place of, the primary row-level result.
3. Target-optimal thresholds would leak target labels if used operationally. They are restricted to post hoc diagnostic use and labeled as oracle values.
4. A full domain-cluster bootstrap for all 300 external configurations would be computationally disproportionate and would invite selective interpretation. The protocol freezes a representative XGBoost r00 comparison of F-All and F-Stable in both directions.

## Issues observed during execution

5. The unit tests emitted 14 `PyparsingDeprecationWarning` messages from Matplotlib's installed compatibility layer. All three numerical tests passed; the warnings do not involve experiment code or calculations, so the frozen environment was not changed.
6. Deduplicated ISCX S3 results showed large repetition-to-repetition variation during execution (for example, some tree runs were near 1.00 while others were below 0.82). This is retained as a substantive domain-composition sensitivity and will be summarized across all fixed repetitions rather than removed as an outlier.
7. The first aggregate run completed all numerical CSVs and figures but failed while formatting the final Markdown report because Pandas' optional `tabulate` package was absent. The report writer was changed to a local deterministic Markdown formatter; no package was installed and no numerical analysis was rerun or altered.
8. Domain-equal Macro-F1 increased markedly for PhiUSIIL-to-ISCX even though domain-equal FPR remained 0.948-1.000. This is not treated as recovery: equal domain weights upweight numerous small phishing-associated domains relative to a few large benign domains. The manuscript therefore reports Macro-F1 together with FPR and class-conditional interpretation.
9. Most post hoc target-oracle thresholds were at or near 1.0 and sometimes approximated an all-negative rule. Oracle gains are reported as evidence of source-threshold incompatibility, not as target-calibrated deployment performance.

## Final status

- Numerical unit tests: 3 passed.
- Artifact validation: 16 checks passed.
- Deduplicated runs: 120/120 complete.
- Domain-equal configurations: 600/600 complete.
- External calibration configurations: 300/300 complete.
- Representative cluster bootstrap: 4 configurations x 2,000 replicates.
