# Part 13 Issues and Resolutions

## Resolved during execution

1. **Float32 probability ties changed forensic AUC recomputation.** Earlier prediction files stored probabilities as float32, while metrics were computed from float64 values. Part 13 uses the validated Stage 11B regenerated audit table and checks the AUC complement identity; it does not reinterpret small Parquet recomputation differences as substantive findings.
2. **Zero-valued or near-zero associations can create unstable sign flips.** A fixed `|Spearman rho| < 0.02` dead-zone is used for the primary sign-flip count. The legacy strict nonzero flip rate is retained as a sensitivity column.
3. **Ordinary random splits could leak domains into source discrimination.** All source-classifier evaluations use `GroupShuffleSplit` with registrable domain as the group.
4. **Source classification could be driven by class prevalence.** Separate models are fitted inside benign and phishing strata, so prevalence cannot explain the source AUC.
5. **SHAP sign and label association answer different questions.** The output names the comparison `direction alignment`, not SHAP correctness, and reports source-label and target-label conflict separately.
6. **The first complete run stopped on a false missing-join alarm.** All 2,100 SHAP rows matched, but constant features legitimately had undefined Spearman correlations. The validation was changed to inspect the merge indicator directly; undefined directions are retained as neutral `0` and excluded from conflicts.

## Residual limitations

1. Source identity remains a bundle of collection-time, curation, and sampling effects; this audit does not isolate one causal driver.
2. The source-discrimination model uses all 35 lexical features and is diagnostic. It is not a phishing detector and is not proposed for deployment.
3. Target labels are used only after model fitting for diagnosis. Reversed-score AUC must never be presented as an achievable zero-target-feedback operating result.
4. Existing Part 5 SHAP cohorts are reused so the analysis remains tied to frozen experiments; no new target-optimized explanations are produced.
