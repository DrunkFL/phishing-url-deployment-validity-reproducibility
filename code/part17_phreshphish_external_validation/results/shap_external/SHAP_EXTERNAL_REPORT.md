# Source-to-PhreshPhish SHAP transportability report

Date: 2026-10-07  
Status: **FULL RUN PASSED AND AUDITED**  
Scope: 2 source datasets × 10 repetitions × 3 model families = 60 comparisons

## Integrity result

The repaired analysis completed all prespecified comparisons without relaxing
the source-replay guards.

- `run_metadata.json`: `FULL_RUN_PASS`;
- `validation_summary.json`: `PASS`;
- comparison rows: 60/60, with exact source/repetition/model keys;
- SHAP value files: 120/120;
- global summary files: 120/120;
- rows per source and target explanation cohort: 200;
- source ROC-AUC replay maximum absolute delta: `1.1185e-08`;
- source Macro-F1 replay maximum absolute delta: `0`;
- maximum probability delta versus archived float32 predictions: `2.9802e-08`;
- frozen-threshold classification changes: 0;
- archived-threshold inconsistencies: 0;
- archived float32 probability vectors: exact for all 60 runs;
- missing comparison values: 0;
- runtime warnings: 0;
- target cohort hashes per source direction: 1.

All detected BLAS/OpenMP pools used the repaired 12-thread numerical topology.
The failed two-thread metadata-only run remains separately archived and is not
included in these results.

## Aggregate findings

Values are means over ten source-domain repetitions. Direction agreement is
calculated over the union of the source and target Top-20 features. Performance
loss is source-test score minus PhreshPhish score.

| Source | Model | Top-15 Jaccard | Rank Spearman | Top-20 direction agreement | Direction conflicts | ROC-AUC loss | Macro-F1 loss | Target FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ISCX-URL2016 | LR | 0.811 | 0.952 | 1.000 | 0.0 | 0.257 | 0.379 | 0.781 |
| ISCX-URL2016 | RF | 0.718 | 0.924 | 0.909 | 1.4 | 0.439 | 0.599 | 0.944 |
| ISCX-URL2016 | XGB | 0.633 | 0.917 | 0.958 | 0.6 | 0.446 | 0.555 | 0.963 |
| PhiUSIIL | LR | 0.856 | 0.954 | 0.986 | 0.0 | 0.623 | 0.657 | 0.908 |
| PhiUSIIL | RF | 0.588 | 0.863 | 0.851 | 2.4 | 0.585 | 0.690 | 0.952 |
| PhiUSIIL | XGB | 0.765 | 0.941 | 0.810 | 4.0 | 0.653 | 0.694 | 0.953 |

The central result is negative but informative: high feature-rank agreement can
coexist with severe loss of discrimination and threshold validity. LR retained
the highest rank and direction agreement in both source directions, including
zero Top-15-union direction conflicts, yet still suffered large ROC-AUC and
Macro-F1 losses. Explanation-ranking stability therefore did not certify
decision-rule portability.

Tree models showed more attribution-direction conflicts. Across all repetitions,
the most recurrent conflicts were `url_length` for PhiUSIIL XGB (10/10),
`subdomain_depth` for PhiUSIIL XGB (10/10) and RF (10/10),
`registrable_domain_length` for PhiUSIIL RF (7/10) and XGB (6/10),
`letter_count` for PhiUSIIL XGB (7/10), and `special_char_ratio` for ISCX RF
(7/10). These are associations with the transported model explanations, not
evidence that any feature caused the performance failure.

## Descriptive association with performance loss

The 60 comparisons share fixed target cohorts, so these correlations are
descriptive and are not inferential tests.

- Top-15 Jaccard versus ROC-AUC loss: `rho = 0.140`;
- Top-15 Jaccard versus Macro-F1 loss: `rho = -0.080`;
- full-rank Spearman versus ROC-AUC loss: `rho = 0.076`;
- full-rank Spearman versus Macro-F1 loss: `rho = -0.158`;
- Top-20 direction agreement versus ROC-AUC loss: `rho = -0.522`;
- Top-20 direction agreement versus Macro-F1 loss: `rho = -0.689`.

Direction agreement was more consistently aligned with performance retention
than rank overlap in this fixed-corpus analysis. This pattern remains
source-dependent: the corresponding source-stratified correlations vary in
magnitude and sometimes in sign. It must not be presented as a universal or
causal relationship.

## Manuscript-safe conclusion

On deterministic balanced source-test and PhreshPhish cohorts, global SHAP
feature rankings were often similar across sources, especially for logistic
regression. This apparent explanation stability did not imply portable ranking
performance or operational thresholds. Direction conflicts were more frequent
for tree models and were concentrated in a small set of lexical URL features,
but the analysis identifies model-and-source-conditioned associations only.

Acceptable claim: source-to-target explanation agreement is a separate audit
dimension and cannot substitute for direct external performance and operating-
point validation.

Do not claim: that SHAP instability caused transport failure; that stable SHAP
rankings validate deployment; or that the listed features are universal causes
of phishing-detection failure.

## Artifact hashes

- `comparison_metrics.csv`: `9acbd3806126dc3b819205b53e42ccc22e86d4d869e946e5e1ed5a1f4594fe90`
- `aggregate_summary.csv`: `02279bd3e045171cd19633642952c3286ddba24da09f68ba4dabbd6699cdb08d`
- `direction_conflicts.csv`: `f496259861f3e979f565f33e84645f0adc3feaad4bc58ff57b295e2343756e6e`
- `feature_conflict_summary.csv`: `ca1af2cbd4a5c97a1c287b9e1869ef81166a4da9c86c99f27af9cd318d39c2e9`
- `stability_performance_association.csv`: `f6f8e04ac85d3f1270319c203fd691969bbf982adcb831004774df3f10b92637`

