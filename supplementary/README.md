# Supporting Information

This directory contains the files cited as Supporting Information by the
manuscript-associated `v1.0.0` release.

## Data and protocol files

| Citation in manuscript | File | Contents |
|---|---|---|
| Supplementary Data S1 | `supplementary_feature_dictionary.csv` | Ordered definitions for the 35 offline web-address features |
| Supplementary Data S2 | `supplementary_model_candidates.json` | The nine prespecified model candidates and fixed parameters |
| Supplementary Protocol S3 | `supplementary_shap_protocol.md` | Backgrounds, cohorts, explainers, scales, and SHAP stability estimands |
| Supplementary Section S6 | `supplementary_section_s6.md` | Third-source sensitivity analysis and interpretation boundary |

## Supplementary tables

| Table | File | Contents |
|---|---|---|
| S1 | `tables/table_s1_dataset_audit.csv` | Dataset cleaning and entity audit |
| S2 | `tables/table_s2_internal_regime_performance.csv` | Internal S0-S3 performance by model and repetition summary |
| S3 | `tables/table_s3_shap_stability.csv` | Seed, partition, regime, and source SHAP stability |
| S4 | `tables/table_s4_s3_feature_methods.csv` | S3 feature-method comparison and retained feature counts |
| S5 | `tables/table_s5_external_transfer.csv` | Zero-target-feedback transfer results |
| S6 | `tables/table_s6_false_positive_domains.csv` | External false-positive domain concentration |
| S7 | `tables/table_s7_low_fpr_operating_points.csv` | Low-FPR operating-point summary |
| S8 | `tables/table_s8_low_fpr_run_level.csv` | All 360 low-FPR run-level records |

## Supplementary figure

`figures/Figure_S1.pdf` is the submission copy of Supplementary Figure S1. The
PNG beside it is an accessible preview of the same fifteen-bin external
reliability diagrams.

All thresholds in Tables S7-S8 were selected on source-validation data and
applied unchanged to source-test and external cohorts. The Wilson and exact
Clopper-Pearson bounds are post-selection descriptions of the frozen
validation cohorts; they are not selection-adjusted or population-level
confidence guarantees.

Author-created documentation and aggregate result tables in this directory
are licensed under CC BY 4.0 as defined in the repository-level
`LICENSE-DOCUMENTATION` and `LICENSE_SCOPE.md`. Those notices do not license
underlying source datasets or record-level dataset derivatives.
