# PhreshPhish full-split result audit

Audit date: 2026-10-07

## Run status

- Status: `FULL_RUN_PASS`
- Target revision: `eabec4b7a66324b79cc8a0ad856d1731dc26fe1a`
- Target rows: 168,060 (91,260 benign; 76,800 phishing)
- Invalid target URLs: 0
- Target feature SHA-256: `5cd003999629e29ad31bd7a01a4f4fd9313b7caf1e44227fe5893d6ed19dc495`
- Evaluation time: 612.72 seconds with `n_jobs=4`

## Integrity checks

- `external_metrics.csv`: 720 rows; stored SHA-256 matches; no duplicate experimental keys.
- `source_replay_check.csv`: all 60 source/repetition/model combinations present; stored SHA-256 matches.
- Maximum absolute source replay drift: ROC-AUC `2.22e-16`; Macro-F1 `0.0`.
- Prediction artifacts: 60 Parquet files; every file contains 168,060 unique samples; no missing probabilities.
- Probability range across artifacts: `[0.0, 1.0]`.
- Expected result design is complete: two source datasets, ten repetitions, three models, two cohorts, three threshold policies, and two weighting policies.
- The only missing values are the 240 intentionally inapplicable `source_validation_fpr_budget` entries for the ordinary Macro-F1 threshold.
- Maximum AUC negation-identity error: `4.44e-16`. Maximum complement-identity discrepancy: `1.22e-6`, consistent with finite-precision/tie effects near saturated probabilities.
- No runtime error or warning was written to the evaluation error log.

## Primary domain-filtered results

Values are mean ± standard deviation across ten source-split repetitions, using the source-selected Macro-F1 threshold and row weighting.

| Source | Model | ROC-AUC | PR-AUC | Macro-F1 | Recall | FPR |
|---|---:|---:|---:|---:|---:|---:|
| ISCX-URL2016 | LR | 0.640 ± 0.039 | 0.585 ± 0.037 | 0.487 ± 0.008 | 0.904 ± 0.012 | 0.781 ± 0.015 |
| ISCX-URL2016 | RF | 0.560 ± 0.020 | 0.492 ± 0.019 | 0.359 ± 0.020 | 0.950 ± 0.011 | 0.944 ± 0.024 |
| ISCX-URL2016 | XGB | 0.553 ± 0.035 | 0.476 ± 0.045 | 0.342 ± 0.009 | 0.956 ± 0.012 | 0.963 ± 0.011 |
| PhiUSIIL | LR | 0.371 ± 0.003 | 0.312 ± 0.003 | 0.329 ± 0.003 | 0.838 ± 0.010 | 0.908 ± 0.003 |
| PhiUSIIL | RF | 0.414 ± 0.010 | 0.335 ± 0.004 | 0.300 ± 0.001 | 0.888 ± 0.002 | 0.952 ± 0.001 |
| PhiUSIIL | XGB | 0.346 ± 0.005 | 0.301 ± 0.004 | 0.300 ± 0.001 | 0.892 ± 0.002 | 0.953 ± 0.000 |

Primary-cohort sizes vary with the prespecified source/repetition-specific domain exclusion: 163,847–164,857 rows for ISCX-URL2016 sources and 123,246–124,390 rows for PhiUSIIL sources.

## Initial interpretation boundary

The third-source test shows severe deployment degradation. ISCX-URL2016 logistic regression is the strongest transferred model but still produces a mean FPR of 0.781. PhiUSIIL-derived models have ROC-AUC below 0.5 after domain-overlap exclusion, indicating systematic ranking inversion rather than merely a threshold mismatch. Source-validation low-FPR thresholds also fail to transfer: mean target FPR ranges from 0.403 to 0.966 for the source 0.1% policy and from 0.458 to 0.970 for the source 1% policy.

These results support a bounded claim about cross-source transport failure and threshold instability. They do not yet establish a causal mechanism. Manuscript integration should wait for aggregate comparison with the original two-source results and the prespecified SHAP/source-versus-target explanation analysis.
