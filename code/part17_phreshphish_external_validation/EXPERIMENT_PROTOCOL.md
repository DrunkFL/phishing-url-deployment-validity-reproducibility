# Locked PhreshPhish External-Validation Protocol

Status: full experiment completed and audited  
Target paper: *Journal of Information Security and Applications*  
Dataset revision: `eabec4b7a66324b79cc8a0ad856d1731dc26fe1a` (PhreshPhish v1.0.1)

## Purpose

This experiment tests whether the deployment-validity findings observed between ISCX-URL2016 and PhiUSIIL persist on an independent third source. A valid run may confirm, weaken, or contradict the two-source pattern. Favorable performance is not a success criterion.

## Locked boundaries

1. Only stored URL strings and non-content metadata are read. The pipeline never visits a URL and never reads the `html` column.
2. The exact frozen Part 2 normalizer, Public Suffix List snapshot, and 35-feature extractor are imported from the archived original experiment.
3. Source model family, selected candidate, random seed, ordinary threshold, and low-FPR threshold selection are determined entirely from the source corpus.
4. PhreshPhish labels are used only after target probabilities have been generated, for evaluation and a prespecified class-balanced SHAP cohort in the later explanation stage.
5. Target rows sharing a registrable domain with source training or validation are excluded from the primary cohort. Unfiltered results and normalized-URL, host, and registrable-domain overlap counts remain visible.
6. Natural test composition is primary. Realistic-base-rate benchmark views are a separate analysis and may be added only after their index semantics are verified against this pinned revision.
7. Smoke subsets are marked `smoke_only` and cannot be cited in the manuscript.

## Primary outputs

- Source-replay check for every source, repetition, and model.
- ROC-AUC, PR-AUC, Macro-F1, recall, FPR, alerts per 10,000 benign URLs, Brier score, log loss, and 15-bin ECE.
- Ordinary row-weighted and registrable-domain-equal estimates.
- Ordinary validation-Macro-F1 threshold plus source-validation 0.1% and 1% FPR-constrained thresholds.
- AUC-complement identity check and overlap counts for every source/repetition/model run.

## Run validity rules

- The target feature order must equal the frozen source feature order.
- The dataset commit, shard LFS hashes, label mapping, expected row counts, PSL hash, and derived feature-file hash must be recorded.
- A full run stops when replayed source-test ROC-AUC or Macro-F1 differs from the saved source result by more than the declared tolerance.
- Full-split results are not integrated into the manuscript until aggregate direction, uncertainty, overlap exclusions, and the explanation-stability analysis have been reviewed.

## Deliberately deferred work

- Top-15 SHAP source-versus-PhreshPhish agreement and direction-conflict analysis.
- Class-conditional three-source discrimination.
- Verified reconstruction of the 975 realistic-base-rate benchmark views.
- Optional PhreshPhish-trained reverse direction.

These are later experiment segments, not missing values to infer from the smoke test.
