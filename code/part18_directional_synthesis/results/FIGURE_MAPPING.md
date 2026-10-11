# Figure Mapping: Directional Validity Synthesis

## Proposed manuscript figure

- **Manuscript number:** Figure 10
- **Editable source:** `Figure_10_directional_validity_synthesis.svg`
- **Rendered manuscript asset:** `Figure_10_directional_validity_synthesis.png`
- **Evidence role:** derived visualization of frozen E1, E2, E3, and original two-source outputs; no new model fitting, threshold selection, or numerical evidence.

## Panel mapping

- **Panel A:** Mean ± SD ROC-AUC for LR, RF, and XGBoost on all six directed source-to-target edges. The dashed line is chance (0.5).
- **Panel B:** Range across model-family mean target-FPR-to-nominal-budget ratios at 0.1% and 1% source-validation FPR budgets.
- **Panel C:** E2 Top-15 SHAP Jaccard agreement versus ROC-AUC loss for 60 dependent source-to-PhreshPhish comparisons. The reported Spearman coefficient is descriptive.

## Design provenance

This is an original deterministic scientific figure generated from the audited CSV outputs in this directory. It does not imitate a specific published figure and does not use generative image synthesis. Color, marker, type, and line conventions are kept consistent with the manuscript's existing performance and SHAP figures.

## Adoption gate

Adopt only if `AUDIT_REPORT.md` is `PASS`, the PNG and SVG hashes are recorded after the final build, and the compiled manuscript confirms legibility at publication scale.

## Adopted artifact record

- `manuscript/figures/Figure_10.png`: SHA-256 `D7577603288E3447224B58450CE5F2A4D8F5B0F1104AC170F4168ED0CB3AB44E`
- `manuscript/figures/Figure_10.svg`: SHA-256 `4274786584F66F351DA3209C4F99830E647EC5966BAD089D17BA2BE693B4A5C4`
- Independent synthesis audit: `PASS` (98 checks).
- Compiled placement: page 29 of the 38-page Stage 18 build; no clipping, overlap, black boxes, or unreadable panel labels observed at 180 dpi.
