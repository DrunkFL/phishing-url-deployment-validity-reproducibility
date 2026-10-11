# Supporting Information for the JISA Manuscript

This directory is the Codex-accepted supporting-information package for the current manuscript. It contains the six items cited in the text and the frozen Public Suffix List (PSL) snapshot used for registrable-domain extraction. The files are evidence-bound to archived project outputs; building this package does not train a model, regenerate a prediction, or recompute a SHAP value.

## Contents

| Manuscript label | File | Role |
|---|---|---|
| Supplementary Data S1 | `supplementary_feature_dictionary.csv` | Ordered definitions of the 35 frozen URL features |
| Supplementary Data S2 | `supplementary_model_candidates.json` | Nine prespecified model candidates and fixed parameters |
| Supplementary Protocol S3 | `supplementary_shap_protocol.md` | Original two-source and PhreshPhish SHAP protocols |
| Supplementary Table S4 | `tables/table_s4_s3_feature_methods.csv` | S3 internal results for five feature methods |
| Supplementary Table S5 | `tables/table_s5_external_transfer.csv` | Primary domain-filtered bidirectional transfer results |
| Supplementary Figure S1 | `figures/Figure_S1.pdf` | Complete external reliability diagrams |
| Supplementary Section S6 | `supplementary_section_s6.md` | URL-Phish sensitivity analysis and interpretation boundary |
| Frozen PSL resource | `resources/public_suffix_list_tldextract_5.1.3.dat` | Exact suffix snapshot used by the parser |

`SUPPLEMENT_BINDING_MATRIX.csv` records each file's source and acceptance basis. `SUPPLEMENT_PACKAGE_MANIFEST.csv` records portable package-relative paths, byte sizes, and SHA-256 values. `SUPPLEMENT_EVIDENCE_INDEX.md` maps manuscript claims to their supporting files and evidence classes.

## Validation and manifest refresh

From this directory, run:

```text
python scripts/build_supporting_information.py
```

The release package already contains the accepted files. The script verifies their structure, row counts, model grid, and frozen PSL hash, then refreshes the binding matrix and package manifest. Paths in both inventories are repository-relative; no server path or user-specific absolute path is embedded.

## Interpretation boundary

The package preserves negative and directionally inconsistent results. Supplementary Table S5 does not establish external portability; Supplementary Protocol S3 does not treat explanation agreement as evidence of predictive transfer; and Supplementary Section S6 is a sensitivity analysis rather than a fourth independent source or prospective deployment validation.
