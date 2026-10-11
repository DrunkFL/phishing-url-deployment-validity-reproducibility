# Directional transfer synthesis

This part derives the manuscript's six-edge transfer summary from frozen Part
11, Part 12, and Part 17 aggregate outputs. It does not fit a model, tune a
threshold, or introduce another independent source.

From the repository root, run:

```powershell
python code\part18_directional_synthesis\scripts\build_directional_synthesis.py
python code\part18_directional_synthesis\scripts\audit_directional_synthesis.py
```

`results/INPUT_MANIFEST.json` records the six hashed inputs. The audit verifies
the design cardinalities, prespecified direction-model means, nominal-budget
ratios, evidence-state boundaries, and the pooled descriptive SHAP association.
The rendered figure is stored under `figures/`.

