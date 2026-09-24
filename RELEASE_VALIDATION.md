# Release Validation Record

Validation date: 23 September 2026

## Repository checks

An isolated Python 3.9 environment reconstructed from
`environment/requirements.txt` completed the following checks on the
`v1.0.0` release candidate:

- all 212 release files were matched to `MANIFEST.csv` by path, canonical byte length, and SHA-256; text line endings are normalized to LF for this check, while binary files are compared byte for byte;
- all 82 Python files passed syntax parsing;
- 180 public text files contained no local user path;
- six public Parquet schemas contained no banned plaintext URL, host, domain, or source-file field;
- representative command-line entry points from Parts 4, 11, 14, and 15 imported successfully and returned a zero exit status for `--help`.
- the release contained no detected credential pattern, raw source archive, or plaintext URL/host/domain identifier field; and
- the largest file was 39,867,801 bytes, below GitHub's 100 MB per-file limit.

The command used was:

```powershell
python tools\validate_repository.py --smoke
```

## Representative training rerun

A clean temporary Part 4 output directory was used to rerun PhiUSIIL, S3-domain, repetition r00, and logistic regression with `--n-jobs 1`. Inputs were read from the public sanitized feature and assignment files. With the frozen Windows environment and the numerical libraries' default thread settings, the rerun reproduced the retained reference row:

| Metric | Rerun | Retained reference |
|---|---:|---:|
| Macro-F1 | 0.9854696115222255 | 0.9854696115222255 |
| ROC-AUC | 0.9947835521789182 | 0.9947835521789182 |
| Validation Macro-F1 | 0.9868251613366142 | 0.9868251613366142 |
| Validation ROC-AUC | 0.9928290724070032 | 0.9928290724070032 |
| False positives | 148 | 148 |
| False negatives | 361 | 361 |

Imposing single-thread BLAS environment variables changed this representative run to Macro-F1 0.985559993956699 and ROC-AUC 0.9949290488066245. The selected candidate and threshold were unchanged. This small difference is recorded as numerical-environment sensitivity; users should not expect bitwise equality across arbitrary platforms or BLAS thread configurations.

This release validation is a representative rerun, not a second execution of all Parts 2-15. The retained Part 11, 14, and 15 validation summaries document the completed full experiment and its deterministic replay checks.
