# Part 15 Issues and Resolutions

## Resolved during execution

1. **The manuscript called the H1 procedure a paired t-test.** The implemented procedure is now named precisely: a one-sample, one-sided t-test on ten paired repetition-level differences against the -0.01 margin.
2. **Ten overlapping repetitions were easy to misread as independent datasets.** All ten differences are disclosed, and effect sizes, intervals, and direction consistency are primary. P-values are explicitly auxiliary.
3. **A repetition bootstrap alone does not remove dependence.** A transparent equicorrelation stress grid from rho=0 to 0.90 reports effective repetition count and widened intervals.
4. **The H2 interval is not a literal nested domain bootstrap.** The audit reuses the locked linearized domain-Gaussian-multiplier plus outer-bootstrap interval and keeps its approximation boundary visible.
5. **A positive result in one transfer direction could be selectively emphasized.** Both directions are kept in the same table and the study-wide strong-contribution condition remains false.

## Residual limitations

1. The equicorrelation grid is a stress analysis, not an estimate of the true correlation among repetitions.
2. With only ten repetitions, high-correlation effective sample sizes approach one; no p-value can substitute for new independent corpora.
3. The fixed two-corpus design supports corpus-conditional conclusions only.
4. The non-inferiority margin of 0.01 is a prespecified engineering tolerance, not an externally validated operational standard.
