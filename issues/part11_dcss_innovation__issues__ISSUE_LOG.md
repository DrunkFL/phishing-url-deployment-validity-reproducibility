# Part 11 Issue Log

This log records experimental problems in a form suitable for later supervisor
or reviewer questions. An issue is not closed until its verification evidence is
saved.

## Summary

| ID | Stage | Severity | Status | Short description |
|---|---|---|---|---|
| P11-001 | 11A | Low | Resolved | Scientific inputs are spread across nine part directories |
| P11-002 | 11A | Medium | Resolved | The proposed DCSS formula initially left several implementation choices ambiguous |
| P11-003 | 11A | Medium | Resolved | The original success criterion combined ROC-AUC and FPR without a single primary endpoint |
| P11-004 | 11A | Low | Resolved | There is no standalone Part 1 directory |
| P11-007 | 11A | Low | Resolved | Assignment files were initially categorized as generic data in the manifest |
| P11-005 | 11B | Major | Resolved | Existing deduplicated partitions did not preserve master sample roles |
| P11-006 | 11B | Major | Resolved | Reverse-transfer XGBoost ROC-AUC is approximately 0.075 |
| P11-008 | 11B | Medium | Resolved | Historical F-All models were retained only for r00 |
| P11-009 | 11B | Low | Resolved | A foreground run stopped after 109 of 120 models |
| P11-010 | 11B | Medium | Resolved | Persisted float32 scores cannot exactly reproduce float64 ranking metrics |
| P11-011 | 11B | Low | Resolved | Validation expected a synonymous but different partition-source value |
| P11-012 | 11C | Low | Resolved | Initial DCSS runner patch contained malformed source fragments |
| P11-013 | 11C | Major | Resolved | A conventional full-table merge would read outer-test labels unnecessarily |
| P11-014 | 11C | Low | Resolved | Static leakage check matched the Part 3 directory name |
| P11-015 | 11C | Medium | Resolved | ISCX held-domain folds are not exactly class-balanced |
| P11-016 | 11C | Medium | Open | Some F-DCSS-15 sets strongly overlap existing selectors |
| P11-017 | 11D | Medium | Resolved | Random-15 was assigned to both Stage 11D and Stage 11E |
| P11-018 | 11D | Medium | Resolved | Historical float32 probabilities cross a few saved decision thresholds |
| P11-019 | 11D | Major | Open | The prespecified strong DCSS contribution condition was not met |
| P11-020 | 11E | Medium | Resolved | Saving every random model and prediction would exceed available disk space |
| P11-021 | 11E | Major | Open | Two DCSS components almost never change Top-15 membership |

## P11-001: Distributed scientific inputs

**Discovery.** Parts 2-10 contain thousands of files across data, assignments,
models, predictions, selection runs, and result directories. Hashing caches and
transient logs would add noise without improving scientific provenance.

**Impact.** A vague statement that inputs were frozen would not be reproducible,
while an indiscriminate manifest would include mutable cache files.

**Resolution.** The Stage 11A manifest recursively includes scientific artifacts
and source files from Parts 2-10, while excluding `.pytest_cache`, `__pycache__`,
`logs`, `.pyc`, and temporary files. The current manuscript is included
separately.

**Verification.** `data/manifests/input_manifest.csv` and
`results/stage11a_validation.json` record the final count, bytes, hashes, and
excluded-file policy.

**Supervisor answer.** We froze every scientific input consumed by the new
experiments, but intentionally excluded caches and execution logs because they
are not analytical inputs and may change without changing a result.

## P11-002: Ambiguous DCSS implementation details

**Discovery.** The planning formula did not specify within-fold normalization,
rank ties, zero or undefined SHAP-direction correlations, exact fold seeds, or
feature-score tie breaking.

**Impact.** Different reasonable implementations could select different
features, creating hidden researcher degrees of freedom.

**Resolution.** Protocol 1.0.0 fixes all five details, uses exactly five
StratifiedGroupKFold domain folds, and declares deterministic fold and model
seeds before training.

**Verification.** The future Stage 11C tests must reproduce identical feature
lists from the same inputs and seeds.

**Supervisor answer.** The method was fully specified before examining its
target performance; even undefined direction correlations are handled by a
predeclared rule rather than a post hoc choice.

## P11-003: Ambiguous primary success endpoint

**Discovery.** The initial plan described improvement in either external
ROC-AUC or FPR, which could allow the favorable metric to be selected after
results were known.

**Impact.** This would weaken the confirmatory claim and invite accusations of
metric cherry-picking.

**Resolution.** External ROC-AUC is the sole primary endpoint because it is
threshold independent and directly addresses the observed ranking reversal.
FPR, Macro-F1, PR-AUC, and calibration measures are secondary outcomes.

**Verification.** Protocol Sections 2, 10, and 11 declare the hypothesis and
comparison family before Part 11 model training.

**Supervisor answer.** We selected ROC-AUC in advance because the central
question is cross-source ordering, not target-specific threshold optimization.

## P11-004: No standalone Part 1 directory

**Discovery.** The experiment root contains the shared `.venv`, `data`, `env`,
and scripts plus Part 2 through Part 10, but no folder named Part 1.

**Impact.** This is an organization and provenance issue, not evidence that the
environment or data-preparation work is missing.

**Resolution.** The current structure was preserved to avoid breaking paths.
The Stage 11A environment snapshot identifies the shared virtual environment,
and the manifest freezes the foundational inputs without renaming historical
folders.

**Verification.** The environment snapshot points to
`${EXPERIMENT_ROOT}\.venv\Scripts\python.exe`, and the
Stage 11A validation confirms that Parts 2-10 and all required foundational
inputs are present.

**Supervisor answer.** The original first stage was stored in shared root-level
environment and data folders. We documented that legacy layout instead of
renaming it after the experiment and risking broken provenance.

## P11-007: Assignment-role classification in the manifest

**Discovery.** Assignment files are stored below `data/assignments`. The first
manifest build checked for the generic `data` directory before checking for the
more specific `assignments` directory, so the files were hashed correctly but
their descriptive role was recorded as data.

**Impact.** There was no loss or hash error, but the role summary did not show a
separate assignment count and was less useful for auditing.

**Resolution.** The role test now checks `assignments` before `data`. The
manifest and validation report are regenerated after this change.

**Verification.** The final Stage 11A validation report must contain a nonzero
`assignment` role count and zero hash mismatches.

**Supervisor answer.** This was a metadata classification issue, not an
experimental-data error. We corrected it before freezing the final manifest and
preserved the problem in the audit log.

## P11-005: Deduplicated role mismatch

**Discovery.** Retained-sample role agreement between master and deduplicated
assignments ranges from 71.70% to 95.92%, not 100%.

**Impact.** The old deduplicated-minus-master comparison mixes deduplication with
repartitioning and cannot isolate the duplicate-row effect.

**Resolution.** Stage 11B filtered each master assignment to the canonical
deduplicated representatives, preserved every surviving sample's role, and
refitted all 120 F-All runs.

**Verification.** Role agreement equals 100% over all 40 checks, S3 domain
overlap is zero, all 120 model classes equal `[0, 1]`, and the independent Stage
11B validation passes.

**Supervisor answer.** The first sensitivity analysis was not a strict matched
comparison. We detected this during independent review, withdrew that causal
interpretation, and designed a fixed-role rerun.

## P11-006: Reverse-transfer AUC inversion

**Discovery.** PhiUSIIL-to-ISCX XGBoost ROC-AUC is approximately 0.075 for
several feature sets.

**Impact.** The result could indicate a class-probability indexing problem or a
real source-dependent reversal. Threshold calibration alone cannot explain it.

**Resolution.** Stage 11B asserted all model class orders, regenerated float64
probabilities, reproduced historical metrics, reported diagnostic `1-p` AUC,
inspected class-conditional scores, and audited feature-label direction changes.

**Verification.** All 300 model configurations have class order `[0, 1]`; all
stored threshold predictions agree; 600 metric rows pass the declared numerical
tolerance. The extreme ordinary AUC remains, while `1-p` AUC is approximately
0.922-0.926 for four XGBoost feature sets. The issue is genuine cross-source
ranking reversal, not probability-column inversion.

**Supervisor answer.** We treat the extreme AUC as a validity question first. It
will become a scientific finding only after the probability and label pipeline
passes explicit checks.

## P11-008: Historical F-All models retained only for r00

**Discovery.** Parts 6 and 8 retain one model file for every repetition and
feature set, but the Part 4 training code explicitly saves F-All models only when
`repetition == "r00"`. All 60 S4 prediction files survived, but fitted F-All
objects for r01-r09 were never retained.

**Impact.** Existing F-All predictions and metrics can be recomputed, but the
`classes_` order of every historical fitted object cannot be inspected directly.

**Resolution.** Reconstructed all 60 source-S3 F-All models from the
frozen master assignments, seeds, candidate grids, and threshold rule. Save each
model under Part 11, assert class order `[0, 1]`, and compare reconstructed
probabilities with the frozen historical predictions by sample ID.

**Verification.** All selected candidates and thresholds match. Maximum
probability difference is 2.98e-08 and every reconstructed class order is
`[0, 1]`.

**Supervisor answer.** The old workflow intentionally preserved every prediction
but only the representative r00 F-All fitted objects. We therefore reconstructed
all repetitions from frozen inputs and verified them against the original
prediction files.

## P11-009: Interrupted foreground training run

**Discovery.** The first 120-run fixed-role command stopped after 109 completed
models when the interactive execution session was interrupted.

**Impact.** A monolithic rerun could have wasted completed computation or
silently duplicated outputs.

**Resolution.** Each run writes metrics, model, prediction, and a completion
marker atomically enough for run-level resumption. The resumed command skipped
109 completed keys and trained only the remaining 11.

**Verification.** The final artifact counts are exactly 120 models, 120
predictions, 120 completion markers, and 120 unique metric keys.

**Supervisor answer.** Training was checkpointed per experimental key, so an
execution interruption did not alter seeds, repeat completed runs, or lose
scientific outputs.

## P11-010: Float32 persistence and ranking-metric reproduction

**Discovery.** Historical ROC-AUC and PR-AUC were calculated from in-memory
float64 probabilities, while prediction Parquet files stored float32 values.
Tiny quantization changes created additional ties and an initial direct-Parquet
ROC-AUC mismatch of approximately 0.0057.

**Impact.** Recomputing a ranking metric only from persisted float32 scores is
not a bit-exact audit of the originally reported float64 metric.

**Resolution.** The final audit regenerated float64 probabilities from each
saved or reconstructed estimator, checked threshold classifications separately,
and recorded both quantization and metric-reproduction errors. A declared 1e-04
tolerance covers tree-model numerical variation without concealing meaningful
differences.

**Verification.** Maximum per-score difference is 2.98e-08, maximum regenerated
metric difference is 2.89e-05, and all 600 rows pass. No threshold classification
differs from the frozen predictions.

**Supervisor answer.** The discrepancy was caused by score serialization and
ties, not a class-label error. We disclosed it and used regenerated full-precision
scores for the forensic audit.

## P11-011: Partition-source validation label

**Discovery.** The validator expected the descriptive value
`master_roles_filtered_to_deduplicated_representatives`, whereas the training
outputs used the shorter `filtered_master_assignment`.

**Impact.** The first validation attempt failed despite both names denoting the
same frozen partition source.

**Resolution.** The validator now checks the actual pre-existing output value;
scientific artifacts were not rewritten.

**Verification.** All 120 rows contain `filtered_master_assignment`, and the
final Stage 11B validation passes.

**Supervisor answer.** This was a metadata-contract mismatch in a new validator,
not an experimental partition error; we corrected the assertion and preserved
the original results.

## P11-012: Malformed initial runner patch

**Discovery.** Static compilation caught malformed text in the first generated
version of the Stage 11C runner before any model was executed.

**Impact.** None on scientific outputs because the file did not compile and no
DCSS run had begun.

**Resolution.** The malformed fragments and unused import were removed, then all
Stage 11C source files were compiled and tested.

**Verification.** `py_compile` passes and `tests/test_dcss_utils.py` reports four
passing tests before the first pilot run.

**Supervisor answer.** This was a pre-execution implementation typo detected by
the quality gate; no data or experimental result was produced by the invalid
version.

## P11-013: Preventing outer-test label reads

**Discovery.** Reusing the historical convenience loader would load labels for
the full master table before filtering to the outer training role. Although the
test labels would not enter a calculation, that behavior would violate the
strict Stage 11C no-test-label-access requirement.

**Impact.** A weaker implementation could make the leakage audit depend on
claims about downstream variable use rather than an enforceable data boundary.

**Resolution.** The DCSS loader first reads the feature table without its label
column, then uses a Parquet row filter on `split_rXX == train` while reading the
assignment labels. Only those source outer-training rows are merged.

**Verification.** Run metadata declares `selection_scope` as
`source_outer_train_only`; Stage 11C validation will inspect every saved cohort
and assignment against the frozen outer roles.

**Supervisor answer.** DCSS does not merely ignore test labels logically; its
data-loading path excludes them before labels enter memory.

## P11-014: False positive in the static leakage check

**Discovery.** The first Stage 11C validation run rejected the runner because
the forbidden substring `s4_` occurs inside the historical directory name
`part3_s0_s4_data_splits`, although the runner referenced only S3 assignment
files.

**Impact.** The validation attempt failed, but no scientific artifact was
incorrect and no external file had been read.

**Resolution.** The coarse substring was removed. The validator now rejects
actual external-module and target-data identifiers while separately checking
all saved IDs against the outer-train role.

**Verification.** The final source-boundary and all 10 Stage 11C validation
groups pass.

**Supervisor answer.** This was a false positive in a defensive static test;
the stronger row-level audit confirms all 300 cohorts came only from source
outer-training rows.

## P11-015: Imperfect class balance under domain grouping

**Discovery.** ISCX held-domain folds have phishing prevalence between 16.00%
and 30.27%, despite stratification. Large registrable-domain groups prevent
simultaneous exact group exclusivity, equal fold size, and exact class balance.

**Impact.** Different leave-domain folds have different class composition and
training size. This is part of the domain-composition perturbation but should
not be described as perfectly balanced five-fold cross-validation.

**Resolution.** The locked `StratifiedGroupKFold` output was retained without
post hoc reassignment. Every fold contains both classes, and each SHAP
explanation cohort uses exactly 100 rows per class. Full fold counts are saved
for disclosure.

**Verification.** `dcss_fold_balance.csv` contains 100 fold records; domain
overlap is zero and no explanation cohort required reduction.

**Supervisor answer.** The folds are approximately stratified subject to strict
domain isolation. We report the residual imbalance rather than altering folds
after observing them.

## P11-016: High overlap with existing selectors

**Discovery.** F-DCSS-15 overlaps F-Single and F-Stable substantially. The
PhiUSIIL random-forest F-DCSS-15 set is identical to F-Single in all ten outer
repetitions.

**Impact.** A new score that usually selects the same features may provide
limited predictive novelty even if its formulation is new.

**Current resolution path.** Stage 11D tested the predeclared internal
non-inferiority and external ROC-AUC comparisons without changing the formula.
The strong condition was not met, so the overlap remains scientifically
important. Stage 11E will quantify component ablations and analyze the features
that differ from existing sets.

**Required verification.** Any innovation claim must be based on paired
performance or diagnostic evidence, not on the existence of a new acronym or
score alone. Stage 11D supplies conditional rather than general performance
support; Stage 11E mechanism evidence is still required.

**Supervisor answer.** DCSS is intentionally related to SHAP stability, so some
overlap is expected. We treat the very high overlap as a falsifiable risk and
will not claim added value unless downstream comparisons support it.

## P11-017: Ambiguous stage assignment for F-Random-15

**Discovery.** The locked comparator list includes 30 F-Random-15 sets, while
the staged experiment plan explicitly places their training in Stage 11E with
the ablations.

**Impact.** Without a pre-performance decision, random results could be added or
omitted opportunistically after seeing F-DCSS performance.

**Resolution.** Before any final F-DCSS model was trained, the Stage 11D analysis
lock assigned confirmatory and deterministic-baseline comparisons to 11D and
all 30 random sets to 11E. Random selection is exploratory and is not part of
H1 or H2.

**Verification.** `STAGE11D_ANALYSIS_LOCK.md` records the boundary and date. The
Stage 11D validator will require only the three F-DCSS sets and five frozen
deterministic comparators.

**Supervisor answer.** We resolved a scheduling ambiguity before viewing the new
method's performance. Random-15 remains mandatory, but it is evaluated with the
predeclared ablation family rather than used to alter the confirmatory result.

## P11-018: Historical threshold changes after float32 persistence

**Discovery.** During the combined Stage 11D audit, recomputing decisions from
historical float32 Parquet probabilities disagreed with the saved prediction
column for one internal ISCX RF F-Stable sample and 15 unique external ISCX-to-
PhiUSIIL RF F-All samples. The external primary and unfiltered cohort views
contain the same affected samples, so the combined table displays 30 external
mismatches rather than 30 unique cases.

**Impact.** Reconstructing historical discrete metrics by thresholding the
quantized probabilities can change a small number of boundary decisions. It
does not affect the new F-DCSS runs and is not evidence of label inversion.

**Resolution.** Historical accuracy, Macro-F1, recall, and FPR are checked
against the persisted prediction column. Differences in ROC-AUC and PR-AUC
after float32 persistence are retained as explicit audit columns. All Stage 11D
DCSS probabilities are saved as float64.

**Verification.** New DCSS predictions have zero threshold mismatches and pass
exact metric reproduction. The largest historical persistence differences are
0.01245 for ROC-AUC and 0.00540 for PR-AUC. The independent Stage 11D validator
passes.

**Supervisor answer.** The legacy files rounded probabilities to float32, so a
few values extremely close to the learned threshold changed sides when metrics
were reconstructed. We preserved the original class decisions, disclosed the
ranking-metric differences, and used float64 for every new run.

## P11-019: Prespecified strong DCSS condition not met

**Discovery.** F-DCSS-15 met the internal mean non-inferiority margin in four of
six dataset-model groups. ISCX passed only one of three models. Against F-Stable,
external ROC-AUC improved for PhiUSIIL-to-ISCX but not for ISCX-to-PhiUSIIL.

**Impact.** The paper cannot honestly claim that F-DCSS-15 is generally
non-inferior internally or consistently superior under bidirectional external
transfer. Doing so would overstate the evidence and invite an innovation and
selective-reporting objection.

**Planned resolution.** Keep the locked primary result unchanged. Stage 11E
will run the predeclared component and random-feature ablations, including a
mechanism analysis of the F-DCSS-20 sensitivity pattern. Stage 11G will add
cluster-aware uncertainty. The manuscript contribution will be framed as a
diagnostic boundary result unless those preplanned analyses justify a narrower
positive claim.

**Required verification.** No formula, weight, primary k, or endpoint may be
changed after Stage 11D. Any F-DCSS-20 statement must be labeled sensitivity or
exploratory evidence.

**Supervisor answer.** The method did not satisfy its prespecified general
success condition, and we report that directly. Its current value is showing
where source-only domain-conditioned stability helps and where it still fails;
the next ablations test the mechanism rather than tune toward the target labels.

## P11-020: Random-baseline artifact storage exceeds free disk

**Discovery.** Stage 11D used approximately 1.96 GB for 180 models and their
internal/external predictions. Expanding the same artifact policy to 1,800
random runs would require roughly 19.6 GB, while the C drive had about 10.0 GB
free when Stage 11E was locked.

**Impact.** Saving every random fitted object and target-level score would risk
exhausting the disk and interrupting the experiment. Omitting all run-level
evidence, however, would make the exploratory baseline difficult to audit.

**Resolution.** Formula ablations retain full models and float64 predictions.
Random runs retain deterministic global feature lists, feature hashes, tuning
rows, complete metrics, warning records, and atomic completion metadata, but
not models or per-sample predictions. A stratified sample of random runs will
be deterministically refitted during validation.

**Verification.** The Stage 11E validator must require all 1,800 random keys,
all 180 ablation keys, valid class order, exact feature hashes, and successful
sampled refits. The random family remains exploratory and is excluded from the
Stage 11G primary two-level bootstrap.

**Supervisor answer.** We preserved every result needed to reconstruct the
random distribution while avoiding approximately 18 GB of redundant fitted
objects and target scores. The primary method and all formal ablations still
retain full per-sample evidence; selected random runs are independently refit.

## P11-021: Direction and rank-dispersion terms rarely change Top-15 membership

**Discovery.** Before inspecting Stage 11E model performance, the frozen
feature lists showed that NoRankDispersion selected exactly the same Top-15 as
DCSS-Full in all 60 dataset-repetition-model keys. NoDirection was identical in
59 of 60 keys, and NoFrequency was identical in 44 of 60 keys.

**Impact.** The multiplicative terms change numerical scores but usually do not
change the decision boundary at k=15. Claims that every formula component makes
an independent predictive contribution would therefore be unsupported, and
the method may be dominated by mean importance and Top-15 frequency.

**Planned resolution.** Retain every predeclared ablation and report identical
feature sets and performance as a result rather than dropping them. Analyze
score-rank changes, the few boundary feature substitutions, and the stronger
F-DCSS-20 sensitivity pattern. Do not invent new weights after seeing target
performance.

**Required verification.** The Stage 11E feature-overlap table must reproduce
60/60, 59/60, and 44/60 identical-key counts and pair each non-identical key
with its performance change.

**Supervisor answer.** The ablation reveals that direction consistency and rank
dispersion have little influence on Top-15 membership in these data. We treat
this as a limitation of the current formulation and evidence that frequency
and mean SHAP importance drive most selections, not as four substantively
different selectors.

## P11-022: Interrupted URL-Phish snapshot download

**Discovery.** The first PowerShell web request for the versioned Mendeley ZIP
terminated with an unexpected end-of-stream error before a usable file was
created.

**Impact.** No experimental result was affected, but treating a partial or
unverified download as the source snapshot would break provenance.

**Resolution.** Retried the official version-1 ZIP endpoint with redirect
following, transport retries, and failure-on-HTTP-error enabled. Inspected all
ZIP entry paths before extraction and rejected neither absolute paths nor
parent traversal because none were present.

**Verification.** The downloaded ZIP is 3,475,005 bytes, contains one CSV, and
both ZIP and extracted CSV receive SHA-256 records before preprocessing.

**Supervisor answer.** The first network transfer failed cleanly and produced
no accepted artifact. We retried the same immutable version endpoint, validated
the archive structure, and froze cryptographic hashes before using the data.

## P11-023: URL-Phish v1 documentation and file counts disagree

**Discovery.** The article and Mendeley page state 111,660 rows with 11,660
phishing URLs, but the downloaded v1 CSV has 116,600 rows with 16,600 phishing
rows. It also contains 1,369 exact duplicate rows/URLs, despite the stated
duplicate-removal step.

**Impact.** The public artifact cannot be treated as a perfectly curated
benchmark, and favorable accuracy could be influenced by duplicate or source-
specific structure. The discrepancy is independent of our model.

**Resolution.** Preserve the raw file and discrepancy, apply a frozen
normalized-URL deduplication and conflict rule, recompute the original 35 Part 2
features from raw URL strings, and use this dataset only as an exploratory
sensitivity target. No target label may alter any fitted model or feature set.

**Required verification.** Stage 11F must report raw and post-cleaning counts by
label, exact and normalized duplicate counts, label conflicts, parsing failures,
and URL/domain overlap with both source corpora.

**Supervisor answer.** We did not silently adopt the repository's advertised
sample count. The immutable v1 file differs from its documentation, so we
recorded the mismatch, cleaned it under a result-independent rule, and reduced
the evidential status of the third-source experiment.

## P11-024: Third-source domain filtering is strongly class-dependent

**Discovery.** After normalized-URL deduplication, 22,830 of 115,037 URL-Phish
rows share a registrable domain with at least one of the two source corpora.
Removing these overlaps retains 85,791 of 98,543 benign rows but only 6,416 of
16,494 phishing rows.

**Impact.** The domain-filtered cohort has a substantially different class
prevalence and the large malicious-domain overlap confirms that this is not an
independent draw from an unrelated phishing population. Unfiltered results can
be optimistic due to source overlap, while filtered results address a smaller
and selected target population.

**Resolution.** Retain and report both frozen cohorts, designate the domain-
filtered one as primary only within the exploratory Stage 11F analysis, and
exclude the entire third-source analysis from H1/H2 and formal generalization
claims.

**Required verification.** Predictions must identify cohort membership, and
the validator must confirm that no source-overlapping domain remains in the
filtered cohort.

**Supervisor answer.** Domain filtering removed phishing rows much more often
than benign rows, so neither filtered nor unfiltered performance is a clean
population estimate. We present both as bounded sensitivity checks and avoid
claiming an independent prospective validation.

## P11-025: Literal large-corpus cluster bootstrap is computationally excessive

**Discovery.** One primary target contains roughly 175,000 registrable domains.
A literal nested bootstrap with 5,000 replicates, ten sampled outer repetitions,
and repeated full-corpus AUC calculations would require billions of domain draws
and hundreds of thousands of large AUC evaluations.

**Impact.** Running the literal algorithm on this workstation would be
disproportionately slow and could fail to complete, while quietly reducing the
replicate count would violate the locked protocol.

**Resolution.** Before computing any Stage 11G interval, preserve protocol 1.0.0
and add execution addendum 1.0.1. Use paired empirical AUC influence values,
sum them at registrable-domain level, draw joint Gaussian cluster multipliers,
and then apply the declared outer-repetition bootstrap for 5,000 replicates.

**Verification.** Point estimates must reproduce Stage 11D exactly, saved
covariance and seeds must regenerate every bootstrap value, and the report must
call the interval a linearized cluster-bootstrap approximation.

**Supervisor answer.** We retained the dependence units and 5,000 Monte Carlo
replicates but used the standard influence-function multiplier implementation
instead of materializing billions of resampled rows. The approximation and its
scope are explicit; it does not change any observed effect.

## P11-026: Mistyped temporary Stage 11G path

**Discovery.** The first patch for `stage11g_utils.py` contained an extra ASCII
fragment in the experiment-root directory name and created an empty temporary
directory tree outside the real experiment folder.

**Impact.** None on data or results. The file was never imported or executed
from the wrong location.

**Resolution.** Moved the source file into the correct Part 11 `scripts`
directory before execution and ran its unit tests there. The mistaken tree
contained no files after the move.

**Verification.** The correct utility compiles and both unit tests pass before
Stage 11G statistics run.

**Supervisor answer.** This was a pre-execution path typo caught by a direct
existence check. No experiment read from or wrote results to the wrong folder.

## P11-027: Stage 11F parallel inference exceeded available memory

**Discovery.** Six Stage 11F source/model workers were started while the two
remaining Stage 11E workers were active. Five Stage 11F workers terminated with
`MemoryError`; one XGBoost worker completed. The forest traceback occurred while
allocating per-tree probability buffers for all 115,037 target rows.

**Impact.** Several Stage 11F keys were incomplete, but no completed key was
corrupted because completion metadata is written only after metrics and the
combined prediction file are finalized.

**Resolution.** Changed inference to deterministic 20,000-row batches and set
estimator prediction parallelism to one thread. Resume from run-level completion
markers with fewer simultaneous memory-heavy workers.

**Required verification.** The final Stage 11F validator must find exactly 60
completed keys and reproduce metrics from every retained float64 prediction.

**Supervisor answer.** This was a workstation memory-pressure failure during
parallel prediction, not a model or data failure. Batched inference is
mathematically identical and completion markers prevent partial runs from being
counted.

## P11-028: Stage 11E random-forest worker repartitioned for completion time

**Discovery.** After five dataset/model workers completed, the remaining
PhiUSIIL random-forest worker still had more than 200 random runs queued in one
sequential repetition range.

**Impact.** Continuing one process would not change results but would extend
unattended wall time substantially. Abruptly duplicating the same keys could
create write races.

**Resolution.** Stopped the active worker once its run-level checkpoint design
was confirmed, then assigned non-overlapping repetition ranges to separate
workers. A final `r09` worker was added only for an as-yet-unstarted repetition.
Completed keys were skipped; an interrupted key without `complete.json` was
recomputed from its frozen seed.

**Verification.** Final counts are exactly 180 ablation and 1,800 random keys,
all keys are unique, all worker error logs are empty, and 12 sampled random
refits reproduce candidates, thresholds, and metrics.

**Supervisor answer.** We changed only scheduling, not the experiment. Run keys
and seeds are deterministic, completion markers prevent double counting, and
the independent validator found no missing or duplicate key.

## P11-029: Constant inputs in exploratory diagnostic correlations

**Discovery.** Four small-scope Spearman calculations emitted
`ConstantInputWarning` because one diagnostic variable had no variation within
that dataset/model subgroup.

**Impact.** A correlation is mathematically undefined in those cells. Replacing
it with zero or omitting the warning would overstate available mechanism
evidence.

**Resolution.** Preserve the corresponding coefficients and p values as
missing, keep the aggregate and variable subgroups that have variation, and
label all 60-key correlations descriptive because outer keys are dependent.

**Verification.** Stage 11E analysis completes, the correlation table contains
the undefined cells as missing values, and no confirmatory conclusion depends
on them.

**Supervisor answer.** Some components were literally constant in small
subgroups, which is itself consistent with the ablation redundancy finding.
We do not assign a numeric correlation where none is identifiable.

## P11-030: Positive directional result could be mistaken for overall success

**Discovery.** F-DCSS-15 had a positive, multiplicity-controlled external AUC
result for PhiUSIIL-to-ISCX, while the opposite direction was null and the
prespecified strong-contribution condition failed.

**Impact.** Reporting only the positive direction would constitute selective
emphasis and invite a justified reviewer objection about post hoc framing.

**Resolution.** Put both transfer directions in the Abstract and the same main
Results table, retain the failed overall decision, and state model-level
heterogeneity including the negative XGBoost difference.

**Verification.** The Abstract, Results 5.11, Discussion 6.3, and Conclusion all
state one-direction improvement and no general superiority.

**Supervisor answer.** The manuscript reports the complete confirmatory family.
One direction was supported, the other was not, so the contribution is
conditional rather than a successful bidirectional selector claim.

## P11-031: DCSS component novelty weakened by ablation

**Discovery.** NoRankDispersion reproduced 60 of 60 full Top-15 sets and
NoDirection reproduced 59 of 60.

**Impact.** Claiming that every multiplicative component is necessary would be
inconsistent with the observed selection boundary and would overstate method
novelty.

**Resolution.** Move the exact identity counts into the main Results and call
the locked formula over-specified at k=15. Mention a simpler
importance-plus-frequency score only as future work, without rerunning or
substituting it after target inspection.

**Verification.** Results 5.12, Discussion 6.3, the Abstract, and Conclusion
all disclose component redundancy.

**Supervisor answer.** The domain-conditioned evaluation design remains useful,
but the experiment does not support independent value for every score term.
We kept this negative ablation visible and did not redesign the method after
seeing target performance.

## P11-032: Third-source cohort is not independent prospective validation

**Discovery.** URL-Phish v1 has a documented-versus-observed row-count mismatch,
class/source coupling, and class-dependent removal during domain filtering.

**Impact.** Calling it independent validation would imply a cleaner target
population than the data support, even though all models and feature sets were
frozen before evaluation.

**Resolution.** Label the experiment third-source sensitivity throughout,
report the data discrepancies before performance, and pair relative AUC changes
with absolute AUC, FPR, and calibration failure.

**Verification.** Methods 4.2, Results 5.13, Threats to Validity, and the source
map use the same bounded evidential status.

**Supervisor answer.** The third dataset checks whether a relative comparison
survives another frozen source; it does not estimate prospective deployment
performance or remove source-construction bias.

## P11-033: Markdown-to-LaTeX conversion initially escaped display equations

**Discovery.** The first Pandoc conversion treated `\\[...\\]` display blocks as
ordinary Markdown text, producing escaped underscores and literal bracket
tokens in `main.tex`.

**Impact.** The manuscript Markdown and all results were unaffected, but the
generated LaTeX source would have displayed several equations incorrectly.

**Resolution.** Enabled Pandoc's `tex_math_single_backslash` and `raw_tex`
extensions in the deterministic build script and regenerated `main.tex`.

**Verification.** The final source contains native `\\[...\\]` blocks and the
exact `\\operatorname{DCSS}_j(k)` expression. LaTeX guard reports zero errors.

**Supervisor answer.** This was an export-format issue caught before delivery.
The corrected source preserves the locked equations; it did not alter any
method or result.

## P11-034: PaperSpine citation-candidate audit remains stricter than the final reference list

**Discovery.** The manuscript contains 26 references, but the PaperSpine
workflow requires an internal pool of three candidates per target citation and
about 80% recent candidates. The current support bank has eight grouped claim
entries rather than 60 paper-level candidates.

**Impact.** No experimental claim is unsupported, and all manuscript citations
are preserved. However, the publication-preparation artifact check cannot pass
until a broader literature-candidate and metadata verification pass is run.

**Resolution.** Preserve the audit failure rather than lowering the configured
20-reference target or duplicating references to satisfy a count. Record the
remaining literature task in the Stage 11H report.

**Verification.** Artifact check reports no missing files and only the two
candidate-pool findings: fewer than 60 candidates and fewer than 48 recent
candidates.

**Supervisor answer.** We did not manipulate the audit threshold. The current
paper draft is evidence-consistent, but its final submission package still
needs a broader recent-literature screening and publisher-level bibliography
check.

## P11-035: Unrelated research dossiers were mixed into the PaperSpine workspace

**Discovery.** A legacy writing folder and six PaperSpine research artifacts
contained material about an unrelated code-generation security topic. The
English phishing manuscript and validated experiment outputs did not contain
that argument, but the auxiliary research files could misdirect a later rewrite.

**Impact.** Leaving the files in place could cause topic contamination during
language polishing, literature positioning, or a future context recovery. It
did not alter the frozen protocol, model outputs, statistical results, or the
current manuscript claims.

**Resolution.** Deleted the complete legacy project, rebuilt the affected
research dossier, exemplar dossier, style profile, SOTA map, motivation options,
and source index for phishing URL evaluation, and changed the PaperSpine
configuration and source map to point only to the phishing workspace and Part
11 evidence.

**Verification.** A recursive keyword audit found no remaining references to
the unrelated topic. The only occurrence of “Large Language Models” is in the
official title of reference [26], which was retained to preserve bibliographic
accuracy.

**Supervisor answer.** The mixed files were auxiliary planning artifacts, not
evidence used in the paper. They were removed before polishing, their successors
were rebuilt from the phishing-specific source map, and the manuscript numbers
remain tied to the validated experiment outputs.
