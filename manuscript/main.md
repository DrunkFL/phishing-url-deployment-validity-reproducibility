# Leakage-Aware Evaluation of Prediction and Explanation Stability under Cross-Source Shift in Phishing Web Address Detection

**Short title:** Cross-Source Phishing Validity

## Abstract

A machine-learning detector for phishing URLs may look accurate and explainable while depending on entity associations and decision thresholds that fail under deployment shift. We use a connected deployment-validity audit to evaluate logistic regression, random forest, and XGBoost on conflict-cleaned ISCX-URL2016 and PhiUSIIL corpora with 35 common offline URL features. The evaluation moves from row-random splits to normalized-URL-, host-, and registrable-domain-disjoint splits, followed by zero-target-feedback transfer. Registrable-domain separation reduced mean Macro-F1 by 0.038-0.109 on ISCX-URL2016, compared with only 0.001-0.002 on PhiUSIIL. Thresholds selected under empirical source-validation false-positive-rate budgets of 0.1% or 1% did not transfer: mean external false-positive rates ranged from 0.532 to 1.000, or 5,318-10,000 false alerts per 10,000 benign URLs. A mechanism audit excluded label-orientation and probability-column errors, identified feature-label direction reversals for 19-20 of 35 features, and found that collection source remained distinguishable within benign and phishing samples (grouped AUC 1.000 and 0.913). Source-level Top-15 SHAP agreement was 0.623-0.788, despite higher agreement across model seeds. As a bounded diagnostic instance within the audit, we also evaluate Domain-Conditioned SHAP Stability (DCSS), a protocol that uses source-corpus training data only. Against random-resampling stability, the mean external ROC-AUC effect was -0.00086 in one direction (domain-plus-outer 95% interval: -0.02014 to 0.01847; 6/10 positive repetitions) and +0.01467 in the other (0.00254 to 0.02739; 10/10 positive). Entity separation, explanation repeatability, score orientation, and frozen-threshold transfer are distinct validity requirements, and no single one certifies deployable cross-source performance.

**Keywords:** phishing URL detection; data leakage; registrable domain; explainable artificial intelligence; explanation stability; domain shift; false-positive rate

**Practitioner points:**

- A low empirical false-positive rate on source validation data is a threshold-selection condition, not a population or deployment guarantee; the frozen-threshold transfer diagnostic produced a mean of 5,318-10,000 false alerts per 10,000 benign target URLs.
- URL, host, and registrable-domain overlap should be reported separately because each supports a different claim about recurring or unseen entities.
- Repeatable SHAP rankings can remain source specific, so explanation stability should be audited independently from predictive and threshold transfer.

## 1. Introduction

Deployed phishing URL classifiers make decisions at fixed thresholds rather than producing rankings for inspection. An email gateway, browser filter, or security operations workflow must classify newly observed URLs while keeping false alerts within a tolerable budget. Published evaluations, however, often use a random split of a public corpus and report accuracy, F1 score, or ROC-AUC. These metrics describe performance on the sampled test data, but they cannot establish whether the detector, threshold, or explanation will remain valid for unseen domains or a different collection source^[1-10]^.

Dependence between URLs creates the first problem. Corpora assembled from threat feeds, popular-site lists, search results, or web crawls may include repeated rows, several paths from the same host, and multiple hosts under one registrable domain. A random split can place these related entities in both the training and test sets. The leakage interpretation depends on the deployment claim: recurring domains may be expected in one setting, while shared hosts invalidate an unseen-host test and shared `eTLD+1` entities invalidate an unseen-registrable-domain test^[11-15]^. We therefore treat row, URL, host, and registrable-domain partitions as separate estimands, without assuming that they form a universal progression from invalid to valid evaluation.

Ranking performance also leaves the operating point unresolved. ROC-AUC integrates over thresholds, while a deployed detector incurs the false alerts produced by one frozen decision rule. A threshold chosen under a low empirical source-validation false-positive-rate budget may violate that budget after entity composition or data source changes; its realized errors diagnose frozen-threshold transfer failure rather than provide a population-level low-FPR guarantee. Recalibrating the threshold with target labels would instead answer a target-supervised question. Threshold portability therefore requires a separate zero-target-feedback test.

Explanations pose a related problem. SHAP can identify influential features for one fitted model, but a single importance plot does not show whether the ranking survives model reseeding, grouped repartitioning, a new deployment regime, or a different data source^[16-20]^. Recent studies have examined cross-dataset degradation, cross-source feature behavior, and explanation stability^[7-10,21-25]^. Ahamed et al.^[10]^, for example, combined normalization, host-disjoint testing, external validation, and explanation analysis but still found residual `eTLD+1` overlap. The remaining question is more specific than "using SHAP for phishing detection": do entity independence, explanation repeatability, and threshold portability provide mutually consistent evidence of deployability?

We examine that question through a connected set of tests. The protocol audits overlap at the normalized-URL, host, and registrable-domain levels and compares four internal deployment estimands. Models and thresholds are then frozen using source data alone before bidirectional transfer is evaluated at empirical source-validation FPR budgets of 0.1% and 1%. We test whether below-chance target ranking reflects an implementation error or a source-conditioned association shift, while separating the seed, partition, regime, and source components of SHAP stability. We then assess whether Domain-Conditioned SHAP Stability (DCSS) provides a bounded diagnostic signal when explanations are aggregated over held-out registrable domains. Each claim is limited to the population and perturbation under study; the design neither treats the most restrictive split as universally realistic nor assumes that DCSS will outperform simpler selectors.

This study addresses three research questions:

- **RQ1:** How do entity separation and source shift change predictive performance, calibration, and the transfer of thresholds selected under low empirical source-validation FPR budgets?
- **RQ2:** How stable are SHAP feature rankings under controlled changes in model seed, group partition, deployment regime, and evaluation source?
- **RQ3:** Within this audit, does aggregating SHAP evidence over held-out registrable domains identify compact feature sets that are less sensitive to training-domain composition than conventional random-resampling screening, and where does that signal fail?

**Table 1. Research questions, evaluation scopes, prespecified comparisons, and result locations**

| Question | Split or selection scope | Primary metric or diagnostic | Prespecified comparison | Result location |
|---|---|---|---|---|
| RQ1 | S0-S3 internal partitions and bidirectional S4 transfer; source-only threshold selection | Macro-F1, ROC-AUC, FPR, calibration, source discrimination | Entity regimes; frozen source threshold versus realized source-test and target errors | Sections 5.1 and 5.3 |
| RQ2 | Fixed SHAP cohorts under seed, partition, regime, and source changes | Top-15 Jaccard, rank correlation, SHAP/label-direction conflict | Controlled perturbation pairs | Sections 5.2 and 5.3 |
| RQ3 | Outer-training-only held-domain folds; F-DCSS-15 fixed before target inspection | S3 Macro-F1 and external ROC-AUC difference | F-DCSS-15 versus F-All internally and F-Stable externally | Section 5.4 |

*Note.* RQ, research question; S0, row-random split; S1, normalized-uniform-resource-locator-disjoint split; S2, host-disjoint split; S3, registrable-domain-disjoint split; S4, zero-target-feedback cross-source transfer; Macro-F1, macro-averaged F1 score; ROC-AUC, area under the receiver operating characteristic curve; FPR, false-positive rate; SHAP, SHapley Additive exPlanations; DCSS, Domain-Conditioned SHAP Stability; F-DCSS-15, the 15-feature DCSS subset; F-All, the full 35-feature set; F-Stable, the random-resampling stability subset.

The main contributions are as follows:

1. We provide a deployment-validity audit that treats entity alignment, score orientation, explanation repeatability, and frozen-threshold portability as separate claims while keeping fitting, selection, and threshold choice within the declared source scope.
2. Applying this audit shows that entity-separation effects are corpus dependent, empirical 0.1% and 1% source-validation thresholds fail as transferable decision rules, and below-chance external AUC reflects source-conditioned ranking reversal rather than a label-orientation error.
3. Within the audit, we decompose SHAP agreement into seed, partition, regime, and source estimands and evaluate a prespecified four-factor DCSS construction as a bounded diagnostic instance. DCSS produces a positive mean effect in one transfer direction but not the other; at the Top-15 boundary, normalized importance and held-domain selection frequency dominate its locked score.

## 2. Related Work

### 2.1 Machine Learning for Phishing URL Detection

Phishing website detection commonly uses list matching, page-similarity analysis, or machine-learning-based classification^[1]^. List-based methods have low inference costs but cannot promptly cover newly registered domains and short-lived attacks. Page-similarity methods detect impersonation from visual appearance, HTML structure, or brand information. Although page-similarity methods use rich evidence, they require active page access and can be disrupted by unavailable pages, network delays, and content obfuscation. Machine-learning methods learn classification rules from URLs, HTML content, domain registration information, and network attributes.

Mamun et al.^[2]^ introduced the ISCX-URL2016 dataset and showed that low-cost lexical analysis can distinguish benign, phishing, spam, malware, and defacement URLs. Their results indicate that the URL string itself contains useful detection signals. URLNet, proposed by Le et al.^[5]^, learns URL representations with character-level and word-level convolutional networks and reduces manual feature engineering. Opara et al.^[4]^ developed WebPhish, which jointly processes raw URLs and HTML content to extract phishing indicators from complementary information sources. Prasad and Chandra^[3]^ created the PhiUSIIL dataset and combined URL similarity, HTML features, and incremental learning to detect attacks involving homographs and combosquatting. Mahesh et al.^[6]^ combined recursive feature elimination with cross-validation, XGBoost, SHAP, and LIME for malicious URL detection, linking feature selection with model explanation.

These studies have improved predictive modeling for phishing URL detection. Many evaluations, however, still rely mainly on random within-dataset splits and use accuracy or F1 score as their primary evidence. Our classifiers serve as representative models for examining data partitioning and explanation stability, rather than as candidates for increasing architectural complexity.

### 2.2 Dataset Bias and Cross-Dataset Generalization

Phishing URL datasets vary in collection source, time span, class balance, and feature definition. In cross-dataset experiments on three URL datasets, Rashid et al.^[7]^ found that models with strong within-dataset performance degraded markedly on a different source. Their unsupervised domain-adaptation analysis, which included distributional differences in URL path length, showed that data collection and representation contribute to cross-dataset degradation alongside classifier capacity. Yi et al.^[8]^ evaluated random forest, XGBoost, and LightGBM across several URL datasets and argued for multi-dataset validation. Mia et al.^[9]^ compared shared features in two phishing datasets and found source-dependent contributions: features with the same name and semantics could still have different distributions and SHAP rankings.

Ahamed et al.^[10]^ proposed an integrated evaluation protocol covering URL normalization, host-disjoint splitting, cross-dataset generalization, adversarial perturbation, and explanation stability. Their audit nevertheless found 1,387 shared `eTLD+1` entities between the training and test data after host separation, plus further overlap with the validation set. Different hostnames therefore need not represent unseen administrative-domain contexts: `a.example.com` and `b.example.com`, for instance, share the registrable domain `example.com`. This residual dependence motivates the comparison in the present study. We compare host-disjoint and registrable-domain-disjoint estimands on the same corpus without claiming that cross-dataset testing or explanation-stability analysis is new.

PhreshPhish introduced a newly collected large-scale corpus and benchmark variants designed to reduce leakage, vary task difficulty, and represent realistic phishing base rates^[11]^. Its contribution centers on dataset and benchmark construction. Using established public corpora, our study asks a complementary question by separating entity-conditioned evaluation, explanation repeatability, and frozen-threshold portability in one source-only protocol; neither corpus is presented as a new benchmark.

### 2.3 Data Leakage in Security Machine Learning

Kaufman et al.^[12]^ defined data leakage as using target-related information during training that would be unavailable at prediction time. They described how leakage arises and how researchers can detect and prevent it. Kapoor and Narayanan^[13]^ argued that leakage may enter during data collection, preprocessing, modeling, or evaluation, inflating performance and reducing reproducibility in machine-learning-based science.

Dependence and evaluation bias are especially consequential in security machine learning. The TESSERACT framework of Pendlebury et al.^[15]^ distinguishes spatial from temporal bias in malware classification and shows how partitions that do not match deployment conditions can substantially overestimate model capability. Arp et al.^[14]^ analyzed recurring experimental pitfalls in security machine learning and showed that dataset construction, feature design, train-test splitting, and evaluation metrics can pull conclusions away from the underlying security problem. Although neither study focuses specifically on phishing URLs, both support the same requirement: training and test data must be independent in a way that matches the task, and preprocessing and feature selection must use training data alone.

We use the conventional definition of preprocessing leakage: information from validation or test data affects fitting, tuning, threshold selection, or feature selection. Entity overlap requires a claim-specific interpretation. We call it related-sample leakage only when the overlap conflicts with the stated deployment target, as shared hosts would in an unseen-host evaluation. Cross-dataset distribution shift is not itself leakage; we evaluate it separately as evidence of dependence on source-specific distributions and collection procedures.

### 2.4 SHAP-Guided Selection and Explanation Stability

Lundberg and Lee^[16]^ introduced SHAP as a Shapley-value framework for attributing model predictions, and Lundberg et al.^[17]^ developed efficient explanations for tree models. Phishing studies already use SHAP to guide feature selection. Mahesh et al.^[6]^ combined RFECV, XGBoost, SHAP, and LIME for malicious URL detection. Shafin et al.^[22]^ proposed an explainable feature-selection framework for web phishing detection, while Kehkashan et al.^[25]^ applied SHAP-based selection to several supervised classifiers. These methods demonstrate explanation-guided reduction, although their main selection evidence comes from one dataset or one fitted ranking. SHAP feature selection itself is therefore not our novelty claim. We ask whether a subset repeatedly selected from training-only runs remains useful when the model seed, grouped partition, deployment regime, or data source changes.

Producing an explanation does not make it trustworthy. Alvarez-Melis and Jaakkola^[18]^ treated robustness as a desirable property of interpretation methods, and Ghorbani et al.^[19]^ showed that small input changes can alter explanations substantially without changing the predicted label. For security applications, Warnecke et al.^[20]^ argued that assessments should also consider completeness, efficiency, and robustness. Tolay^[21]^ studied SHAP cost, local stability, and selective explanation in intrusion detection, while Calzarossa et al.^[23,24]^ included explanation robustness in a broader assessment. Stability establishes neither faithfulness nor causality: a ranking can reproduce source bias, and correlated features can exchange ranks without a meaningful change in model behavior. We therefore use separate stability estimands, fixed explainer settings, uncertainty intervals, and external-source diagnostics.

Existing work has moved beyond within-dataset accuracy toward deployment-aligned partitions, external-source transfer, and explicit assessment of explanations. Our question is more specific than whether SHAP can explain a classifier. We examine performance under registrable-domain separation, the response of feature rankings to different perturbations, and the evidence added by holding out registrable domains during SHAP aggregation compared with repeated random-resampling stability. Negative and directional results mark the conditions under which explanation stability stops supporting a transfer claim.

**Table 2. Comparison with the closest prior studies**

| Study | Entity-aware split | Cross-source test | Frozen threshold audit | Explanation analysis | Distinctive scope |
|---|---|---|---|---|---|
| Rashid et al.^[7]^ and Yi et al.^[8]^ | Not the central contribution | Yes | No linked low-FPR transfer diagnosis | Limited or not central | Cross-dataset degradation and adaptation/multi-dataset evaluation |
| Ahamed et al.^[10]^ | Host-disjoint, with residual registrable-domain overlap audited | Yes | No linked empirical-budget transfer audit | Yes | Integrated normalization, external validation, perturbation, and explanation protocol |
| PhreshPhish^[11]^ | Leakage-reducing benchmark variants | Benchmark-centered | Realistic base-rate motivation | Not the central contribution | New corpus and benchmark-construction pipeline |
| This study | URL-, host-, and registrable-domain-disjoint estimands | Bidirectional, zero target feedback | Yes; interpreted only as frozen-threshold transfer failure | Seed, partition, regime, source, and held-domain diagnostics | Linked separation of entity, decision, ranking, and explanation validity on established corpora |

*Note.* URL, uniform resource locator; FPR, false-positive rate; eTLD+1, effective top-level domain plus one label (the registrable domain); SHAP, SHapley Additive exPlanations.

## 3. Problem Formulation

### 3.1 Phishing URL Detection Task

Let the dataset be

\[
\mathcal{D}=\{(u_i,y_i)\}_{i=1}^{N},
\]

where \(u_i\) is the \(i\)-th raw URL and \(y_i\in\{0,1\}\) is its label, with 0 denoting a benign URL and 1 denoting a phishing URL. A feature extraction function \(F(\cdot)\) maps each URL to a \(d\)-dimensional vector:

\[
x_i=F(u_i)\in\mathbb{R}^{d}.
\]

A classifier \(f_{\theta}\) outputs a phishing probability from the feature vector:

\[
\hat{p}_i=f_{\theta}(x_i), \qquad \hat{y}_i=\mathbb{I}(\hat{p}_i\geq t),
\]

where \(t\) is the classification threshold. This study does not propose a new classifier. Instead, it investigates model performance and explanation behavior under different leakage-control conditions.

### 3.2 Deployment-Conditioned Entity Associations

We define a URL normalization function \(G(\cdot)\) that standardizes scheme and host case, default ports, surrounding whitespace, and internationalized domain-name representations while preserving path, query, and fragment content that may carry phishing-related information. For each normalized URL, three mappings are defined:

- \(U(u)=G(u)\): the complete normalized URL;
- \(H(u)\): the complete hostname parsed from the URL;
- \(R(u)\): the registrable domain, or `eTLD+1`, extracted using a frozen snapshot of the Public Suffix List^[26]^.

For example, the hostname of `https://login.a.example.co.uk/reset?id=1` is `login.a.example.co.uk`, whereas its registrable domain is `example.co.uk`. Two URLs may therefore have different hostnames while sharing an administrative-domain association. They are not assumed to share the same physical infrastructure.

For a training set \(\mathcal{D}_{tr}\) and a test set \(\mathcal{D}_{te}\), the overlap sets at the complete-URL, hostname, and registrable-domain levels are defined as

\[
O_U=U(\mathcal{D}_{tr})\cap U(\mathcal{D}_{te}),
\]

\[
O_H=H(\mathcal{D}_{tr})\cap H(\mathcal{D}_{te}),
\]

and

\[
O_R=R(\mathcal{D}_{tr})\cap R(\mathcal{D}_{te}).
\]

The corresponding overlap rate is calculated using the number of unique entities in the test set as the denominator. A URL-, host-, or registrable-domain-disjoint split requires the relevant overlap set to be empty. The implementation freezes the Public Suffix List cache with network updates disabled and applies deterministic handling to IP literals, single-label hosts, malformed URLs, and suffix-only hosts.

These overlap conditions estimate different deployment regimes. Row-random and URL-grouped evaluation approximate settings in which known domains may recur. Host grouping estimates performance on unseen hostnames, while `eTLD+1` grouping estimates performance on unseen registrable-domain contexts. Consequently, a stricter grouping level is not asserted to be universally more realistic; it is appropriate only when it matches the intended claim.

### 3.3 SHAP Explanation Stability Estimands

For feature \(j\) in repeated experiment \(r\), global SHAP importance is defined as

\[
I_j^{(r)}=\frac{1}{n}\sum_{i=1}^{n}|\phi_{ij}^{(r)}|,
\]

where \(\phi_{ij}^{(r)}\) is the SHAP value of feature \(j\) for sample \(i\). Ranking features by \(I_j^{(r)}\) produces the Top-k feature set \(T_k^{(r)}\). The stability of two Top-k sets is measured using the Jaccard coefficient:

\[
J(T_k^{(a)},T_k^{(b)})=\frac{|T_k^{(a)}\cap T_k^{(b)}|}{|T_k^{(a)}\cup T_k^{(b)}|}.
\]

Agreement between complete feature rankings is measured using Spearman's rank correlation coefficient. Four estimands are reported separately: (1) seed stability, which varies the model seed while holding the split and explanation cohort fixed; (2) partition stability, which varies the group split and uses equal-size, class-stratified explanation cohorts; (3) regime stability, which compares rankings across S0-S3 while preserving the declared cohort protocol; and (4) source stability, which compares a frozen source-trained model's ranking on fixed source and external-target cohorts. Partition and regime stability necessarily include some sample-composition variation and are interpreted accordingly.

For feature \(j\) in run \(r\), contribution direction is represented by

\[
d_j^{(r)}=\operatorname{sign}\left(\rho_s(x_j,\phi_j^{(r)})\right),
\]

where \(\rho_s\) is Spearman's correlation between observed feature values and their SHAP values in the controlled explanation cohort. This measure is preferred to the mean signed SHAP value, which may cancel heterogeneous local effects. Dependence plots are inspected when the correlation is weak or non-monotonic. The explained class, model-output scale, background data, SHAP version, explainer type, and feature-perturbation setting are fixed and reported. Raw SHAP magnitudes are not compared across different model families.

### 3.4 Definition of Stable Features

For \(m\) repeated experiments, the frequency with which feature \(j\) appears in the Top-k set is

\[
q_j=\frac{1}{m}\sum_{r=1}^{m}\mathbb{I}(j\in T_k^{(r)}).
\]

Let \(c_j\) denote the proportion of eligible runs in which \(d_j^{(r)}\) agrees with the modal nonzero direction. Runs in which a feature is constant or the correlation is undefined are excluded from this direction calculation and counted separately. The prespecified rule includes feature \(j\) in the stable feature set when \(q_j\geq0.8\) and \(c_j\geq0.8\). Features with consistently weak or non-monotonic value-contribution relationships may satisfy the frequency condition without satisfying the direction condition. Sensitivity analyses compare several values of \(k\), selection-frequency thresholds, and a frequency-only variant. Pairwise Jaccard and Spearman values share runs and are therefore not treated as independent observations.

## 4. Methodology

### 4.1 Overall Framework

The deployment-validity audit follows four evidence stages aligned with the three research questions. The first stage establishes protocol integrity: data acquisition, label harmonization, conservative URL normalization, deterministic feature extraction, and explicit URL, host, and registrable-domain mappings precede four deployment-conditioned partitions. The second stage tests operational transfer by freezing source-trained models and thresholds before bidirectional external evaluation, with deduplication, entity-equal weighting, calibration, threshold-transfer failure, score-orientation checks, and false-positive structure used to test alternative explanations. The third stage separates seed, partition, regime, and source components of SHAP stability and compares compact training-only feature sets. The fourth stage evaluates DCSS as a bounded diagnostic instance by holding out registrable-domain folds within each outer training set, followed by prespecified non-inferiority and transfer comparisons, component ablations, random same-size subsets, and a third-source sensitivity analysis.

All fitting, tuning, threshold selection, SHAP aggregation, and feature selection remain within the declared source training and validation scope. The framework is designed to test a sequence of claims rather than to maximize within-dataset accuracy: entity-conditioned performance establishes what is being estimated, frozen transfer tests whether that evidence survives a new source, explanation analysis asks whether model reasoning is equally portable, and DCSS probes whether held-domain explanations add information beyond random-resampling stability. Unfavorable results are retained rather than used to redefine the endpoint or scoring rule after target inspection.

### 4.2 Datasets

The primary experiments use the UCI release of PhiUSIIL and a third-party preserved copy corresponding to ISCX-URL2016. Throughout the manuscript, `ISCX-URL2016` denotes this preserved copy unless the official dataset description is explicitly discussed. PhiUSIIL contains 235,795 observations and 54 provided features, including 134,850 legitimate and 100,945 phishing observations^[3]^. Its original label convention is 1 for legitimate and 0 for phishing; this study explicitly recodes it to 0 for benign and 1 for phishing. The official ISCX-URL2016 description lists benign, phishing, spam, malware, and defacement categories^[2]^. For the binary task, the preserved copy's benign URLs are coded 0 and phishing URLs are coded 1, while the other malicious categories are excluded.

Because the original feature sets differ in number and definition, provided features are not concatenated or aligned by name. The same lexical and structural variables are recalculated from raw URLs in both datasets. Dataset provenance is treated as a potential confounder: ISCX benign URLs are largely derived from Alexa and its phishing URLs from OpenPhish, while PhiUSIIL uses a different collection pipeline^[2,3]^. External testing is therefore described as external-source transfer, not as pure real-world generalization. S4 estimates source-conditioned incompatibility between the evaluated benchmark construction pipelines; it does not estimate the magnitude or frequency of drift in arbitrary live deployments.

URL-Phish v1 is used only as an additional sensitivity source^[27]^. Its immutable archive is identified by DOI `10.17632/65z9twcx3r.1` and SHA-256 hashes. Although its article and repository describe 111,660 rows, the downloaded v1 CSV contained 116,600 rows, including 16,600 phishing URLs. The discrepancy and the coupling of class with collection source prevent this cohort from serving as independent prospective validation. We therefore use it only to test whether the direction of the DCSS-versus-F-Stable contrast survives another frozen target. No active URL is visited, and the same 35 features are recomputed from stored URL strings rather than using its provided 22 features.

### 4.3 URL Normalization and Data Auditing

URL normalization follows the principle of merging only forms that are unambiguously equivalent. The procedure removes leading and trailing whitespace, lowercases schemes and hostnames, normalizes default ports, converts internationalized domain names into a comparable representation, and records missing hosts and parsing failures. Paths and query parameters are neither arbitrarily deleted nor reordered, because doing so may incorrectly merge semantically different addresses.

The data audit records the following information:

1. the original sample count and class distribution;
2. the number of missing URLs and normalization failures;
3. exact duplicate URLs with identical labels;
4. identical URLs associated with conflicting labels;
5. the number of unique hostnames and registrable domains;
6. URL, hostname, and registrable-domain overlap rates across data partitions; and
7. the number of identical feature vectors produced by unified feature extraction.

Samples that share a normalized URL but have conflicting labels are removed from the master corpus rather than assigned either label. Same-label duplicate rows are retained in that corpus so that S0-row and S1-URL use the same observations and differ only in whether equal URLs may cross the partition boundary. A separate fully deduplicated sensitivity analysis retains one row per normalized URL. This design prevents a change in sample count from being mistaken for an effect of grouping. No test information is used to choose cleaning, parsing, or feature-extraction rules.

### 4.4 Unified URL Features

The frozen core set contains 35 deterministic URL features that do not require active page access, live infrastructure queries, or target-dataset statistics. The ordered name, data type, and executable definition of every feature are provided in Supplementary Data S1 (`supplementary_feature_dictionary.csv`). They are divided into five groups:

1. **Length features:** total URL length, hostname length, path length, query-string length, and fragment length;
2. **Character-composition features:** counts and proportions of digits, letters, and special characters, together with counts of dots, hyphens, underscores, slashes, question marks, equals signs, percent signs, and `@` characters;
3. **Structural features:** subdomain depth, number of path segments, number of query parameters, port presence, and user-information presence;
4. **Domain features:** direct IP-address usage, top-level-domain length, registrable-domain length, and the proportions of digits and hyphens in the domain; and
5. **String-statistical features:** URL character entropy, longest repeated-character run, character-type transition counts, and token-length summaries.

WHOIS records, live DNS responses, page HTML, certificate status, suspicious-term lexicons, and brand-domain mismatch indicators are excluded. These restrictions keep feature computation offline and reproducible and reduce the risk of turning target-source vocabulary into a hidden source identifier.

### 4.5 Data Partitioning Strategies

To answer RQ1, four internal deployment regimes and one transfer setting are defined:

- **S0-row - row-random split:** rows from the conflict-cleaned master corpus are randomly partitioned, so same-label duplicate URLs and related entities may cross subsets;
- **S1-URL - normalized-URL-disjoint split:** the same master corpus is grouped by normalized URL, so duplicate rows remain but every normalized URL belongs to only one subset;
- **S2 - host-disjoint split:** complete hostnames are used as groups, and all URLs belonging to one host are assigned to only one subset;
- **S3 - registrable-domain-disjoint split:** `eTLD+1` is the grouping unit, and all subdomains and paths under one registrable domain are assigned to one subset; and
- **S4 - external-source transfer:** a model and threshold fixed using one source are evaluated on the other source after cross-source normalized-URL, host, and `eTLD+1` overlap has been audited.

S0-S3 use training, validation, and test proportions of approximately 70%, 15%, and 15%. S0 uses two-stage stratified row sampling. S1-S3 use a deterministic weighted greedy group-stratification procedure: complete groups are seed-shuffled, processed from largest to smallest, and assigned to the split minimizing the incremental squared relative error in benign rows, phishing rows, and total rows. Achieved row and entity proportions are reported, and no repetition is discarded for unfavorable balance. Scaling, resampling, hyperparameter tuning, threshold selection, and feature selection are fitted using training data, with validation data used only for prespecified model selection. Internal test labels remain unavailable until the pipeline is frozen. For S4, target labels are hidden from tuning and threshold selection. Any target `eTLD+1` also observed in source training or validation data is removed in the primary transfer analysis, with overlap counts and results before and after this removal reported separately.

### 4.6 Classification Models

Three models are used: logistic regression, random forest, and XGBoost. Logistic regression provides a simple linear baseline; random forest represents a bootstrap-aggregation ensemble; and XGBoost represents gradient-boosted decision trees. These models cover different levels of complexity and learning mechanisms and are supported by mature SHAP explainers. CNN, LSTM, and Transformer models are not included, thereby reducing the possibility that architectural differences obscure the effects of leakage and explanation stability.

For each model, one of three prespecified candidate configurations is selected by validation Macro-F1, with validation ROC-AUC and candidate order used as deterministic tie breakers. Logistic regression uses `StandardScaler`, the `lbfgs` solver, and a maximum of 2,000 iterations; tree candidates fix their estimator counts, depth/leaf controls, subsampling, and class weighting. All nine exact candidate definitions and fixed parameters are provided in Supplementary Data S2 (`supplementary_model_candidates.json`). The decision threshold is then selected on the validation set from 0.05 to 0.95 in increments of 0.01. Large-scale AutoML is not used. Ten prespecified partition and model seeds are evaluated for every internal regime. Data indices, entity assignments, selected hyperparameters, thresholds, predicted probabilities, metrics, and SHAP summaries are retained for reproducibility.

### 4.7 Evaluation of SHAP Stability

LinearExplainer is used for logistic regression, whereas TreeExplainer is used for random forest and XGBoost. Two explanation pipelines are kept separate. The *selection pipeline* computes SHAP values only on training-fold or training-validation observations and produces F-Single and F-Stable before final testing. The *evaluation pipeline* computes post hoc explanations on the frozen internal or external test cohort only after model, threshold, and feature set choices are complete. Test explanations never feed back into feature selection.

For seed stability, the partition, background data, and class-stratified explanation cohort are fixed while only the model seed changes. For partition stability, group assignments change and equal-size stratified cohorts are sampled using a prespecified rule. We define *source stability* as the ranking agreement obtained when the same frozen source-trained configuration explains fixed, class-stratified cohorts from its source and from the external corpus. The following quantities are calculated for each model and estimand:

1. mean Jaccard similarity of the Top-10, Top-15, and Top-20 feature sets within each stability estimand;
2. Spearman correlation between complete feature rankings within each stability estimand;
3. the count and frequency with which each feature enters a Top-k set;
4. consistency of the feature-value/SHAP-value correlation direction; and
5. the time required to explain 100, 500, 1,000, and larger numbers of samples.

The explained class, model-output scale, background sample, explainer perturbation mode, and SHAP software version are fixed within each model family. The exact 200-row class-balanced backgrounds/cohorts, deterministic SHA-256 sampling rule, output scales, and perturbation modes are specified in Supplementary Protocol S3 (`supplementary_shap_protocol.md`). Rankings, rather than raw SHAP magnitudes, are used for cross-model comparisons. A feature that is important only under S0-row but loses importance under S3 or external transfer is treated as a candidate source- or entity-associated feature, not as proof of memorization. Correlated and substitutable features are examined before interpreting rank turnover as substantive instability.

### 4.8 Stability-Guided Feature Selection

Stability-guided feature selection consists of three steps. Within each outer split, Top-k selection frequency and direction consistency are first calculated from repeated seeds and inner resamples of that outer training partition. The stable feature set is then constructed using thresholds declared before final testing, after which the classifier is retrained and evaluated on the untouched outer test set. F-Single and F-Stable are constructed separately for each model family; SHAP magnitudes are not pooled across logistic regression, random forest, and XGBoost. Five configurations are compared:

- **F-All:** all unified URL features;
- **F-Single:** the Top-k SHAP features from one prespecified training run and seed;
- **F-Stable:** features satisfying the stability criteria across repeated training runs;
- **F-MI:** the Top-15 mutual-information features selected from the inner training scope; and
- **F-Permutation:** the Top-15 model-specific permutation-importance features selected from the same inner scope.

The configurations are compared in terms of Macro-F1, recall, false-positive rate, ROC-AUC, PR-AUC, training time, per-sample inference time, and SHAP computation time. In S4, F-Single and F-Stable are built using only the source corpus; target SHAP values are diagnostic and cannot alter either set. The procedure is presented as a reproducible screening rule, not as a new optimization algorithm.

### 4.9 Domain-Conditioned SHAP Stability

DCSS is calculated separately for each dataset, outer S3 repetition, and model using only the outer training subset. Intuitively, it rewards a feature that remains important, repeatedly enters the Top-k set, preserves the direction of its value-SHAP association, and avoids large rank swings when entire registrable domains are withheld from fitting. The multiplicative construction is a prespecified diagnostic, not a claim that these four factors form an optimal objective.

The source validation subset may select the already declared model candidate and threshold but is never explained or used in the DCSS score. Registrable domains in the outer training subset are assigned to five deterministic `StratifiedGroupKFold` folds. For held-domain fold \(h\), the model is fitted on the other four folds, a class-balanced 200-row background is sampled from those fitting folds, and a class-balanced 200-row cohort from fold \(h\) is explained. Sampling is row-balanced within class rather than domain-equal; if fewer than 200 eligible rows exist, the complete eligible fold is used and logged. Five folds and 200-row cohorts were locked as a computationally feasible design before target inspection; their separate sensitivity was not estimated and remains a limitation.

Let \(H=5\), \(p=35\), and \(I_{jh}\) denote the mean absolute SHAP value of feature \(j\) on held-domain fold \(h\). Within-fold normalized importance is

\[
N_{jh}=\frac{I_{jh}}{\sum_{l=1}^{p}I_{lh}}.
\]

Let \(r_{jh}\) be the descending rank of \(N_{jh}\), with bytewise feature-name tie breaking, and let \(d_{jh}\) be the sign of the Spearman correlation between the observed feature and its SHAP value. Non-finite or zero correlations are assigned direction zero and remain in the denominator. For selection size \(k\in\{10,15,20\}\), the components are

\[
\bar{N}_j=\frac{1}{H}\sum_h N_{jh}, \qquad
q_j(k)=\frac{1}{H}\sum_h \mathbb{I}(r_{jh}\leq k),
\]

\[
c_j=\frac{\max\{\#(d_{jh}=+1),\#(d_{jh}=-1)\}}{H}, \qquad
v_j=\frac{\operatorname{median}_h|r_{jh}-\operatorname{median}_h(r_{jh})|}{p-1}.
\]

The locked score is

\[
\operatorname{DCSS}_j(k)=\bar{N}_j q_j(k)c_j(1-v_j).
\]

F-DCSS-\(k\) contains exactly the \(k\) highest-scoring features. Ties are resolved by larger mean importance, smaller mean rank, and ascending feature name. F-DCSS-15 is the prespecified primary size; F-DCSS-10 and F-DCSS-20 are sensitivity analyses. Four mechanism controls are evaluated without changing the locked method: NoDirection sets \(c_j=1\), NoRankDispersion sets \(1-v_j=1\), NoFrequency sets \(q_j(k)=1\), and F-Random-15 uses 30 globally fixed random 15-feature sets without selecting the best seed.

### 4.10 Statistical Analysis

The outer repetition is the primary uncertainty unit. The ten repetitions are paired partition perturbations of each fixed corpus, not ten independently collected datasets. Mean effects, intervals, all repetition-level paired differences, and directional consistency are therefore the primary evidence; they quantify sensitivity to the evaluated partitions of fixed corpora rather than population uncertainty over independently collected future sources. The prespecified internal comparison contrasts F-DCSS-15 with F-All using S3 Macro-F1 and an absolute non-inferiority margin of 0.01. This margin encodes the study-design tolerance that a 15-feature representation may lose at most one Macro-F1 percentage point relative to all 35 features; it is not an established clinical or operational standard. For each dataset-model group, an exact implementation of a one-sample, one-sided t-test is applied to paired repetition-level differences relative to the -0.01 margin, with Benjamini-Hochberg adjustment across six tests. The prespecified external comparison contrasts F-DCSS-15 with F-Stable using ROC-AUC averaged over the three model families within each repetition, with the same one-sample, one-sided formulation relative to zero and Benjamini-Hochberg adjustment across two transfer directions. These p-values are auxiliary to the effects and intervals. The multiplicity adjustment controls the reported family of tests but does not remove dependence created by overlapping repetitions.

For external uncertainty, 5,000 replicates combine paired empirical AUC influence values aggregated by target registrable domain, joint Gaussian domain multipliers, and outer-repetition resampling. The resulting *domain-plus-outer interval* accounts for clustering of target URLs within registrable domains and variation across the evaluated outer partitions. It is a linearized two-level cluster-bootstrap approximation fixed in execution addendum 1.0.1 before interval calculation, not literal nested resampling or an interval for future collection processes. An outer-only bootstrap and a dependence-stress analysis are retained as sensitivities. The latter inflates standard errors under assumed positive pairwise correlations among repetitions; it does not estimate the true dependence or convert repeated partitions into independent datasets. Pairwise SHAP similarities and source-only diagnostic correlations share fitted runs and are treated descriptively.

One robustness evaluation changes the weighting unit. In the registrable-domain-equal analysis, every domain contributes total weight one, and each URL in domain \(g\) receives weight \(w_i=1/n_g\). Weighted confusion components yield domain-equal Macro-F1, balanced accuracy, recall, and false-positive rate. A separate focused 2,000-replicate domain-cluster bootstrap is reported for the prespecified XGBoost r00 F-All/F-Stable comparison in both transfer directions; it does not represent all model-selection uncertainty.

External calibration is measured using Brier score, log loss, and expected calibration error (ECE) with 15 fixed equal-width probability bins. The frozen source-validation threshold is the only threshold eligible for zero-target-feedback evaluation, but it is not presumed deployable. An exact target-oracle Macro-F1 threshold is calculated after prediction solely to quantify threshold-transfer regret. Because this oracle uses target labels, it is diagnostic and is never used to refit a model or support an operational performance claim.

### 4.11 Frozen-Threshold Transfer-Failure Diagnostic

To diagnose whether an apparently low source-validation FPR survives an entity or source change, we add two frozen operating points for F-All in the S3 setting. For each dataset, outer repetition, and model, the already selected model candidate is refitted on the original source-training partition. Using only source-validation labels and probabilities, we choose the threshold that maximizes validation true-positive rate subject to an empirical FPR no greater than 0.1% or 1%. Ties are resolved in favor of the lower threshold among operating points with the same true-positive rate. These low-FPR operating points are selected from the complete ROC threshold sequence derived from the distinct source-validation scores (`drop_intermediate=False`) and are independent of the 0.05-0.95 grid used for the general Macro-F1 operating point in Section 4.6. The selected threshold is then applied unchanged to the source S3 test partition and the primary domain-filtered external target cohort. These budgets define empirical threshold-selection rules on finite cohorts; they are not population-level low-FPR guarantees.

Target labels are used only to report true- and false-positive rates after prediction; they do not affect model selection, threshold selection, calibration, or feature selection. For each source-validation cohort, the audit retains integer false positives and one-sided 95% Wilson and exact Clopper-Pearson upper bounds. It also applies the immediately safer and more permissive adjacent score thresholds and repeats threshold selection on a registrable-domain-grouped audit holdout. This grouped holdout isolates threshold-selection reuse, although the model candidate was still selected using the full validation cohort. In addition to rates, we report false alerts per 10,000 benign URLs as \(10{,}000\times\mathrm{FPR}\). These corpus-normalized counts do not estimate production prevalence. The deterministic replay reproduced the original 120 thresholds to a maximum absolute difference of \(2.22\times10^{-16}\).

## 5. Results

### 5.1 Protocol Integrity and Entity-Conditioned Performance

The first part of RQ1 establishes that the performance comparison changes the deployment population while keeping preprocessing and audit rules fixed. It first records the reproducibility environment, then verifies entity separation, and finally tests whether stricter entity independence changes internal performance.

**Reproducibility environment.**

- Python 3.9.13, NumPy 1.26.4, pandas 2.2.3, scikit-learn 1.5.2, XGBoost 2.1.4, SHAP 0.46.0, and tldextract 5.1.3;
- Windows 10 Pro (64 bit), AMD Ryzen 5 5600H (6 cores/12 logical processors), and 16 GB RAM;
- ten partition seeds beginning at 20260902 and ten model seeds from 20261001 to 20261010; and
- a frozen local Public Suffix List cache with network updates disabled during preprocessing.

All raw URLs remained offline. Intermediate assignments use sample and entity hashes, and every completed stage includes machine-readable metrics, predicted probabilities, logs, and SHA-256 manifests.

**Corpus audit.**

PhiUSIIL contained 235,795 input rows, including 134,850 benign and 100,945 phishing URLs. Normalization identified 1,138 duplicate rows beyond the first occurrence and one conflicting normalized-URL group containing two rows. Removing only that conflict produced a master corpus of 235,793 rows. The binary ISCX-URL2016 corpus contained 45,343 rows, including 35,378 benign and 9,965 phishing URLs. It contained 119 duplicate rows beyond the first occurrence and no conflicting-label group.

**Table 3. Corpus audit after normalization and conflict removal**

| Dataset | Input rows | Master rows | Benign | Phishing | Unique normalized URLs | Unique eTLD+1 | Conflicting rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| ISCX-URL2016 | 45,343 | 45,343 | 35,378 | 9,965 | 45,224 | 4,304 | 0 |
| PhiUSIIL | 235,795 | 235,793 | 134,849 | 100,944 | 234,656 | 175,507 | 2 |

*Note.* URL, uniform resource locator; eTLD+1, effective top-level domain plus one label (the registrable domain). ISCX-URL2016 and PhiUSIIL are the two evaluated phishing web-address corpora.

No normalized URL was shared between the two corpora, although 123 registrable domains occurred in both. All 360 internal entity-overlap checks passed. The primary external cohorts had zero normalized-URL, host, and registrable-domain overlap with the corresponding source training and validation sets. The fully deduplicated sensitivity corpora retained 45,224 ISCX and 234,656 PhiUSIIL rows.

**Entity-conditioned internal performance.**

Figure 1 and Table 4 show that separation effects were corpus dependent. ISCX changed little between S0 and URL-disjoint S1, but performance decreased and dispersion increased under host and registrable-domain separation. From S0 to S3, mean Macro-F1 changed by -0.1089 for LR, -0.0383 for RF, and -0.0998 for XGBoost. Mean S3 false-positive rates were 0.1422, 0.0408, and 0.1124, respectively. For LR, this was a 24.5-fold increase from the S0 false-positive rate of 0.0058, making the entity-conditioned change operationally consequential rather than merely a reduction in an aggregate score.

![Figure 1. Internal macro-averaged F1 score (Macro-F1) across row-random, normalized-uniform-resource-locator-disjoint, host-disjoint, and registrable-domain-disjoint regimes. Points are means over ten outer repetitions, and error bars are repetition-bootstrap 95% intervals.](figures/Figure_1.pdf)

PhiUSIIL showed a different pattern: S0-to-S3 Macro-F1 changes were only -0.0009, -0.0014, and -0.0018 for LR, RF, and XGBoost. Thus, random splitting did not inflate performance uniformly. The wide ISCX S3 dispersion reflects movement of a small number of high-volume domain groups between partitions and is not ordinary independent-row uncertainty.

**Table 4. Mean Macro-F1 and false-positive rate under S0 and S3 (10 repetitions)**

| Dataset | Model | S0 Macro-F1 | S3 Macro-F1 | Change | S0 FPR | S3 FPR |
|---|---|---:|---:|---:|---:|---:|
| ISCX-URL2016 | LR | 0.9750 | 0.8661 | -0.1089 | 0.0058 | 0.1422 |
| ISCX-URL2016 | RF | 0.9962 | 0.9578 | -0.0383 | 0.0016 | 0.0408 |
| ISCX-URL2016 | XGBoost | 0.9976 | 0.8978 | -0.0998 | 0.0010 | 0.1124 |
| PhiUSIIL | LR | 0.9866 | 0.9857 | -0.0009 | 0.0110 | 0.0098 |
| PhiUSIIL | RF | 0.9916 | 0.9902 | -0.0014 | 0.0082 | 0.0102 |
| PhiUSIIL | XGBoost | 0.9954 | 0.9936 | -0.0018 | 0.0015 | 0.0037 |

*Note.* Macro-F1, macro-averaged F1 score; FPR, false-positive rate; S0, row-random split; S3, registrable-domain-disjoint split; LR, logistic regression; RF, random forest. ISCX-URL2016 and PhiUSIIL are the two evaluated phishing web-address corpora.

### 5.2 Explanation Stability and Compact Feature Sets

To answer RQ2, the second evidence block asks whether the model's leading feature evidence is as stable as its source-side predictions appear. It separates controlled sources of explanation variation before testing whether repeated training-only selection yields a compact model without assuming that compactness implies transferability.

**Controlled SHAP stability.**

Figure 2 and Table 5 separate four sources of variation. Model-seed changes produced the greatest agreement. LR was deterministic under the fixed split and cohort (Top-15 Jaccard and rank Spearman both 1.000); tree-model seed Jaccard remained 0.876-0.939. Partition changes reduced Top-15 Jaccard to 0.798-0.920, although full-ranking Spearman correlations remained 0.951-0.989.

\clearpage

![Figure 2. Top-15 Jaccard and full-ranking Spearman agreement for SHapley Additive exPlanations (SHAP) under seed, partition, deployment-regime, and explanation-source variation. For each model and estimand, boxes pool the declared pairwise comparisons from both corpora; boxes show the median and interquartile range, whiskers extend to 1.5 times the interquartile range, and outliers are omitted from display.](figures/Figure_2.pdf)

Regime and source changes produced larger set-level differences. Regime Top-15 agreement ranged from 0.704 to 0.897, while source agreement ranged from 0.623 to 0.788 for RF/XGBoost and from 0.719 to 0.730 for LR. Source rank correlations ranged from 0.810 to 0.941. A reproducible ranking under reseeding is therefore weak evidence that the same leading features describe a different deployment regime or source.

**Table 5. Range of SHAP agreement across datasets and models**

| Stability estimand | Top-15 Jaccard range | Full-rank Spearman range | Interpretation |
|---|---:|---:|---|
| Model seed | 0.876-1.000 | 0.986-1.000 | Highest agreement |
| Group partition | 0.798-0.920 | 0.951-0.989 | Moderate set turnover |
| Evaluation regime | 0.704-0.897 | 0.848-0.980 | Split semantics matter |
| Explanation source | 0.623-0.788 | 0.810-0.941 | Largest source sensitivity |

*Note.* SHAP, SHapley Additive exPlanations; Top-15 Jaccard, Jaccard similarity between the sets of the 15 highest-ranked features; full-rank Spearman, Spearman rank correlation across all 35 features.

**Stable feature sets and within-source compactness.**

F-Stable retained 12.3-13.5 of 35 features on ISCX and 13.8-14.3 on PhiUSIIL, a reduction of 59.1%-64.9% (Figure 3 and Supplementary Table S4). Compactness did not imply universal predictive improvement. On PhiUSIIL, F-Stable differed from F-All by -0.0087 for LR, +0.0002 for RF, and -0.0002 for XGBoost. On ISCX, the differences were -0.0445, -0.0311, and +0.0340, with substantial repetition-level dispersion.

![Figure 3. Feature-set size and internal registrable-domain-disjoint macro-averaged F1 score (Macro-F1) relative to the full 35-feature model (F-All). Bars show mean retained features. Boxes pool paired differences over three models and ten outer repetitions within each corpus; the dashed line marks no change from F-All.](figures/Figure_3.pdf)

The conventional controls qualify the feature-selection claim. Across 60 dataset-model-repetition comparisons, F-MI was below F-All in 52 cases. F-Permutation was below F-All in 37, above it in 21, and tied in two. F-Stable exceeded F-MI in 48 of 60 internal comparisons but exceeded F-Permutation in only 26 and was lower in 34. On PhiUSIIL, the F-Stable advantage over F-MI was clearest for RF and XGBoost (mean differences 0.0204 and 0.0137; BH-adjusted Wilcoxon p=0.0074 for both). F-Stable is therefore a compact and more defensible control than model-independent MI in these settings, but it is not generally superior to model-specific permutation selection.

### 5.3 Cross-Source Operational Failure

The operational part of RQ1 freezes the source-side pipeline and asks what survives a change of corpus. It begins with predictive transfer, then tests duplicate removal, entity weighting, calibration, score orientation, frozen-threshold transfer, and false-positive concentration as competing descriptions of the same deployment failure.

**Bidirectional zero-target-feedback transfer.**

All five feature schemes failed severely under zero-target-feedback transfer (Figure 4 and Supplementary Table S5). From ISCX to the domain-filtered PhiUSIIL cohort, mean Macro-F1 across models and repetitions ranged from 0.2908 to 0.2922. In the reverse direction, it ranged from 0.1703 to 0.1937. Mean row-level false-positive rates were approximately 1.000 and 0.978-1.000, respectively. For F-All, the external-minus-internal Macro-F1 gaps were -0.6150 and -0.7985.

![Figure 4. Zero-target-feedback transfer performance for five source-selected feature schemes. Points are descriptive means over three models and ten outer repetitions, and error bars are bootstrap 95% intervals over those model-repetition values. Blue circles and red squares denote transfer from the ISCX-URL2016 corpus to the PhiUSIIL corpus and the reverse direction, respectively.](figures/Figure_4.pdf)

Feature selection did not resolve the failure. F-Stable exceeded F-MI in 50 of 60 external comparisons but exceeded F-Permutation in only 27 and was lower in 31. F-MI was lower than F-All in 54 of 60 comparisons. The association between outer-split feature-set stability and external Macro-F1 was also not positive: Spearman rho was 0.19 (p=0.554) for ISCX-trained models and -0.37 (p=0.241) for PhiUSIIL-trained models. These 12-point source analyses are descriptive and underpowered, but they directly oppose the assumption that repeatability guarantees transfer.

Several configurations also produced ROC-AUC below 0.5, meaning that the learned score ranked target examples in the wrong direction. A 300-configuration orientation audit ruled out an inverted label or probability-column implementation: all class orders were `[0,1]`, stored decisions used the phishing-probability column, probability reconstruction differed by less than \(3\times10^{-8}\), and \(\mathrm{AUC}(\hat p)+\mathrm{AUC}(1-\hat p)=1\) held within \(5.09\times10^{-6}\). For PhiUSIIL-to-ISCX XGBoost, ordinary AUC ranged from 0.0745 to 0.2555 across feature methods, whereas the target-label diagnostic \(\mathrm{AUC}(1-\hat p)\) ranged from 0.7445 to 0.9255. The latter values expose score reversal; they are not corrected model performance.

The completed mechanism audit associated this reversal with source-conditioned feature and class distributions. With a conservative Spearman dead zone of \(|\rho|<0.02\), feature-label direction reversed for 20/35 features from PhiUSIIL to ISCX and 19/35 in the opposite comparison. A logistic source classifier evaluated with registrable-domain-grouped splits distinguished the corpora with mean AUC 1.0000 among benign URLs and 0.9129 among phishing URLs, showing that source differences persist within class. Across fixed external explanation cohorts, Top-10 target SHAP/label-direction conflict ranged from 38% to 80%. These converging diagnostics support the term *source-conditioned ranking reversal associated with class-conditional covariate and association shift*. They do not identify one causal collection mechanism, because source is confounded with time, curation, URL status, and sampling policy.

**Duplicate-removal sensitivity.**

The fully deduplicated analysis repeated F-All training for S0 and S3 with both datasets, all three models, and all ten fixed repetitions (120 matched runs). Removing repeated normalized URLs barely changed S0 estimates: mean paired Macro-F1 differences ranged from -0.0004 to +0.0004. S3 differences were also small for most combinations. The largest mean was +0.0187 for ISCX RF, but its 95% repetition-bootstrap interval was -0.0288 to 0.0635; ISCX XGBoost was +0.0027 (-0.0390 to 0.0418). PhiUSIIL LR increased by 0.0019 (0.0007 to 0.0031), while the remaining PhiUSIIL S3 changes were within 0.0003 in absolute mean.

These matched results do not support duplicate rows as the main explanation for the ISCX S3 instability. The more plausible measured factor is which heterogeneous, high-volume registrable domains enter each grouped test partition.

**Table 6. Mean paired Macro-F1 change after full deduplication**

| Dataset | Regime | LR | RF | XGBoost |
|---|---|---:|---:|---:|
| ISCX-URL2016 | S0 | +0.0001 | +0.0004 | -0.0003 |
| ISCX-URL2016 | S3 | -0.0001 | +0.0187 | +0.0027 |
| PhiUSIIL | S0 | -0.0002 | -0.0004 | -0.0001 |
| PhiUSIIL | S3 | +0.0019 | -0.0002 | +0.0003 |

*Note.* Macro-F1, macro-averaged F1 score; S0, row-random split; S3, registrable-domain-disjoint split; LR, logistic regression; RF, random forest. ISCX-URL2016 and PhiUSIIL are the two evaluated phishing web-address corpora.

**Entity weighting and calibration diagnostics.**

Row-level and domain-equal metrics answer different deployment questions. Under S3 internal evaluation, domain-equal weighting changed Macro-F1 by -0.180 to +0.051 across the five feature methods and three models, confirming that URL-rich domains can materially affect row-level summaries. Under external transfer, ISCX-to-PhiUSIIL domain-equal Macro-F1 was 0.191-0.197, below the row-level range. In the reverse direction it increased to 0.481-0.529 because numerous small phishing-associated domains received the same total weight as a few large benign domains. This higher Macro-F1 does not indicate usable benign discrimination: domain-equal FPR still ranged from 0.948 to 1.000. In the representative XGBoost r00 domain-cluster bootstrap, F-All and F-Stable intervals overlapped closely in both directions (0.1955-0.1980 versus 0.1956-0.1983, and 0.4801-0.4841 versus 0.4799-0.4840).

Calibration results further separate threshold mismatch from transferable discrimination (Figure 5 and Supplementary Figure S1). Across the five methods, mean 15-bin ECE was 0.581-0.592 from ISCX to PhiUSIIL and 0.747-0.794 in the reverse direction. Mean Brier scores were 0.580-0.587 and 0.745-0.794. The complete reliability diagrams show that predicted probabilities cluster at extreme values rather than following the identity line. Exact post hoc target-oracle thresholds improved Macro-F1 by 0.074-0.292 and 0.202-0.333, respectively, but most selected thresholds were at or near 1.0 and often reduced to an almost all-negative decision. The oracle therefore demonstrates severe source-threshold incompatibility, not a deployable correction; substantial ranking failure remained for most configurations.

![Figure 5. Difference between the macro-averaged F1 score (Macro-F1) obtained by a post hoc target-oracle threshold and the Macro-F1 obtained by the frozen source-selected threshold. Boxes summarize ten outer repetitions; the oracle is diagnostic only and is never used for model fitting or a deployment claim.](figures/Figure_5.pdf)

**Frozen-threshold transfer failure.**

The validation constraints were satisfied by construction: mean validation FPR was 0.00092-0.00099 at the 0.1% budget and 0.00949-0.00999 at the 1% budget. The corresponding one-sided 95% Wilson upper bounds were approximately 0.00140-0.00192 and 0.01118-0.01239. Each ISCX validation split allowed only about five false positives at the 0.1% budget, so the threshold is a discrete order statistic with coarse finite-sample resolution. Half of the 60 locked model selections also had a validation Macro-F1 margin below 0.001, and adjacent-score thresholds exposed additional decision sensitivity. These facts preclude interpreting the selected point as an overall low-FPR guarantee.

Untouched source S3 test behavior was already corpus dependent. PhiUSIIL-trained thresholds remained close to their intended budgets, with mean source-test FPRs of 0.00097-0.00116 and 0.00935-0.01054. ISCX-trained thresholds did not: source-test FPRs rose to 0.0785-0.1412 at the 0.1% budget and 0.1371-0.1672 at the 1% budget. Held-domain composition therefore broke threshold portability even before the data source changed.

Cross-source application failed more severely (Table 7). At the 0.1% source-validation budget, mean external FPR ranged from 0.5318 to 1.0000. Relaxing the source budget to 1% did not make the decision rule portable: mean external FPR ranged from 0.9068 to 1.0000. These rates correspond to approximately 5,318-10,000 false alerts per 10,000 benign target URLs. High target recall in several rows is not operationally favorable because it is achieved by labeling almost every benign URL as phishing.

**Table 7. Frozen source-validation operating points on the primary external cohort (10-repetition means)**

| Source to target | Model | External TPR at 0.1% source FPR | External FPR | False alerts per 10,000 benign | External TPR at 1% source FPR | External FPR | False alerts per 10,000 benign |
|---|---|---:|---:|---:|---:|---:|---:|
| ISCX-URL2016 to PhiUSIIL | LR | 0.5654 | 0.5318 | 5,318 | 0.7295 | 0.9068 | 9,068 |
| ISCX-URL2016 to PhiUSIIL | RF | 0.9856 | 1.0000 | 10,000 | 0.9895 | 1.0000 | 10,000 |
| ISCX-URL2016 to PhiUSIIL | XGBoost | 0.9881 | 1.0000 | 10,000 | 0.9903 | 1.0000 | 10,000 |
| PhiUSIIL to ISCX-URL2016 | LR | 0.9951 | 0.9291 | 9,291 | 0.9973 | 0.9407 | 9,407 |
| PhiUSIIL to ISCX-URL2016 | RF | 0.9879 | 0.9993 | 9,993 | 0.9986 | 1.0000 | 10,000 |
| PhiUSIIL to ISCX-URL2016 | XGBoost | 0.9987 | 1.0000 | 10,000 | 0.9999 | 1.0000 | 10,000 |

*Note.* TPR, true-positive rate; FPR, false-positive rate; LR, logistic regression; RF, random forest. The percentages in the column headings are the empirical source-validation FPR budgets. ISCX-URL2016 and PhiUSIIL are the two evaluated phishing web-address corpora.

These operating points reinforce rather than replace the calibration analysis. They are source-only threshold choices under the stated zero-target-feedback protocol, but the experiment diagnoses failure rather than certifying deployment: constraining empirical source-validation FPR did not constrain source-test or target FPR.

![Figure 6. Realized false-positive rates (FPRs) after applying source-validation thresholds to unseen-domain source tests and primary external cohorts. Points are ten-repetition means for full-feature models. Circles and squares denote 0.1% and 1% source-validation FPR budgets; dotted and dashed lines mark those budgets.](figures/Figure_6.pdf)

**False-positive structure.**

The two transfer directions produced different domain-level false-positive structures (Figure 7). ISCX-to-PhiUSIIL yielded approximately 134,700 false positives per run across about 132,100 registrable domains; the ten most frequent domains contributed only 0.48% of false positives. The error was diffuse across the target source. PhiUSIIL-to-ISCX yielded approximately 32,100-32,900 false positives across about 281 domains, with the ten largest domains contributing about 25%. Here the source mismatch was amplified by several high-volume target entities. This asymmetry explains why row- and domain-equal Macro-F1 move in opposite directions while both domain-equal FPRs remain operationally unacceptable.

![Figure 7. Registrable-domain structure of external false positives for five feature schemes, averaged descriptively over three models and ten outer repetitions. The left panel uses a logarithmic count axis; the right panel reports the share contributed by the ten highest-volume registrable domains. Blue forward-hatched and red cross-hatched bars denote transfer from the ISCX-URL2016 corpus to the PhiUSIIL corpus and the reverse direction, respectively.](figures/Figure_7.pdf)

### 5.4 Bounded Evidence for Domain-Conditioned SHAP Stability

The final evidence block tests the deliberately narrower RQ3 claim. DCSS is evaluated first as a reproducible held-domain construction, then against locked internal and external criteria, and finally through component, random-subset, and third-source checks that determine which parts of the method contribute and where the signal fails.

**Selection reproducibility and internal non-inferiority.**

DCSS completed 300 held-domain SHAP subruns and produced one deterministic Top-10, Top-15, and Top-20 set for each of 60 dataset-repetition-model keys. Across outer repetitions, mean pairwise Jaccard agreement for F-DCSS-15 was 0.825, 0.882, and 0.839 for ISCX LR, RF, and XGBoost, respectively, and 0.908, 1.000, and 0.933 for the corresponding PhiUSIIL models. Its average overlap with F-Stable ranged from 10.8 to 14.3 of 15 features, indicating that domain conditioning changed a minority rather than the entirety of the selected set.

Table 8 reports the locked F-DCSS-15 versus F-All internal comparison. The mean effects fail the -0.01 margin for ISCX LR (-0.0384) and RF (-0.0275). The prespecified auxiliary test met its non-inferiority criterion for ISCX XGBoost and all three PhiUSIIL models, but the ISCX XGBoost decision weakened under the illustrative positive-dependence stress analysis. The requirement of at least two qualifying model families in each dataset was therefore not met. The positive ISCX XGBoost mean should be read as model-specific rather than evidence that feature removal generally improves unseen-domain prediction.

**Table 8. F-DCSS-15 minus F-All S3 Macro-F1 and auxiliary non-inferiority decision**

| Dataset | Model | Mean difference | Repetition-bootstrap 95% interval | Auxiliary BH-adjusted one-sided p | Auxiliary decision |
|---|---|---:|---:|---:|---|
| ISCX-URL2016 | LR | -0.0384 | -0.0779 to -0.0082 | 0.9186 | Not supported |
| ISCX-URL2016 | RF | -0.0275 | -0.0843 to 0.0188 | 0.8756 | Not supported |
| ISCX-URL2016 | XGBoost | +0.0885 | 0.0290 to 0.1515 | 0.0119 | Supported |
| PhiUSIIL | LR | -0.0074 | -0.0085 to -0.0064 | 0.0015 | Supported |
| PhiUSIIL | RF | +0.0001 | -0.0004 to 0.0007 | 5.81e-11 | Supported |
| PhiUSIIL | XGBoost | +0.0003 | -0.0001 to 0.0006 | 4.93e-12 | Supported |

*Note.* DCSS, Domain-Conditioned SHAP Stability; SHAP, SHapley Additive exPlanations; F-DCSS-15, the 15-feature DCSS subset; F-All, the full 35-feature set; S3, registrable-domain-disjoint split; Macro-F1, macro-averaged F1 score; BH, Benjamini-Hochberg; LR, logistic regression; RF, random forest. The decision is the prespecified auxiliary test outcome for paired perturbations of the fixed corpus, not a population-level inference for future datasets.

**Prespecified cross-source comparison.**

The primary external comparison was directional (Table 9). From ISCX to PhiUSIIL, F-DCSS-15 changed mean ROC-AUC by -0.00086 relative to F-Stable; the domain-plus-outer interval crossed zero and 6/10 paired repetition effects were positive. From PhiUSIIL to ISCX, the mean difference was +0.01467, the interval excluded zero, and all ten repetition-level differences were positive. The adjusted p values (0.9648 and 0.0336) are reported as auxiliary evidence. By model, the latter-direction differences were +0.0402 for LR, +0.0081 for RF, and -0.0043 for XGBoost, so the persistent XGBoost ranking reversal was not repaired. All 80 paired differences underlying the six internal and two external comparisons are retained in the anonymous artifact.

**Table 9. Prespecified F-DCSS-15 minus F-Stable external ROC-AUC**

| Source to target | Mean difference | Domain-plus-outer 95% interval | Outer-only 95% interval | Auxiliary BH-adjusted p | Positive repetitions |
|---|---:|---:|---:|---:|---:|
| ISCX-URL2016 to PhiUSIIL | -0.00086 | -0.02014 to 0.01847 | -0.01933 to 0.01719 | 0.9648 | 6/10 |
| PhiUSIIL to ISCX-URL2016 | +0.01467 | 0.00254 to 0.02739 | 0.01037 to 0.01898 | 0.0336 | 10/10 |

*Note.* DCSS, Domain-Conditioned SHAP Stability; SHAP, SHapley Additive exPlanations; F-DCSS-15, the 15-feature DCSS subset; F-Stable, the random-resampling stability subset; ROC-AUC, area under the receiver operating characteristic curve; BH, Benjamini-Hochberg. ISCX-URL2016 and PhiUSIIL are the two evaluated phishing web-address corpora.

The size sensitivity did not justify replacing the prespecified choice. F-DCSS-20 was directionally above F-Stable in both transfers, with mean ROC-AUC values of 0.4143 and 0.3704 compared with 0.4013 and 0.3538 for F-Stable, but this result was examined only as a sensitivity analysis. The overall strong-contribution condition failed because internal non-inferiority was not supported in two ISCX model families and external improvement was not observed in both directions. Under an illustrative assumed repetition correlation of 0.50, the positive PhiUSIIL-to-ISCX effect retained a lower bound of +0.00241, whereas several internal non-inferiority decisions weakened; this stress test does not estimate the actual dependence or make the repetitions independent.

**Ablation, random subsets, and mechanism diagnostics.**

At k=15, NoRankDispersion selected exactly the same set as the full DCSS score in all 60 keys. NoDirection was identical in 59 of 60 keys, and NoFrequency in 44 of 60; their mean Jaccard similarities with the full score were 1.000, 0.998, and 0.965. Thus, rank dispersion made no boundary-level contribution and direction consistency almost none under the locked data and dimensionality. Mean normalized importance and Top-15 frequency dominated selection, while the remaining terms mostly changed scores away from the selection boundary.

The 30 fixed random Top-15 sets further limited the superiority claim. Internally, no random set matched mean ISCX XGBoost DCSS Macro-F1, but 43.3% and 46.7% matched or exceeded it for ISCX LR and RF. For PhiUSIIL, the corresponding fractions ranged from 3.3% to 26.7%. Externally, DCSS beat every random set for PhiUSIIL-to-ISCX LR, but every random set exceeded it for the reverse-ranking XGBoost configuration. For ISCX-to-PhiUSIIL, 83.3%, 30.0%, and 20.0% of random sets matched or exceeded DCSS for LR, RF, and XGBoost.

Held-domain Top-15 fold agreement averaged approximately 0.72-0.85 on ISCX and 0.87-0.98 on PhiUSIIL. Across the 60 dependent outer keys, no source-only stability diagnostic had a strong descriptive association with external AUC gain; the largest listed correlation was between direction conflict and AUC difference (Spearman rho=-0.213, nominal p=0.103). These analyses support DCSS as a domain-sensitivity probe in selected settings, not as a general optimization rule.

**Third-source sensitivity.**

An additional URL-Phish v1 sensitivity analysis produced positive mean F-DCSS-15-minus-F-Stable AUC differences for models trained on each primary source, but model-specific effects had mixed signs and absolute operating performance remained unusable. The downloaded row count also differed from the accompanying article, and filtering was class dependent because class and collection source were coupled. URL-Phish v1 is therefore not treated as independent validation. Dataset auditing, filtering, model-level effects, and absolute metrics are reported in Supplementary Section S6.

## 6. Discussion

### 6.1 Principal Findings and Their Boundaries

The deployment-validity audit keeps entity alignment, score orientation, explanation repeatability, and frozen-threshold portability as separate claims. Entity separation changed ISCX performance substantially and had little effect on PhiUSIIL, which shows that random-split optimism depends on corpus construction and on the type of entity novelty represented in the test. Source shift disrupted score orientation, frozen decision rules, and aggregate performance. The mechanism audit excluded a probability-orientation error and found class-conditional source separability, feature-label association reversals, and SHAP-direction conflicts. Seed-level SHAP agreement was high, but agreement declined under regime and source changes; repeatability within one corpus therefore does not establish external invariance. Within this audit, DCSS detected sensitivity to held-out domains and produced a positive effect in one transfer direction, yet its failure in the opposite direction and the weak contribution of two score components preclude a general claim of selector superiority.

The internal results narrow the explanation for this pattern. The near-equality of S0 and S1, together with the fixed-role deduplication analysis, indicates that repeated normalized URLs were not the main source of the ISCX gap. Performance changed chiefly after separating hosts and registrable domains, while PhiUSIIL remained stable. The large ISCX S3 variance and the domain-equal sensitivity also show that group composition is part of the estimand rather than nuisance variation to be hidden by a pooled row-level interval. S0-S3 represent different deployment targets and should not be interpreted as a universal progression from invalid to valid evaluation.

### 6.2 Relation to Prior Work

The cross-source degradation agrees with the findings of Rashid et al.^[7]^ and Yi et al.^[8]^, who reported that strong within-dataset phishing URL performance can weaken on another corpus. Our results extend this observation from ranking and classification metrics to decision policy: thresholds constrained to a 0.1% or 1% source-validation FPR still produced unacceptable external false-alert rates. A detector may preserve some ranking information and remain unusable at its frozen operating point.

PhreshPhish argues that credible phishing benchmarks require leakage control, difficult negatives, temporal structure, and realistic base rates^[11]^. That work builds a prospective benchmark around a newly collected corpus. Our results complement it on established public sources by showing that registrable-domain independence, explanation repeatability, and frozen decision thresholds fail in different ways and require separate evidence.

The explanation results extend the cross-dataset feature differences reported by Mia et al.^[9]^. Instead of comparing one importance profile from each source, our design separates variation due to model seed, grouped partition, deployment regime, and explanation source. The resulting agreement pattern shows which perturbations alter the interpretation and places reseeding stability as the least demanding evidence. Compared with the integrated protocol of Ahamed et al.^[10]^, this study treats residual `eTLD+1` dependence as an explicit registrable-domain-disjoint estimand and connects it to frozen-threshold behavior. Our contribution lies in separating entity validity, explanation repeatability, and threshold portability within one source-only decision protocol, rather than in the first use of external testing or SHAP. This framing follows broader warnings that evidence in security machine learning must match the intended deployment population^[14,15]^.

### 6.3 Security and Operational Implications

The frozen-threshold diagnostic gives the transfer results an operational meaning. A threshold allowed at most 10 or 100 false alerts per 10,000 benign source-validation URLs, yet it produced an average of 5,318 to 10,000 false alerts per 10,000 benign external URLs. Even with high target recall, this volume would overwhelm triage in an email gateway, browser filter, or security operations workflow. The counts are corpus-normalized workload indicators; they are not estimates of production prevalence or guarantees for a future population. At this scale, the evidence supports rejecting deployment of the frozen models unless target-aware monitoring, recalibration, or retraining is available.

For ISCX, threshold instability appeared even before source transfer: validation-constrained operating points yielded mean S3 test FPRs of 7.85%-16.72%, depending on the model and budget. At the 0.1% budget, the validation upper bounds already exceeded the nominal point, and only a small number of false positives was allowed; adjacent-score and grouped-holdout checks showed further sensitivity in threshold selection. Entity composition therefore affected the decision policy as well as the ranking metrics. An operational evaluation should define the entity novelty under test, freeze the threshold without test feedback, and report both uncertainty and realized class-conditional error rates. Neither a source ROC-AUC nor a stable explanation set replaces this check.

### 6.4 Explanation Stability and the Bounded Role of DCSS

Predictive performance and explanation stability address different objectives, while stability across seeds, partitions, regimes, and sources answers different questions. High seed agreement showed that the fitted procedure reproduced its output under a controlled perturbation; lower regime and source agreement showed that the leading explanation set depended on the population. Stability does not establish faithfulness, correctness, or causality because a model may reproduce a source bias consistently^[18-20,23,24]^. Rank turnover is also not sufficient evidence of unusable logic, since correlated features can exchange positions. The lack of a positive stability-transfer association, together with the poor transfer of the perfectly repeated MI set, argues directly against using repeatability as a certificate of external validity.

**What DCSS adds and what it does not.**

DCSS changes the perturbation used to estimate stability by evaluating a feature on domains withheld from fitting, rather than only across random inner resamples. In the PhiUSIIL-to-ISCX direction, the mean difference was +0.01467, the domain-plus-outer interval was 0.00254 to 0.02739, and all ten paired partition effects were positive. This finite-corpus pattern indicates that the conditioning can matter in that transfer direction, particularly for LR; it is not population-level evidence for independently collected sources. The lower fold agreement on ISCX also identifies domain composition as an observable source of explanation variation. These findings do not establish general superiority, but they address a practical question that a conventional seed-stability plot cannot answer.

The negative results limit the interpretation of the method. DCSS did not improve the opposite transfer direction, establish internal non-inferiority for ISCX LR or RF, or repair the reverse-ranking XGBoost failure. The direction and rank-dispersion terms also rarely changed Top-15 membership. The locked four-factor formula is therefore an evaluated diagnostic construction, not a validated optimal scoring rule. The ablations motivate a simpler future selector based on normalized importance and held-domain frequency, but we did not substitute that selector post hoc.

### 6.5 Practical Meaning and Reporting Implications

F-Stable has a limited practical use. For PhiUSIIL RF and XGBoost, approximately 14 features retained nearly all S3 performance, suggesting that repeated training-only screening can support compact within-source models when feature computation or explanation cost matters. DCSS serves a different purpose by evaluating selected features on held-out domains before the outer model is frozen. These findings do not extend uniformly to ISCX or to cross-source deployment. F-Stable and F-DCSS are reproducible screening sets under specified perturbations, not invariant phishing indicators.

The experiments indicate that an evaluation report should define the deployment target before labeling entity overlap as leakage; disclose normalization, conflict handling, and duplicate handling; report overlap at the URL, host, and registrable-domain levels; include row-weighted and entity-weighted metrics when domain sizes are skewed; keep fitting, thresholding, SHAP aggregation, and selection within source-side training data; give finite-sample bounds and realized errors for frozen thresholds; compare proposed selectors with same-size random sets; and report calibration for transferred thresholds. A target-oracle threshold is useful only as a post hoc diagnostic. It cannot be combined with zero-target-feedback results or reported as deployment performance.

### 6.6 Future Work

A useful next test requires more than adding another selector to the same two corpora. Temporally ordered or prospectively collected URLs, sampled at the production base rate, would allow direct measurement of alert volume, delayed labels, and domain churn. Estimating operating points below 0.1% FPR with useful precision also requires larger benign validation cohorts. Our class-conditional source-discrimination and feature-drift analyses show that collection source is detectable within class, but they cannot distinguish the effects of collection pipeline, labeling criteria, temporal change, and genuine attack evolution. That causal distinction requires prospective metadata and collection-matched controls. The resulting diagnostics could inform source-shift alarms, conservative abstention, or recalibration policies without assuming immediate target labels.

A simplified held-domain selector based on normalized importance and selection frequency should be evaluated as a new prespecified method, rather than substituted after the present ablation. Future comparisons should add temporal groups, more public and organizational sources, and URL features paired with independently collected HTML, DNS, certificate, redirect, or runtime evidence. Such experiments would test whether domain-conditioned explanation evidence remains useful when source identity is less tightly coupled to feature collection, and whether any compact set supports a decision rule with stable realized FPR under prospectively measured shift.

### 6.7 Threats to Validity

**Internal validity.** URL normalization may merge distinct addresses incorrectly or fail to merge equivalent forms, and registrable-domain extraction depends on the frozen Public Suffix List and its private-suffix policy. We retain the normalization rules, parser outcomes, hashes, and issue logs. Fixed candidate grids and ten repetitions reduce researcher degrees of freedom, although model and split uncertainty remain.

**Construct validity.** Jaccard similarity, ranking correlation, direction consistency, and the DCSS score measure agreement under specified perturbations; they do not measure faithfulness or causality. In the ablation, two score terms rarely affected Top-15 membership, and no source-only stability diagnostic had a strong descriptive association with external gain. Domain-equal Macro-F1 may increase when many small positive domains receive more total weight even if FPR remains near one, so it requires joint interpretation with class-conditional rates. ECE depends on the chosen bins, and the target-oracle threshold uses target labels only for diagnosis.

**External validity.** The primary design covers two public datasets, three conventional classifiers, and 35 offline URL features. The third source remains exploratory because the documented and observed row counts disagree, filtering is class dependent, and class is coupled with collection source. S4 identifies source-conditioned incompatibility between the evaluated public benchmark construction pipelines rather than universal, temporal, or live-deployment generalization. The analyzed ISCX artifact is a third-party preserved copy from a pinned public mirror and was not byte-verified against an official release. URL-only features are inexpensive and reproducible, but attacks visible only through HTML, certificates, redirects, DNS, or runtime behavior remain outside their scope.

**Label validity.** Public benign and phishing labels may include annotation errors or obsolete records. Removing conflicting labels reduces direct contradictions but does not revalidate every URL. Different labeling criteria may account for part of the cross-dataset degradation, which limits the scope of the conclusions.

**Statistical validity.** Ten overlapping outer repetitions provide paired perturbation evidence about fixed corpora, not ten independent datasets or a sampling frame for future sources. We therefore treat mean effects, intervals, all paired differences, and direction counts as primary, while one-sample one-sided t-tests and multiplicity-adjusted p-values remain auxiliary. The external interval combines a linearized empirical-AUC influence approximation at the target-domain level with outer resampling; this approach preserves the declared clustering unit, but it is not an exact literal nested bootstrap, and its finite-corpus uncertainty does not cover future collection processes. In an illustrative dependence-stress analysis, the positive PhiUSIIL-to-ISCX effect remains above zero under an assumed repetition correlation of 0.50, whereas some internal non-inferiority decisions are less stable; the assumed value is not an estimate of the actual dependence. The separate representative domain-cluster bootstrap covers only XGBoost r00 and F-All/F-Stable. At the 0.1% operating point, each ISCX validation split permits about five false positives; the empirical constraint and Wilson bound are reproducible for the frozen cohort but provide no population-level guarantee. The ablations, random-set distributions, dimensionality checks, and third-source contrasts remain sensitivity evidence.

## 7. Conclusion

This deployment-validity audit separates four requirements for evaluating phishing URL detectors: entity-aligned evaluation, score orientation, explanation stability, and frozen-threshold transfer. Registrable-domain separation materially reduced ISCX performance and changed PhiUSIIL only slightly, so the effect of entity overlap depends on the corpus and deployment target. Thresholds selected under 0.1% or 1% source-validation FPR budgets produced mean external FPRs of 0.532-1.000; these results diagnose transfer failure and do not establish an overall low-FPR guarantee. The below-chance target AUC did not result from a label-orientation error; class-conditional source separability, 19-20 feature-association reversals, and 38%-80% Top-10 SHAP-direction conflict support a bounded diagnosis of source-conditioned ranking reversal between the evaluated benchmarks. SHAP agreement also declined from seed variation to regime and source variation. Within the audit, DCSS produced a positive finite-corpus effect in one transfer direction (+0.01467; domain-plus-outer interval 0.00254-0.02739; 10/10 positive repetitions), but not in the other (-0.00086; -0.02014 to 0.01847; 6/10 positive), and it failed internal non-inferiority for two ISCX model families. The methodological conclusion is limited but direct: entity alignment, score orientation, source-only thresholds, and explanation repeatability require separate tests, and none alone certifies calibrated cross-source defense.

## Declarations

### Data and Code Availability

The source datasets are publicly described, but the availability and licensing of the analyzed files differ. PhiUSIIL was downloaded from the UCI Machine Learning Repository (dataset 967; https://archive.ics.uci.edu/dataset/967/phiusiil%2Bphishing%2Burl%2Bdataset) on 2 September 2026 under CC BY 4.0; the downloaded archive has SHA-256 `0a639fd03aea6308c5b1c10c92aa23c2ce1505447a9137271865cd0badc9a59a`. ISCX-URL2016 is described by the Canadian Institute for Cybersecurity (https://www.unb.ca/cic/datasets/url-2016.html). Because its registered download endpoint was unavailable on 2 September 2026, the experiment used a third-party preserved copy of the raw URL lists from https://github.com/Raj-S-Singh/Real-Real-Time-Detection-of-Malicious-URLs-using-Machine-Learning at commit `9f8ffd697ce3250a7db5f4bce9c4870722b70ef5`; the mirror archive has SHA-256 `6ff4e3bbd7aeb2ecb30624a8142d5de517e2c8ef5cc20adcbc574bd657483858`. The manuscript uses `ISCX-URL2016` as shorthand for this preserved copy, not as a claim of byte identity with the official release. The filenames, category structure, and counts match the official description, but byte identity with the official archive and the mirror's redistribution terms could not be established. URL-Phish v1 was downloaded from Mendeley Data (version 1, DOI `10.17632/65z9twcx3r.1`) on 8 September 2026 under CC BY 4.0; the ZIP and extracted CSV have SHA-256 `946c670c9f88721f5f770ceda28bab84931a275db020fd3db5ba411693591b60` and `d68b3cd0648dcf9c775347416ad1a8995e8a025921fbe3871ca6158d4db3c3a1`, respectively. The downloaded CSV contained 116,600 rows rather than the 111,660 described on its landing page, and this discrepancy is retained in the audit record. A validated anonymous reproducibility artifact has been assembled locally with code, acquisition instructions, source hashes, sanitized derived features, frozen split roles, repetition-level results, protocols, and issue logs. It excludes raw URL strings, plaintext host/domain identifiers, original archives, and trained binaries. Its public repository URL and the license for the authors' code and derived materials will be added before submission; the ISCX raw lists will not be redistributed unless their terms are established.

### Funding

This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.

### Conflict of Interest

The authors declare no conflicts of interest.

### Author Contributions

A CRediT contribution statement will be completed after the author list is finalized.

## References

[1] R. Zieni, L. Massari, and M. C. Calzarossa, "Phishing or Not Phishing? A Survey on the Detection of Phishing Websites," *IEEE Access*, vol. 11, pp. 18499-18519, 2023. https://doi.org/10.1109/ACCESS.2023.3247135.

[2] M. S. I. Mamun, M. A. Rathore, A. H. Lashkari, et al., "Detecting Malicious URLs Using Lexical Analysis," in *Network and System Security*, Cham: Springer, pp. 467-482, 2016. https://doi.org/10.1007/978-3-319-46298-1_30.

[3] A. Prasad and S. Chandra, "PhiUSIIL: A Diverse Security Profile Empowered Phishing URL Detection Framework Based on Similarity Index and Incremental Learning," *Computers & Security*, vol. 136, art. 103545, 2024. https://doi.org/10.1016/j.cose.2023.103545.

[4] C. Opara, Y. Chen, and B. Wei, "Look Before You Leap: Detecting Phishing Web Pages by Exploiting Raw URL and HTML Characteristics," *Expert Systems with Applications*, vol. 236, art. 121183, 2024. https://doi.org/10.1016/j.eswa.2023.121183.

[5] H. Le, Q. Pham, D. Sahoo, and S. C. H. Hoi, "URLNet: Learning a URL Representation with Deep Learning for Malicious URL Detection," arXiv:1802.03162, 2018. https://arxiv.org/abs/1802.03162.

[6] A. Mahesh, A. Nitheka, S. Shivvanii, et al., "An Explainable Machine Learning and Deep Learning Framework for Malicious URL Detection," *Procedia Computer Science*, vol. 283, pp. 1600-1609, 2026. https://doi.org/10.1016/j.procs.2026.06.236.

[7] F. Rashid, B. Doyle, S. C. Han, and S. Seneviratne, "Phishing URL Detection Generalisation Using Unsupervised Domain Adaptation," *Computer Networks*, vol. 245, art. 110398, 2024. https://doi.org/10.1016/j.comnet.2024.110398.

[8] L. Yi, A. Omotosho, and H. Balogun, "Phishing URL Detection and Interpretability With Machine Learning: A Cross-Dataset Approach," *Security and Privacy*, vol. 9, no. 1, art. e70175, 2026. https://doi.org/10.1002/spy2.70175.

[9] M. Mia, D. Derakhshan, and M. M. A. Pritom, "Can Features for Phishing URL Detection Be Trusted Across Diverse Datasets? A Case Study with Explainable AI," in *Proceedings of the 11th International Conference on Networking, Systems, and Security*, New York: ACM, pp. 137-145, 2024. https://doi.org/10.1145/3704522.3704532.

[10] T. Ahamed, S. C. Kakon, F. Al Farid, et al., "An Integrated Evaluation Protocol for Adversarial Robustness, Generalization, and Explanation Stability in URL-Based Phishing Detection," *Frontiers in Computer Science*, vol. 8, art. 1834407, 2026. https://doi.org/10.3389/fcomp.2026.1834407.

[11] T. Dalton, H. Gowda, G. Rao, et al., "PhreshPhish: A Real-World, High-Quality, Large-Scale Phishing Website Dataset and Benchmark," arXiv:2507.10854, 2025. https://arxiv.org/abs/2507.10854.

[12] S. Kaufman, S. Rosset, C. Perlich, and O. Stitelman, "Leakage in Data Mining: Formulation, Detection, and Avoidance," *ACM Transactions on Knowledge Discovery from Data*, vol. 6, no. 4, pp. 1-21, 2012. https://doi.org/10.1145/2382577.2382579.

[13] S. Kapoor and A. Narayanan, "Leakage and the Reproducibility Crisis in Machine-Learning-Based Science," *Patterns*, vol. 4, no. 9, art. 100804, 2023. https://doi.org/10.1016/j.patter.2023.100804.

[14] D. Arp, E. Quiring, F. Pendlebury, et al., "Dos and Don'ts of Machine Learning in Computer Security," in *31st USENIX Security Symposium (USENIX Security 22)*, pp. 3971-3988, 2022. https://www.usenix.org/conference/usenixsecurity22/presentation/arp.

[15] F. Pendlebury, F. Pierazzi, R. Jordaney, J. Kinder, and L. Cavallaro, "TESSERACT: Eliminating Experimental Bias in Malware Classification across Space and Time," in *28th USENIX Security Symposium (USENIX Security 19)*, pp. 729-746, 2019. https://www.usenix.org/conference/usenixsecurity19/presentation/pendlebury.

[16] S. M. Lundberg and S. I. Lee, "A Unified Approach to Interpreting Model Predictions," in *Advances in Neural Information Processing Systems 30*, pp. 4765-4774, 2017. https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html.

[17] S. M. Lundberg, G. Erion, H. Chen, et al., "From Local Explanations to Global Understanding with Explainable AI for Trees," *Nature Machine Intelligence*, vol. 2, no. 1, pp. 56-67, 2020. https://doi.org/10.1038/s42256-019-0138-9.

[18] D. Alvarez-Melis and T. S. Jaakkola, "On the Robustness of Interpretability Methods," arXiv:1806.08049, 2018. https://arxiv.org/abs/1806.08049.

[19] A. Ghorbani, A. Abid, and J. Zou, "Interpretation of Neural Networks Is Fragile," in *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 33, no. 1, pp. 3681-3688, 2019. https://doi.org/10.1609/aaai.v33i01.33013681.

[20] A. Warnecke, D. Arp, C. Wressnegger, and K. Rieck, "Evaluating Explanation Methods for Deep Learning in Security," in *2020 IEEE European Symposium on Security and Privacy*, pp. 158-174, 2020. https://doi.org/10.1109/EuroSP48549.2020.00018.

[21] A. Tolay, "Beyond Detection Accuracy: Measuring Explanation Cost, Stability, and Utility for Resource-Aware IoT Intrusion Detection," arXiv:2608.10349, 2026. https://arxiv.org/abs/2608.10349.

[22] S. S. Shafin, "An Explainable Feature Selection Framework for Web Phishing Detection with Machine Learning," *Data Science and Management*, vol. 8, no. 2, pp. 127-136, 2025. https://doi.org/10.1016/j.dsm.2024.08.004.

[23] M. C. Calzarossa, P. Giudici, and R. Zieni, "How Robust Are Ensemble Machine Learning Explanations?" *Neurocomputing*, vol. 630, art. 129686, 2025. https://doi.org/10.1016/j.neucom.2025.129686.

[24] M. C. Calzarossa, P. Giudici, and R. Zieni, "An Assessment Framework for Explainable AI with Applications to Cybersecurity," *Artificial Intelligence Review*, vol. 58, no. 5, art. 150, 2025. https://doi.org/10.1007/s10462-025-11141-w.

[25] T. Kehkashan, M. Abdelhaq, A. S. Al-Shamayleh, et al., "Explainable Phishing Website Detection for Secure and Sustainable Cyber Infrastructure," *Scientific Reports*, vol. 15, no. 1, art. 41751, 2025. https://doi.org/10.1038/s41598-025-27984-w.

[26] Public Suffix List community, "Public Suffix List," https://publicsuffix.org/list/index.html, accessed September 17, 2026.

[27] D. M. Linh and T. C. Hung, "A Feature-Engineered Dataset of Benign and Phishing URLs for Machine Learning and Large Language Models Evaluation," *Data in Brief*, vol. 63, art. 112162, 2025. https://doi.org/10.1016/j.dib.2025.112162. Dataset version used: https://doi.org/10.17632/65z9twcx3r.1.
