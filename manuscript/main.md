# Auditing Deployment Validity in Phishing URL Detection: Entity Independence, Threshold Portability, and Explanation Stability under Cross-Source Shift

**Dongze Li^1,*^ and Jiahui Liu^1^**  
^1^ School of Computer Science and Technology, Harbin University of Science and Technology, Harbin 150080, China  
*Corresponding author: Dongze Li, ldz0101fl@163.com  
Email: ldz0101fl@163.com (D.L.); liujiahui@hrbust.edu.cn (J.L.)  
ORCID: Dongze Li, 0009-0002-1583-4326

**Short title:** Deployment Validity of Phishing URL Detection

## Abstract

Phishing URL detectors are commonly validated on random splits, but deployment requires frozen rules to survive entity and source changes. We present a source-only security evaluation protocol that treats entity independence, score orientation, predictive transfer, threshold portability, and explanation stability as separate claims. Logistic regression, random forest, and XGBoost serve as controlled audit objects on two primary corpora and an independent third source, represented by the same 35 offline URL features. Across ten repetitions, registrable-domain separation reduced mean Macro-F1 by 0.038-0.109 on ISCX-URL2016 but by only 0.001-0.002 on PhiUSIIL. Source-validation thresholds constrained to 0.1% or 1% false-positive rate (FPR) yielded mean external FPR of 0.532-1.000 between the primary sources. On the complete 168,060-row PhreshPhish v1.0.1 test split, six frozen source-model families yielded mean ROC-AUC of 0.346-0.640 and FPR of 0.781-0.963 on prespecified domain-filtered cohorts; every PhiUSIIL-derived family ranked below chance. Yet logistic-regression explanations retained source-to-PhreshPhish Top-15 SHAP Jaccard agreement of 0.811 and 0.856. The complete six-edge graph among the three sources was asymmetric: PhreshPhish-trained models were near chance on ISCX (ROC-AUC 0.518-0.564) but stronger on PhiUSIIL (0.729-0.885), and PhreshPhish-to-PhiUSIIL was the only direction whose model-family mean low-FPR budget ratios approached or fell below unity. A secondary held-domain selector helped in one original transfer direction but not the other. High benchmark discrimination, repeatable explanations, and source-constrained thresholds provide non-interchangeable evidence; none alone establishes operational portability.

**Keywords:** phishing URL detection; data leakage; registrable domain; explainable artificial intelligence; explanation stability; domain shift; false-positive rate

**Practitioner points:**

- A low empirical false-positive rate on source validation data is a threshold-selection condition, not a population or deployment guarantee; the original transfer diagnostic produced 5,318-10,000 false alerts per 10,000 benign target URLs, and the third-source audit again produced high realized false-positive rates.
- URL, host, and registrable-domain overlap should be reported separately because each supports a different claim about recurring or unseen entities.
- High SHAP rank agreement can coexist with severe discrimination and threshold failure, so explanation transportability must be audited separately from predictive and operating-point transfer.

## 1. Introduction

Phishing URL detection is an operating-point problem. An email gateway, browser filter, or security operations workflow must act on newly observed URLs at a fixed threshold while keeping false alerts within a tolerable workload. Yet many published evaluations use random splits of public corpora and emphasize accuracy, F1 score, or ROC-AUC^[1-10]^. These quantities can characterize the sampled test set, but they do not establish that the same detector, threshold, or explanatory rationale remains valid for unseen entities or a new collection source.

Entity dependence is the first source of ambiguity. Corpora assembled from threat feeds, popular-site lists, search results, or web crawls may contain repeated URLs, several paths from one host, or several hosts under one registrable domain. A random split can distribute these related observations across training and test sets. Whether that overlap is leakage depends on the deployment claim: recurring domains may be expected in one application, whereas shared hosts invalidate an unseen-host test and shared `eTLD+1` entities invalidate an unseen-registrable-domain test^[11-15]^. Row-, URL-, host-, and registrable-domain-disjoint partitions should therefore be treated as different deployment estimands, not as a universal ladder from invalid to valid evaluation.

Ranking performance leaves a second question unresolved. ROC-AUC averages discrimination over possible thresholds, whereas a deployed system incurs the errors produced by one frozen rule. A threshold chosen to satisfy a low empirical false-positive-rate (FPR) budget on source validation data can violate that budget after the entity composition or collection source changes. Recalibrating with target labels would answer a different, target-supervised question. Operational portability must therefore be tested by freezing the source model and threshold before target evaluation and reporting the realized benign-alert workload.

Explanations create a third validity claim. SHAP can rank influential features for one fitted model, but one importance plot does not show whether that ranking survives model reseeding, grouped repartitioning, a change in deployment regime, or a new source^[16-20]^. Recent work has documented cross-dataset degradation, source-dependent feature behavior, and explanation instability^[7-10,21-25]^. Ahamed et al.^[10]^ is the closest identified protocol: it combines normalization, host-disjoint testing, external validation, adversarial robustness, leave-one-feature-out analysis, and explanation stability, while also identifying residual `eTLD+1` overlap. We extend that protocol by fixing the registrable-domain estimand, verifying score orientation, and freezing source-selected low-FPR thresholds, then compare the resulting conclusions with cross-source performance and explanation transportability.

We address this gap with a connected deployment-validity audit. The audit measures normalized-URL, host, and registrable-domain overlap across four internal deployment estimands, then freezes source models and thresholds before bidirectional transfer at empirical source-validation FPR budgets of 0.1% and 1%. Orientation and source-discrimination checks distinguish implementation failure from source-conditioned ranking reversal. Controlled SHAP comparisons isolate variation from model seed, group partition, deployment regime, and evaluation source. We then evaluate the frozen models and thresholds on the complete PhreshPhish v1.0.1 test split without target feedback and pair that test with a source-to-third-source SHAP analysis. A separate PhreshPhish-trained experiment provides a reciprocal directional diagnostic. The frozen outputs form a complete six-edge transfer graph among the three sources, which allows direct comparison of predictive transfer, threshold portability, and explanation stability. A prespecified held-domain analysis supplies a secondary evaluation of DCSS. Logistic regression, random forest, and XGBoost serve as conventional audit objects; the contribution is the linked validity protocol and its security interpretation.

This study addresses two primary research questions:

- **RQ1:** How do entity separation and source shift change predictive performance, calibration, score orientation, and the transfer of thresholds selected under low empirical source-validation FPR budgets?
- **RQ2:** How stable are SHAP feature rankings under controlled changes in model seed, group partition, deployment regime, and evaluation source, and what additional evidence is provided by holding out registrable domains during explanation aggregation?

Table: Research questions, evaluation scopes, prespecified comparisons, and result locations

| Question | Split or selection scope | Primary metric or diagnostic | Prespecified comparison | Result location |
|---|---|---|---|---|
| RQ1 | S0-S3 internal partitions, bidirectional S4 transfer, frozen evaluation on PhreshPhish, and the bounded six-edge directional synthesis; source-only model and threshold selection | Macro-F1, ROC-AUC, FPR, calibration, orientation, and source discrimination | Entity regimes; frozen source threshold versus realized source-test and target errors; directional transfer and budget portability | Sections 5.1, 5.3, 5.4, 5.6, and 5.7 |
| RQ2 | Fixed SHAP cohorts under seed, partition, regime, and source changes; outer-training-only held-domain folds | Top-15 Jaccard, rank correlation, SHAP-direction agreement, S3 Macro-F1, and external ROC-AUC difference | Controlled perturbation pairs; source-to-PhreshPhish explanation transport; F-DCSS-15 versus F-All internally and F-Stable externally | Sections 5.2, 5.5, 5.7, and 5.8 |

*Note.* RQ, research question; S0, row-random split; S1, normalized-uniform-resource-locator-disjoint split; S2, host-disjoint split; S3, registrable-domain-disjoint split; S4, zero-target-feedback cross-source transfer; Macro-F1, macro-averaged F1 score; ROC-AUC, area under the receiver operating characteristic curve; FPR, false-positive rate; SHAP, SHapley Additive exPlanations; F-DCSS-15, the 15-feature Domain-Conditioned SHAP Stability subset; F-All, the full 35-feature set; F-Stable, the random-resampling stability subset.

The main contributions are as follows:

1. We operationalize deployment validity as a vector of separately testable claims: entity alignment, score orientation, predictive transfer, frozen-threshold portability, and explanation stability. Each claim has a declared decision rule and evidence boundary.
2. Applying the audit reveals an operational failure that aggregate benchmark scores obscure. Entity-separation effects differ sharply by corpus, the same frozen models degrade severely on the independent PhreshPhish test split, and the complete six-edge graph among the three evaluated sources shows that both discrimination and low-FPR portability are directional rather than source-invariant.
3. We show that explanatory and predictive validity can disagree. High source-to-PhreshPhish SHAP rank agreement coexists with poor discrimination and high FPR. DCSS remains a bounded secondary result: it helps in one original transfer direction, fails in the other, and does not support a general selector-superiority claim.

## 2. Related Work

### 2.1 Machine Learning for Phishing URL Detection

Phishing website detection commonly uses list matching, page-similarity analysis, or machine-learning-based classification^[1]^. List-based methods have low inference costs but cannot promptly cover newly registered domains and short-lived attacks. Page-similarity methods detect impersonation from visual appearance, HTML structure, or brand information. Although page-similarity methods use rich evidence, they require active page access and can be disrupted by unavailable pages, network delays, and content obfuscation. Machine-learning methods learn classification rules from URLs, HTML content, domain registration information, and network attributes.

Mamun et al.^[2]^ introduced the ISCX-URL2016 dataset and showed that low-cost lexical analysis can distinguish benign, phishing, spam, malware, and defacement URLs. Their results indicate that the URL string itself contains useful detection signals. URLNet, proposed by Le et al.^[5]^, learns URL representations with character-level and word-level convolutional networks and reduces manual feature engineering. Huang et al.^[31]^ combined RoBERTa URL representations with a sparse autoencoder trained on benign samples, illustrating the recent move toward learned URL representations. Opara et al.^[4]^ developed WebPhish, which jointly processes raw URLs and HTML content to extract phishing indicators from complementary information sources. Prasad and Chandra^[3]^ created the PhiUSIIL dataset and combined URL similarity, HTML features, and incremental learning to detect attacks involving homographs and combosquatting. Mahesh et al.^[6]^ combined recursive feature elimination with cross-validation, XGBoost, SHAP, and LIME for malicious URL detection, linking feature selection with model explanation.

These studies have improved predictive modeling for phishing URL detection. Many evaluations, however, still rely mainly on random within-dataset splits and use accuracy or F1 score as their primary evidence. We use representative classifiers to isolate the effects of data partitioning, frozen thresholds, and explanation stability.

### 2.2 Dataset Bias and Cross-Dataset Generalization

Phishing URL datasets vary in collection source, time span, class balance, and feature definition. In cross-dataset experiments on three URL datasets, Rashid et al.^[7]^ found that models with strong within-dataset performance degraded markedly on a different source. Their unsupervised domain-adaptation analysis, which included distributional differences in URL path length, showed that data collection and representation contribute to cross-dataset degradation alongside classifier capacity. Yi et al.^[8]^ evaluated random forest, XGBoost, and LightGBM across several URL datasets and argued for multi-dataset validation. Mia et al.^[9]^ compared shared features in two phishing datasets and found source-dependent contributions: features with the same name and semantics could still have different distributions and SHAP rankings.

Ahamed et al.^[10]^ proposed an integrated evaluation protocol covering URL normalization, host-disjoint splitting, cross-dataset generalization, adversarial perturbation, and explanation stability. Their audit nevertheless found 1,387 shared `eTLD+1` entities between the training and test data after host separation, plus further overlap with the validation set. Different hostnames therefore need not represent unseen administrative-domain contexts: `a.example.com` and `b.example.com`, for instance, share the registrable domain `example.com`. This residual dependence motivates our direct comparison of host-disjoint and registrable-domain-disjoint estimands on the same corpus.

Lo Re et al.^[32]^ used heuristic search to generate problem-space evasions against phishing webpage detectors. That work evaluates deliberate adversarial manipulation; our source-shift analysis evaluates frozen models on independently collected, unmodified target corpora.

PhreshPhish introduced a newly collected large-scale corpus and benchmark variants designed to reduce leakage, vary task difficulty, and represent realistic phishing base rates^[11]^. Its contribution centers on dataset and benchmark construction. Using established public corpora, we study entity-conditioned evaluation, explanation repeatability, and frozen-threshold portability within one source-only protocol.

Recent JISA studies frame distribution shift mainly as an adaptation problem. Escudero García and DeCastro-García^[28]^ studied conformal labelling and online model updating under concept drift, while Rustam and Jurcut^[29]^ combined ensemble drift detection, incremental learning, and model explanation in multi-environment networks. Those studies address adaptation after a change is observed. We examine the evidence available earlier, when the source model and threshold are frozen and target labels, updating, and recalibration are unavailable.

### 2.3 Data Leakage in Security Machine Learning

Kaufman et al.^[12]^ defined data leakage as using target-related information during training that would be unavailable at prediction time. They described how leakage arises and how researchers can detect and prevent it. Kapoor and Narayanan^[13]^ argued that leakage may enter during data collection, preprocessing, modeling, or evaluation, inflating performance and reducing reproducibility in machine-learning-based science.

Dependence and evaluation bias are especially consequential in security machine learning. The TESSERACT framework of Pendlebury et al.^[15]^ distinguishes spatial from temporal bias in malware classification and shows how partitions that do not match deployment conditions can substantially overestimate model capability. Arp et al.^[14]^ analyzed recurring experimental pitfalls in security machine learning and showed that dataset construction, feature design, train-test splitting, and evaluation metrics can pull conclusions away from the underlying security problem. Although neither study focuses specifically on phishing URLs, both support the same requirement: training and test data must be independent in a way that matches the task, and preprocessing and feature selection must use training data alone.

We use the conventional definition of preprocessing leakage: information from validation or test data affects fitting, tuning, threshold selection, or feature selection. Entity overlap requires a claim-specific interpretation. We call it related-sample leakage only when the overlap conflicts with the stated deployment target, as shared hosts would in an unseen-host evaluation. Cross-dataset distribution shift is not itself leakage; we evaluate it separately as evidence of dependence on source-specific distributions and collection procedures.

### 2.4 SHAP-Guided Selection and Explanation Stability

Lundberg and Lee^[16]^ introduced SHAP as a Shapley-value framework for attributing model predictions, and Lundberg et al.^[17]^ developed efficient explanations for tree models. Phishing studies already use SHAP to guide feature selection. Mahesh et al.^[6]^ combined RFECV, XGBoost, SHAP, and LIME for malicious URL detection. Shafin et al.^[22]^ proposed an explainable feature-selection framework for web phishing detection, while Kehkashan et al.^[25]^ applied SHAP-based selection to several supervised classifiers. These methods demonstrate explanation-guided reduction, although their main selection evidence comes from one dataset or one fitted ranking. We evaluate whether a subset repeatedly selected from training-only runs remains useful when the model seed, grouped partition, deployment regime, or data source changes.

In adjacent security work, Najibi and Bidgoly^[30]^ used explainable deep learning to identify robust features for Android malware detection. Our study tests the separate question of whether explanation rankings remain stable when a frozen model is moved between URL sources.

Explanation robustness requires its own evaluation. Alvarez-Melis and Jaakkola^[18]^ treated robustness as a desirable property of interpretation methods, and Ghorbani et al.^[19]^ showed that small input changes can alter explanations substantially without changing the predicted label. For security applications, Warnecke et al.^[20]^ argued that assessments should also consider completeness, efficiency, and robustness. Tolay^[21]^ studied SHAP cost, local stability, and selective explanation in intrusion detection, while Calzarossa et al.^[23,24]^ included explanation robustness in a broader assessment. A stable ranking can still reproduce source bias, and correlated features can exchange ranks without a meaningful change in model behavior. We therefore report separate stability estimands, fixed explainer settings, uncertainty intervals, and external-source diagnostics while reserving faithfulness and causal claims.

Existing work has moved beyond within-dataset accuracy toward deployment-aligned partitions, external-source transfer, and explicit assessment of explanations. We examine performance under registrable-domain separation, the response of feature rankings to different perturbations, and the evidence added by holding out registrable domains during SHAP aggregation compared with repeated random-resampling stability. The negative and directional results define when explanation stability fails to support a transfer claim.

Table: Comparison with the closest prior studies

| Study | Entity-aware split | Cross-source test | Frozen threshold audit | Score-orientation audit | Explanation analysis | Distinctive scope |
|---|---|---|---|---|---|---|
| Rashid et al.^[7]^ and Yi et al.^[8]^ | Not the central contribution | Yes | No linked low-FPR transfer diagnosis | Not reported as a separate audit | Limited or not central | Cross-dataset degradation and adaptation/multi-dataset evaluation |
| Ahamed et al.^[10]^ | Host-disjoint, with residual registrable-domain overlap audited | Yes | No source-selected 0.1%/1% FPR transfer audit identified | Not reported as a separate audit | Yes | Integrated normalization, external validation, perturbation, and explanation protocol |
| PhreshPhish^[11]^ | Leakage-reducing benchmark variants | Benchmark-centered | Realistic base-rate motivation | Not the central contribution | Not the central contribution | New corpus and benchmark-construction pipeline |
| This study | URL-, host-, and registrable-domain-disjoint estimands | Bidirectional between primary sources plus independent frozen evaluation on PhreshPhish | Yes; interpreted only as frozen-threshold transfer failure | Yes; probability, label, complement, and negation identities | Seed, partition, regime, source, and held-domain diagnostics | Linked separation of entity, decision, ranking, and explanation validity |

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

where \(t\) is the classification threshold. We use fixed conventional classifiers to study model performance and explanation behavior under different leakage-control conditions.

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

The deployment-validity audit has five evidence stages aligned with the two primary research questions. Protocol integrity comes first: data acquisition, label harmonization, conservative URL normalization, deterministic feature extraction, and explicit URL, host, and registrable-domain mappings precede four deployment-conditioned partitions. Operational transfer then freezes source-trained models and thresholds before bidirectional external evaluation. Deduplication, entity-equal weighting, calibration, score-orientation checks, and false-positive structure test alternative accounts of transfer failure. The SHAP analysis separates seed, partition, regime, and source variation and compares compact training-only feature sets. The independent-source stage applies the frozen models and thresholds to the PhreshPhish test split and compares source and target explanations; a separately trained reciprocal experiment diagnoses transfer direction. The final stage evaluates DCSS as a secondary held-domain diagnostic with an auxiliary internal margin comparison, a directional transfer comparison, component ablations, random same-size subsets, and an exploratory URL-Phish sensitivity analysis.

All fitting, tuning, threshold selection, SHAP aggregation, and feature selection remain within the declared source training and validation scope. The framework tests a sequence of claims: entity-conditioned performance establishes the estimand, frozen transfer tests whether the evidence survives a new source, explanation analysis measures the portability of model reasoning, and DCSS tests whether held-domain explanations add information beyond random-resampling stability. The endpoints and scoring rules remain fixed after target inspection, including when results are unfavorable.

### 4.2 Datasets

The primary two-source experiments use the UCI release of PhiUSIIL and a third-party preserved copy corresponding to ISCX-URL2016. Throughout the manuscript, `ISCX-URL2016` denotes this preserved copy unless the official dataset description is explicitly discussed. PhiUSIIL contains 235,795 observations and 54 provided features, including 134,850 legitimate and 100,945 phishing observations^[3]^. Its original label convention is 1 for legitimate and 0 for phishing; this study explicitly recodes it to 0 for benign and 1 for phishing. The official ISCX-URL2016 description lists benign, phishing, spam, malware, and defacement categories^[2]^. For the binary task, the preserved copy's benign URLs are coded 0 and phishing URLs are coded 1, while the other malicious categories are excluded.

Because the original feature sets differ in number and definition, provided features are not concatenated or aligned by name. The same lexical and structural variables are recalculated from raw URLs in both datasets. Dataset provenance is treated as a potential confounder: ISCX benign URLs are largely derived from Alexa and its phishing URLs from OpenPhish, while PhiUSIIL uses a different collection pipeline^[2,3]^. External testing is therefore described as external-source transfer, not as pure real-world generalization. S4 estimates source-conditioned incompatibility between the evaluated benchmark construction pipelines; it does not estimate the magnitude or frequency of drift in arbitrary live deployments.

PhreshPhish v1.0.1 provides the independent third source^[11]^. The pinned revision is commit `eabec4b7a66324b79cc8a0ad856d1731dc26fe1a`. Its complete test split contains 168,060 URLs (91,260 benign and 76,800 phishing), and its training split contains 498,255 URLs. The original train/test roles are preserved. For the primary independent evaluation, only the test split is used: models, hyperparameters, feature sets, and thresholds remain frozen from ISCX or PhiUSIIL, and no PhreshPhish label is used for fitting, selection, calibration, or threshold adjustment. For the reciprocal diagnostic, models are trained only on the PhreshPhish training split and then applied to the already studied ISCX and PhiUSIIL targets. This reverse direction is not counted as another independent source or as a second independent validation.

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

Three models are used: logistic regression, random forest, and XGBoost. Logistic regression provides a simple linear baseline; random forest represents a bootstrap-aggregation ensemble; and XGBoost represents gradient-boosted decision trees. These models cover different levels of complexity and learning mechanisms and are supported by mature SHAP explainers. Restricting the comparison to these conventional families helps isolate the effects of entity separation and explanation stability from architectural differences.

For each candidate configuration, a decision threshold is first selected on the validation set from 0.05 to 0.95 in increments of 0.01 by maximizing validation Macro-F1. The resulting candidate-threshold pairs are then compared by validation Macro-F1, with validation ROC-AUC and candidate order used as deterministic tie breakers. Logistic regression uses `StandardScaler`, the `lbfgs` solver, and a maximum of 2,000 iterations; tree candidates fix their estimator counts, depth/leaf controls, subsampling, and class weighting. All nine exact candidate definitions and fixed parameters are provided in Supplementary Data S2 (`supplementary_model_candidates.json`). Large-scale AutoML is not used. Ten prespecified partition and model seeds are evaluated for every internal regime. Data indices, entity assignments, selected hyperparameters, thresholds, predicted probabilities, metrics, and SHAP summaries are retained for reproducibility.

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

The configurations are compared in terms of Macro-F1, recall, false-positive rate, ROC-AUC, PR-AUC, training time, per-sample inference time, and SHAP computation time. In S4, F-Single and F-Stable are built using only the source corpus; target SHAP values are diagnostic and cannot alter either set. The resulting feature sets are reproducible screening rules with fixed construction procedures.

### 4.9 Domain-Conditioned SHAP Stability

DCSS is calculated separately for each dataset, outer S3 repetition, and model using only the outer training subset. It rewards a feature that remains important, repeatedly enters the Top-k set, preserves the direction of its value-SHAP association, and avoids large rank swings when entire registrable domains are withheld from fitting. The four-factor multiplicative score is a prespecified diagnostic whose contribution is tested through ablation.

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

The outer repetition is the primary uncertainty unit. The ten repetitions are paired partition perturbations of each fixed corpus, not ten independently collected datasets. Mean effects, intervals, all repetition-level paired differences, and directional consistency are therefore the primary evidence; they quantify sensitivity to the evaluated partitions of fixed corpora rather than population uncertainty over independently collected future sources. The prespecified auxiliary internal comparison contrasts F-DCSS-15 with F-All using S3 Macro-F1 and a finite-corpus tolerance margin of 0.01. This margin encodes the study-design tolerance that a 15-feature representation may lose at most one Macro-F1 percentage point relative to all 35 features; it is not an established clinical or operational standard. For each dataset-model group, a one-sample, one-sided t-test evaluates whether the mean paired repetition-level difference exceeds the -0.01 margin, with Benjamini-Hochberg adjustment across the six internal tests. The prespecified external comparison contrasts F-DCSS-15 with F-Stable using ROC-AUC averaged over the three model families within each repetition. Its p value is the two-sided empirical tail probability around zero from the 5,000-replicate domain-plus-outer distribution, with Benjamini-Hochberg adjustment across the two transfer directions. These p-values and their pass/fail labels are auxiliary finite-corpus summaries, not formal non-inferiority conclusions for a population of future sources. The multiplicity adjustment controls the reported test family but does not remove dependence created by overlapping repetitions.

For each transfer direction, the external estimand is the unweighted mean F-DCSS-15-minus-F-Stable ROC-AUC difference across the three prespecified model families and ten outer repetitions. Within each model and repetition, we subtract the paired empirical AUC influence values of the two methods and sum those differences by target registrable domain. The three model-specific domain contributions are then averaged with equal weight, consistent with an estimand that gives the prespecified model families equal standing rather than weighting them by target sample size or observed performance. Domains absent after filtering in a repetition contribute zero to that repetition's covariance column. The resulting domain-by-repetition matrix defines the covariance of a joint Gaussian perturbation, so target domains that recur across repetitions induce cross-repetition covariance while the within-run pairing of the two feature methods is preserved. Each of 5,000 replicates independently resamples the ten outer repetitions and averages the selected observed differences plus their jointly drawn domain perturbations. We call the resulting percentile interval the *domain-plus-outer interval*. This linearized two-level approximation was prespecified in analysis protocol version 1.0.1 before interval calculation; it is not a literal nested cluster bootstrap and does not describe future collection processes. An outer-only bootstrap and a dependence-stress analysis are retained as sensitivities. The latter inflates standard errors under assumed positive pairwise correlations among repetitions; it does not estimate the true dependence or convert repeated partitions into independent datasets. Pairwise SHAP similarities and source-only diagnostic correlations share fitted runs and are treated descriptively.

One robustness evaluation changes the weighting unit. In the registrable-domain-equal analysis, every domain contributes total weight one, and each URL in domain \(g\) receives weight \(w_i=1/n_g\). Weighted confusion components yield domain-equal Macro-F1, balanced accuracy, recall, and false-positive rate. A separate focused 2,000-replicate domain-cluster bootstrap is reported for the prespecified XGBoost repetition 00 F-All/F-Stable comparison in both transfer directions; it does not represent all model-selection uncertainty.

External calibration is measured using Brier score, log loss, and expected calibration error (ECE) with 15 fixed equal-width probability bins. The frozen source-validation threshold is the only threshold eligible for zero-target-feedback evaluation, but it is not presumed deployable. An exact target-oracle Macro-F1 threshold is calculated after prediction solely to quantify threshold-transfer regret. Because this oracle uses target labels, it is diagnostic and is never used to refit a model or support an operational performance claim.

### 4.11 Frozen-Threshold Transfer-Failure Diagnostic

To diagnose whether an apparently low source-validation FPR survives an entity or source change, we add two frozen operating points for F-All in the S3 setting. For each dataset, outer repetition, and model, the already selected model candidate is refitted on the original source-training partition. Using only source-validation labels and probabilities, we choose the threshold that maximizes validation true-positive rate subject to an empirical FPR no greater than 0.1% or 1%. Ties are resolved in favor of the lower threshold among operating points with the same true-positive rate. These low-FPR operating points are selected from the complete ROC threshold sequence derived from the distinct source-validation scores (`drop_intermediate=False`) and are independent of the 0.05-0.95 grid used for the general Macro-F1 operating point in Section 4.6. The selected threshold is then applied unchanged to the source S3 test partition and the primary domain-filtered external target cohort. The budgets are empirical threshold-selection rules for the finite validation cohorts.

Target labels are used only to report true- and false-positive rates after prediction; they do not affect model selection, threshold selection, calibration, or feature selection. For each source-validation cohort, the audit retains integer false positives and one-sided 95% Wilson and exact Clopper-Pearson upper bounds. Because the same validation labels determine both the threshold and these bounds, the bounds are post-selection descriptions of the frozen validation cohorts. They do not account for the threshold search and do not provide selection-adjusted or population-level confidence guarantees. The audit also applies the immediately safer and more permissive adjacent score thresholds and repeats threshold selection on a registrable-domain-grouped audit holdout. This grouped holdout isolates threshold-selection reuse, although the model candidate was still selected using the full validation cohort. In addition to rates, we report false alerts per 10,000 benign URLs as \(10{,}000\times\mathrm{FPR}\). These corpus-normalized counts do not estimate production prevalence. The deterministic replay reproduced the original 120 thresholds to a maximum absolute difference of \(2.22\times10^{-16}\).

### 4.12 Independent Third-Source and Reciprocal Evaluation

The independent third-source evaluation reuses the 60 frozen S3 F-All models from the two primary sources (two sources, ten repetitions, and three model families). The same 35 features are recomputed from stored PhreshPhish URL strings, with no page retrieval or live network features. Each frozen model is applied to the complete PhreshPhish test split and to a primary cohort that excludes every target `eTLD+1` observed in the corresponding source training or validation set. We report row- and domain-equal metrics under the source-selected Macro-F1 threshold and the source-selected 0.1% and 1% validation-FPR thresholds. Stored class order, the phishing-probability column, source-test replay, probability ranges, sample identifiers, and the AUC complement and score-negation identities are checked before interpretation.

The paired explanation audit uses the same frozen model in its source and PhreshPhish contexts. For each source direction and model run, deterministic class-balanced cohorts of 200 source-test and 200 target examples are explained with the original explainer settings. Global mean-absolute-SHAP rankings, Top-15 Jaccard similarity, full-ranking Spearman correlation, and Top-20 contribution-direction agreement are compared. Performance loss is reported beside explanation agreement, but their association is descriptive because the 60 comparisons share datasets, model families, and fixed target cohorts. Direction conflicts identify source- and model-conditioned attribution changes; they do not establish that a feature caused the performance loss.

The reciprocal diagnostic uses ten registrable-domain-grouped 70/15/15 partitions of the complete 498,255-row PhreshPhish training split. LR, RF, and XGBoost use the same candidate grids and seed schedule as the primary study, with candidate and threshold selection confined to PhreshPhish training and validation data. Each of the 30 selected models is applied without target feedback to the master and deduplicated ISCX and PhiUSIIL views, both before and after target-domain overlap exclusion. The diagnostic preserves 150 prediction files: 30 source-test files and 120 target files. All target identifiers, probability ranges, file hashes, design combinations, and 120 source-fit-domain versus primary-target isolation checks are audited. Because the targets were already used in the original study, this experiment tests reciprocity and directionality only.

### 4.13 Directed Transfer and Validity-Disagreement Synthesis

To compare the directionality of the audited claims without introducing a new experiment, we derive a six-edge source-to-target synthesis from frozen run-level outputs. The graph contains every directed edge among ISCX-URL2016, PhiUSIIL, and PhreshPhish. Each edge uses the common 35-feature representation, LR, RF, and XGBoost, ten repetitions, row weighting, and the primary cohort after source-fit-domain exclusion. ROC-AUC is summarized under the ordinary source-selected Macro-F1 threshold as a threshold-independent ranking measure. Operating-point portability is summarized separately at the source-selected 0.1% and 1% validation-FPR thresholds.

For a nominal source-validation FPR budget \(b\in\{0.001,0.01\}\), the target budget-exceedance ratio is \(r=\mathrm{FPR}_{\mathrm{target}}/b\). The denominator is the declared nominal budget. Ratios are first averaged within each direction-model cell across ten repetitions and then displayed as the range across the three model families. The ISCX/PhiUSIIL edges are retained from the primary two-source analysis; the two source-to-PhreshPhish edges are the primary third-source additions; and the PhreshPhish-trained reverse edges remain secondary reciprocal diagnostics. Training sizes, source distributions, and selected hyperparameters differ across sources, so the direction contrasts are descriptive. The explanation-performance panel uses the 60 paired source-to-PhreshPhish comparisons and reports a descriptive Spearman association separately from the six-edge performance graph.

## 5. Results

### 5.1 Protocol Integrity and Entity-Conditioned Performance

RQ1 begins by changing the deployment population while keeping preprocessing and audit rules fixed. We record the reproducibility environment, verify entity separation, and compare internal performance across the four entity regimes.

**Reproducibility environment.**

- Python 3.9.13, NumPy 1.26.4, pandas 2.2.3, scikit-learn 1.5.2, XGBoost 2.1.4, SHAP 0.46.0, and tldextract 5.1.3;
- Windows 10 Pro (64 bit), AMD Ryzen 5 5600H (6 cores/12 logical processors), and 16 GB RAM;
- ten partition seeds beginning at 20260902 and ten model seeds from 20261001 to 20261010; and
- a frozen local Public Suffix List cache with network updates disabled during preprocessing.

All raw URLs remained offline. Intermediate assignments use sample and entity hashes, and every completed stage includes machine-readable metrics, predicted probabilities, logs, and SHA-256 manifests.

**Corpus audit.**

PhiUSIIL contained 235,795 input rows, including 134,850 benign and 100,945 phishing URLs. Normalization identified 1,138 duplicate rows beyond the first occurrence and one conflicting normalized-URL group containing two rows. Removing only that conflict produced a master corpus of 235,793 rows. The binary ISCX-URL2016 corpus contained 45,343 rows, including 35,378 benign and 9,965 phishing URLs. It contained 119 duplicate rows beyond the first occurrence and no conflicting-label group.

Table: Corpus audit after normalization and conflict removal

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

Table: Mean Macro-F1 and false-positive rate under S0 and S3 (10 repetitions)

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

RQ2 separates controlled sources of explanation variation and then tests whether repeated training-only selection yields a compact model. Compactness and transferability are evaluated as distinct outcomes.

**Controlled SHAP stability.**

Figure 2 and Table 5 separate four sources of variation. Model-seed changes produced the greatest agreement. LR was deterministic under the fixed split and cohort (Top-15 Jaccard and rank Spearman both 1.000); tree-model seed Jaccard remained 0.876-0.939. Partition changes reduced Top-15 Jaccard to 0.798-0.920, although full-ranking Spearman correlations remained 0.951-0.989.

![Figure 2. Top-15 Jaccard and full-ranking Spearman agreement for SHapley Additive exPlanations (SHAP) under seed, partition, deployment-regime, and explanation-source variation. For each model and estimand, boxes pool the declared pairwise comparisons from both corpora; boxes show the median and interquartile range, whiskers extend to 1.5 times the interquartile range, and outliers are omitted from display.](figures/Figure_2.pdf)

Regime and source changes produced larger set-level differences. Regime Top-15 agreement ranged from 0.704 to 0.897, while source agreement ranged from 0.623 to 0.788 for RF/XGBoost and from 0.719 to 0.730 for LR. Source rank correlations ranged from 0.810 to 0.941. A reproducible ranking under reseeding is therefore weak evidence that the same leading features describe a different deployment regime or source.

Table: Range of SHAP agreement across datasets and models

| Stability estimand | Top-15 Jaccard range | Full-rank Spearman range | Interpretation |
|---|---:|---:|---|
| Model seed | 0.876-1.000 | 0.986-1.000 | Highest agreement |
| Group partition | 0.798-0.920 | 0.951-0.989 | Moderate set turnover |
| Evaluation regime | 0.704-0.897 | 0.848-0.980 | Split semantics matter |
| Explanation source | 0.623-0.788 | 0.810-0.941 | Largest source sensitivity |

*Note.* SHAP, SHapley Additive exPlanations; Top-15 Jaccard, Jaccard similarity between the sets of the 15 highest-ranked features; full-rank Spearman, Spearman rank correlation across all 35 features.

**Stable feature sets and within-source compactness.**

F-Stable retained 12.3-13.5 of 35 features on ISCX and 13.8-14.3 on PhiUSIIL, a reduction of 59.1%-64.9% (Figure 3 and Supplementary Table S4). Compactness did not imply universal predictive improvement. On PhiUSIIL, F-Stable differed from F-All by -0.0087 for LR, +0.0002 for RF, and -0.0002 for XGBoost. On ISCX, the differences were -0.0445, -0.0311, and +0.0340, with substantial repetition-level dispersion.

![Figure 3. Feature-set size and internal registrable-domain-disjoint macro-averaged F1 score (Macro-F1) relative to the full 35-feature model (F-All). Bars show mean retained features. Boxes pool paired differences over three models and ten outer repetitions within each corpus; boxes show the median and interquartile range, whiskers extend to 1.5 times the interquartile range, and outliers are omitted from display. The dashed line marks no change from F-All.](figures/Figure_3.pdf)

The conventional controls qualify the feature-selection claim. Across 60 dataset-model-repetition comparisons, F-MI was below F-All in 52 cases. F-Permutation was below F-All in 37, above it in 21, and tied in two. F-Stable exceeded F-MI in 48 of 60 internal comparisons but exceeded F-Permutation in only 26 and was lower in 34. On PhiUSIIL, the F-Stable advantage over F-MI was clearest for RF and XGBoost (mean differences 0.0204 and 0.0137; two-sided Wilcoxon signed-rank p=0.0074 for both after BH adjustment across all 144 internal dataset-model-comparison-metric tests). These exploratory paired results support the observed contrast within the evaluated repetitions; F-Stable delivered compact models and outperformed model-independent MI more often in these settings, while model-specific permutation selection remained competitive.

### 5.3 Cross-Source Operational Failure

For the operational part of RQ1, we freeze the source-side pipeline before changing the corpus. Predictive transfer is examined alongside duplicate removal, entity weighting, calibration, score orientation, frozen-threshold transfer, and false-positive concentration.

**Bidirectional zero-target-feedback transfer.**

All five feature schemes failed severely under zero-target-feedback transfer (Figure 4 and Supplementary Table S5). From ISCX to the domain-filtered PhiUSIIL cohort, mean Macro-F1 across models and repetitions ranged from 0.2908 to 0.2922. In the reverse direction, it ranged from 0.1703 to 0.1937. Mean row-level false-positive rates were approximately 1.000 and 0.978-1.000, respectively. For F-All, the external-minus-internal Macro-F1 gaps were -0.6150 and -0.7985.

![Figure 4. Zero-target-feedback transfer performance for five source-selected feature schemes. Points are descriptive means over three models and ten outer repetitions, and error bars are bootstrap 95% intervals over those model-repetition values. Blue and red circles denote transfer from the ISCX-URL2016 corpus to the PhiUSIIL corpus and the reverse direction, respectively. Both vertical axes are truncated to resolve differences within the observed ranges.](figures/Figure_4.pdf)

Feature selection did not resolve the failure. F-Stable exceeded F-MI in 50 of 60 external comparisons but exceeded F-Permutation in only 27 and was lower in 31. F-MI was lower than F-All in 54 of 60 comparisons. The association between outer-split feature-set stability and external Macro-F1 was also not positive: Spearman rho was 0.19 (nominal unadjusted p=0.554) for ISCX-trained models and -0.37 (nominal unadjusted p=0.241) for PhiUSIIL-trained models. These 12-point source analyses are descriptive and underpowered, and they provide no evidence that repeatability predicts transfer.

Several configurations also produced ROC-AUC below 0.5, meaning that the learned score ranked target examples in the wrong direction. A 300-configuration orientation audit ruled out an inverted label or probability-column implementation: all class orders were `[0,1]`, stored decisions used the phishing-probability column, probability reconstruction differed by less than \(3\times10^{-8}\), and \(\mathrm{AUC}(\hat p)+\mathrm{AUC}(1-\hat p)=1\) held within \(5.09\times10^{-6}\). For PhiUSIIL-to-ISCX XGBoost, ordinary AUC ranged from 0.0745 to 0.2555 across feature methods, whereas the target-label diagnostic \(\mathrm{AUC}(1-\hat p)\) ranged from 0.7445 to 0.9255. The latter values expose score reversal; they are not corrected model performance.

The mechanism audit associated this reversal with source-conditioned feature and class distributions. With a conservative Spearman dead zone of \(|\rho|<0.02\), feature-label direction reversed for 20/35 features from PhiUSIIL to ISCX and 19/35 in the opposite comparison. A logistic source classifier evaluated with registrable-domain-grouped splits distinguished the corpora with mean AUC 1.0000 among benign URLs and 0.9129 among phishing URLs, showing that source differences persist within class. Across fixed external explanation cohorts, Top-10 target SHAP/label-direction conflict ranged from 38% to 80%. We describe this pattern as *source-conditioned ranking reversal associated with class-conditional covariate and association shift*. Source remains confounded with time, curation, URL status, and sampling policy, so the audit cannot isolate a single collection mechanism.

**Duplicate-removal sensitivity.**

The fully deduplicated analysis repeated F-All training for S0 and S3 with both datasets, all three models, and all ten fixed repetitions (120 matched runs). Removing repeated normalized URLs barely changed S0 estimates: mean paired Macro-F1 differences ranged from -0.0004 to +0.0004. S3 differences were also small for most combinations. The largest mean was +0.0187 for ISCX RF, but its 95% repetition-bootstrap interval was -0.0288 to 0.0635; ISCX XGBoost was +0.0027 (-0.0390 to 0.0418). PhiUSIIL LR increased by 0.0019 (0.0007 to 0.0031), while the remaining PhiUSIIL S3 changes were within 0.0003 in absolute mean.

These matched results do not support duplicate rows as the main explanation for the ISCX S3 instability. The more plausible measured factor is which heterogeneous, high-volume registrable domains enter each grouped test partition.

Table: Mean paired Macro-F1 change after full deduplication

| Dataset | Regime | LR | RF | XGBoost |
|---|---|---:|---:|---:|
| ISCX-URL2016 | S0 | +0.0001 | +0.0004 | -0.0003 |
| ISCX-URL2016 | S3 | -0.0001 | +0.0187 | +0.0027 |
| PhiUSIIL | S0 | -0.0002 | -0.0004 | -0.0001 |
| PhiUSIIL | S3 | +0.0019 | -0.0002 | +0.0003 |

*Note.* Macro-F1, macro-averaged F1 score; S0, row-random split; S3, registrable-domain-disjoint split; LR, logistic regression; RF, random forest. ISCX-URL2016 and PhiUSIIL are the two evaluated phishing web-address corpora.

**Entity weighting and calibration diagnostics.**

Row-level and domain-equal metrics answer different deployment questions. Under S3 internal evaluation, domain-equal weighting changed Macro-F1 by -0.180 to +0.051 across the five feature methods and three models, confirming that URL-rich domains can materially affect row-level summaries. Under external transfer, ISCX-to-PhiUSIIL domain-equal Macro-F1 was 0.191-0.197, below the row-level range. In the reverse direction it increased to 0.481-0.529 because numerous small phishing-associated domains received the same total weight as a few large benign domains. This higher Macro-F1 does not indicate usable benign discrimination: domain-equal FPR still ranged from 0.948 to 1.000. In the representative XGBoost repetition 00 domain-cluster bootstrap, F-All and F-Stable intervals overlapped closely in both directions (0.1955-0.1980 versus 0.1956-0.1983, and 0.4801-0.4841 versus 0.4799-0.4840).

Calibration results further separate threshold mismatch from transferable discrimination (Figure 5 and Supplementary Figure S1). Across the five methods, mean 15-bin ECE was 0.581-0.592 from ISCX to PhiUSIIL and 0.747-0.794 in the reverse direction. Mean Brier scores were 0.580-0.587 and 0.745-0.794. The complete reliability diagrams show that predicted probabilities cluster at extreme values rather than following the identity line. Exact post hoc target-oracle thresholds improved Macro-F1 by 0.074-0.292 and 0.202-0.333, respectively, but most selected thresholds were at or near 1.0 and often reduced to an almost all-negative decision. The oracle therefore demonstrates severe source-threshold incompatibility, not a deployable correction; substantial ranking failure remained for most configurations.

![Figure 5. Difference between the macro-averaged F1 score (Macro-F1) obtained by a post hoc target-oracle threshold and the Macro-F1 obtained by the frozen source-selected threshold. Boxes summarize ten outer repetitions; boxes show the median and interquartile range, whiskers extend to 1.5 times the interquartile range, and outliers are omitted from display. The oracle is diagnostic only and is never used for model fitting or a deployment claim.](figures/Figure_5.pdf)

**Frozen-threshold transfer failure.**

The validation constraints were satisfied by construction: mean validation FPR was 0.00092-0.00099 at the 0.1% budget and 0.00949-0.00999 at the 1% budget. The corresponding one-sided 95% Wilson upper bounds were approximately 0.00140-0.00192 and 0.01118-0.01239. Each ISCX validation split allowed only about five false positives at the 0.1% budget, so the threshold is a discrete order statistic with coarse finite-sample resolution. Half of the 60 locked model selections also had a validation Macro-F1 margin below 0.001, and adjacent-score thresholds exposed additional decision sensitivity. These facts preclude interpreting the selected point as an overall low-FPR guarantee.

Untouched source S3 test behavior was already corpus dependent. PhiUSIIL-trained thresholds remained close to their intended budgets, with mean source-test FPRs of 0.00097-0.00116 and 0.00935-0.01054. ISCX-trained thresholds did not: source-test FPRs rose to 0.0785-0.1412 at the 0.1% budget and 0.1371-0.1672 at the 1% budget. Held-domain composition therefore broke threshold portability even before the data source changed.

Cross-source application failed more severely (Table 7). At the 0.1% source-validation budget, mean external FPR ranged from 0.5318 to 1.0000. Relaxing the source budget to 1% did not make the decision rule portable: mean external FPR ranged from 0.9068 to 1.0000. These rates correspond to approximately 5,318-10,000 false alerts per 10,000 benign target URLs. High target recall in several rows is not operationally favorable because it is achieved by labeling almost every benign URL as phishing.

Table: Frozen source-validation operating points on the primary external cohort (10-repetition means)

| Source to target | Model | External TPR at 0.1% source FPR | External FPR | False alerts per 10,000 benign | External TPR at 1% source FPR | External FPR | False alerts per 10,000 benign |
|---|---|---:|---:|---:|---:|---:|---:|
| ISCX-URL2016 to PhiUSIIL | LR | 0.5654 | 0.5318 | 5,318 | 0.7295 | 0.9068 | 9,068 |
| ISCX-URL2016 to PhiUSIIL | RF | 0.9856 | 1.0000 | 10,000 | 0.9895 | 1.0000 | 10,000 |
| ISCX-URL2016 to PhiUSIIL | XGBoost | 0.9881 | 1.0000 | 10,000 | 0.9903 | 1.0000 | 10,000 |
| PhiUSIIL to ISCX-URL2016 | LR | 0.9951 | 0.9291 | 9,291 | 0.9973 | 0.9407 | 9,407 |
| PhiUSIIL to ISCX-URL2016 | RF | 0.9879 | 0.9993 | 9,993 | 0.9986 | 1.0000 | 10,000 |
| PhiUSIIL to ISCX-URL2016 | XGBoost | 0.9987 | 1.0000 | 10,000 | 0.9999 | 1.0000 | 10,000 |

*Note.* TPR, true-positive rate; FPR, false-positive rate; LR, logistic regression; RF, random forest. The percentages in the column headings are the empirical source-validation FPR budgets. ISCX-URL2016 and PhiUSIIL are the two evaluated phishing web-address corpora.

Together with the calibration analysis, these operating points diagnose the same transfer failure. Figure 6 shows that constraining empirical source-validation FPR did not constrain source-test or target FPR under the zero-target-feedback protocol.

![Figure 6. Realized false-positive rates (FPRs) after applying source-validation thresholds to unseen-domain source tests and primary external cohorts. Points are ten-repetition means for full-feature models. Circles and squares denote 0.1% and 1% source-validation FPR budgets; dotted and dashed lines mark those budgets.](figures/Figure_6.pdf)

**False-positive structure.**

The two transfer directions produced different domain-level false-positive structures (Figure 7). ISCX-to-PhiUSIIL yielded approximately 134,700 false positives per run across about 132,100 registrable domains; the ten most frequent domains contributed only 0.48% of false positives. The error was diffuse across the target source. PhiUSIIL-to-ISCX yielded approximately 32,100-32,900 false positives across about 281 domains, with the ten largest domains contributing about 25%. Here the source mismatch was amplified by several high-volume target entities. This asymmetry explains why row- and domain-equal Macro-F1 move in opposite directions while both domain-equal FPRs remain operationally unacceptable.

![Figure 7. Registrable-domain structure of external false positives for five feature schemes, averaged descriptively over three models and ten outer repetitions. The left panel uses a logarithmic count axis; the right panel reports the share contributed by the ten highest-volume registrable domains. Blue and red bars denote transfer from the ISCX-URL2016 corpus to the PhiUSIIL corpus and the reverse direction, respectively.](figures/Figure_7.pdf)

### 5.4 Independent Third-Source Performance and Threshold Transfer

The independent PhreshPhish evaluation tests whether the original two-source diagnosis survives a separately curated test source. All 60 frozen source-model combinations replayed their archived source metrics before target evaluation. The audit covered 168,060 target rows (91,260 benign and 76,800 phishing), 720 metric rows, and 60 prediction files; every prediction file contained 168,060 unique sample identifiers and finite probabilities in \([0,1]\). The primary cohort removed target registrable domains found in the corresponding source training or validation data. Across repetitions, it retained 163,847-164,857 rows for ISCX-derived models (89,125-89,466 benign and 74,722-75,405 phishing) and 123,246-124,390 rows for PhiUSIIL-derived models (76,689-77,670 benign and 46,324-47,317 phishing).

Performance degraded for every source-model family (Table 8). ISCX-derived LR was the strongest transferred model, but its mean ROC-AUC was 0.640 and its mean FPR was 0.781 at the source-selected Macro-F1 threshold. The other ISCX-derived models had mean ROC-AUC of 0.553-0.560 and FPR of 0.944-0.963. All three PhiUSIIL-derived families ranked below chance, with mean ROC-AUC of 0.346-0.414, while their mean FPR remained 0.908-0.953. Figure 8 summarizes these six source-model profiles and their across-repetition variability. The result therefore extends the original cross-source failure beyond the original dataset pair.

Table: Frozen source models on the primary domain-filtered PhreshPhish cohort

| Source | Model | ROC-AUC | PR-AUC | Macro-F1 | Recall | FPR |
|---|---|---:|---:|---:|---:|---:|
| ISCX-URL2016 | LR | 0.640 ± 0.039 | 0.585 ± 0.037 | 0.487 ± 0.008 | 0.904 ± 0.012 | 0.781 ± 0.015 |
| ISCX-URL2016 | RF | 0.560 ± 0.020 | 0.492 ± 0.019 | 0.359 ± 0.020 | 0.950 ± 0.011 | 0.944 ± 0.024 |
| ISCX-URL2016 | XGBoost | 0.553 ± 0.035 | 0.476 ± 0.045 | 0.342 ± 0.009 | 0.956 ± 0.012 | 0.963 ± 0.011 |
| PhiUSIIL | LR | 0.371 ± 0.003 | 0.312 ± 0.003 | 0.329 ± 0.003 | 0.838 ± 0.010 | 0.908 ± 0.003 |
| PhiUSIIL | RF | 0.414 ± 0.010 | 0.335 ± 0.004 | 0.300 ± 0.001 | 0.888 ± 0.002 | 0.952 ± 0.001 |
| PhiUSIIL | XGBoost | 0.346 ± 0.005 | 0.301 ± 0.004 | 0.300 ± 0.001 | 0.892 ± 0.002 | 0.953 ± 0.000 |

*Note.* Values are mean ± standard deviation across ten source-split repetitions under the source-selected Macro-F1 threshold and row weighting. ROC-AUC, area under the receiver operating characteristic curve; PR-AUC, area under the precision-recall curve; Macro-F1, macro-averaged F1 score; FPR, false-positive rate; LR, logistic regression; RF, random forest.

![Figure 8. External performance of six source-dataset-model combinations on the prespecified primary domain-filtered PhreshPhish cohort. Points show means and whiskers show standard deviations across ten source-split repetitions, using the source-selected Macro-F1 threshold and row weighting. PhiUSIIL-derived models yielded target ROC-AUC values below 0.5, while all combinations produced high false-positive rates (0.781-0.963). Whiskers are descriptive standard deviations, not confidence intervals.](figures/Figure_8.png)

The low-FPR rules also failed to transfer. Mean target FPR ranged from 0.403 to 0.966 under the source 0.1% policy and from 0.458 to 0.970 under the source 1% policy. These fixed-corpus errors do not estimate production prevalence. The score-orientation audit again excluded an implementation reversal: the maximum AUC-negation identity error was \(4.44\times10^{-16}\), and the maximum complement-identity discrepancy was \(1.22\times10^{-6}\), consistent with finite-precision ties near saturated probabilities. We retain the below-chance AUC of the PhiUSIIL-derived models instead of flipping scores after target inspection.

### 5.5 Explanation Agreement Does Not Certify Transfer

The paired SHAP audit separates explanation transportability from predictive transportability. All 60 comparisons passed source-model replay, threshold, probability-vector, cohort, and artifact checks; the analysis produced 120 SHAP matrices and 120 global summaries. Table 9 reports the numerical comparisons, and Figure 9 displays the three explanation-agreement measures across source-model combinations. Explanation rankings often remained similar even when predictive behavior did not. LR retained the greatest source-to-PhreshPhish agreement in both directions: mean Top-15 Jaccard was 0.811 for ISCX-trained LR and 0.856 for PhiUSIIL-trained LR, with full-ranking Spearman correlations of 0.952 and 0.954. Nevertheless, their mean ROC-AUC losses were 0.257 and 0.623, and their target FPRs were 0.781 and 0.908.

Table: Source-to-PhreshPhish SHAP agreement and performance loss

| Source | Model | Top-15 Jaccard | Rank Spearman | Top-20 direction agreement | Direction conflicts | ROC-AUC loss | Macro-F1 loss | Target FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ISCX-URL2016 | LR | 0.811 | 0.952 | 1.000 | 0.0 | 0.257 | 0.379 | 0.781 |
| ISCX-URL2016 | RF | 0.718 | 0.924 | 0.909 | 1.4 | 0.439 | 0.599 | 0.944 |
| ISCX-URL2016 | XGBoost | 0.633 | 0.917 | 0.958 | 0.6 | 0.446 | 0.555 | 0.963 |
| PhiUSIIL | LR | 0.856 | 0.954 | 0.986 | 0.0 | 0.623 | 0.657 | 0.908 |
| PhiUSIIL | RF | 0.588 | 0.863 | 0.851 | 2.4 | 0.585 | 0.690 | 0.952 |
| PhiUSIIL | XGBoost | 0.765 | 0.941 | 0.810 | 4.0 | 0.653 | 0.694 | 0.953 |

*Note.* Values are means over ten source-domain repetitions. Performance loss is source-test score minus PhreshPhish score. Direction agreement is calculated over the union of source and target Top-20 features. SHAP, SHapley Additive exPlanations; FPR, false-positive rate; LR, logistic regression; RF, random forest.

![Figure 9. Transportability of global SHAP summaries from each source to PhreshPhish. Points show means and whiskers show standard deviations across ten repetitions for Top-15 feature-set Jaccard similarity, full-ranking Spearman correlation, and attribution-direction agreement over the union of source and target Top-20 features. Ranking agreement was often high, particularly for logistic regression, despite severe external performance and operating-point degradation. These measures audit explanation transportability and do not establish deployment validity.](figures/Figure_9.png)

Tree models showed more attribution-direction conflicts, concentrated in a small set of lexical and structural features. The most recurrent conflicts included `url_length`, `subdomain_depth`, `registrable_domain_length`, `letter_count`, and `special_char_ratio`, depending on source and model. We treat these conflicts as descriptive model-source associations because the analysis does not identify a causal mechanism for the performance loss.

Across the 60 dependent comparisons, rank overlap was weakly associated with performance loss: Top-15 Jaccard versus ROC-AUC loss had Spearman \(\rho=0.140\), and versus Macro-F1 loss had \(\rho=-0.080\). Direction agreement was more consistently aligned with retained performance (\(\rho=-0.522\) and \(-0.689\), respectively), but source-stratified correlations varied in magnitude and sometimes in sign. High explanation-rank agreement therefore coexisted with severe ranking and operating-point failure. Explanation agreement must be evaluated separately from transportability.

### 5.6 Reciprocal Transfer Is Target Dependent

The reciprocal experiment asks whether training on PhreshPhish produces the same outcome on both original targets. The verified execution completed 30 model fits, 180 internal metric rows, 1,440 external metric rows, 90 tuning rows, and 150 prediction files without warnings. All 245 recorded hashes matched, all prediction identifiers were unique after correction of the ISCX input-file mapping, and all 120 primary-cohort domain-isolation checks found zero registrable-domain overlap.

PhreshPhish-trained models achieved mean internal ROC-AUC of 0.944 for LR, 0.987 for RF, and 0.988 for XGBoost, but external performance was strongly target dependent (Table 10). Transfer to ISCX was near chance, with mean ROC-AUC of 0.518-0.564 across target views and models. Transfer to PhiUSIIL retained substantially stronger discrimination, with mean ROC-AUC of 0.729-0.885. Master and deduplicated views gave nearly identical conclusions.

Table: PhreshPhish-trained reciprocal transfer on primary domain-filtered target cohorts

| Target view | Model | ROC-AUC | Macro-F1 | FPR |
|---|---|---:|---:|---:|
| ISCX master | LR | 0.521 ± 0.010 | 0.519 ± 0.009 | 0.357 ± 0.030 |
| ISCX master | RF | 0.564 ± 0.024 | 0.572 ± 0.040 | 0.302 ± 0.011 |
| ISCX master | XGBoost | 0.535 ± 0.017 | 0.532 ± 0.019 | 0.380 ± 0.072 |
| ISCX deduplicated | LR | 0.518 ± 0.010 | 0.516 ± 0.009 | 0.357 ± 0.030 |
| ISCX deduplicated | RF | 0.561 ± 0.024 | 0.569 ± 0.040 | 0.302 ± 0.011 |
| ISCX deduplicated | XGBoost | 0.531 ± 0.017 | 0.529 ± 0.019 | 0.380 ± 0.072 |
| PhiUSIIL master | LR | 0.730 ± 0.019 | 0.665 ± 0.012 | 0.019 ± 0.008 |
| PhiUSIIL master | RF | 0.885 ± 0.025 | 0.756 ± 0.068 | 0.048 ± 0.016 |
| PhiUSIIL master | XGBoost | 0.806 ± 0.017 | 0.719 ± 0.037 | 0.032 ± 0.025 |
| PhiUSIIL deduplicated | LR | 0.729 ± 0.019 | 0.665 ± 0.011 | 0.019 ± 0.008 |
| PhiUSIIL deduplicated | RF | 0.884 ± 0.025 | 0.755 ± 0.068 | 0.048 ± 0.016 |
| PhiUSIIL deduplicated | XGBoost | 0.805 ± 0.017 | 0.718 ± 0.037 | 0.032 ± 0.025 |

*Note.* Values are mean ± standard deviation across ten repetitions under the source-selected Macro-F1 threshold and row weighting. ROC-AUC, area under the receiver operating characteristic curve; Macro-F1, macro-averaged F1 score; FPR, false-positive rate; LR, logistic regression; RF, random forest.

Operating-point behavior was equally directional. At the source-validation 0.1% FPR budget, mean target FPR ranged from 0.273 to 0.301 on the two ISCX views but from 0.000047 to 0.000637 on the PhiUSIIL views. At the 1% budget, the corresponding ranges were 0.289-0.322 and 0.00284-0.0225. This asymmetry shows that source-internal performance does not determine transfer behavior. Because the original targets were already part of the primary study, this reciprocal experiment supports directionality but does not provide independent validation or identify a causal collection mechanism.

### 5.7 Deployment-Validity Claims Are Directional and Can Disagree

Figure 10 brings the frozen results into one directional audit. Across the complete six-edge graph among the three evaluated sources, no source had uniformly successful outgoing transfer. The ordinary-threshold ROC-AUC pattern was markedly asymmetric. For example, PhreshPhish-to-PhiUSIIL retained mean ROC-AUC of 0.730-0.885 across model families, whereas the reciprocal PhiUSIIL-to-PhreshPhish means were 0.346-0.414. PhreshPhish-to-ISCX remained near chance (0.521-0.564), while the original PhiUSIIL-to-ISCX result included a severe XGBoost reversal (mean ROC-AUC 0.076). The graph characterizes direction-specific behavior in the evaluated fixed corpora; it does not rank intrinsic dataset quality.

The low-FPR synthesis sharpened the same conclusion. At the nominal 0.1% budget, the ranges across model-family mean exceedance ratios were 531.85-1,000.00 for ISCX-to-PhiUSIIL, 929.12-1,000.00 for PhiUSIIL-to-ISCX, 402.51-965.84 for ISCX-to-PhreshPhish, 872.29-952.69 for PhiUSIIL-to-PhreshPhish, and 272.93-301.22 for PhreshPhish-to-ISCX. PhreshPhish-to-PhiUSIIL was the only direction whose model-family means remained below the nominal budget (0.047-0.637 times the budget). At the 1% budget, its range was 0.284-2.251, while every other direction exceeded the budget by at least 28.85 times. The comparatively favorable reverse edge therefore remained model- and budget-dependent.

The validity claims also disagreed within the 60 source-to-PhreshPhish comparisons. Top-15 SHAP Jaccard agreement and ROC-AUC loss had only a weak pooled descriptive association (\(\rho=0.140\)); high agreement could coexist with a large loss. Because these comparisons share datasets, model families, and target cohorts, the coefficient visualizes coexistence and is not an independent inferential test. The three panels support separate acceptance criteria for entity independence, ranking transfer, threshold portability, and explanation stability.

![Figure 10. Directional transfer and disagreement among deployment-validity claims. Panel A shows mean ± standard deviation ROC-AUC across ten repetitions for LR, RF, and XGBoost on all six directed source-to-target edges; the dashed line marks chance. Panel B shows the range across model-family mean target-FPR-to-nominal-budget ratios at the 0.1% and 1% source-validation FPR budgets. Panel C shows Top-15 SHAP Jaccard agreement versus ROC-AUC loss for the 60 dependent source-to-PhreshPhish comparisons; the pooled Spearman coefficient is descriptive. The ISCX/PhiUSIIL edges are retained from the primary two-source analysis, and PhreshPhish-trained reverse edges are secondary reciprocal diagnostics.](figures/Figure_10.png)

### 5.8 Bounded Evidence for Domain-Conditioned SHAP Stability

DCSS is a secondary RQ2 analysis evaluated against locked internal and external criteria, ablations, same-size random subsets, and a third source.

**Selection reproducibility and the auxiliary internal margin comparison.**

Across 300 held-domain SHAP subruns, DCSS returned deterministic Top-10, Top-15, and Top-20 sets for 60 dataset-repetition-model keys. F-DCSS-15 mean pairwise Jaccard agreement was 0.825, 0.882, and 0.839 for ISCX LR, RF, and XGBoost, and 0.908, 1.000, and 0.933 for PhiUSIIL. Its average overlap with F-Stable was 10.8 to 14.3 of 15 features.

In the locked F-DCSS-15 versus F-All comparison (Table 11), ISCX LR (-0.0384) and RF (-0.0275) missed the -0.01 margin. The auxiliary test met the finite-corpus criterion for ISCX XGBoost and all three PhiUSIIL models, although the XGBoost decision weakened in the illustrative positive-dependence analysis. The requirement for at least two qualifying model families per dataset was not met; the positive ISCX XGBoost mean was model-specific.

Table: F-DCSS-15 minus F-All S3 Macro-F1 and auxiliary finite-corpus margin decision

| Dataset | Model | Mean difference | Repetition-bootstrap 95% interval | Auxiliary BH-adjusted one-sided p | Margin criterion |
|---|---|---:|---:|---:|---|
| ISCX-URL2016 | LR | -0.0384 | -0.0779 to -0.0082 | 0.9186 | Not supported |
| ISCX-URL2016 | RF | -0.0275 | -0.0843 to 0.0188 | 0.8756 | Not supported |
| ISCX-URL2016 | XGBoost | +0.0885 | 0.0290 to 0.1515 | 0.0119 | Supported |
| PhiUSIIL | LR | -0.0074 | -0.0085 to -0.0064 | 0.0015 | Supported |
| PhiUSIIL | RF | +0.0001 | -0.0004 to 0.0007 | 5.81e-11 | Supported |
| PhiUSIIL | XGBoost | +0.0003 | -0.0001 to 0.0006 | 4.93e-12 | Supported |

*Note.* DCSS, Domain-Conditioned SHAP Stability; SHAP, SHapley Additive exPlanations; F-DCSS-15, the 15-feature DCSS subset; F-All, the full 35-feature set; S3, registrable-domain-disjoint split; Macro-F1, macro-averaged F1 score; BH, Benjamini-Hochberg; LR, logistic regression; RF, random forest. The decision records whether the prespecified auxiliary test met the -0.01 margin for paired perturbations of the fixed corpus; it is not a formal population-level non-inferiority conclusion for future datasets.

**Prespecified cross-source comparison.**

In the directional external comparison (Table 12), F-DCSS-15 changed mean ROC-AUC relative to F-Stable by -0.00086 from ISCX to PhiUSIIL; the domain-plus-outer interval crossed zero and 6/10 effects were positive. From PhiUSIIL to ISCX, the mean difference was +0.01467, the interval excluded zero, and all ten effects were positive. The two BH-adjusted p values, 0.9648 and 0.0336, are auxiliary finite-corpus evidence. Latter-direction differences were +0.0402 for LR, +0.0081 for RF, and -0.0043 for XGBoost, leaving the XGBoost ranking reversal. The versioned materials retain all 80 paired differences from six internal and two external comparisons.

Table: Prespecified F-DCSS-15 minus F-Stable external ROC-AUC

| Source to target | Mean difference | Domain-plus-outer 95% interval | Outer-only 95% interval | Auxiliary BH-adjusted p | Positive repetitions |
|---|---:|---:|---:|---:|---:|
| ISCX-URL2016 to PhiUSIIL | -0.00086 | -0.02014 to 0.01847 | -0.01933 to 0.01719 | 0.9648 | 6/10 |
| PhiUSIIL to ISCX-URL2016 | +0.01467 | 0.00254 to 0.02739 | 0.01037 to 0.01898 | 0.0336 | 10/10 |

*Note.* DCSS, Domain-Conditioned SHAP Stability; SHAP, SHapley Additive exPlanations; F-DCSS-15, the 15-feature DCSS subset; F-Stable, the random-resampling stability subset; ROC-AUC, area under the receiver operating characteristic curve; BH, Benjamini-Hochberg. The reported p values are BH-adjusted two-sided empirical tail probabilities from the domain-plus-outer distribution. ISCX-URL2016 and PhiUSIIL are the two evaluated phishing web-address corpora.

F-DCSS-20 exceeded F-Stable in both transfers, with mean ROC-AUC of 0.4143 and 0.3704 versus 0.4013 and 0.3538, but this size sensitivity did not replace the prespecified choice. The strong-contribution condition failed because two ISCX model families missed the auxiliary internal margin and one external direction did not improve. Assuming an illustrative repetition correlation of 0.50 retained a +0.00241 lower bound for PhiUSIIL-to-ISCX but weakened several auxiliary internal decisions. The assumption neither estimates actual dependence nor creates independent repetitions.

**Ablation, random subsets, and mechanism diagnostics.**

At k=15, NoRankDispersion matched the full DCSS set in all 60 keys, NoDirection in 59 of 60, and NoFrequency in 44 of 60; their mean Jaccard similarities were 1.000, 0.998, and 0.965. Rank dispersion did not affect membership, direction consistency affected one key, and mean normalized importance plus Top-15 frequency dominated selection.

Of 30 fixed random Top-15 sets, none matched mean ISCX XGBoost DCSS Macro-F1; 43.3% and 46.7% matched or exceeded it for ISCX LR and RF, and PhiUSIIL fractions were 3.3% to 26.7%. Externally, DCSS beat every random set for PhiUSIIL-to-ISCX LR, while every set beat it for reverse-ranking XGBoost. From ISCX to PhiUSIIL, 83.3%, 30.0%, and 20.0% matched or exceeded DCSS for LR, RF, and XGBoost.

Held-domain Top-15 fold agreement averaged approximately 0.72-0.85 on ISCX and 0.87-0.98 on PhiUSIIL. Across 60 dependent outer keys, the largest listed association with external AUC gain was for direction conflict (Spearman rho=-0.213, nominal unadjusted p=0.103). DCSS is a domain-sensitivity probe in selected settings, not a general optimization rule.

**Third-source sensitivity.**

An additional URL-Phish v1 analysis produced positive mean F-DCSS-15-minus-F-Stable AUC differences for models trained on each primary source, but model-specific signs were mixed and absolute operating performance remained unusable. The downloaded row count differed from the accompanying article, and filtering was class dependent because class and collection source were coupled. URL-Phish v1 therefore remains sensitivity evidence; Supplementary Section S6 reports its audit, filtering, model-level effects, and absolute metrics.

## 6. Discussion

### 6.1 Principal Findings

The deployment-validity audit keeps entity alignment, score orientation, predictive transfer, explanation repeatability, and frozen-threshold portability as separate claims. Entity separation changed ISCX performance substantially and had little effect on PhiUSIIL, which shows that random-split optimism depends on corpus construction and on the type of entity novelty represented in the test. Source shift disrupted score orientation, frozen decision rules, and aggregate performance, and the independent PhreshPhish test reproduced severe degradation without target feedback. At the same time, LR explanation rankings remained highly concordant between each source and PhreshPhish, directly showing that explanation-rank agreement can coexist with poor discrimination and high FPR. The orientation and mechanism audits excluded a probability-column or label inversion and retained source-conditioned association differences as descriptive evidence rather than a causal mechanism. The complete six-edge synthesis made the directional structure explicit: PhreshPhish-trained models were near chance on ISCX but retained useful ranking performance and comparatively portable low-FPR thresholds on PhiUSIIL, while the reciprocal PhiUSIIL-to-PhreshPhish edge failed. DCSS detected sensitivity to held-out domains and produced a positive effect in one original transfer direction, yet its failure in the opposite direction and the weak contribution of two score components preclude a general claim of selector superiority.

The internal results narrow the explanation for this pattern. The near-equality of S0 and S1, together with the fixed-role deduplication analysis, indicates that repeated normalized URLs were not the main source of the ISCX gap. Performance changed chiefly after separating hosts and registrable domains, while PhiUSIIL remained stable. The large ISCX S3 variance and the domain-equal sensitivity also show that group composition is part of the estimand. S0-S3 represent different deployment targets rather than a universal progression from invalid to valid evaluation.

### 6.2 Relation to Prior Work

The cross-source degradation agrees with the findings of Rashid et al.^[7]^ and Yi et al.^[8]^, who reported that strong within-dataset phishing URL performance can weaken on another corpus. Our results extend this observation from ranking and classification metrics to decision policy: thresholds constrained to a 0.1% or 1% source-validation FPR still produced unacceptable external false-alert rates. A detector may preserve some ranking information and remain unusable at its frozen operating point.

PhreshPhish argues that credible phishing benchmarks require leakage control, difficult negatives, temporal structure, and realistic base rates^[11]^. Its pinned test split provides an independent target for our frozen models, and the resulting degradation extends the original two-source diagnosis. The six-edge synthesis shows that reversing source and target can change both ranking and threshold behavior. These results describe the evaluated fixed corpora; temporal and production validation remain open.

The explanation results extend the cross-dataset feature differences reported by Mia et al.^[9]^. Our design separates variation due to model seed, grouped partition, deployment regime, and explanation source, then tests explanation transportability beside the performance of the same frozen model. The resulting agreement pattern places reseeding stability as the least demanding evidence and shows that even high cross-source rank agreement is not a portability certificate. Building on the host-disjoint evaluation, external data, adversarial robustness, leave-one-feature-out analysis, and explanation stability of Ahamed et al.^[10]^, we link an explicit `eTLD+1` estimand, a score-orientation audit, source-selected 0.1%/1% FPR thresholds that remain frozen across sources, and matched predictive and explanatory transfer. This protocol-level extension follows broader warnings that security-machine-learning evidence must match the intended deployment population^[14,15]^.

### 6.3 Security and Operational Implications

The frozen-threshold diagnostic gives the transfer results an operational meaning. A threshold allowed at most 10 or 100 false alerts per 10,000 benign source-validation URLs, yet it produced an average of 5,318 to 10,000 false alerts per 10,000 benign URLs in the original cross-source transfer. The complete PhreshPhish test split was evaluated, and the prespecified source-specific domain-filtered primary cohorts reached the same operational conclusion: across the six source-model families (60 frozen runs), mean FPR at the ordinary source-selected threshold was 0.781-0.963, and the nominal low-FPR rules realized mean target FPR of 0.403-0.970. Only PhreshPhish-to-PhiUSIIL approached or met the nominal budgets at the model-family level, and even that edge exceeded the 1% budget for one family. Even with high target recall, the other rates imply a very high target-corpus triage workload for an email gateway, browser filter, or security operations workflow. These corpus-normalized counts do not estimate production prevalence or future-population performance. The frozen models would therefore require target-aware monitoring, recalibration, or retraining before deployment.

For ISCX, threshold instability appeared even before source transfer: validation-constrained operating points yielded mean S3 test FPRs of 7.85%-16.72%, depending on the model and budget. At the 0.1% budget, the validation upper bounds already exceeded the nominal point, and only a small number of false positives was allowed; adjacent-score and grouped-holdout checks showed further sensitivity in threshold selection. Entity composition therefore affected the decision policy as well as the ranking metrics. An operational evaluation should define the entity novelty under test, freeze the threshold without test feedback, and report both uncertainty and realized class-conditional error rates. Neither a source ROC-AUC nor a stable explanation set replaces this check.

### 6.4 Explanation Stability and the Interpretation of DCSS

Predictive performance and explanation stability address different objectives. High seed agreement showed repeatability under a controlled perturbation, whereas lower regime and source agreement showed population dependence. With PhreshPhish, LR retained Top-15 Jaccard agreement above 0.81 in both source directions while losing 0.257-0.623 ROC-AUC and producing FPR of 0.781-0.908. Stability does not establish faithfulness, causality, or external validity^[18-20,23,24]^, and rank turnover may reflect correlated features exchanging positions. The weak, source-dependent association between agreement and performance loss makes explanation stability an unsuitable portability certificate.

**Interpretation of DCSS.**

DCSS estimates stability on domains withheld from fitting rather than only on random inner resamples. From PhiUSIIL to ISCX, the mean difference was +0.01467, the domain-plus-outer interval was 0.00254 to 0.02739, and all ten paired partition effects were positive, particularly for LR. Lower fold agreement on ISCX likewise identifies domain composition as a source of explanation variation. Evidence from more independently collected sources is needed to assess general superiority.

DCSS did not improve the opposite transfer direction, meet the auxiliary internal margin criterion for ISCX LR or RF, or repair the reverse-ranking XGBoost failure. Direction and rank dispersion also rarely changed Top-15 membership. The locked four-factor formula is an evaluated diagnostic, not a validated optimal scoring rule. The ablations motivate prospective evaluation of a simpler selector based on normalized importance and held-domain frequency.

### 6.5 Practical Meaning and Reporting Implications

For PhiUSIIL RF and XGBoost, approximately 14 features retained nearly all S3 performance, so repeated training-only screening can support compact within-source models when feature computation or explanation cost matters. DCSS serves a different purpose by evaluating selected features on held-out domains before the outer model is frozen. The results were not uniform for ISCX or cross-source deployment; F-Stable and F-DCSS should therefore be treated as screening sets under their specified perturbations.

The experiments support a concrete reporting checklist. Define the deployment target before labeling entity overlap as leakage; disclose normalization, conflict handling, and duplicate handling; and report overlap at the URL, host, and registrable-domain levels. Include row-weighted and entity-weighted metrics when domain sizes are skewed. Keep fitting, thresholding, SHAP aggregation, and selection within source-side training data, verify probability orientation, and report finite-sample bounds and realized errors for frozen thresholds. Compare predictive and explanatory transfer on the same frozen models, compare proposed selectors with same-size random sets, and retain an independent source that cannot influence model or threshold selection. Report a target-oracle threshold only as a post hoc diagnostic, separately from zero-target-feedback deployment results.

### 6.6 Future Work

The next evaluation should use temporally ordered or prospectively collected URLs sampled at the production base rate, allowing direct measurement of alert volume, delayed labels, and domain churn. Estimating operating points below 0.1% FPR with useful precision also requires larger benign validation cohorts. Our class-conditional source-discrimination and feature-drift analyses show that collection source is detectable within class, but they cannot distinguish the effects of collection pipeline, labeling criteria, temporal change, and genuine attack evolution. That causal distinction requires prospective metadata and collection-matched controls. The resulting diagnostics could inform source-shift alarms, conservative abstention, or recalibration policies without assuming immediate target labels.

A simplified held-domain selector based on normalized importance and selection frequency should be evaluated as a new prespecified method, rather than substituted after the present ablation. Future comparisons should add temporal groups, more public and organizational sources, and URL features paired with independently collected HTML, DNS, certificate, redirect, or runtime evidence. Such experiments would test whether domain-conditioned explanation evidence remains useful when source identity is less tightly coupled to feature collection, and whether any compact set supports a decision rule with stable realized FPR under prospectively measured shift.

### 6.7 Threats to Validity

**Internal validity.** URL normalization may merge distinct addresses incorrectly or fail to merge equivalent forms, and registrable-domain extraction depends on the frozen Public Suffix List and its private-suffix policy. We retain the normalization rules, parser outcomes, hashes, and issue logs. Fixed candidate grids and ten repetitions reduce researcher degrees of freedom, although model and split uncertainty remain.

**Construct validity.** Jaccard similarity, ranking correlation, direction consistency, and the DCSS score measure agreement under specified perturbations; they do not measure faithfulness or causality. In the ablation, two score terms rarely affected Top-15 membership, and no source-only stability diagnostic had a strong descriptive association with external gain. Domain-equal Macro-F1 may increase when many small positive domains receive more total weight even if FPR remains near one, so it requires joint interpretation with class-conditional rates. ECE depends on the chosen bins, and the target-oracle threshold uses target labels only for diagnosis.

**External validity.** The design covers three public URL sources, three conventional classifiers, and 35 offline URL features. PhreshPhish provides an independent fixed test source for the frozen ISCX/PhiUSIIL models, but it is not a prospective deployment stream, and its training split is used only in the separately identified reciprocal diagnostic. URL-Phish remains exploratory because the documented and observed row counts disagree, filtering is class dependent, and class is coupled with collection source. The experiments identify source-conditioned incompatibility among the evaluated public benchmark construction pipelines rather than universal, temporal, or live-deployment generalization. The analyzed ISCX artifact is a third-party preserved copy from a pinned public mirror and was not byte-verified against an official release. URL-only features are inexpensive and reproducible, but attacks visible only through HTML, certificates, redirects, DNS, or runtime behavior remain outside their scope.

**Label validity.** Public benign and phishing labels may include annotation errors or obsolete records. Removing conflicting labels reduces direct contradictions but does not revalidate every URL. Different labeling criteria may account for part of the cross-dataset degradation, which limits the scope of the conclusions.

**Statistical validity.** Ten overlapping outer repetitions provide paired perturbation evidence about fixed corpora, not ten independent datasets or a sampling frame for future sources. We therefore treat mean effects, intervals, all paired differences, and direction counts as primary; the internal one-sample one-sided t-tests and external empirical tail probabilities remain auxiliary even after multiplicity adjustment. The external interval combines a linearized empirical-AUC influence approximation at the target-domain level with outer resampling; this approach preserves the declared clustering unit, but it is not an exact literal nested bootstrap, and its finite-corpus uncertainty does not cover future collection processes. In an illustrative dependence-stress analysis, the positive PhiUSIIL-to-ISCX effect remains above zero under an assumed repetition correlation of 0.50, whereas some auxiliary internal margin decisions are less stable; the assumed value is not an estimate of the actual dependence. The separate representative domain-cluster bootstrap covers only XGBoost repetition 00 and F-All/F-Stable. At the 0.1% operating point, each ISCX validation split permits about five false positives; the empirical constraint and Wilson bound are reproducible for the frozen cohort but do not support a population-level guarantee. The ablations, random-set distributions, dimensionality checks, and third-source contrasts remain sensitivity evidence.

## 7. Conclusion

This deployment-validity audit separates five requirements for evaluating phishing URL detectors: entity-aligned evaluation, score orientation, predictive transfer, explanation transportability, and frozen-threshold transfer. Registrable-domain separation materially reduced ISCX performance and changed PhiUSIIL only slightly, so the effect of entity overlap depends on the corpus and deployment target. Thresholds selected under 0.1% or 1% source-validation FPR budgets produced mean external FPRs of 0.532-1.000 between the original sources. For PhreshPhish, the complete test split was evaluated; across the prespecified source-specific domain-filtered primary cohorts, the six source-model-family means were 0.346-0.640 for ROC-AUC and 0.781-0.963 for FPR at the ordinary source-selected thresholds. Orientation checks excluded inverted labels and probability columns. High source-to-PhreshPhish SHAP rank agreement, especially for LR, nevertheless coexisted with severe performance loss, so explanation agreement did not certify transportability. Across all six directed edges among the three evaluated sources, transfer remained asymmetric: PhreshPhish-trained models were near chance on ISCX but substantially stronger on PhiUSIIL, and PhreshPhish-to-PhiUSIIL was the only direction with model-family mean low-FPR budget ratios that approached or fell below unity. DCSS remained secondary: it produced a positive finite-corpus effect in one original transfer direction, not the other, and two ISCX model families did not meet the auxiliary internal margin criterion. Each validity claim therefore requires its own evidence under cross-source shift.

## Declarations

### Data and Code Availability

The source datasets are publicly described, but the availability and licensing of the analyzed files differ. PhiUSIIL was downloaded from the UCI Machine Learning Repository (dataset 967; https://archive.ics.uci.edu/dataset/967/phiusiil%2Bphishing%2Burl%2Bdataset) on 2 September 2026 under CC BY 4.0; the downloaded archive has SHA-256 `0a639fd03aea6308c5b1c10c92aa23c2ce1505447a9137271865cd0badc9a59a`. ISCX-URL2016 is described by the Canadian Institute for Cybersecurity (https://www.unb.ca/cic/datasets/url-2016.html). Because its registered download endpoint was unavailable on 2 September 2026, the experiment used a third-party preserved copy of the raw URL lists from https://github.com/Raj-S-Singh/Real-Real-Time-Detection-of-Malicious-URLs-using-Machine-Learning at commit `9f8ffd697ce3250a7db5f4bce9c4870722b70ef5`; the mirror archive has SHA-256 `6ff4e3bbd7aeb2ecb30624a8142d5de517e2c8ef5cc20adcbc574bd657483858`. The manuscript uses `ISCX-URL2016` as shorthand for this preserved copy, not as a claim of byte identity with the official release. Its five-category layout and observed benign/phishing counts (35,378/9,965) are consistent with the official page's approximate descriptions (more than 35,300/around 10,000), but byte identity, the exact official version, and redistribution terms could not be established. A category-level comparison, member hashes, and the limits of two feature-only cross-checks are recorded in `ISCX_PROVENANCE_NOTE.md`. PhreshPhish v1.0.1 was obtained from https://huggingface.co/datasets/phreshphish/phreshphish and pinned to commit `eabec4b7a66324b79cc8a0ad856d1731dc26fe1a`; the derived train and test feature tables have SHA-256 `a3de94991600eb6eff16a26ca809b192ae763650cc91cfef198c068e5aefeeba` and `5cd003999629e29ad31bd7a01a4f4fd9313b7caf1e44227fe5893d6ed19dc495`. URL-Phish v1 was downloaded from Mendeley Data (version 1, DOI `10.17632/65z9twcx3r.1`) on 8 September 2026 under CC BY 4.0; the ZIP and extracted CSV have SHA-256 `946c670c9f88721f5f770ceda28bab84931a275db020fd3db5ba411693591b60` and `d68b3cd0648dcf9c775347416ad1a8995e8a025921fbe3871ca6158d4db3c3a1`, respectively. The downloaded CSV contained 116,600 rows rather than the 111,660 described on its landing page, and this discrepancy is retained in the audit record.

Version 1.1.0 of the public reproducibility repository is available at https://github.com/DrunkFL/phishing-url-deployment-validity-reproducibility. It provides the released analysis code, acquisition and verification instructions, source hashes, sanitized derived features, frozen split roles, protocols, and compact result records. Raw URL strings, plaintext host and domain identifiers, original archives, trained binaries, and third-party materials are excluded. The repository records the applicable licensing and security boundaries and explains how the excluded source data were obtained and verified. Author-created code is licensed under MIT; author-created documentation, protocols, and aggregate results are licensed under CC BY 4.0.

### Funding

This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.

### Conflict of Interest

The authors declare no conflicts of interest.

### Generative Artificial Intelligence Disclosure

OpenAI GPT-based tools assisted with preliminary information retrieval, consistency checks, and language refinement. The human authors retained responsibility for the research questions, approval of the experimental protocol, interpretation of the results, and the conclusions. The authors reviewed the source materials, code, experimental outputs, factual claims, and references used in the manuscript and take full responsibility for the accuracy and integrity of the work. No AI tool was listed as an author or used to fabricate or alter research data or results.

### Author Contributions

Dongze Li: Conceptualization, Methodology, Investigation, Formal analysis, Software, Data curation, Visualization, and Writing - original draft. Jiahui Liu: Supervision and Writing - review and editing. Both authors reviewed and approved the final manuscript.

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

[28] D. Escudero García and N. DeCastro-García, "Conformal Prediction for Labelling and Updating Online Models in the Presence of Concept Drift in Cybersecurity," *Journal of Information Security and Applications*, vol. 93, art. 104120, 2025. https://doi.org/10.1016/j.jisa.2025.104120.

[29] F. Rustam and A. D. Jurcut, "Adaptive Security Framework for Multi-Environment Networks Using Ensemble Data Drift Detection and Incremental Deep Learning," *Journal of Information Security and Applications*, vol. 94, art. 104219, 2025. https://doi.org/10.1016/j.jisa.2025.104219.

[30] M. Najibi and A. J. Bidgoly, "Towards a Robust Android Malware Detection Model Using Explainable Deep Learning," *Journal of Information Security and Applications*, vol. 93, art. 104191, 2025. https://doi.org/10.1016/j.jisa.2025.104191.

[31] Z. Huang, T. Ban, and Y. Zhang, "A Novel Approach for Malicious URL Detection Using RoBERTa and Sparse Autoencoder," *Journal of Information Security and Applications*, vol. 94, art. 104214, 2025. https://doi.org/10.1016/j.jisa.2025.104214.

[32] G. Lo Re, M. Morana, and G. Rizzo, "Adversarial Attacks on Phishing Webpage Detectors via Heuristic Search Techniques," *Journal of Information Security and Applications*, vol. 100, art. 104486, 2026. https://doi.org/10.1016/j.jisa.2026.104486.
