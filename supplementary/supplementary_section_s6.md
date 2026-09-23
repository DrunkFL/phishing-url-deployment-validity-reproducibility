# Supplementary Section S6. Third-Source Sensitivity Analysis

## S6.1 Scope and Dataset Audit

URL-Phish v1 was used only as a post-confirmatory sensitivity source. Its immutable archive is identified by DOI `10.17632/65z9twcx3r.1`. The downloaded file contained 116,600 rows rather than the 111,660 rows reported in the accompanying article. After normalization and removal of 1,563 duplicate rows, 115,037 unique URLs remained. Removing normalized URLs and registrable domains shared with either confirmatory source produced a domain-filtered cohort of 92,207 URLs, including 85,791 benign and 6,416 phishing observations.

Filtering was strongly class dependent, and the original benign and phishing classes came from different collection sources. The cohort therefore cannot represent independent prospective validation. It is used only to test whether the direction of the F-DCSS-15-versus-F-Stable contrast persists on an additional frozen target. The same 35 URL features were recomputed from stored URL strings; the 22 features distributed with URL-Phish were not used.

## S6.2 Results

On the domain-filtered cohort, the mean F-DCSS-15 minus F-Stable ROC-AUC was +0.0020 for ISCX-trained models and +0.0078 for PhiUSIIL-trained models. Effects varied by model: ISCX-trained RF and PhiUSIIL-trained RF were negative, whereas the largest positive difference was +0.0281 for PhiUSIIL-trained LR.

Absolute performance remained poor. ISCX-trained F-DCSS-15 achieved AUCs of 0.258-0.391 with false-positive rates of 0.967-0.999. PhiUSIIL-trained models achieved AUCs of 0.787-0.848, but false-positive rates still ranged from 0.396 to 0.803 and calibration error remained high. The third source therefore does not convert a relative feature-set contrast into a deployment claim.

## S6.3 Interpretation Boundary

The positive mean contrasts are directionally consistent at the aggregate source level but are not uniform across models. The row-count discrepancy, class-dependent filtering, and collection-source coupling prevent causal attribution or a claim of general external validation. These results are retained as sensitivity evidence only.
