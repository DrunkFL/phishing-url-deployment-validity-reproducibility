# ISCX-URL2016 provenance note

This note records the provenance boundary for the ISCX-URL2016 files used in the study. It supports the Data and Code Availability statement; it does not establish that the analyzed files are byte-identical to an official archive.

## Official description

The Canadian Institute for Cybersecurity page, checked on 10 October 2026, describes five URL categories and gives approximate or lower-bound counts: more than 35,300 benign URLs, around 12,000 spam URLs, around 10,000 phishing URLs, more than 11,500 malware URLs, and more than 45,450 defacement URLs. It attributes the benign set to Alexa-ranked sites followed by crawling and VirusTotal checking, and the phishing set to OpenPhish.

- Official description: https://www.unb.ca/cic/datasets/url-2016.html
- Associated paper: https://doi.org/10.1007/978-3-319-46298-1_30
- Official registration endpoint checked on 2 September 2026: https://cicresearch.ca/CICDataset/ISCX-URL-2016/

The registration endpoint did not yield an official archive during acquisition. The official description does not publish archive-level or member-level hashes, exact version identifiers, or redistribution terms.

## Analyzed preserved copy

The study used a public repository copy pinned to commit `9f8ffd697ce3250a7db5f4bce9c4870722b70ef5`:

- Repository: https://github.com/Raj-S-Singh/Real-Real-Time-Detection-of-Malicious-URLs-using-Machine-Learning
- Archived source URL: https://raw.githubusercontent.com/Raj-S-Singh/Real-Real-Time-Detection-of-Malicious-URLs-using-Machine-Learning/9f8ffd697ce3250a7db5f4bce9c4870722b70ef5/ISCXURL2016.zip
- Acquisition date: 2 September 2026
- Archive size: 11,181,885 bytes
- Archive SHA-256: `6ff4e3bbd7aeb2ecb30624a8142d5de517e2c8ef5cc20adcbc574bd657483858`

| Category | Official description | Preserved-copy file | Observed rows | File SHA-256 |
|---|---:|---|---:|---|
| Benign | more than 35,300 | `Benign_list_big_final.csv` | 35,378 | `07cd35b469ad2c1cbe8c4a56c59e76560b6253978e8e625e90286ff0f6bc18b2` |
| Spam | around 12,000 | `spam_dataset.csv` | 12,000 | `10dae0780e52bea4f88e2af9e21e59db579e6477b946454a942f34870f2e89e7` |
| Phishing | around 10,000 | `phishing_dataset.csv` | 9,965 | `070df5f7ab2d9cd9023f6ecac1526b1c3abb33a84e019515e6ee61141c77413e` |
| Malware | more than 11,500 | `Malware_dataset.csv` | 11,566 | `1419670dcf8e87ae25f4d3302a5cfd3d3624d32805678a6ef49cb8488572e5d6` |
| Defacement | more than 45,450 | `DefacementSitesURLFiltered.csv` | 96,457 | `15af0e5922132ef0bb0ebebbab0173b289a709c57e4769b2d8519f2baa3f89d4` |

The five files contain 165,366 rows in total. The binary benign/phishing analysis uses 45,343 rows: 35,378 benign and 9,965 phishing. Category assignment follows the source filename; each file contains one URL per row and no header.

## Correspondence and limits

The filenames, five-category layout, and category scales are consistent with the official description. The benign and phishing counts used in the binary analysis fall within the official page's approximate descriptions. This is category-level correspondence, not proof of archive identity.

Two independent feature-only mirrors were also inspected during acquisition: Hugging Face `bencorn/ISCX-URL-2016` at revision `411f1413600d989c11b283228852efb07e4c2c12`, and Kaggle `sarrazer/url-dataset-iscx-url2016`, version 1 as observed on 2 September 2026. Because those mirrors contain precomputed features rather than the five raw URL lists, they were excluded from the experiment and cannot establish raw-file byte identity.

Accordingly:

- confidence that the analyzed files follow the published ISCX-URL2016 category structure is **medium**;
- byte identity with an official archive is **not established**;
- the official archive's exact version date is **not established**;
- redistribution rights for the preserved raw URL lists are **not established**, so the raw lists are excluded from the reproducibility release.

If an official archive becomes available, the required reconciliation is an archive-hash comparison followed by member hashes, per-category row counts, and normalized-URL set comparisons. Until then, `ISCX-URL2016` in the manuscript is shorthand for the pinned preserved copy described above.
