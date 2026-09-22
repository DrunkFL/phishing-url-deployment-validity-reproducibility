# Dataset Source Notes

Access date: 2026-09-02

## PhiUSIIL

- Dataset: PhiUSIIL Phishing URL (Website), UCI dataset ID 967
- Official page: https://archive.ics.uci.edu/dataset/967/phiusiil%2Bphishing%2Burl%2Bdataset
- Official archive: https://archive.ics.uci.edu/static/public/967/phiusiil%2Bphishing%2Burl%2Bdataset.zip
- Reported instances: 235,795
- Reported features: 54
- Reported class counts: 134,850 legitimate and 100,945 phishing
- Original labels: 1 = legitimate, 0 = phishing
- Experimental labels, applied only in Part 2: 0 = benign, 1 = phishing
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Associated paper DOI: https://doi.org/10.1016/j.cose.2023.103545

## ISCX-URL2016

- Dataset: URL dataset (ISCX-URL2016), Canadian Institute for Cybersecurity, University of New Brunswick
- Official description: https://www.unb.ca/cic/datasets/url-2016.html
- Official download registration page: https://cicresearch.ca/CICDataset/ISCX-URL-2016/
- Official provenance: benign URLs were collected from Alexa-ranked sites and checked with VirusTotal; phishing URLs were collected from OpenPhish.
- Official reported scale: more than 35,300 benign URLs and around 10,000 phishing URLs, plus spam, malware, and defacement categories.
- Associated paper DOI: https://doi.org/10.1007/978-3-319-46298-1_30

The official registration page returned a server error on 2026-09-02 and requires personal information. No personal information was submitted. The experiment copy was recovered from a public GitHub repository that preserves the original `FinalDataset/URL/` directory and all five raw URL-list filenames:

- Mirror repository: https://github.com/Raj-S-Singh/Real-Real-Time-Detection-of-Malicious-URLs-using-Machine-Learning
- Pinned repository commit: `9f8ffd697ce3250a7db5f4bce9c4870722b70ef5`
- Mirrored archive path: `data/raw/iscx_url2016_repository_mirror/ISCXURL2016_repository_mirror.zip`
- Raw members: `Benign_list_big_final.csv`, `DefacementSitesURLFiltered.csv`, `Malware_dataset.csv`, `phishing_dataset.csv`, and `spam_dataset.csv`
- File format: one URL per line, no header
- Line endings: most files use LF or CRLF; `spam_dataset.csv` primarily uses legacy CR-only line endings, which must be normalized in Part 2
- Dataset license: not stated on the official description page and not established by the mirror; redistribution terms must be verified before publishing the data itself

Confidence that the files follow the ISCX-URL2016 raw layout is medium: names, category structure, and scale match the official description and independent research code, but byte identity with the unavailable official download cannot yet be verified. If the official archive is later obtained, compare hashes, per-class row counts, and normalized URL sets before replacing this mirror.

Two feature-only mirrors were also inspected and retained under `data/reference/feature_only_mirrors/`. Their tables contain precomputed lexical features rather than raw URL strings, so they are excluded from the experiment inputs:

- Hugging Face: https://huggingface.co/datasets/bencorn/ISCX-URL-2016, pinned revision `411f1413600d989c11b283228852efb07e4c2c12`
- Kaggle: `sarrazer/url-dataset-iscx-url2016`, version 1 as observed on 2026-09-02
