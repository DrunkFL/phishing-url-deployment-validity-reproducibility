# PhreshPhish data provenance

The experiment uses PhreshPhish v1.0.1 at commit
`eabec4b7a66324b79cc8a0ad856d1731dc26fe1a`. The complete shard inventory is
stored in `manifests/phreshphish_files.json`.

Raw URLs, row-level predictions, and the derived train/test feature Parquet
files are not redistributed. `src/prepare_dataset.py` reconstructs the 35
frozen URL features from the public dataset while omitting raw URLs from its
output. The retained result metadata records SHA-256 hashes for the exact
derived feature files used in the reported analyses.
