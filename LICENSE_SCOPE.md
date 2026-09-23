# License scope

This repository uses separate licenses because it contains author-created
software, author-created documentation, record-level research artifacts, and
third-party material.

## MIT-licensed software

`LICENSE-CODE` applies to author-created source code and build utilities in:

- `code/**/scripts/`
- `tools/`
- `manuscript/build_submission.py`

It does not apply to data files, model outputs, figures, prose, or third-party
files stored beside the scripts.

## CC BY 4.0 documentation and aggregate results

`LICENSE-DOCUMENTATION` applies to author-created:

- protocol and reproduction documents in `protocols/`, `issues/`, and the
  repository root;
- feature and configuration documentation in `features/` and `configs/`;
- aggregate CSV tables and supporting-information documentation in
  `supplementary/`; and
- aggregate result tables in `code/**/results/` that contain no row-level
  records or source identifiers.

The license covers only rights held by the authors. It does not change the
terms of any underlying dataset.

## Material not covered by the repository licenses

The following material is excluded from the MIT and CC BY 4.0 grants:

- raw source datasets, URL lists, original archives, and trained binaries,
  which are not distributed here;
- sanitized record-level feature tables, hashed identifiers, and frozen role
  assignments derived from source datasets, including files under `splits/`
  and `code/**/data/`; these are supplied for research verification without a
  new license grant and remain subject to applicable source terms and law;
- the manuscript source, manuscript PDF, and main-text figures in
  `manuscript/`, except for the MIT-licensed build script identified above;
- `resources/public_suffix_list_tldextract_5.1.3.dat` and the duplicate copy
  under `code/part2_url_normalization_leakage_audit/resources/`, which retain
  the Mozilla Public License 2.0 notice embedded in those files; and
- any other file that carries its own license or attribution notice.

ISCX-URL2016 redistribution terms and byte identity with the official archive
were not established. No ISCX raw URL lists are redistributed. PhiUSIIL and
URL-Phish v1 remain subject to their source licenses and citation requirements.

Questions about reuse should be resolved from the most specific applicable
notice. If this scope file conflicts with an embedded third-party notice, the
third-party notice controls for that material.
