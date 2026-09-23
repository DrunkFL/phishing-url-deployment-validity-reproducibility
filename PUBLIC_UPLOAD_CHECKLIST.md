# Public Upload Checklist

## Required before the repository is public

- [x] Confirm that every included experiment script is author-created; third-party source data and the MPL-2.0 Public Suffix List are scoped separately.
- [x] Add the MIT License for author-created code and CC BY 4.0 for the author-created documentation, protocols, and aggregate results defined in `LICENSE_SCOPE.md`.
- [x] Re-run `python tools/validate_repository.py --smoke` and retain a `PASS` result; see `RELEASE_VALIDATION.md` (23 September 2026).
- [x] Confirm that raw URL strings, plaintext hosts/domains, source archives, credentials, personal paths, and author-private notes are absent.
- [x] Review `DATA_AVAILABILITY.md` and keep the ISCX-URL2016 redistribution restriction explicit.
- [x] Publish the repository under the corresponding author's GitHub account; the submission is not using an anonymous repository workflow.
- [x] Replace the repository placeholder in `manuscript/main.md` and regenerate `main.tex` and `paper.pdf` before submission.
- [ ] Complete the CRediT author-contribution statement after the author list is finalized.

## Recommended release steps

- [ ] Use a concise repository name, for example `phishing-url-deployment-validity`.
- [x] Create the first Git tag, `v1.0.0`, for the exact manuscript-associated snapshot.
- [ ] Archive the tagged release with Zenodo or another long-term repository if a DOI is desired.
- [x] Add the final public repository URL to the manuscript's Data and Code Availability statement.
- [x] Re-run `python tools/generate_manifest.py` after the release changes, then validate again.

## Current status

- Raw source datasets: excluded.
- Sanitized features and hashed split assignments: included.
- Final manuscript PDF and source: included.
- Individual files at or above 100 MB: none at assembly time.
- Software license: MIT for author-created code.
- Documentation and aggregate-result license: CC BY 4.0 within `LICENSE_SCOPE.md`.
- Public repository URL: https://github.com/DrunkFL/phishing-url-deployment-validity-reproducibility
- Archived DOI: not assigned; the release is identified by Git tag `v1.0.0`.
