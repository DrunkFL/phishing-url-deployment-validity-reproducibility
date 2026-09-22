# Public Upload Checklist

## Required before the repository is public

- [ ] Confirm that every script is owned by the authors or may be redistributed.
- [ ] Select and add an explicit license for code and, if appropriate, a separate license for derived data and documentation.
- [x] Re-run `python tools/validate_repository.py --smoke` and retain a `PASS` result; see `RELEASE_VALIDATION.md` (22 September 2026).
- [ ] Confirm that raw URL strings, plaintext hosts/domains, source archives, credentials, personal paths, and author-private notes are absent.
- [ ] Review `DATA_AVAILABILITY.md` and keep the ISCX-URL2016 redistribution restriction explicit.
- [ ] Decide whether the repository should remain anonymous during peer review; if so, use an anonymous review link or delay public attribution in accordance with the journal workflow.
- [ ] Replace the repository placeholder in `manuscript/main.md` and regenerate `main.tex` and `paper.pdf` before submission.
- [ ] Complete the CRediT author-contribution statement after the author list is finalized.

## Recommended release steps

- [ ] Use a concise repository name, for example `phishing-url-deployment-validity`.
- [ ] Create the first Git tag, such as `v1.0.0`, for the exact manuscript-associated snapshot.
- [ ] Archive the tagged release with Zenodo or another long-term repository if a DOI is desired.
- [ ] Add the final repository URL or DOI to the manuscript's Data and Code Availability statement.
- [ ] Re-run `python tools/generate_manifest.py` after any file change, then validate again.

## Current status

- Raw source datasets: excluded.
- Sanitized features and hashed split assignments: included.
- Final manuscript PDF and source: included.
- Individual files at or above 100 MB: none at assembly time.
- Software/derived-material license: not yet selected.
- Public repository URL and DOI: not yet assigned.
