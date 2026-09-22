# Environment Notes

- Operating system: Windows
- Project Python: 3.9.13, isolated in `.venv`
- System Python is not modified beyond creating the virtual environment.
- Package versions are pinned in `requirements.txt`.
- Installation completed on 2026-09-02. The Tsinghua PyPI mirror was used only as the package transport endpoint; package names and versions remain pinned locally.
- `pip check` completed with no broken requirements.
- A complete transitive package snapshot is stored in `requirements.lock.txt`.
- The original environment verification output was retained in the internal experiment log and is not required to use this repository.
- The frozen Public Suffix List snapshot used for URL parsing is included at `resources/public_suffix_list_tldextract_5.1.3.dat`; its SHA-256 is recorded in the repository documentation and manifest.
