# Code directory

The analysis map is in `../METHOD_TO_OUTPUT_MAP.md`. Shared implementations are under `src/`; deterministic tests are under `tests/`.

Important packaging corrections:

- `matr_estimator_signal_analysis.py` accepts an explicit `--fold-assignment` path.
- `matr_positive_selection_ridge_sensitivity.py` passes each ridge run's own primary fold assignment to the signal audit.
- Root paths are resolved at run time; no Windows path is hard-coded.

The legacy AIPW development scripts are retained for provenance but are not used for manuscript comparisons. See `README_ORIGINAL.md` for the complete script inventory.
