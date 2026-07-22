# Analysis script map

Shared implementations are under `src/`; deterministic tests are under `tests/`.
The manuscript-facing revision workflow is documented in `../REVISION_README.md`.

## Reviewer-requested revision analyses

- `revision_metrics.py` -- final signed/absolute error, paired absolute-error
  improvement, relative reduction, and overlay Monte Carlo standard errors.
- `matr_nested_preprocessing_audit.py` -- fully nested nuisance preprocessing,
  out-of-fold calibration, weight, ESS, event-time, and unit-dominance diagnostics.
- `build_revision_macros.py` -- writes the reviewer-requested results into
  `revision_results.tex` for conditional insertion into the manuscript.
- `run_tests.py` -- cross-platform deterministic scientific test runner. Use
  `--require-artifacts` after the full raw-data workflow.

## Simulation

- `simulation_benchmark.py` -- known-DGP implementation and misspecification grid.

## MATR cohort and endpoint benchmark

- `matr_data.py` -- raw-file, schema, endpoint, and signal inventory audit.
- `matr_endpoint_reconstruction.py` -- 0.88-Ah endpoint reconstruction and final
  124-cell cohort audit.
- `matr_primary_analysis.py` -- batch-stratified endpoint-benchmark TV-IPCW analysis.

## MATR joint cell-and-policy redesign

- `matr_bootstrap_design.py` -- deterministic cell-resampling and overlay manifests.
- `matr_bootstrap_run.py` -- sample-adaptive redesign replicate computation.
- `matr_bootstrap_summarize.py` -- descriptive redesign distributions and support summaries.
- `matr_bootstrap_validate.py` -- redesign-result validation without inferential thresholding.

## MATR estimator and sensitivity analyses

- `matr_positive_selection_ridge_sensitivity.py` -- non-selective ridge 4/16/64
  audit for the mechanism-aligned and policy-driver-omission analyses.
- `matr_estimator_signal_analysis.py` and validator -- policy-driver omission audit.
  Its legacy augmented-estimator rows are not used in the revised manuscript.
- `matr_temperature_extension.py` and validator -- four-batch temperature extension.
- `matr_policy_sensitivity.py` and validator -- replacement-intensity and
  maintenance-rule analyses.
- `matr_horizon_eol_sensitivity.py` and validator -- RMST-horizon and EOL-threshold
  sensitivity.
- `matr_sensitivity_summary.py` -- compact sensitivity tables and numerical macros.
- `matr_aipw_regularization.py` and validator -- retained as legacy development
  code only; the revised manuscript makes no AIPW comparison.

## XJTU-SY support-boundary analyses

- `xjtu_primary_analysis.py` -- paired overlays and same-sample/held-out-unit fits.
- `xjtu_support_diagnostics.py` -- weight, unit-dominance, and support diagnostics.
- `xjtu_horizon_sensitivity.py` -- horizon and tail-support sensitivity.
- `xjtu_condition_sensitivity.py` -- operating-condition sensitivity.
- `xjtu_policy_sensitivity.py` -- policy and measurement-noise sensitivity.
- `xjtu_static_fit_diagnostics.py` -- static-IPCW fit-failure summary.
- `xjtu_oracle_check.py` -- independent known-DGP oracle-IPCW check.

## Submission assembly and validation

- `build_manuscript_results.py` -- installs audited legacy numerical macros. It now
  supports both the original nested repository layout and this flat submission layout.
- `validate_figures.py` -- checks the four cited figure PDFs.
- `validate_release.py` -- validates a complete release after generated artifacts exist.
