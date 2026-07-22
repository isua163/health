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
  audit for the near-aligned and policy-driver-omission analyses.
- `matr_estimator_signal_analysis.py` and validator -- policy-driver omission audit
  and augmented-estimator implementation diagnostic.
- `matr_temperature_extension.py` and validator -- four-batch temperature extension.
- `matr_policy_sensitivity.py` and validator -- completed replacement-intensity
  and maintenance-rule analyses.
- `matr_aipw_regularization.py` and validator -- non-selective AIPW
  implementation-stability grid.
- `matr_horizon_eol_sensitivity.py` and validator -- RMST-horizon and EOL-threshold
  sensitivity.
- `matr_sensitivity_summary.py` -- compact sensitivity tables and numerical macros.

## XJTU-SY support-boundary analyses

- `xjtu_primary_analysis.py` -- paired overlays and same-sample/held-out-unit fits.
- `xjtu_support_diagnostics.py` -- weight, unit-dominance, and support diagnostics.
- `xjtu_horizon_sensitivity.py` -- horizon and tail-support sensitivity.
- `xjtu_condition_sensitivity.py` -- operating-condition sensitivity.
- `xjtu_policy_sensitivity.py` -- policy and measurement-noise sensitivity.
- `xjtu_static_fit_diagnostics.py` -- static-IPCW fit-failure summary.
- `xjtu_oracle_check.py` -- independent known-DGP oracle-IPCW check.

## Submission assembly and validation

- `build_manuscript_results.py` -- installs audited numerical macros.
- `validate_figures.py` -- checks the four cited figure PDFs.
- `run_tests.py` -- runs the deterministic scientific test suite.
- `validate_release.py` -- validates the final package.
