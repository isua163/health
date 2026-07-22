# External data requirements

Raw research data are not redistributed in this package. Place the public data outside the package and pass their directories at run time.

## MATR battery data

The MATR directory must directly contain these four MATLAB v7.3/HDF5 files:

```text
2017-05-12_batchdata_updated_struct_errorcorrect.mat
2017-06-30_batchdata_updated_struct_errorcorrect.mat
2018-04-12_batchdata_updated_struct_errorcorrect.mat
2019-01-24_batchdata_updated_struct_errorcorrect.mat
```

The author's completed audit read 185 raw cells, retained 169 units under the 95% completeness rule, and formed the primary internal-resistance cohort of 124 cells with batch counts 41, 43, and 40. The scripts independently audit these conditions before analysis.

## XJTU-SY bearing data

`code/src/data_xjtu.py` accepts either of the commonly distributed layouts:

1. one `.mat` file per bearing; or
2. per-record CSV files under condition and bearing subdirectories.

Pass the dataset root through `--xjtu`. The package does not treat test termination as independently adjudicated physical failure.

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
