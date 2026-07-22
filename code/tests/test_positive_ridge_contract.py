from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def test_primary_script_has_no_stale_ridge_namespace_reference() -> None:
    text = (ROOT / "code" / "matr_primary_analysis.py").read_text(encoding="utf-8")
    assert "a.require_selection_audit" not in text
    assert "ridge_lock_required = bool(a.verify_ridge_selection)" in text


def test_positive_selection_ridge_results_contract() -> None:
    directory = ROOT / "results" / "matr_positive_ridge_sensitivity"
    primary = pd.read_csv(directory / "primary_positive_selection_ridge_summary.csv")
    support = pd.read_csv(directory / "primary_positive_selection_support_summary.csv")
    fits = pd.read_csv(directory / "primary_positive_selection_fit_summary.csv")
    signal = pd.read_csv(directory / "policy_driver_ridge_summary.csv")

    assert set(primary["ridge_slope"].astype(float)) == {4.0, 16.0, 64.0}
    assert set(primary["batch"]) == {"MATR-05-12", "MATR-06-30", "MATR-04-12"}
    assert len(primary) == len(support) == len(fits) == len(signal) == 9
    assert (primary["crossfit_minus_naive_pp"] < 0).all()
    assert not primary.groupby("batch")["crossfit_minus_same_sample_pp"].apply(
        lambda x: (x >= 0).all() or (x <= 0).all()
    ).any()
    assert (fits["fit_failure_fraction"] == 0).all()
    assert (support["fraction_replicates_with_exp_clipping"] == 0).all()
    assert (signal["omitted_driver_extra_pp"] > 0).all()
