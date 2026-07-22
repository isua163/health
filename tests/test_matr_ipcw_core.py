import numpy as np

from src.matr_ipcw import (
    build_ir_signal,
    fitted_cumhaz_before,
    make_stratified_folds,
    overlay_from_uniforms,
    unit_equal_quantile,
    weighted_product_limit,
)
from src._survival import km, rmrl_from_survival


def test_build_ir_signal_is_causal_finite_and_endpoint_aligned():
    ir = np.array([1.0, 1.1, np.nan, 1.2, 1.3])
    z = build_ir_signal(ir, lifetime=6, baseline_cycles=2, smooth_window=2)
    assert len(z) == 6
    assert np.all(np.isfinite(z))
    assert np.isclose(z[-1], z[-2])


def test_unit_equal_quantile_does_not_give_long_path_extra_total_mass():
    short = np.array([0.0, 10.0])
    long = np.zeros(100)
    # Each unit has one-half total weight, so the 75th percentile reaches the
    # upper observation of the short unit rather than being dominated by length.
    assert unit_equal_quantile([short, long], 0.75) == 0.0
    assert unit_equal_quantile([short, long], 0.99) == 10.0


def test_zero_censoring_hazard_weighted_km_matches_naive_km():
    times = np.array([2.0, 3.0, 4.0, 4.0])
    events = np.array([1, 0, 1, 1])
    paths = [np.zeros(int(t)) for t in times]
    cumhaz = [fitted_cumhaz_before(x, -25.0, 0.0) for x in paths]
    g1, s1 = weighted_product_limit(times, events, cumhaz)
    g2, s2, _ = km(times, events)
    assert np.isclose(rmrl_from_survival(g1, s1, 0.0, 4.0),
                      rmrl_from_survival(g2, s2, 0.0, 4.0))


def test_overlay_uses_event_before_terminal_censoring():
    path = [np.array([0.0, 0.0, 0.0])]
    # Even a terminal uniform of zero cannot censor because only records j<T are eligible.
    times, events, observed, before = overlay_from_uniforms(
        path, beta=0.0, tau=0.0, lambda0=0.0, uniforms=[np.array([0.0, 0.0])]
    )
    assert times.tolist() == [3.0]
    assert events.tolist() == [1]
    assert len(observed[0]) == 3
    assert np.allclose(before[0], [0.0, 0.0, 0.0, 0.0])


def test_stratified_folds_keep_each_batch_balanced():
    labels = ["a"] * 11 + ["b"] * 12
    fold = make_stratified_folds(labels, n_folds=5, seed=7)
    for label in ("a", "b"):
        counts = [int(np.sum((np.asarray(labels) == label) & (fold == f))) for f in range(5)]
        assert max(counts) - min(counts) <= 1


def test_policy_run_in_prevents_early_replacement_and_weights_are_one():
    path = [np.zeros(6)]
    times, events, _, before = overlay_from_uniforms(
        path, beta=0.0, tau=0.0, lambda0=100.0,
        uniforms=[np.zeros(5)], policy_start=3,
    )
    # First eligible record is 4 (index 3), despite zero uniforms earlier.
    assert times.tolist() == [4.0]
    assert events.tolist() == [0]
    assert np.allclose(before[0][:5], 0.0)


def test_exact_crude_equals_net_without_replacement():
    from src.matr_ipcw import exact_crude_rmst

    paths = [np.zeros(3), np.zeros(5)]
    H = 4.0
    crude, g = exact_crude_rmst(paths, beta=1.0, tau=0.0, lambda0=0.0, horizon=H)
    truth = np.mean([min(len(x), H) for x in paths])
    assert np.allclose(g, 1.0)
    assert np.isclose(crude, truth)


def test_exact_crude_is_not_below_net_under_preventive_replacement():
    from src.matr_ipcw import exact_crude_rmst

    paths = [np.ones(3), np.ones(5)]
    H = 5.0
    crude, g = exact_crude_rmst(
        paths, beta=0.0, tau=0.0, lambda0=0.5, horizon=H, policy_start=0
    )
    truth = np.mean([min(len(x), H) for x in paths])
    assert np.all((g > 0.0) & (g < 1.0))
    assert crude > truth


def test_weighted_event_diagnostics_records_tied_failure_fraction():
    from src.matr_ipcw import weighted_event_diagnostics

    times = np.array([2.0, 2.0, 3.0, 4.0])
    events = np.array([1, 1, 0, 1])
    paths = [np.zeros(int(t)) for t in times]
    cumhaz = [fitted_cumhaz_before(x, -25.0, 0.0) for x in paths]
    rows = weighted_event_diagnostics(times, events, cumhaz, horizon=4.0)
    first = rows[0]
    assert first["time"] == 2.0
    assert first["n_at_risk"] == 4
    assert first["n_failures"] == 2
    assert np.isclose(first["weighted_hazard_increment"], 0.5)



def test_ridge_slope_zero_recovers_unpenalized_fit_and_positive_ridge_shrinks():
    from src.matr_ipcw import fit_cloglog_fast

    x = np.linspace(-2.0, 2.0, 100)
    y = (x > 1.0).astype(float)
    fit0 = fit_cloglog_fast(x, y, ridge_slope=0.0)
    fit0b = fit_cloglog_fast(x, y)
    fitr = fit_cloglog_fast(x, y, ridge_slope=16.0)
    assert fit0.success and fit0b.success and fitr.success
    assert np.isclose(fit0.intercept, fit0b.intercept)
    assert np.isclose(fit0.slope, fit0b.slope)
    assert abs(fitr.slope) < abs(fit0.slope)


def test_ht_ipcw_rmst_equals_finite_fleet_truth_without_censoring():
    from src.matr_ipcw import ht_ipcw_rmst

    times = np.array([2.0, 3.0, 5.0])
    events = np.ones(3, dtype=int)
    cumhaz = [np.zeros(int(t) + 1) for t in times]
    H = 4.0
    truth = np.mean(np.minimum(times, H))
    assert np.isclose(ht_ipcw_rmst(times, events, cumhaz, H), truth)


def test_cloglog_scipy_fallback_optimizes_same_penalized_objective():
    from src.matr_ipcw import fit_cloglog_fast

    x = np.linspace(-3.0, 3.0, 240)
    y = np.zeros_like(x)
    y[[170, 185, 200, 220, 235]] = 1.0
    reference = fit_cloglog_fast(x, y, ridge_slope=4.0)
    fallback = fit_cloglog_fast(x, y, max_iter=0, ridge_slope=4.0)
    assert reference.success and fallback.success
    assert fallback.method.startswith("fallback_")
    assert fallback.grad_norm <= 1e-5
    assert np.allclose(
        [fallback.intercept, fallback.slope],
        [reference.intercept, reference.slope],
        atol=2e-5,
        rtol=2e-5,
    )
    assert np.isclose(fallback.objective, reference.objective, atol=1e-8, rtol=1e-8)


def test_cloglog_sparse_ridge_fit_has_finite_recorded_solver_method():
    from src.matr_ipcw import fit_cloglog_fast

    x = np.concatenate([np.linspace(-12.0, 1.0, 500), np.array([8.0, 10.0])])
    y = np.zeros_like(x)
    y[-2:] = 1.0
    fit = fit_cloglog_fast(x, y, ridge_slope=4.0)
    assert fit.success
    assert np.isfinite(fit.intercept)
    assert np.isfinite(fit.slope)
    assert np.isfinite(fit.objective)
    assert np.isfinite(fit.grad_norm)
    assert fit.method in {
        "newton", "newton_precision", "fallback_nelder-mead", "fallback_bfgs"
    }


def test_exact_any_exit_equals_net_without_replacement():
    from src.matr_ipcw import exact_any_exit_rmst
    paths = [np.zeros(5), np.zeros(9)]
    H = 7.0
    expected = np.mean([5.0, 7.0])
    got = exact_any_exit_rmst(paths, beta=1.0, tau=0.0, lambda0=0.0, horizon=H)
    assert np.isclose(got, expected)


def test_exact_any_exit_is_not_above_net_under_replacement():
    from src.matr_ipcw import exact_any_exit_rmst
    paths = [np.ones(10), np.ones(12)]
    H = 10.0
    net = np.mean([10.0, 10.0])
    got = exact_any_exit_rmst(
        paths, beta=1.0, tau=0.0, lambda0=0.2, horizon=H, policy_start=2
    )
    assert 0.0 < got <= net


def test_weight_cap_changes_extreme_weight_product_limit():
    times = np.array([1.0, 1.0, 2.0])
    events = np.array([1, 0, 1])
    cumhaz = [np.array([0.0, 10.0, 10.0]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])]
    _, uncapped = weighted_product_limit(times, events, cumhaz)
    _, capped = weighted_product_limit(times, events, cumhaz, weight_cap=10.0)
    assert np.isfinite(capped).all()
    assert not np.allclose(uncapped, capped)

