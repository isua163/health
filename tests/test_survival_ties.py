"""Regression tests for tied-time survival and exact restricted-mean calculations."""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from src._survival import km, rmrl_from_survival
from src.arms.competing_risks import aalen_johansen_cif1


def test_km_failure_and_censoring_at_same_time_is_order_invariant():
    time = np.array([1.0, 1.0])
    event = np.array([0, 1])
    expected_t = np.array([1.0])
    expected_s = np.array([0.5])
    expected_h = np.array([0.5])

    for order in itertools.permutations(range(2)):
        t, s, h = km(time[list(order)], event[list(order)])
        np.testing.assert_array_equal(t, expected_t)
        np.testing.assert_allclose(s, expected_s)
        np.testing.assert_allclose(h, expected_h)


def test_km_multiple_failures_and_censoring_at_tied_times():
    time = np.array([1, 1, 1, 1, 2], dtype=float)
    event = np.array([1, 1, 0, 0, 1])
    t, s, h = km(time, event)
    np.testing.assert_array_equal(t, [1.0, 2.0])
    np.testing.assert_allclose(s, [0.6, 0.0])
    np.testing.assert_allclose(h, [0.4, 1.4])


def test_km_random_permutations_do_not_change_result():
    time = np.array([1, 1, 2, 2, 2, 3, 4, 4], dtype=float)
    event = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    expected = km(time, event)
    rng = np.random.default_rng(20260713)
    for _ in range(100):
        p = rng.permutation(len(time))
        actual = km(time[p], event[p])
        for got, want in zip(actual, expected):
            np.testing.assert_allclose(got, want)


def test_km_matches_statsmodels_reference_with_ties():
    statsmodels = pytest.importorskip("statsmodels.duration.survfunc")
    SurvfuncRight = statsmodels.SurvfuncRight
    time = np.array([1, 1, 1, 1, 2, 3, 3, 4], dtype=float)
    event = np.array([1, 1, 0, 0, 1, 0, 1, 1])

    t, s, _ = km(time, event)
    reference = SurvfuncRight(time, event)
    # statsmodels reports only event times; compare the KM curve at those knots.
    idx = np.searchsorted(t, reference.surv_times)
    np.testing.assert_allclose(t[idx], reference.surv_times)
    np.testing.assert_allclose(s[idx], reference.surv_prob, rtol=0, atol=1e-12)


def test_aalen_johansen_tied_causes_and_admin_censoring():
    time = np.array([1, 1, 1, 2], dtype=float)
    is_cause1 = np.array([1, 0, 0, 1])
    is_event = np.array([1, 1, 0, 1])
    t, cif = aalen_johansen_cif1(time, is_cause1, is_event)
    np.testing.assert_array_equal(t, [1.0, 2.0])
    np.testing.assert_allclose(cif, [0.25, 0.75])


def test_aalen_johansen_is_order_invariant_with_ties():
    time = np.array([1, 1, 1, 2, 2, 3], dtype=float)
    is_cause1 = np.array([1, 0, 0, 1, 0, 1])
    is_event = np.array([1, 1, 0, 1, 0, 1])
    expected = aalen_johansen_cif1(time, is_cause1, is_event)
    rng = np.random.default_rng(7303)
    for _ in range(100):
        p = rng.permutation(len(time))
        actual = aalen_johansen_cif1(time[p], is_cause1[p], is_event[p])
        for got, want in zip(actual, expected):
            np.testing.assert_allclose(got, want)


def test_rmrl_exact_step_integral_from_origin():
    xs = np.array([1.0, 3.0])
    surv = np.array([0.8, 0.4])
    assert rmrl_from_survival(xs, surv, t_L=0.0, H=4.0) == pytest.approx(3.0)


def test_rmrl_exact_conditional_step_integral():
    xs = np.array([1.0, 3.0])
    surv = np.array([0.8, 0.4])
    assert rmrl_from_survival(xs, surv, t_L=1.0, H=3.0) == pytest.approx(2.5)


def test_rmrl_exact_when_horizon_ends_between_knots():
    xs = np.array([1.0, 3.0, 8.0])
    surv = np.array([0.75, 0.5, 0.2])
    # 0--1: 1; 1--3: .75; 3--4.5: .5
    assert rmrl_from_survival(xs, surv, t_L=0.0, H=4.5) == pytest.approx(3.25)


def test_aalen_johansen_matches_statsmodels_reference_with_ties():
    statsmodels = pytest.importorskip("statsmodels.duration.survfunc")
    CumIncidenceRight = statsmodels.CumIncidenceRight
    time = np.array([1, 1, 1, 2, 2, 3, 4, 4], dtype=float)
    # 0=administrative censoring, 1=cause 1, 2=cause 2
    status = np.array([1, 2, 0, 1, 0, 2, 1, 0])
    is_cause1 = (status == 1).astype(int)
    is_event = (status > 0).astype(int)

    t, cif = aalen_johansen_cif1(time, is_cause1, is_event)
    reference = CumIncidenceRight(time, status)
    np.testing.assert_allclose(t, reference.times, rtol=0, atol=0)
    np.testing.assert_allclose(cif, reference.cinc[0], rtol=0, atol=1e-12)


def test_absolute_risk_set_support_diagnostics_and_conservative_endpoint():
    from src._survival import risk_set_diagnostics, maximum_supported_time

    time = np.array([2.0, 3.0, 3.0, 5.0, 8.0])
    event = np.array([1, 0, 1, 0, 1])
    rows = risk_set_diagnostics(time, event, [3.0, 6.0, 8.0])
    assert [r["n_at_risk"] for r in rows] == [4, 1, 1]
    assert rows[0]["failures_at_time"] == 1
    assert maximum_supported_time(time, event, min_at_risk=3) == 3.0
    assert maximum_supported_time(time, event, min_at_risk=2, require_observed_failure=True) == 5.0
