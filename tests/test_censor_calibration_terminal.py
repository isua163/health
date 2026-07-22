import numpy as np

from src.data_xjtu import (
    _expected_cens_fraction,
    calibrate_lambda0,
    impose_informative_censoring,
)


def test_terminal_record_does_not_affect_calibration():
    base = [
        np.array([0.0, 0.2, 0.4, 10.0]),
        np.array([0.0, 0.1, 0.3, 12.0]),
    ]
    altered_terminal = [x.copy() for x in base]
    altered_terminal[0][-1] = 1000.0
    altered_terminal[1][-1] = 2000.0

    lam_a = calibrate_lambda0(base, beta=2.0, tau=0.25, c_target=0.4)
    lam_b = calibrate_lambda0(
        altered_terminal, beta=2.0, tau=0.25, c_target=0.4
    )
    assert np.isclose(lam_a, lam_b, rtol=1e-12, atol=0.0)


def test_calibration_matches_pre_failure_event_definition():
    hi = [
        np.array([0.0, 0.1, 0.3, 2.0]),
        np.array([0.0, 0.2, 0.5, 3.0]),
        np.array([0.0, 0.4, 0.7, 4.0]),
    ]
    target = 0.4
    lam0 = calibrate_lambda0(hi, beta=3.0, tau=0.25, c_target=target)
    expected = _expected_cens_fraction(hi, 3.0, 0.25, lam0)
    assert abs(expected - target) < 1e-10


def test_monte_carlo_realized_fraction_matches_calibrated_target():
    hi = [
        np.linspace(0.0, 1.0 + 0.1 * i, 25 + i)
        for i in range(8)
    ]
    target = 0.4
    lam0 = calibrate_lambda0(hi, beta=2.0, tau=0.5, c_target=target)
    rng = np.random.default_rng(20260714)
    realized = []
    for _ in range(4000):
        ds, _ = impose_informative_censoring(
            hi, beta=2.0, tau=0.5, lam0=lam0, sigma=0.0, rng=rng
        )
        realized.append(np.mean(ds.event == 0))
    assert abs(float(np.mean(realized)) - target) < 0.015
