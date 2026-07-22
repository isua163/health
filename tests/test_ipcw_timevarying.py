import numpy as np

from src.arms.base import Dataset
from src.arms.ipcw_correction import (
    TimeVaryingIPCWArm,
    _cumhaz_before_times,
    _person_period,
)


def test_person_period_event_before_censoring_convention():
    hi = [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5])]
    event = np.array([1, 0])
    x, y = _person_period(hi, event)
    # Failure at L=3 contributes records 1 and 2 only; censoring at L=2
    # contributes both records with the last outcome equal to one.
    assert np.allclose(x, [0.1, 0.2, 0.4, 0.5])
    assert np.allclose(y, [0.0, 0.0, 0.0, 1.0])


def test_cumulative_hazard_is_k_tminus():
    hi = np.array([0.0, 0.0, 0.0])
    before = _cumhaz_before_times(hi, a=0.0, g=0.0)
    # Unit record hazards are all one.  Before time 1 there is no exposure;
    # before times 2 and 3 there are one and two intervals respectively.
    assert np.allclose(before, [0.0, 0.0, 1.0, 2.0])


def test_time_varying_arm_records_convergence_and_multiple_time_diagnostics():
    ds = Dataset(
        unit_id=np.arange(5),
        Ttil=np.array([2.0, 3.0, 4.0, 4.0, 5.0]),
        event=np.array([0, 1, 0, 1, 1]),
        x_obs=np.ones(5),
        hi_obs=[
            np.array([0.1, 0.2]),
            np.array([0.1, 0.2, 0.3]),
            np.array([0.0, 0.2, 0.4, 0.6]),
            np.array([0.1, 0.3, 0.5, 0.7]),
            np.array([0.0, 0.1, 0.2, 0.4, 0.8]),
        ],
    )
    arm = TimeVaryingIPCWArm().fit(ds)
    assert arm.fit_success_
    assert arm.fit_method_ in {"BFGS", "Nelder-Mead", "boundary"}
    assert len(arm.weight_diagnostics_) >= 3
    assert all("n_at_risk" in row and "ess_over_n_at_risk" in row for row in arm.weight_diagnostics_)
    # K(1-) = 1 for every subject, so requested time-one weights are all one.
    row = arm.diagnostics_at_times([1])[0]
    assert row["n_at_risk"] == 5
    assert np.isclose(row["max_weight"], 1.0)
    assert np.isclose(row["ess_over_n_at_risk"], 1.0)


def test_time_varying_weighted_km_is_row_order_invariant():
    base = Dataset(
        unit_id=np.arange(6),
        Ttil=np.array([2.0, 2.0, 3.0, 4.0, 4.0, 5.0]),
        event=np.array([0, 1, 1, 0, 1, 1]),
        x_obs=np.ones(6),
        hi_obs=[
            np.array([0.1, 0.2]),
            np.array([0.1, 0.3]),
            np.array([0.0, 0.2, 0.4]),
            np.array([0.1, 0.2, 0.4, 0.6]),
            np.array([0.0, 0.2, 0.5, 0.7]),
            np.array([0.0, 0.1, 0.3, 0.5, 0.8]),
        ],
    )
    a = TimeVaryingIPCWArm().fit(base)
    order = np.array([5, 2, 0, 4, 1, 3])
    perm = Dataset(
        unit_id=base.unit_id[order],
        Ttil=base.Ttil[order],
        event=base.event[order],
        x_obs=base.x_obs[order],
        hi_obs=[base.hi_obs[i] for i in order],
    )
    b = TimeVaryingIPCWArm().fit(perm)
    assert np.allclose(a.grid_, b.grid_)
    assert np.allclose(a.surv_, b.surv_)


def test_static_ipcw_records_optimizer_status():
    from src.arms.ipcw_correction import IPCWCorrectionArm

    ds = Dataset(
        unit_id=np.arange(6),
        Ttil=np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        event=np.array([0, 1, 0, 1, 1, 1]),
        x_obs=np.array([1.0, 1.2, 1.1, 1.5, 1.7, 2.0]),
    )
    arm = IPCWCorrectionArm("linear").fit(ds)
    assert arm.fit_success_
    assert arm.fit_status_ == 0
    assert arm.fit_method_ in {"newton_exp_ph", "boundary"}
    assert np.isfinite(arm.fit_objective_)


def test_static_ipcw_supports_requested_time_weight_diagnostics():
    from src.arms.ipcw_correction import IPCWCorrectionArm

    ds = Dataset(
        unit_id=np.arange(5),
        Ttil=np.array([2.0, 3.0, 4.0, 5.0, 6.0]),
        event=np.array([0, 1, 0, 1, 1]),
        x_obs=np.array([1.0, 1.2, 1.4, 1.6, 1.8]),
    )
    arm = IPCWCorrectionArm("linear").fit(ds)
    rows = arm.diagnostics_at_times([1.0, 5.0, 7.0])
    assert rows[0]["n_at_risk"] == 5
    assert rows[1]["n_at_risk"] == 2
    assert rows[2]["n_at_risk"] == 0
