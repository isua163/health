"""Arm (3): competing-risks reference using an Aalen--Johansen CIF.

Preventive replacement is treated as cause 2 rather than as censoring.  The arm
reports the crude/subdistribution restricted mean time free of cause-1 failure;
it is not a net-lifetime estimator.
"""
from __future__ import annotations

import numpy as np

from .base import Arm, Dataset
from .._survival import rmrl_from_survival


def aalen_johansen_cif1(time, is_cause1, is_event=None):
    """Aalen--Johansen cumulative incidence for cause 1 with tied times.

    Parameters
    ----------
    time : array-like
        Event or administrative-censoring times.
    is_cause1 : array-like
        ``1`` for a cause-1 event and ``0`` otherwise.
    is_event : array-like, optional
        ``1`` for any event and ``0`` for administrative censoring.  If omitted,
        every observation is treated as an event (cause 1 or cause 2).

    At each unique time all subjects present immediately before that time enter
    the risk set.  Cause-specific events are aggregated at that time, and
    administrative censoring removes subjects only from subsequent risk sets.
    """
    t = np.asarray(time, dtype=float)
    c1 = np.asarray(is_cause1)
    ev = np.ones_like(t, dtype=int) if is_event is None else np.asarray(is_event)

    if t.ndim != 1 or c1.ndim != 1 or ev.ndim != 1:
        raise ValueError("time, is_cause1, and is_event must be one-dimensional")
    if not (len(t) == len(c1) == len(ev)):
        raise ValueError("time, is_cause1, and is_event must have equal length")
    if len(t) == 0:
        raise ValueError("inputs must not be empty")
    if not np.all(np.isfinite(t)) or np.any(t < 0):
        raise ValueError("time must contain finite non-negative values")
    if not np.all(np.isin(c1, [0, 1])) or not np.all(np.isin(ev, [0, 1])):
        raise ValueError("is_cause1 and is_event must contain only 0/1 values")
    if np.any((c1 == 1) & (ev == 0)):
        raise ValueError("a cause-1 event must also be marked as an event")

    unique_t, inverse, counts = np.unique(t, return_inverse=True, return_counts=True)
    d_all = np.bincount(inverse, weights=ev, minlength=len(unique_t)).astype(float)
    d_cause1 = np.bincount(inverse, weights=c1, minlength=len(unique_t)).astype(float)
    removed_before = np.concatenate(([0], np.cumsum(counts[:-1])))
    at_risk = (len(t) - removed_before).astype(float)

    overall_survival = 1.0
    cif1 = 0.0
    cif_values = np.empty(len(unique_t), dtype=float)
    for j in range(len(unique_t)):
        if at_risk[j] > 0:
            cif1 += overall_survival * d_cause1[j] / at_risk[j]
            overall_survival *= max(0.0, 1.0 - d_all[j] / at_risk[j])
        cif_values[j] = cif1

    return unique_t, cif_values


class CompetingRisksArm(Arm):
    name = "competing_risks"
    ANSWERS = "cause-1-free crude subdistribution functional (NOT net lifetime)"
    METRIC_TYPE = "crude_estimand_gap_pct_vs_net_rmst"
    ESTIMAND = r"mu_1_crude(H)=integral_0^H {1-F_1(t)} dt"

    def fit(self, ds: Dataset) -> "CompetingRisksArm":
        is_cause1 = (ds.event == 1).astype(float)
        t, F1 = aalen_johansen_cif1(ds.Ttil, is_cause1)
        self.grid_ = t
        self.subsurv_ = np.clip(1.0 - F1, 0.0, 1.0)
        return self

    def predict_rmrl(self, ds: Dataset, t_L: float, H: float) -> np.ndarray:
        # This integrates 1-F_1, so a unit replaced for cause 2 remains in the
        # cause-1-free subdistribution after replacement.  The result is not
        # time remaining in service and must not be labelled net-RMST bias.
        val = rmrl_from_survival(self.grid_, self.subsurv_, t_L, H)
        return np.full(len(ds.unit_id), val)
