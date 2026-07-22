"""Shared survival-analysis primitives.

The implementations in this module explicitly aggregate observations at each
unique event time.  This is required for discrete-time reliability data, where
failures and censoring commonly share the same recorded time.  At a tied time,
all subjects still under observation immediately before that time contribute to
the risk set; failures are counted first and censoring then removes subjects from
subsequent risk sets.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _validated_survival_inputs(time, event):
    """Return validated one-dimensional ``float`` time and binary event arrays."""
    t = np.asarray(time, dtype=float)
    e = np.asarray(event)
    if t.ndim != 1 or e.ndim != 1:
        raise ValueError("time and event must be one-dimensional arrays")
    if len(t) != len(e):
        raise ValueError("time and event must have the same length")
    if len(t) == 0:
        raise ValueError("time and event must not be empty")
    if not np.all(np.isfinite(t)):
        raise ValueError("time must contain only finite values")
    if np.any(t < 0):
        raise ValueError("time values must be non-negative")
    if not np.all(np.isin(e, [0, 1])):
        raise ValueError("event must contain only 0/1 values")
    return t, e.astype(int, copy=False)


def km(time, event):
    """Kaplan--Meier survival and Nelson--Aalen cumulative hazard.

    Parameters
    ----------
    time : array-like
        Observed failure or censoring times.
    event : array-like
        ``1`` for an observed failure and ``0`` for censoring.

    Returns
    -------
    unique_time, survival, cumulative_hazard : ndarray
        Values after processing all failures at each unique observed time.

    Notes
    -----
    For a unique time ``t_j``, the risk set is
    ``Y_j = sum_i I(time_i >= t_j)`` and the number of failures is
    ``d_j = sum_i I(time_i == t_j and event_i == 1)``.  Subjects censored at
    ``t_j`` remain in ``Y_j`` for the event contribution at that time and leave
    only before the next unique time.
    """
    t, e = _validated_survival_inputs(time, event)

    unique_t, inverse, counts = np.unique(t, return_inverse=True, return_counts=True)
    failures = np.bincount(inverse, weights=e, minlength=len(unique_t)).astype(float)
    removed_before = np.concatenate(([0], np.cumsum(counts[:-1])))
    at_risk = (len(t) - removed_before).astype(float)

    increments = np.divide(
        failures,
        at_risk,
        out=np.zeros_like(failures, dtype=float),
        where=at_risk > 0,
    )
    factors = np.clip(1.0 - increments, 0.0, 1.0)
    survival = np.cumprod(factors)
    cumulative_hazard = np.cumsum(increments)
    return unique_t, survival, cumulative_hazard


def step_eval(x, xs, ys, left=1.0):
    """Evaluate a right-continuous step function at ``x``.

    The value is ``left`` before the first knot and ``ys[j]`` from ``xs[j]``
    through the interval immediately preceding the next knot.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.ndim != 1 or ys.ndim != 1 or len(xs) != len(ys):
        raise ValueError("xs and ys must be one-dimensional arrays of equal length")
    if len(xs) == 0:
        return np.full_like(x, float(left), dtype=float)
    if np.any(np.diff(xs) <= 0):
        raise ValueError("xs must be strictly increasing")
    i = np.searchsorted(xs, x, side="right") - 1
    return np.where(i >= 0, ys[np.clip(i, 0, len(ys) - 1)], float(left))


def risk_set_diagnostics(time, event, times):
    """Observed support diagnostics at requested times.

    The diagnostics deliberately report absolute risk-set sizes in addition to
    any weight-based effective sample size.  A high ESS ratio is not evidence
    of adequate support when only one or two units remain under observation.
    """
    t, e = _validated_survival_inputs(time, event)
    out = []
    for u in sorted({float(x) for x in times}):
        out.append({
            "time": u,
            "n_at_risk": int(np.sum(t >= u)),
            "failures_at_time": int(np.sum((t == u) & (e == 1))),
            "censorings_at_time": int(np.sum((t == u) & (e == 0))),
            "failures_by_time": int(np.sum((t <= u) & (e == 1))),
            "censorings_by_time": int(np.sum((t <= u) & (e == 0))),
        })
    return out


def maximum_supported_time(time, event, min_at_risk=1, require_observed_failure=False):
    """Largest time with a requested observed risk-set size.

    When ``require_observed_failure`` is true, the endpoint is also capped at
    the last observed failure time.  This provides a conservative sensitivity
    endpoint alongside the usual fixed-horizon KM/RMST estimate.
    """
    t, e = _validated_survival_inputs(time, event)
    min_at_risk = int(min_at_risk)
    if min_at_risk < 1:
        raise ValueError("min_at_risk must be at least one")
    if len(t) < min_at_risk:
        return 0.0
    risk_endpoint = float(np.sort(t)[-min_at_risk])
    if require_observed_failure:
        failures = t[e == 1]
        if len(failures) == 0:
            return 0.0
        risk_endpoint = min(risk_endpoint, float(np.max(failures)))
    return risk_endpoint


def rmrl_from_survival(xs, surv, t_L, H, ngrid=None):
    """Exact restricted mean residual life for a step survival curve.

    Computes

    ``integral[t_L, t_L + H] S(u) / S(t_L) du``

    by summing the exact rectangular areas between successive survival-curve
    knots.  ``ngrid`` is accepted only for backward API compatibility and is
    ignored; no numerical quadrature is used.
    """
    xs = np.asarray(xs, dtype=float)
    surv = np.asarray(surv, dtype=float)
    if xs.ndim != 1 or surv.ndim != 1 or len(xs) != len(surv):
        raise ValueError("xs and surv must be one-dimensional arrays of equal length")
    if len(xs) and np.any(np.diff(xs) <= 0):
        raise ValueError("xs must be strictly increasing")
    if not np.isfinite(t_L) or not np.isfinite(H) or H < 0:
        raise ValueError("t_L must be finite and H must be finite and non-negative")
    if H == 0:
        return 0.0

    end = float(t_L + H)
    S_tL = float(step_eval(t_L, xs, surv)[0])
    if S_tL <= 1e-12:
        return 0.0

    interior = xs[(xs > t_L) & (xs < end)]
    boundaries = np.concatenate(([float(t_L)], interior, [end]))
    widths = np.diff(boundaries)
    interval_survival = step_eval(boundaries[:-1], xs, surv)
    conditional_survival = np.clip(interval_survival / S_tL, 0.0, 1.0)
    return float(np.sum(widths * conditional_survival))


@dataclass(frozen=True)
class ExpPHFit:
    intercept: float
    slope: float
    success: bool
    status: int
    message: str
    method: str
    n_iter: int
    objective: float


def fit_exp_ph(Ttil, is_cens, hx, iters=40, tol=1e-9, return_fit=False):
    """MLE of an exponential proportional-hazards censoring model.

    ``lambda_C(x) = exp(a + g * hx)`` is constant in time.  Returns ``(a, g)``.
    """
    Ttil = np.asarray(Ttil, float)
    is_cens = np.asarray(is_cens, float)
    hx = np.asarray(hx, float)
    if Ttil.ndim != 1 or is_cens.ndim != 1 or hx.ndim != 1:
        raise ValueError("Ttil, is_cens, and hx must be one-dimensional")
    if not (len(Ttil) == len(is_cens) == len(hx)) or len(Ttil) == 0:
        raise ValueError("Ttil, is_cens, and hx must have equal non-zero length")
    if np.any(Ttil < 0) or not np.all(np.isfinite(Ttil)):
        raise ValueError("Ttil must be finite and non-negative")
    if not np.all(np.isin(is_cens, [0.0, 1.0])) or not np.all(np.isfinite(hx)):
        raise ValueError("is_cens must be binary and hx must be finite")

    # A no-censoring sample is a valid boundary fit.  Using a finite numerical
    # approximation keeps downstream weights equal to one to machine precision.
    if float(is_cens.sum()) == 0.0:
        fit = ExpPHFit(-30.0, 0.0, True, 0, "no censoring events", "boundary", 0, 0.0)
        return fit if return_fit else (fit.intercept, fit.slope)

    Xd = np.column_stack([np.ones_like(hx), hx])
    a0 = np.log(max(is_cens.sum(), 0.5) / max(Ttil.sum(), 1e-6))
    theta = np.array([a0, 0.0])
    success = False
    status = 1
    message = f"maximum iterations ({iters}) reached"
    n_iter = 0
    for iteration in range(1, iters + 1):
        n_iter = iteration
        eta = np.clip(Xd @ theta, -30, 30)
        mu = np.exp(eta) * Ttil
        g = Xd.T @ (is_cens - mu)
        Hn = (Xd * mu[:, None]).T @ Xd + 1e-6 * np.eye(2)
        try:
            step = np.linalg.solve(Hn, g)
        except np.linalg.LinAlgError:
            status = 2
            message = "singular Newton system"
            break
        theta = theta + step
        if not np.all(np.isfinite(theta)):
            status = 3
            message = "non-finite coefficient encountered"
            break
        if np.max(np.abs(step)) < tol:
            success = True
            status = 0
            message = "converged"
            break
    eta = np.clip(Xd @ theta, -30, 30)
    objective = float(np.sum(np.exp(eta) * Ttil - is_cens * eta))
    fit = ExpPHFit(
        float(theta[0]), float(theta[1]), success, status, message,
        "newton_exp_ph", n_iter, objective,
    )
    return fit if return_fit else (fit.intercept, fit.slope)
