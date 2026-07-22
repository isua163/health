"""Inverse-probability-of-censoring weighted product-limit estimators.

The static arm fits an exponential proportional-hazards censoring model from a
baseline covariate.  The time-varying arm fits a discrete-time complementary-
log-log censoring model to observed person-period health records.

Time convention used throughout
--------------------------------
Recorded times are positive integers.  At a tied time ``u`` an observed failure
is processed before censoring at ``u``.  Consequently the IPC weight entering
the failure risk set is ``1 / K_i(u-)``.  For a unit that fails at record ``L``,
the terminal failure record is not treated as a non-censoring person-period
observation.  For a unit censored at ``L``, the record that triggers censoring is
included with censoring outcome one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import Arm, Dataset
from .._survival import fit_exp_ph, rmrl_from_survival


class IPCWCorrectionArm(Arm):
    """Static fitted IPCW using one observed baseline covariate."""

    def __init__(self, transform="log", weight_cap=None, exp_clip=30.0):
        self.transform = transform
        self.weight_cap = weight_cap
        self.exp_clip = exp_clip
        self.name = f"ipcw_{transform}"

    def _hx(self, x):
        if self.transform == "log":
            return np.log(np.maximum(x, 1e-8))
        if self.transform == "linear":
            return x
        return np.zeros_like(x)

    def _weighted_km(self, Ttil, event, lam):
        """Tie-safe weighted product limit with ``w_i(u)=1/K_i(u)``.

        The exponential static model is continuous-time, so ``K(u-)=K(u)``.
        """
        fails = np.unique(Ttil[event == 1])
        if fails.size == 0:
            return np.array([float(np.max(Ttil))]), np.array([1.0])
        logW = np.clip(lam[:, None] * fails[None, :], None, self.exp_clip)
        W = np.exp(logW)
        if self.weight_cap is not None:
            W = np.minimum(W, self.weight_cap * np.median(W, axis=0, keepdims=True))
        atrisk = Ttil[:, None] >= fails[None, :]
        dead = (Ttil[:, None] == fails[None, :]) & (event[:, None] == 1)
        Yw = (W * atrisk).sum(0)
        dNw = (W * dead).sum(0)
        factors = np.where(Yw > 0, 1.0 - dNw / np.where(Yw > 0, Yw, 1.0), 1.0)
        surv = np.cumprod(np.clip(factors, 0.0, 1.0))
        return fails, surv

    def _weights_at_time(self, u):
        u = float(u)
        idx = np.flatnonzero(self.Ttil_ >= u)
        if len(idx) == 0:
            return idx, np.array([], float)
        w = np.exp(np.clip(self.lam_[idx] * u, None, self.exp_clip))
        if self.weight_cap is not None:
            w = np.minimum(w, self.weight_cap * np.median(w))
        return idx, w

    def diagnostics_at_times(self, times):
        out = []
        for u in sorted({float(t) for t in times}):
            idx, w = self._weights_at_time(u)
            if len(w) == 0:
                out.append({"time": u, "n_at_risk": 0, "ess": 0.0,
                            "ess_over_n_at_risk": np.nan, "weight_median": np.nan,
                            "weight_p95": np.nan, "weight_p99": np.nan,
                            "max_weight": np.nan})
                continue
            ess = float(w.sum() ** 2 / np.sum(w ** 2))
            out.append({
                "time": u, "n_at_risk": int(len(idx)), "ess": ess,
                "ess_over_n_at_risk": ess / len(idx),
                "weight_median": float(np.median(w)),
                "weight_p95": float(np.percentile(w, 95)),
                "weight_p99": float(np.percentile(w, 99)),
                "max_weight": float(np.max(w)),
            })
        return out

    def fit(self, ds: Dataset) -> "IPCWCorrectionArm":
        is_cens = 1.0 - ds.event
        fit = fit_exp_ph(ds.Ttil, is_cens, self._hx(ds.x_obs), return_fit=True)
        if not fit.success:
            raise RuntimeError(
                "static exponential censoring model failed to converge: "
                f"status={fit.status}, message={fit.message}, objective={fit.objective}"
            )
        a, g = fit.intercept, fit.slope
        lam = np.exp(np.clip(a + g * self._hx(ds.x_obs), -30, 30))
        self.Ttil_ = np.asarray(ds.Ttil, float)
        self.lam_ = np.asarray(lam, float)
        self.grid_, self.surv_ = self._weighted_km(ds.Ttil, ds.event, lam)
        tmid = float(np.median(ds.Ttil))
        _, w = self._weights_at_time(tmid)
        if len(w) == 0:
            raise RuntimeError("no units at risk at the median diagnostic time")
        ess = float((w.sum() ** 2) / np.sum(w ** 2))
        self.weight_time_ = tmid
        self.n_at_risk_ = int(len(w))
        self.ess_ = ess
        self.ess_over_n_at_risk_ = float(ess / len(w))
        # Backward-compatible alias; the denominator is now explicitly the
        # time-specific risk set rather than the original fleet size.
        self.ess_over_n_ = self.ess_over_n_at_risk_
        self.weight_median_ = float(np.median(w))
        self.weight_p95_ = float(np.percentile(w, 95))
        self.weight_p99_ = float(np.percentile(w, 99))
        self.max_weight_ = float(np.max(w))
        self.weight_cap_rule_ = (
            "none" if self.weight_cap is None
            else f"cap at {self.weight_cap:g} x time-specific median"
        )
        self.numerical_exp_clip_ = float(self.exp_clip)
        self.coef_ = (a, g)
        self.fit_success_ = fit.success
        self.fit_status_ = fit.status
        self.fit_message_ = fit.message
        self.fit_method_ = fit.method
        self.fit_n_iter_ = fit.n_iter
        self.fit_objective_ = fit.objective
        return self

    def predict_rmrl(self, ds: Dataset, t_L: float, H: float) -> np.ndarray:
        return np.full(len(ds.unit_id), rmrl_from_survival(self.grid_, self.surv_, t_L, H))


@dataclass(frozen=True)
class CloglogFit:
    intercept: float
    slope: float
    success: bool
    status: int
    message: str
    method: str
    n_iter: int
    objective: float


def _cloglog_nll(beta, X, y):
    eta = np.clip(X @ beta, -25.0, 25.0)
    mu = np.exp(eta)
    # For y=0: -log(1-p)=mu. For y=1: -log(p) with p=1-exp(-mu).
    log_p = np.log(np.clip(-np.expm1(-mu), 1e-300, 1.0))
    return float(np.sum((1.0 - y) * mu - y * log_p))


def _fit_cloglog(x, y):
    """Fit ``p=1-exp{-exp(a+g*x)}`` and return coefficients plus status.

    BFGS is attempted first and Nelder--Mead is used as a deterministic fallback.
    A failed fit is not silently accepted.
    """
    from scipy.optimize import minimize

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) == 0:
        raise ValueError("x and y must be non-empty one-dimensional arrays of equal length")
    if not np.all(np.isin(y, [0.0, 1.0])):
        raise ValueError("y must contain only 0/1 values")

    # Boundary cases are valid censoring fits and should not be sent to an optimiser.
    if float(y.sum()) == 0.0:
        return CloglogFit(-25.0, 0.0, True, 0, "no censoring events", "boundary", 0, 0.0)
    if float(y.sum()) == float(len(y)):
        return CloglogFit(10.0, 0.0, True, 0, "all records censored", "boundary", 0, 0.0)

    X = np.column_stack([np.ones_like(x), x])
    p0 = np.clip(y.mean(), 1e-5, 1 - 1e-5)
    a0 = np.log(-np.log1p(-p0))
    start = np.array([a0, 0.0])

    candidates = []
    for method, options in (
        ("BFGS", {"gtol": 1e-7, "maxiter": 1200}),
        ("Nelder-Mead", {"xatol": 1e-7, "fatol": 1e-9, "maxiter": 4000}),
    ):
        res = minimize(_cloglog_nll, start, args=(X, y), method=method, options=options)
        candidates.append((res, method))
        if bool(res.success) and np.all(np.isfinite(res.x)) and np.isfinite(res.fun):
            return CloglogFit(
                float(res.x[0]), float(res.x[1]), True, int(res.status), str(res.message),
                method, int(getattr(res, "nit", -1)), float(res.fun)
            )
        if np.all(np.isfinite(res.x)):
            start = np.asarray(res.x, float)

    best, method = min(candidates, key=lambda z: np.inf if not np.isfinite(z[0].fun) else z[0].fun)
    raise RuntimeError(
        "cloglog censoring model failed to converge: "
        f"method={method}, status={best.status}, message={best.message}, objective={best.fun}"
    )


def _person_period(hi_obs_list, event):
    """Build censoring person-period data under event-before-censoring timing.

    A censored unit observed for ``L`` records contributes ``L`` rows and the last
    outcome equals one.  A failing unit contributes only the first ``L-1`` rows;
    the terminal failure record is excluded from the censoring likelihood.
    """
    xs, ys = [], []
    for hi, e in zip(hi_obs_list, event):
        hi = np.asarray(hi, float)
        L = len(hi)
        if int(e) == 0:
            if L == 0:
                continue
            xs.append(hi)
            yy = np.zeros(L, float)
            yy[-1] = 1.0
            ys.append(yy)
        else:
            if L <= 1:
                continue
            xs.append(hi[:-1])
            ys.append(np.zeros(L - 1, float))
    if not xs:
        raise ValueError("no censoring risk intervals available")
    return np.concatenate(xs), np.concatenate(ys)


def _cumhaz_before_times(hi, a, g):
    """Return cumulative censoring hazard before integer times 0..L.

    The returned array has length ``L+1`` and element ``u`` equals the sum of
    record hazards strictly before time ``u``.  Thus element 1 is zero and
    element ``L`` excludes the terminal record, matching ``K(L-)``.
    """
    hi = np.asarray(hi, float)
    L = len(hi)
    mu = np.exp(np.clip(a + g * hi, -25.0, 25.0))
    out = np.zeros(L + 1, float)
    if L > 1:
        out[2:] = np.cumsum(mu[:-1])
    return out


class TimeVaryingIPCWArm(Arm):
    """Fitted time-varying IPCW using observed health trajectories."""

    def __init__(self, exp_clip=30.0):
        self.exp_clip = float(exp_clip)
        self.name = "ipcw_tv"

    def _weights_at_time(self, u):
        u = int(round(float(u)))
        mask = self.Ttil_ >= u
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            return idx, np.array([], float)
        logw = np.array([
            self.cumhaz_before_[i][min(max(u, 0), len(self.cumhaz_before_[i]) - 1)]
            for i in idx
        ])
        return idx, np.exp(np.clip(logw, None, self.exp_clip))

    def _diagnostics_at(self, times):
        out = []
        for u in sorted({int(max(1, round(float(t)))) for t in times}):
            idx, w = self._weights_at_time(u)
            if len(w) == 0:
                out.append({"time": u, "n_at_risk": 0, "ess": 0.0, "ess_over_n_at_risk": np.nan,
                            "weight_median": np.nan, "weight_p95": np.nan,
                            "weight_p99": np.nan, "max_weight": np.nan})
                continue
            ess = float((w.sum() ** 2) / np.sum(w ** 2))
            out.append({
                "time": u,
                "n_at_risk": int(len(idx)),
                "ess": ess,
                "ess_over_n_at_risk": ess / len(idx),
                "weight_median": float(np.median(w)),
                "weight_p95": float(np.percentile(w, 95)),
                "weight_p99": float(np.percentile(w, 99)),
                "max_weight": float(np.max(w)),
            })
        return out

    def fit(self, ds: Dataset) -> "TimeVaryingIPCWArm":
        if ds.hi_obs is None:
            raise ValueError("TimeVaryingIPCWArm requires ds.hi_obs")
        x, y = _person_period(ds.hi_obs, ds.event)
        fit = _fit_cloglog(x, y)
        a, g = fit.intercept, fit.slope
        self.coef_ = (a, g)
        self.fit_success_ = fit.success
        self.fit_status_ = fit.status
        self.fit_message_ = fit.message
        self.fit_method_ = fit.method
        self.fit_n_iter_ = fit.n_iter
        self.fit_objective_ = fit.objective
        self.n_person_period_rows_ = int(len(x))
        self.n_censoring_events_ = int(y.sum())

        self.cumhaz_before_ = [_cumhaz_before_times(hi, a, g) for hi in ds.hi_obs]
        self.Ttil_ = np.asarray(ds.Ttil, float)
        self.event_ = np.asarray(ds.event, int)
        self.grid_, self.surv_ = self._weighted_km()

        q = np.percentile(self.Ttil_, [25, 50, 75, 90, 100])
        self.weight_diagnostics_ = self._diagnostics_at(q)
        med = min(self.weight_diagnostics_, key=lambda r: abs(r["time"] - np.median(self.Ttil_)))
        self.weight_time_ = float(med["time"])
        self.n_at_risk_ = int(med["n_at_risk"])
        self.ess_ = float(med["ess"])
        self.ess_over_n_at_risk_ = float(med["ess_over_n_at_risk"])
        # Backward-compatible alias; the denominator is the time-specific
        # risk set.  Earlier code divided by the original fleet size, which
        # made static and time-varying diagnostics non-comparable.
        self.ess_over_n_ = self.ess_over_n_at_risk_
        self.weight_median_ = float(med["weight_median"])
        self.weight_p95_ = float(med["weight_p95"])
        self.weight_p99_ = float(med["weight_p99"])
        self.max_weight_ = float(med["max_weight"])
        self.weight_cap_rule_ = "none (numerical exponent clip only)"
        self.numerical_exp_clip_ = self.exp_clip
        return self

    def _weighted_km(self):
        fails = np.unique(self.Ttil_[self.event_ == 1])
        if fails.size == 0:
            return np.array([float(np.max(self.Ttil_))]), np.array([1.0])
        survival = 1.0
        times, values = [], []
        for u in fails:
            idx, w = self._weights_at_time(u)
            Yw = float(w.sum())
            if Yw <= 0:
                continue
            dead_mask = (self.Ttil_[idx] == u) & (self.event_[idx] == 1)
            dNw = float(w[dead_mask].sum())
            survival *= max(0.0, 1.0 - dNw / Yw)
            times.append(float(u))
            values.append(float(survival))
        return np.asarray(times, float), np.asarray(values, float)

    def diagnostics_at_times(self, times):
        """Public weight/risk-set diagnostics at requested integer times."""
        return self._diagnostics_at(times)

    def predict_rmrl(self, ds: Dataset, t_L: float, H: float) -> np.ndarray:
        return np.full(len(ds.unit_id), rmrl_from_survival(self.grid_, self.surv_, t_L, H))
