"""Core utilities for the MATR endpoint-benchmark replacement-overlay analysis.

The functions in this module preserve the event-before-censoring convention used
throughout the package.  At an observed failure time ``u``, IPC weights use the
censoring survival immediately before ``u``.  A preventive replacement at record
``L`` contributes the triggering record to the censoring-model likelihood.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class FastCloglogFit:
    intercept: float
    slope: float
    success: bool
    n_iter: int
    objective: float
    grad_norm: float
    message: str
    method: str = "newton"


def _finite_trailing_median(x: np.ndarray, window: int) -> np.ndarray:
    """Causal trailing median, ignoring non-finite observations."""
    arr = np.asarray(x, float).ravel()
    if window < 1:
        raise ValueError("window must be at least one")
    out = np.full(len(arr), np.nan, float)
    for j in range(len(arr)):
        w = arr[max(0, j - window + 1):j + 1]
        w = w[np.isfinite(w)]
        if w.size:
            out[j] = float(np.median(w))
    return out


def build_ir_signal(ir: Sequence[float], lifetime: int, baseline_cycles: int = 50,
                    smooth_window: int = 5) -> np.ndarray:
    """Construct the preregistered causal within-cell log-IR-ratio signal.

    Non-positive and non-finite IR records are ignored by the causal trailing
    median.  Any initial gap is assigned zero (the early-life baseline), and
    subsequent isolated gaps are carried forward.  If the reconstructed endpoint
    is one record beyond the final IR observation, the last signal is carried to
    the endpoint, matching the frozen endpoint-alignment rule.
    """
    raw = np.asarray(ir, float).ravel()
    T = int(lifetime)
    if T < 1:
        raise ValueError("lifetime must be positive")
    positive = np.where(np.isfinite(raw) & (raw > 0), raw, np.nan)
    n0 = min(int(baseline_cycles), len(positive))
    base_vals = positive[:n0]
    base_vals = base_vals[np.isfinite(base_vals)]
    if base_vals.size == 0:
        raise ValueError("no positive finite IR value in the baseline window")
    baseline = float(np.median(base_vals))
    smoothed = _finite_trailing_median(positive, int(smooth_window))
    signal = np.log(smoothed / baseline)

    # Causal filling: baseline zero before the first finite record, then LOCF.
    last = 0.0
    for j in range(len(signal)):
        if np.isfinite(signal[j]):
            last = float(signal[j])
        else:
            signal[j] = last

    if T <= len(signal):
        out = signal[:T].copy()
    else:
        out = np.concatenate([signal, np.full(T - len(signal), last, float)])
    if len(out) != T or not np.all(np.isfinite(out)):
        raise RuntimeError("IR signal construction did not produce a finite endpoint-aligned path")
    return out


def unit_equal_quantile(paths: Sequence[np.ndarray], q: float) -> float:
    """Weighted quantile where every unit contributes equal total mass."""
    if not 0.0 <= float(q) <= 1.0:
        raise ValueError("q must lie in [0,1]")
    if len(paths) == 0:
        raise ValueError("paths must be non-empty")
    values = []
    weights = []
    n_units = len(paths)
    for path in paths:
        x = np.asarray(path, float).ravel()
        x = x[np.isfinite(x)]
        if x.size == 0:
            raise ValueError("every path must contain a finite value")
        values.append(x)
        weights.append(np.full(x.size, 1.0 / (n_units * x.size), float))
    value = np.concatenate(values)
    weight = np.concatenate(weights)
    order = np.argsort(value, kind="mergesort")
    value = value[order]
    cdf = np.cumsum(weight[order])
    index = min(int(np.searchsorted(cdf, float(q), side="left")), len(value) - 1)
    return float(value[index])


def unit_equal_iqr_scale(paths: Sequence[np.ndarray]) -> float:
    q25 = unit_equal_quantile(paths, 0.25)
    q75 = unit_equal_quantile(paths, 0.75)
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError(f"unit-equal IQR is not positive: {scale}")
    return scale


def standardize_policy_paths(paths: Sequence[np.ndarray], policy_start: int = 0) -> tuple[list[np.ndarray], float, float]:
    """Scale the policy signal using unit-equal eligible pre-failure records.

    The baseline run-in and the terminal failure record do not enter the policy
    scale or threshold because preventive replacement is not eligible there.
    The fitted scale is then applied to each complete path for bookkeeping and
    weight prediction.
    """
    start = max(0, int(policy_start))
    reference = [np.asarray(x, float)[start:-1] for x in paths if len(np.asarray(x)) > start + 1]
    if len(reference) != len(paths):
        raise ValueError("every path must extend beyond the policy run-in")
    scale = unit_equal_iqr_scale(reference)
    z = [np.asarray(x, float) / scale for x in paths]
    tau = unit_equal_quantile([x[start:-1] for x in z], 0.70)
    return z, float(scale), float(tau)


def expected_censor_fraction(paths: Sequence[np.ndarray], beta: float, tau: float,
                             lambda0: float, policy_start: int = 0) -> float:
    if lambda0 <= 0:
        return 0.0
    probs = []
    for path in paths:
        z = np.asarray(path, float)
        start = max(0, int(policy_start))
        if len(z) <= start + 1:
            probs.append(0.0)
            continue
        eta = np.clip(float(beta) * (z[start:-1] - float(tau)), -700.0, 700.0)
        total_hazard = float(lambda0) * float(np.exp(eta).sum())
        probs.append(float(-np.expm1(-min(total_hazard, 745.0))))
    return float(np.mean(probs))


def calibrate_lambda0(paths: Sequence[np.ndarray], beta: float, tau: float,
                      target: float, iterations: int = 100, policy_start: int = 0) -> float:
    """Calibrate a common per-record baseline hazard to the target replacement rate."""
    target = float(target)
    if not 0.0 <= target < 1.0:
        raise ValueError("target must lie in [0,1)")
    if target == 0.0:
        return 0.0
    log_lo, log_hi = -745.0, np.log(1e3)
    for _ in range(iterations):
        log_mid = 0.5 * (log_lo + log_hi)
        value = expected_censor_fraction(paths, beta, tau, float(np.exp(log_mid)), policy_start=policy_start)
        if value < target:
            log_lo = log_mid
        else:
            log_hi = log_mid
    result = float(np.exp(0.5 * (log_lo + log_hi)))
    attained = expected_censor_fraction(paths, beta, tau, result, policy_start=policy_start)
    if not np.isfinite(attained) or abs(attained - target) > 1e-8:
        raise RuntimeError(f"lambda calibration failed: target={target}, attained={attained}")
    return result


def overlay_from_uniforms(paths: Sequence[np.ndarray], beta: float, tau: float,
                          lambda0: float, uniforms: Sequence[np.ndarray],
                          policy_start: int = 0) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Apply preventive replacement using supplied common random numbers.

    Returns observed times, event indicators, observed signal prefixes, and the
    known oracle cumulative censoring hazards before integer times 0..Ttil.
    """
    if len(paths) != len(uniforms):
        raise ValueError("paths and uniforms must have equal length")
    times: list[float] = []
    events: list[int] = []
    observed: list[np.ndarray] = []
    oracle_before: list[np.ndarray] = []
    for path, uu in zip(paths, uniforms):
        z = np.asarray(path, float)
        u = np.asarray(uu, float)
        T = len(z)
        start = max(0, int(policy_start))
        if T < 1 or len(u) < max(T - 1, 0):
            raise ValueError("uniform vector does not cover all pre-failure records")
        mu = float(lambda0) * np.exp(np.clip(float(beta) * (z - float(tau)), -700.0, 700.0))
        p = -np.expm1(-np.minimum(mu, 745.0))
        eligible = np.arange(start, max(T - 1, start), dtype=int)
        fired_local = np.flatnonzero(u[eligible] < p[eligible]) if eligible.size else np.array([], int)
        if fired_local.size:
            fired_index = int(eligible[int(fired_local[0])])
            L = fired_index + 1
            event = 0
        else:
            L = T
            event = 1
        prefix = z[:L].copy()
        before = np.zeros(L + 1, float)
        if L > start + 1:
            before[start + 2:] = np.cumsum(mu[start:L - 1])
        times.append(float(L))
        events.append(event)
        observed.append(prefix)
        oracle_before.append(before)
    return np.asarray(times, float), np.asarray(events, int), observed, oracle_before


def person_period(paths: Sequence[np.ndarray], events: Sequence[int], indices: Iterable[int] | None = None,
                  policy_start: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Build censoring person-period rows under event-before-censoring timing."""
    ev = np.asarray(events, int)
    use = range(len(paths)) if indices is None else list(indices)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    start = max(0, int(policy_start))
    for i in use:
        z = np.asarray(paths[i], float)
        if ev[i] == 0:
            if len(z) <= start:
                continue
            zz = z[start:]
            y = np.zeros(len(zz), float)
            y[-1] = 1.0
            xs.append(zz)
            ys.append(y)
        else:
            if len(z) <= start + 1:
                continue
            zz = z[start:-1]
            xs.append(zz)
            ys.append(np.zeros(len(zz), float))
    if not xs:
        raise ValueError("no person-period rows available")
    return np.concatenate(xs), np.concatenate(ys)


def _cloglog_terms(beta: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    eta = beta[0] + beta[1] * x
    if np.max(np.abs(eta)) > 40:
        # Large values are not expected in the preregistered model and usually
        # indicate a failed Newton proposal.  The caller's line search will shrink it.
        return float("inf"), np.full(2, np.nan), np.full((2, 2), np.nan)
    mu = np.exp(eta)
    is_event = y > 0.5
    nll = float(mu[~is_event].sum())
    if np.any(is_event):
        nll -= float(np.log(np.clip(-np.expm1(-mu[is_event]), 1e-300, 1.0)).sum())

    d = mu.copy()
    h = mu.copy()
    if np.any(is_event):
        m = mu[is_event]
        r = np.empty_like(m)
        h1 = np.empty_like(m)
        small = m < 1e-5
        large = m > 50.0
        middle = ~(small | large)
        if np.any(small):
            ms = m[small]
            r[small] = 1.0 - ms / 2.0 + ms * ms / 12.0
            h1[small] = ms / 2.0 - ms * ms / 6.0
        if np.any(middle):
            mm = m[middle]
            em1 = np.expm1(mm)
            em = em1 + 1.0
            r[middle] = mm / em1
            h1[middle] = mm * (((mm - 1.0) * em) + 1.0) / (em1 * em1)
        if np.any(large):
            r[large] = 0.0
            h1[large] = 0.0
        d[is_event] = -r
        h[is_event] = np.maximum(h1, 1e-12)

    grad = np.array([d.sum(), np.dot(d, x)], float)
    h00 = float(h.sum())
    h01 = float(np.dot(h, x))
    h11 = float(np.dot(h, x * x))
    hess = np.array([[h00, h01], [h01, h11]], float)
    return nll, grad, hess


def _cloglog_fallback_fit(
    xx: np.ndarray,
    yy: np.ndarray,
    ridge: float,
    start_coef: np.ndarray,
    prior_iterations: int,
    tol: float,
    reason: str,
) -> FastCloglogFit:
    """Deterministic SciPy fallback for rare damped-Newton precision failures.

    The fallback optimizes the identical penalized likelihood.  It does not
    change the ridge penalty, the data, or the estimator.  Nelder--Mead is used
    to obtain a robust finite point and BFGS then polishes it with the analytic
    gradient.  A result is accepted only when the analytic score is small.
    """
    from scipy.optimize import minimize

    def terms(coef_value: np.ndarray) -> tuple[float, np.ndarray]:
        obj_value, grad_value, _ = _cloglog_terms(np.asarray(coef_value, float), xx, yy)
        if ridge > 0.0 and np.isfinite(obj_value):
            obj_value = float(obj_value + 0.5 * ridge * float(coef_value[1]) ** 2)
            grad_value = np.asarray(grad_value, float).copy()
            grad_value[1] += ridge * float(coef_value[1])
        return float(obj_value), np.asarray(grad_value, float)

    def objective(coef_value: np.ndarray) -> float:
        obj_value, _ = terms(coef_value)
        return obj_value if np.isfinite(obj_value) else 1e300

    def gradient(coef_value: np.ndarray) -> np.ndarray:
        _, grad_value = terms(coef_value)
        if np.all(np.isfinite(grad_value)):
            return grad_value
        # This branch should only be visited after an invalid proposal.  A large
        # finite vector lets SciPy retreat rather than abort on NaNs.
        return np.full(2, 1e100, float)

    start = np.asarray(start_coef, float).copy()
    candidates: list[tuple[object, str]] = []

    nm = minimize(
        objective,
        start,
        method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 4000},
    )
    candidates.append((nm, "nelder-mead"))
    polish_start = np.asarray(nm.x if np.all(np.isfinite(nm.x)) else start, float)

    bfgs = minimize(
        objective,
        polish_start,
        jac=gradient,
        method="BFGS",
        options={"gtol": max(float(tol), 1e-9), "maxiter": 1500},
    )
    candidates.append((bfgs, "bfgs"))

    acceptable: list[tuple[float, float, object, str]] = []
    fallback_grad_tol = max(1e-5, 100.0 * float(tol))
    for result, method in candidates:
        coef_value = np.asarray(getattr(result, "x", [np.nan, np.nan]), float)
        if coef_value.shape != (2,) or not np.all(np.isfinite(coef_value)):
            continue
        obj_value, grad_value = terms(coef_value)
        if not np.isfinite(obj_value) or not np.all(np.isfinite(grad_value)):
            continue
        grad_norm = float(np.linalg.norm(grad_value, ord=np.inf))
        if grad_norm <= fallback_grad_tol:
            acceptable.append((obj_value, grad_norm, result, method))

    if acceptable:
        obj_value, grad_norm, result, method = min(acceptable, key=lambda item: (item[0], item[1]))
        coef_value = np.asarray(result.x, float)
        n_iter = int(prior_iterations) + max(0, int(getattr(result, "nit", 0)))
        status = "success" if bool(getattr(result, "success", False)) else "score-tolerance"
        return FastCloglogFit(
            float(coef_value[0]),
            float(coef_value[1]),
            True,
            n_iter,
            float(obj_value),
            float(grad_norm),
            f"{status} via {method} fallback after {reason}",
            f"fallback_{method}",
        )

    # Preserve the best finite candidate in the failure report.
    finite: list[tuple[float, float, object, str]] = []
    for result, method in candidates:
        coef_value = np.asarray(getattr(result, "x", [np.nan, np.nan]), float)
        if coef_value.shape != (2,) or not np.all(np.isfinite(coef_value)):
            continue
        obj_value, grad_value = terms(coef_value)
        if np.isfinite(obj_value) and np.all(np.isfinite(grad_value)):
            finite.append((float(obj_value), float(np.linalg.norm(grad_value, ord=np.inf)), result, method))
    if finite:
        obj_value, grad_norm, result, method = min(finite, key=lambda item: (item[0], item[1]))
        coef_value = np.asarray(result.x, float)
        return FastCloglogFit(
            float(coef_value[0]), float(coef_value[1]), False,
            int(prior_iterations) + max(0, int(getattr(result, "nit", 0))),
            obj_value, grad_norm,
            f"fallback {method} did not satisfy score tolerance after {reason}",
            f"fallback_{method}_failed",
        )
    return FastCloglogFit(
        float(start[0]), float(start[1]), False, int(prior_iterations),
        float("inf"), float("inf"),
        f"all fallback optimizers failed after {reason}", "fallback_failed",
    )


def fit_cloglog_fast(x: Sequence[float], y: Sequence[float], max_iter: int = 80,
                     tol: float = 1e-7, ridge_slope: float = 0.0) -> FastCloglogFit:
    """Fit ``p=1-exp{-exp(a+g*x)}`` by damped Newton iteration.

    ``ridge_slope`` applies the fixed penalty ``0.5 * ridge_slope * g**2``
    to the time-varying signal coefficient only.  The intercept is left
    unpenalized.  A value of zero exactly recovers the original estimator.

    Rare numerical line-search or maximum-iteration failures are passed to a
    deterministic SciPy fallback that optimizes the same penalized objective.
    The fallback is recorded in ``FastCloglogFit.method`` and is accepted only
    if the analytic score is below a strict tolerance.
    """
    xx = np.asarray(x, float).ravel()
    yy = np.asarray(y, float).ravel()
    ridge = float(ridge_slope)
    if not np.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge_slope must be finite and non-negative")
    if len(xx) == 0 or len(xx) != len(yy) or not np.all(np.isin(yy, [0.0, 1.0])):
        raise ValueError("x and y must be equal-length non-empty arrays with binary y")
    n_events = int(yy.sum())
    if n_events == 0:
        return FastCloglogFit(-25.0, 0.0, True, 0, 0.0, 0.0, "no censoring events", "boundary")
    if n_events == len(yy):
        return FastCloglogFit(10.0, 0.0, True, 0, 0.0, 0.0, "all rows censored", "boundary")

    def penalized_terms(coef_value: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        obj_value, grad_value, hess_value = _cloglog_terms(coef_value, xx, yy)
        if ridge > 0.0 and np.isfinite(obj_value):
            obj_value = float(obj_value + 0.5 * ridge * coef_value[1] ** 2)
            grad_value = grad_value.copy()
            hess_value = hess_value.copy()
            grad_value[1] += ridge * coef_value[1]
            hess_value[1, 1] += ridge
        return float(obj_value), grad_value, hess_value

    rate = np.clip(float(yy.mean()), 1e-10, 1.0 - 1e-10)
    coef = np.array([np.log(-np.log1p(-rate)), 0.0], float)
    last_obj = float("inf")
    last_iteration = 0
    for iteration in range(1, int(max_iter) + 1):
        last_iteration = iteration
        obj, grad, hess = penalized_terms(coef)
        if not np.isfinite(obj) or not np.all(np.isfinite(grad)) or not np.all(np.isfinite(hess)):
            return _cloglog_fallback_fit(
                xx, yy, ridge, coef, iteration, tol, "non-finite Newton quantities"
            )
        grad_norm = float(np.linalg.norm(grad, ord=np.inf))
        if grad_norm < tol:
            return FastCloglogFit(
                float(coef[0]), float(coef[1]), True, iteration,
                obj, grad_norm, "converged", "newton"
            )
        hess = hess + np.eye(2) * 1e-10
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hess) @ grad
        directional = float(np.dot(grad, step))
        if not np.isfinite(directional) or directional <= 0.0:
            # The penalized Hessian should be positive definite.  A normalized
            # gradient direction is a safe deterministic fallback proposal.
            denom = max(float(np.linalg.norm(grad)), 1.0)
            step = grad / denom
            directional = float(np.dot(grad, step))
        accepted = False
        factor = 1.0
        for _ in range(40):
            candidate = coef - factor * step
            cand_obj, cand_grad, _ = penalized_terms(candidate)
            armijo = obj - 1e-4 * factor * directional
            roundoff = 1e-12 * max(1.0, abs(obj))
            improved_score = (
                np.all(np.isfinite(cand_grad))
                and float(np.linalg.norm(cand_grad, ord=np.inf)) < grad_norm
            )
            if np.isfinite(cand_obj) and (
                cand_obj <= armijo or (cand_obj <= obj + roundoff and improved_score)
            ):
                coef = candidate
                last_obj = cand_obj
                accepted = True
                break
            factor *= 0.5
        if not accepted:
            if grad_norm <= max(1e-6, 10.0 * float(tol)):
                return FastCloglogFit(
                    float(coef[0]), float(coef[1]), True, iteration,
                    obj, grad_norm, "line-search precision tolerance", "newton_precision"
                )
            return _cloglog_fallback_fit(
                xx, yy, ridge, coef, iteration, tol, "line search failed"
            )
        if float(np.linalg.norm(factor * step, ord=np.inf)) < tol * (1.0 + float(np.linalg.norm(coef, ord=np.inf))):
            obj2, grad2, _ = penalized_terms(coef)
            return FastCloglogFit(
                float(coef[0]), float(coef[1]), True, iteration,
                obj2, float(np.linalg.norm(grad2, ord=np.inf)), "step tolerance", "newton"
            )
    obj, grad, _ = penalized_terms(coef)
    if np.isfinite(obj) and np.all(np.isfinite(grad)):
        grad_norm = float(np.linalg.norm(grad, ord=np.inf))
        if grad_norm <= max(1e-6, 10.0 * float(tol)):
            return FastCloglogFit(
                float(coef[0]), float(coef[1]), True, last_iteration,
                obj, grad_norm, "maximum-iteration score tolerance", "newton_precision"
            )
    return _cloglog_fallback_fit(
        xx, yy, ridge, coef, last_iteration, tol, "maximum iterations reached"
    )


def fitted_cumhaz_before(path: Sequence[float], intercept: float, slope: float,
                           policy_start: int = 0) -> np.ndarray:
    z = np.asarray(path, float)
    mu = np.exp(np.clip(float(intercept) + float(slope) * z, -25.0, 25.0))
    out = np.zeros(len(z) + 1, float)
    start = max(0, int(policy_start))
    if len(z) > start + 1:
        out[start + 2:] = np.cumsum(mu[start:-1])
    return out


def make_stratified_folds(labels: Sequence[str], n_folds: int = 5,
                          seed: int = 20261001) -> np.ndarray:
    """Deterministic unit-level folds, stratified by batch."""
    lab = np.asarray(labels, object)
    if n_folds < 2:
        raise ValueError("n_folds must be at least two")
    fold = np.full(len(lab), -1, int)
    rng = np.random.default_rng(int(seed))
    for value in sorted(set(lab.tolist())):
        idx = np.flatnonzero(lab == value)
        rng.shuffle(idx)
        for j, unit in enumerate(idx):
            fold[unit] = j % n_folds
    if np.any(fold < 0):
        raise RuntimeError("some units were not assigned to a fold")
    return fold


def fit_crossfit_cumhaz(observed_paths: Sequence[np.ndarray], events: Sequence[int],
                        folds: Sequence[int], policy_start: int = 0,
                        ridge_slope: float = 0.0) -> tuple[list[np.ndarray], list[FastCloglogFit]]:
    """Fit on training units and predict cumulative hazards for held-out units."""
    fold = np.asarray(folds, int)
    event = np.asarray(events, int)
    predictions: list[np.ndarray | None] = [None] * len(observed_paths)
    fits: list[FastCloglogFit] = []
    for f in sorted(np.unique(fold)):
        train = np.flatnonzero(fold != f)
        test = np.flatnonzero(fold == f)
        x, y = person_period(observed_paths, event, train, policy_start=policy_start)
        fit = fit_cloglog_fast(x, y, ridge_slope=ridge_slope)
        fits.append(fit)
        if not fit.success:
            raise RuntimeError(f"cross-fit censoring model failed in fold {f}: {fit.message}")
        for i in test:
            predictions[i] = fitted_cumhaz_before(observed_paths[i], fit.intercept, fit.slope, policy_start=policy_start)
    if any(x is None for x in predictions):
        raise RuntimeError("cross-fit predictions are incomplete")
    return [np.asarray(x, float) for x in predictions], fits  # type: ignore[arg-type]


def fit_same_sample_cumhaz(observed_paths: Sequence[np.ndarray], events: Sequence[int],
                            policy_start: int = 0,
                            ridge_slope: float = 0.0) -> tuple[list[np.ndarray], FastCloglogFit]:
    x, y = person_period(observed_paths, events, policy_start=policy_start)
    fit = fit_cloglog_fast(x, y, ridge_slope=ridge_slope)
    if not fit.success:
        raise RuntimeError(f"same-sample censoring model failed: {fit.message}")
    return [fitted_cumhaz_before(z, fit.intercept, fit.slope, policy_start=policy_start) for z in observed_paths], fit


def weighted_product_limit(times: Sequence[float], events: Sequence[int],
                           cumhaz_before: Sequence[np.ndarray], exp_clip: float = 30.0,
                           weight_cap: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Tie-safe IPCW product-limit estimator using K(u-) weights.

    ``weight_cap`` is an optional absolute cap applied after the numerical
    exponent clip. It is intended for prespecified positivity sensitivity
    analyses; the uncapped estimator remains the default.
    """
    tt = np.asarray(times, float)
    ev = np.asarray(events, int)
    failures = np.unique(tt[ev == 1])
    if failures.size == 0:
        return np.array([float(np.max(tt))]), np.array([1.0])
    survival = 1.0
    grid: list[float] = []
    values: list[float] = []
    for u0 in failures:
        u = int(round(float(u0)))
        idx = np.flatnonzero(tt >= u0)
        logw = np.array([cumhaz_before[i][min(u, len(cumhaz_before[i]) - 1)] for i in idx], float)
        w = np.exp(np.minimum(logw, float(exp_clip)))
        if weight_cap is not None:
            if not np.isfinite(weight_cap) or float(weight_cap) <= 0:
                raise ValueError("weight_cap must be finite and positive")
            w = np.minimum(w, float(weight_cap))
        denom = float(w.sum())
        if denom <= 0:
            continue
        dead = (tt[idx] == u0) & (ev[idx] == 1)
        numer = float(w[dead].sum())
        survival *= max(0.0, 1.0 - numer / denom)
        grid.append(float(u0))
        values.append(float(survival))
    return np.asarray(grid, float), np.asarray(values, float)


def weight_diagnostics(times: Sequence[float], cumhaz_before: Sequence[np.ndarray],
                       checkpoints: Sequence[float], exp_clip: float = 30.0,
                       weight_cap: float | None = None) -> list[dict[str, float | int]]:
    tt = np.asarray(times, float)
    rows: list[dict[str, float | int]] = []
    for t0 in checkpoints:
        u = max(1, int(round(float(t0))))
        idx = np.flatnonzero(tt >= u)
        if len(idx) == 0:
            rows.append({"time": u, "n_at_risk": 0, "ess": 0.0,
                         "ess_over_n_at_risk": float("nan"), "weight_median": float("nan"),
                         "weight_p95": float("nan"), "weight_p99": float("nan"),
                         "max_weight": float("nan"), "n_exp_clipped": 0,
                         "fraction_exp_clipped": float("nan")})
            continue
        logw = np.array([cumhaz_before[i][min(u, len(cumhaz_before[i]) - 1)] for i in idx], float)
        clipped = logw > float(exp_clip)
        w = np.exp(np.minimum(logw, float(exp_clip)))
        n_weight_capped = 0
        if weight_cap is not None:
            if not np.isfinite(weight_cap) or float(weight_cap) <= 0:
                raise ValueError("weight_cap must be finite and positive")
            n_weight_capped = int(np.sum(w > float(weight_cap)))
            w = np.minimum(w, float(weight_cap))
        ess = float(w.sum() ** 2 / np.sum(w ** 2))
        rows.append({
            "time": u,
            "n_at_risk": int(len(idx)),
            "ess": ess,
            "ess_over_n_at_risk": float(ess / len(idx)),
            "weight_median": float(np.median(w)),
            "weight_p95": float(np.percentile(w, 95)),
            "weight_p99": float(np.percentile(w, 99)),
            "max_weight": float(np.max(w)),
            "n_exp_clipped": int(clipped.sum()),
            "fraction_exp_clipped": float(np.mean(clipped)),
            "n_weight_capped": n_weight_capped,
            "fraction_weight_capped": float(n_weight_capped / len(idx)),
        })
    return rows



def ht_ipcw_rmst(times: Sequence[float], events: Sequence[int],
                 cumhaz_before: Sequence[np.ndarray], horizon: float,
                 exp_clip: float | None = None) -> float:
    """Direct Horvitz--Thompson IPCW estimator of finite-fleet net RMST.

    For the frozen complete trajectories,

    ``mu_0(H) = H - N^{-1} sum_i (H - T_i)_+``.

    Under event-before-censoring timing, ``Delta_i / G_i(T_i-)`` is an
    unbiased estimator of one for each failure contribution.  This produces
    an estimand-aligned diagnostic that is linear in the inverse weights,
    unlike the ratio/product-limit estimator.  It may fall outside ``[0,H]``
    in a small sample, so it is primarily used here to distinguish
    finite-sample product-limit bias from a timing or weighting error.
    """
    tt = np.asarray(times, float)
    ev = np.asarray(events, int)
    H = float(horizon)
    if len(tt) == 0 or len(tt) != len(ev) or len(tt) != len(cumhaz_before):
        raise ValueError("times, events, and cumulative hazards must be non-empty and aligned")
    if not np.isfinite(H) or H < 0:
        raise ValueError("horizon must be finite and non-negative")
    total = 0.0
    for i in np.flatnonzero(ev == 1):
        t = float(tt[i])
        loss = max(H - t, 0.0)
        if loss == 0.0:
            continue
        u = int(round(t))
        logw = float(cumhaz_before[int(i)][min(u, len(cumhaz_before[int(i)]) - 1)])
        if exp_clip is not None:
            logw = min(logw, float(exp_clip))
        total += float(np.exp(logw)) * loss
    return float(H - total / len(tt))

def exact_crude_rmst(paths: Sequence[np.ndarray], beta: float, tau: float,
                      lambda0: float, horizon: float,
                      policy_start: int = 0) -> tuple[float, np.ndarray]:
    """Exact finite-fleet crude RMST under the known overlay policy.

    For a complete trajectory with endpoint ``T``, failure is observed before
    replacement at the terminal record.  Therefore the probability of a
    cause-1 failure is the censoring survival immediately before ``T``.  On a
    fixed finite fleet,

    ``mu_crude(H) = H - mean_i G_i(T_i-) * max(H - T_i, 0)``.

    The returned vector contains ``G_i(T_i-)`` for auditing.
    """
    H = float(horizon)
    if not np.isfinite(H) or H < 0:
        raise ValueError("horizon must be finite and non-negative")
    start = max(0, int(policy_start))
    g_terminal: list[float] = []
    contributions: list[float] = []
    for path in paths:
        z = np.asarray(path, float).ravel()
        T = len(z)
        if T < 1:
            raise ValueError("every path must contain at least one record")
        # Replacement is eligible at path indices start,...,T-2.  The final
        # record is protected by the event-before-censoring convention.
        if T > start + 1 and float(lambda0) > 0.0:
            eta = np.clip(float(beta) * (z[start:T - 1] - float(tau)), -700.0, 700.0)
            cumhaz = float(lambda0) * float(np.exp(eta).sum())
        else:
            cumhaz = 0.0
        g = float(np.exp(-min(cumhaz, 745.0)))
        g_terminal.append(g)
        contributions.append(g * max(H - float(T), 0.0))
    crude = H - float(np.mean(contributions))
    return float(crude), np.asarray(g_terminal, float)



def exact_any_exit_rmst(paths: Sequence[np.ndarray], beta: float, tau: float,
                        lambda0: float, horizon: float,
                        policy_start: int = 0) -> float:
    """Exact finite-fleet restricted mean time to failure or replacement.

    For each deterministic complete endpoint path, preventive replacement is
    random under the imposed discrete-time policy.  This function integrates
    the exact distribution of ``min(T, C, H)`` using the same event-before-
    censoring convention as :func:`overlay_from_uniforms`.  Unlike the
    cause-1-free crude functional, time stops when a unit is replaced, making
    this an operational time-to-any-removal estimand.
    """
    H = float(horizon)
    if not np.isfinite(H) or H < 0:
        raise ValueError("horizon must be finite and non-negative")
    start = max(0, int(policy_start))
    values: list[float] = []
    for path in paths:
        z = np.asarray(path, float).ravel()
        T = len(z)
        if T < 1:
            raise ValueError("every path must contain at least one record")
        survival = 1.0
        expected = 0.0
        if float(lambda0) > 0.0:
            for j in range(start, max(T - 1, start)):
                exit_time = float(j + 1)
                if exit_time >= H:
                    expected += survival * H
                    survival = 0.0
                    break
                mu = float(lambda0) * float(np.exp(np.clip(float(beta) * (z[j] - float(tau)), -700.0, 700.0)))
                p = float(-np.expm1(-min(mu, 745.0)))
                expected += survival * p * exit_time
                survival *= 1.0 - p
        if survival > 0.0:
            expected += survival * min(float(T), H)
        values.append(expected)
    return float(np.mean(values))

def weighted_event_diagnostics(times: Sequence[float], events: Sequence[int],
                               cumhaz_before: Sequence[np.ndarray],
                               horizon: float | None = None,
                               exp_clip: float = 30.0) -> list[dict[str, float | int]]:
    """Event-time support diagnostics for a weighted product-limit curve.

    Checkpoint ESS can look benign while one tied failure time consumes a large
    fraction of the weighted risk set.  This function records every observed
    failure time, including the weighted hazard increment that drives the
    product-limit update.
    """
    tt = np.asarray(times, float)
    ev = np.asarray(events, int)
    failures = np.unique(tt[ev == 1])
    if horizon is not None:
        failures = failures[failures <= float(horizon)]
    rows: list[dict[str, float | int]] = []
    for u0 in failures:
        u = int(round(float(u0)))
        idx = np.flatnonzero(tt >= u0)
        logw = np.array([
            cumhaz_before[i][min(u, len(cumhaz_before[i]) - 1)] for i in idx
        ], float)
        clipped = logw > float(exp_clip)
        w = np.exp(np.minimum(logw, float(exp_clip)))
        dead = (tt[idx] == u0) & (ev[idx] == 1)
        denom = float(w.sum())
        numer = float(w[dead].sum())
        ess = float(w.sum() ** 2 / np.sum(w ** 2)) if len(w) else 0.0
        rows.append({
            "time": float(u0),
            "n_at_risk": int(len(idx)),
            "n_failures": int(dead.sum()),
            "weighted_risk": denom,
            "weighted_failures": numer,
            "weighted_hazard_increment": float(numer / denom) if denom > 0 else float("nan"),
            "ess": ess,
            "ess_over_n_at_risk": float(ess / len(idx)) if len(idx) else float("nan"),
            "max_weight": float(np.max(w)) if len(w) else float("nan"),
            "n_exp_clipped": int(clipped.sum()),
        })
    return rows
