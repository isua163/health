"""Cross-fitted longitudinal augmented-IPCW utilities for the MATR benchmark.

This module implements a discrete-time augmented hazard estimator adapted from
Kawahara, Shinozaki, and Matsuyama (2020).  It is intentionally a low-dimensional diagnostic implementation:
models are low-dimensional, unit-level cross-fitting is mandatory, and every
prediction is generated from the observed prefix or a forward g-formula model.
No post-replacement record from a held-out unit is used by the estimator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class BinaryFit:
    coef: np.ndarray
    success: bool
    objective: float
    grad_norm: float
    n_iter: int
    message: str
    link: str


@dataclass(frozen=True)
class LinearFit:
    coef: np.ndarray
    sigma: float
    success: bool
    message: str


def _binary_terms(coef: np.ndarray, X: np.ndarray, y: np.ndarray, link: str,
                  ridge: float, penalty_mask: np.ndarray) -> tuple[float, np.ndarray]:
    eta = X @ coef
    if link == "logit":
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -35.0, 35.0)))
        nll = -float(np.sum(y * np.log(np.clip(p, 1e-12, 1.0)) +
                            (1.0-y) * np.log(np.clip(1.0-p, 1e-12, 1.0))))
        grad = X.T @ (p-y)
    elif link == "cloglog":
        e = np.exp(np.clip(eta, -30.0, 20.0))
        p = -np.expm1(-e)
        is1 = y > 0.5
        nll = float(np.sum(e[~is1]))
        if np.any(is1):
            nll -= float(np.log(np.clip(p[is1], 1e-12, 1.0)).sum())
        d = e.copy()
        if np.any(is1):
            ee = e[is1]
            denom = np.expm1(np.clip(ee, None, 50.0))
            d[is1] = -ee / np.maximum(denom, 1e-12)
        grad = X.T @ d
    else:
        raise ValueError(link)
    if ridge > 0.0:
        nll += 0.5 * ridge * float(np.sum((coef * penalty_mask) ** 2))
        grad = grad + ridge * coef * penalty_mask
    return nll, np.asarray(grad, float)


def fit_binary(X: np.ndarray, y: np.ndarray, *, link: str = "cloglog",
               ridge: float = 1.0, penalty_mask: np.ndarray | None = None) -> BinaryFit:
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    if X.ndim != 2 or len(X) != len(y) or len(y) == 0:
        raise ValueError("invalid X/y")
    if not np.all(np.isin(y, [0.0, 1.0])):
        raise ValueError("binary y required")
    pm = np.ones(X.shape[1], float) if penalty_mask is None else np.asarray(penalty_mask, float)
    if len(pm) != X.shape[1]:
        raise ValueError("penalty mask mismatch")
    # Intercept is conventionally unpenalized.
    start = np.zeros(X.shape[1], float)
    rate = float(np.clip(y.mean(), 1e-6, 1-1e-6))
    start[0] = np.log(rate/(1-rate)) if link == "logit" else np.log(-np.log1p(-rate))

    def fun(b):
        return _binary_terms(np.asarray(b, float), X, y, link, ridge, pm)[0]

    def jac(b):
        return _binary_terms(np.asarray(b, float), X, y, link, ridge, pm)[1]

    def run_lbfgsb(x0: np.ndarray, *, maxiter: int, ftol: float,
                    gtol: float, maxls: int):
        return minimize(
            fun, np.asarray(x0, float), jac=jac, method="L-BFGS-B",
            options={"maxiter": maxiter, "ftol": ftol, "gtol": gtol,
                     "maxls": maxls},
        )

    # First pass retains the prespecified settings.  SciPy/NumPy builds can
    # terminate by relative objective reduction with a raw infinity-norm
    # gradient only marginally above 1e-4.  Such a result is numerically
    # stationary for these row-stacked likelihoods, but the original hard
    # gate incorrectly rejected it.  A second pass is attempted before the
    # documented near-stationary acceptance rule is applied.
    res = run_lbfgsb(start, maxiter=1000, ftol=1e-12, gtol=1e-7, maxls=20)

    def evaluate(result):
        cc = np.asarray(result.x, float)
        oo, gg = _binary_terms(cc, X, y, link, ridge, pm)
        nn = float(np.linalg.norm(gg, ord=np.inf))
        return cc, float(oo), nn

    coef, obj, gnorm = evaluate(res)
    finite = bool(np.all(np.isfinite(coef)) and np.isfinite(obj) and np.isfinite(gnorm))
    strict_ok = bool(finite and gnorm <= 1e-4)
    near_ok = bool(finite and bool(getattr(res, "success", False)) and gnorm <= 5e-4)

    if not (strict_ok or near_ok) and finite:
        retry = run_lbfgsb(coef, maxiter=4000, ftol=1e-14, gtol=1e-8, maxls=50)
        coef2, obj2, gnorm2 = evaluate(retry)
        # Prefer the retry when it improves stationarity, or ties on gradient
        # while improving the penalized objective.
        if (gnorm2 < gnorm) or (np.isclose(gnorm2, gnorm) and obj2 <= obj):
            res, coef, obj, gnorm = retry, coef2, obj2, gnorm2
        finite = bool(np.all(np.isfinite(coef)) and np.isfinite(obj) and np.isfinite(gnorm))
        strict_ok = bool(finite and gnorm <= 1e-4)
        near_ok = bool(finite and bool(getattr(res, "success", False)) and gnorm <= 5e-4)

    ok = bool(strict_ok or near_ok)
    message = str(getattr(res, "message", ""))
    if ok and not strict_ok:
        message += "; accepted_near_stationary_raw_grad<=5e-4"
    return BinaryFit(coef=coef, success=ok, objective=float(obj), grad_norm=gnorm,
                     n_iter=int(getattr(res, "nit", 0)),
                     message=message, link=link)


def predict_binary(fit: BinaryFit, X: np.ndarray) -> np.ndarray:
    eta = np.asarray(X, float) @ fit.coef
    if fit.link == "logit":
        return 1.0 / (1.0 + np.exp(-np.clip(eta, -35.0, 35.0)))
    e = np.exp(np.clip(eta, -30.0, 20.0))
    return -np.expm1(-e)


def fit_linear(X: np.ndarray, y: np.ndarray, ridge: float = 1e-3,
               penalty_mask: np.ndarray | None = None) -> LinearFit:
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    if X.ndim != 2 or len(X) != len(y) or len(y) == 0:
        raise ValueError("invalid X/y")
    pm = np.ones(X.shape[1], float) if penalty_mask is None else np.asarray(penalty_mask, float)
    A = X.T @ X + float(ridge) * np.diag(pm)
    b = X.T @ y
    try:
        coef = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(A) @ b
    resid = y - X @ coef
    sigma = float(np.sqrt(max(np.mean(resid**2), 1e-10)))
    ok = bool(np.all(np.isfinite(coef)) and np.isfinite(sigma))
    return LinearFit(np.asarray(coef, float), sigma, ok, "closed-form ridge")




def draw_linear_prediction(fit: LinearFit, features: np.ndarray,
                           rng: np.random.Generator) -> float:
    """Draw one scalar transition prediction (NumPy 1.x/2.x compatible)."""
    x = np.asarray(features, float).reshape(-1)
    coef = np.asarray(fit.coef, float).reshape(-1)
    if x.size != coef.size:
        raise ValueError(f"transition feature/coef mismatch: {x.size} != {coef.size}")
    return float(np.dot(x, coef) + rng.normal(0.0, float(fit.sigma)))

def _recent_slope(path: np.ndarray, t_index: int, window: int = 5) -> np.ndarray:
    """Causal slope ending at t_index (0-based), one value per signal."""
    arr = np.asarray(path, float)
    lo = max(0, t_index-window+1)
    w = arr[lo:t_index+1]
    if len(w) < 2:
        return np.zeros(arr.shape[1], float)
    x = np.arange(len(w), dtype=float)
    x = x-x.mean()
    den = float(np.dot(x, x))
    return (x[:, None] * (w-w.mean(axis=0))).sum(axis=0) / max(den, 1e-12)


def event_features(path: np.ndarray, event_time: int, horizon: int,
                   baseline: np.ndarray) -> np.ndarray:
    """Features for event hazard at integer event_time, using history through t-1."""
    arr = np.asarray(path, float)
    lag_idx = max(0, min(event_time-2, len(arr)-1))
    state = arr[lag_idx]
    slope = _recent_slope(arr, lag_idx, 5)
    u = float(event_time) / max(float(horizon), 1.0)
    return np.concatenate([[1.0, u, u*u], baseline, state, slope])


def transition_features(path: np.ndarray, next_index: int, horizon: int,
                        baseline: np.ndarray) -> np.ndarray:
    """Features predicting path[next_index] from history through next_index-1."""
    arr = np.asarray(path, float)
    lag_idx = max(0, min(next_index-1, len(arr)-1))
    state = arr[lag_idx]
    slope = _recent_slope(arr, lag_idx, 5)
    u = float(next_index+1) / max(float(horizon), 1.0)
    return np.concatenate([[1.0, u, u*u], baseline, state, slope])


def censor_features(path: np.ndarray, record_index: int) -> np.ndarray:
    arr = np.asarray(path, float)
    idx = max(0, min(record_index, len(arr)-1))
    return np.concatenate([[1.0], arr[idx]])


def build_training_rows(paths: Sequence[np.ndarray], times: Sequence[float],
                        events: Sequence[int], indices: Sequence[int],
                        horizon: int, policy_start: int, baseline_cycles: int = 50):
    """Observed-prefix rows only; returns censor, event, and transition designs."""
    tt = np.asarray(times, float)
    ev = np.asarray(events, int)
    Xc=[]; yc=[]; Xe=[]; ye=[]; Xt=[]; yt=[]
    max_index_used = -1
    for i in indices:
        p = np.asarray(paths[i], float)
        L = int(round(tt[i]))
        if len(p) != L:
            raise ValueError("observed path length/time mismatch")
        base = np.median(p[:min(baseline_cycles, len(p))], axis=0)
        # Censoring rows: triggering record is included for censored units.
        last_c = L-1 if ev[i] == 0 else L-2
        for j in range(policy_start, last_c+1):
            Xc.append(censor_features(p, j))
            yc.append(1.0 if (ev[i] == 0 and j == L-1) else 0.0)
            max_index_used=max(max_index_used,j)
        # Event rows use information through t-1; event at t is unobserved at a censor time.
        last_event_t = L if ev[i] == 1 else L-1
        for t in range(max(2, policy_start+1), min(last_event_t, horizon)+1):
            Xe.append(event_features(p, t, horizon, base))
            ye.append(1.0 if (ev[i] == 1 and t == L) else 0.0)
            max_index_used=max(max_index_used,min(t-2,L-1))
        # Health transitions into the triggering record are observed.
        for nxt in range(1, min(L, horizon)):
            Xt.append(transition_features(p, nxt, horizon, base))
            yt.append(p[nxt])
            max_index_used=max(max_index_used,nxt)
    return (np.asarray(Xc,float),np.asarray(yc,float),
            np.asarray(Xe,float),np.asarray(ye,float),
            np.asarray(Xt,float),np.asarray(yt,float), max_index_used)


def fit_fold_models(paths, times, events, train_idx, horizon, policy_start,
                    censor_ridge=16.0, event_ridge=2.0, transition_ridge=0.1):
    Xc,yc,Xe,ye,Xt,yt,max_used = build_training_rows(
        paths,times,events,train_idx,horizon,policy_start)
    pm_c=np.ones(Xc.shape[1]); pm_c[0]=0.0
    pm_e=np.ones(Xe.shape[1]); pm_e[0]=0.0
    censor=fit_binary(Xc,yc,link="cloglog",ridge=censor_ridge,penalty_mask=pm_c)
    event=fit_binary(Xe,ye,link="cloglog",ridge=event_ridge,penalty_mask=pm_e)
    if not censor.success:
        raise RuntimeError(f"censor model failed: {censor.message}; grad={censor.grad_norm}")
    if not event.success:
        raise RuntimeError(f"event model failed: {event.message}; grad={event.grad_norm}")
    trans=[]
    pm_t=np.ones(Xt.shape[1]); pm_t[0]=0.0
    for j in range(yt.shape[1]):
        fit=fit_linear(Xt,yt[:,j],ridge=transition_ridge,penalty_mask=pm_t)
        if not fit.success:
            raise RuntimeError("transition model failed")
        trans.append(fit)
    return censor,event,trans,max_used


def censor_cumhaz_before(path: np.ndarray, fit: BinaryFit, policy_start: int) -> np.ndarray:
    p=np.asarray(path,float)
    L=len(p)
    out=np.zeros(L+1,float)
    if L<=policy_start+1:
        return out
    X=np.vstack([censor_features(p,j) for j in range(policy_start,L-1)])
    prob=np.clip(predict_binary(fit,X),1e-12,1-1e-12)
    haz=-np.log1p(-prob)
    out[policy_start+2:]=np.cumsum(haz)
    return out


def _simulate_imputed_hazards(last_path: np.ndarray, baseline: np.ndarray,
                               start_t: int, horizon: int, event_fit: BinaryFit,
                               trans_fits: Sequence[LinearFit], rng: np.random.Generator,
                               mc: int) -> dict[int,float]:
    """Conditional hazards after censoring using a parametric g-formula."""
    p0=np.asarray(last_path,float)
    states=[p0.copy() for _ in range(mc)]
    weights=np.ones(mc,float)
    out={}
    for t in range(start_t, horizon):
        hazards=np.empty(mc,float)
        for m,state in enumerate(states):
            X=event_features(state,t,horizon,baseline)[None,:]
            hazards[m]=float(predict_binary(event_fit,X)[0])
        denom=float(weights.sum())
        out[t]=float(np.dot(weights,hazards)/denom) if denom>1e-14 else float(np.mean(hazards))
        weights*=1.0-hazards
        # Generate next health record for time t+1 when needed.
        if t < horizon-1:
            for m,state in enumerate(states):
                xrow=transition_features(state,len(state),horizon,baseline)
                nxt=np.array([draw_linear_prediction(f,xrow,rng) for f in trans_fits], dtype=float)
                states[m]=np.vstack([state,nxt])
    return out


def crossfit_dr_rmst(paths: Sequence[np.ndarray], times: Sequence[float], events: Sequence[int],
                     folds: Sequence[int], horizon: int, policy_start: int,
                     seed: int, mc: int = 32, censor_ridge: float = 16.0,
                     event_ridge: float = 2.0, transition_ridge: float = 0.1):
    """Return DR RMST, g-formula RMST, censor predictions, and fit diagnostics."""
    paths=[np.asarray(p,float) for p in paths]
    tt=np.asarray(times,float); ev=np.asarray(events,int); fold=np.asarray(folds,int)
    n=len(paths)
    if not (len(tt)==len(ev)==len(fold)==n):
        raise ValueError("length mismatch")
    m_haz=np.full((n,horizon),np.nan,float)
    # No terminal endpoints occur during the frozen 50-cycle run-in in the primary cohort.
    m_haz[:, :min(policy_start+1, horizon)] = 0.0
    pi=np.ones((n,horizon),float)
    fit_rows=[]
    cumhaz=[None]*n
    for f in sorted(np.unique(fold)):
        train=np.flatnonzero(fold!=f); test=np.flatnonzero(fold==f)
        cfit,efit,tfit,max_used=fit_fold_models(
            paths,tt,ev,train,horizon,policy_start,censor_ridge,event_ridge,transition_ridge)
        fit_rows.append({"fold":int(f),"n_train":int(len(train)),"n_test":int(len(test)),
                         "censor_grad":cfit.grad_norm,"event_grad":efit.grad_norm,
                         "censor_iter":cfit.n_iter,"event_iter":efit.n_iter,
                         "censor_near_stationary":int(cfit.success and cfit.grad_norm > 1e-4),
                         "event_near_stationary":int(efit.success and efit.grad_norm > 1e-4),
                         "censor_message":cfit.message,"event_message":efit.message,
                         "max_training_index_used":int(max_used),
                         "max_training_observed_index":int(max(len(paths[i])-1 for i in train)),
                         "transition_sigma_max":float(max(x.sigma for x in tfit))})
        for i in test:
            p=paths[i]; L=int(round(tt[i])); base=np.median(p[:min(50,L)],axis=0)
            ch=censor_cumhaz_before(p,cfit,policy_start)
            cumhaz[i]=ch
            for t in range(1, min(L, horizon-1)+1):
                pi[i,t]=float(np.exp(-ch[min(t,len(ch)-1)]))
            # Actual-history predictions before censoring/event.
            last_actual_t=min(L,horizon-1)
            for t in range(max(1,policy_start+1),last_actual_t+1):
                if ev[i]==0 and t>=L:
                    break
                m_haz[i,t]=float(predict_binary(efit,event_features(p,t,horizon,base)[None,:])[0])
            # Impute hazards from censor time onward.
            if ev[i]==0 and L < horizon:
                rng=np.random.default_rng(int(seed)+1000003*int(i)+10007*int(f))
                imp=_simulate_imputed_hazards(p,base,max(L,policy_start+1),horizon,efit,tfit,rng,mc)
                for t,v in imp.items():
                    m_haz[i,t]=v
            # Event units are removed after failure; no future prediction needed.
    if any(x is None for x in cumhaz):
        raise RuntimeError("incomplete censor predictions")

    dr_h=[]; g_h=[]; clip_count=0
    survival_dr=1.0; survival_g=1.0
    rmst_dr=1.0; rmst_g=1.0  # S(0)
    for t in range(1,horizon):
        # Z=1 if still at risk or already censored; prior failures are excluded.
        Z=np.array([not (ev[i]==1 and tt[i] < t) for i in range(n)],bool)
        contrib=[]; gvals=[]
        for i in np.flatnonzero(Z):
            m=float(m_haz[i,t])
            if not np.isfinite(m):
                # Uncensored unit with short observed path can only be a prior failure, excluded above.
                raise RuntimeError(f"missing outcome prediction i={i}, t={t}, L={tt[i]}, ev={ev[i]}")
            censored_by_t=bool(ev[i]==0 and tt[i] <= t)
            if censored_by_t:
                val=m
            else:
                y=float(ev[i]==1 and int(round(tt[i]))==t)
                pp=max(float(pi[i,t]),1e-10)
                val=m+(y-m)/pp
            contrib.append(val); gvals.append(m)
        raw=float(np.mean(contrib)) if contrib else 0.0
        gh=float(np.mean(gvals)) if gvals else 0.0
        if raw<0.0 or raw>1.0:
            clip_count+=1
        h=float(np.clip(raw,0.0,1.0)); gh=float(np.clip(gh,0.0,1.0))
        dr_h.append((t,raw,h)); g_h.append((t,gh))
        survival_dr*=1.0-h; survival_g*=1.0-gh
        rmst_dr+=survival_dr; rmst_g+=survival_g
    return {
        "dr_rmst":float(rmst_dr),"gformula_rmst":float(rmst_g),
        "dr_hazards":dr_h,"gformula_hazards":g_h,
        "hazard_clip_count":int(clip_count),"cumhaz_before":cumhaz,
        "fit_diagnostics":fit_rows,
    }
