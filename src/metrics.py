"""Metrics (protocol section 6): bias BY METHOD CLASS + positivity diagnostic."""
import numpy as np


def bias_by_class(pred_rmrl, truth_val, at_risk_mask):
    """Signed net restricted-mean lifetime bias on units at risk at the landmark. Labels over/under."""
    p = np.asarray(pred_rmrl)[at_risk_mask]
    t = np.asarray(truth_val)[at_risk_mask]
    b = float(np.mean(p - t))
    denom = float(np.mean(t)) if np.mean(t) != 0 else np.nan
    return dict(bias=b, bias_pct=100.0 * b / denom, sign=("over" if b > 0 else "under"),
                n=int(at_risk_mask.sum()))


def ess_over_n(weights):
    w = np.asarray(weights, float)
    return float((w.sum() ** 2) / (np.sum(w**2) * len(w)))


def bootstrap_bias_ci(pred_rmrl, truth_val, at_risk_mask, unit_id=None, B=1000, seed=0):
    """Bootstrap CI of the bias, resampling UNITS (protocol section 6)."""
    rng = np.random.default_rng(seed)
    p = np.asarray(pred_rmrl)[at_risk_mask]
    t = np.asarray(truth_val)[at_risk_mask]
    n = len(t)
    stats = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        stats[b] = np.mean(p[idx] - t[idx])
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))
