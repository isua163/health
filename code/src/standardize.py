"""HI standardization for cross-fleet comparability (pre-reg v2 addendum, D1).

WHY: the censoring overlay lambda_C = lam0*exp(beta*(HI - tau)) makes a given beta
a DIFFERENT-strength intervention across fleets whose HI live on different scales.
To put beta on a common scale across fleets, standardize each fleet's HI BEFORE the
overlay. This module is used ONLY for the cross-fleet dose-response and its synthetic
backbone (so real & synthetic share a scale for the D3 prediction test); the single-
fleet five-arm table keeps raw HI.

FROZEN (pre-reg v3, D6-i): the PRIMARY transform is 'iqr'. The module still exposes all
three behind `method` (default 'none' = no silent commitment):
  * 'iqr'   PRIMARY -> divide each fleet's HI by its within-fleet IQR: puts beta on a common
                      "IQR-of-HI" scale across fleets WHILE PRESERVING the HI's dynamic shape
                      (the end-of-life steep rise that carries the informative-censoring signal).
                      Outlier-robust (deliberately NOT SD-based, per the CV-failure result).
  * 'rank'  REJECTED (v3) -> within-fleet empirical CDF (HI ~ Uniform[0,1]). It equalizes scale
                      but ABOLISHES the end-of-life HI steepness: compressing the marginal to
                      uniform flattens the sharp end rise, so near-failure units no longer stand
                      out -> selection is artificially weakened and the bias is drastically
                      UNDER-estimated (empirically: XJTU real bias +34% raw / +32% iqr -> +5.6%
                      rank). Kept only as a documented counter-example; see test_standardize.
  * 'none'  raw HI: scale not comparable across fleets (single-fleet diagnostics only).

All transforms are within-fleet and MONOTONE, so they preserve: the Proposition direction
(lambda_obs <= lambda_T), the beta=0 null cell, the frail->short->high-HI mechanism, and
time-varying IPCW correctness (overlay & arms share the transform). Dispersion log(p90/p10) is
a LIFETIME property, untouched here -> a fleet's x-coordinate is invariant; only its bias moves.
"""
from dataclasses import dataclass

import numpy as np


def _pooled(hi_trajs):
    return np.concatenate([np.asarray(h, float) for h in hi_trajs])


@dataclass(frozen=True)
class HIStandardizer:
    """Fitted fleet-level HI transform that can be reused in bootstrap samples."""

    method: str
    scale: float = 1.0
    reference: object = None


def fit_hi_standardizer(hi_trajs, method="none"):
    """Fit a transform once, without applying it.

    Reusing the fitted object is essential when a bootstrap analysis is meant
    to hold the intervention policy and HI scale fixed across resamples.
    """
    pooled = _pooled(hi_trajs)
    if method == "none":
        return HIStandardizer("none")
    if method == "iqr":
        q1, q3 = np.percentile(pooled, [25, 75])
        return HIStandardizer("iqr", scale=max(float(q3 - q1), 1e-9))
    if method == "rank":
        return HIStandardizer("rank", reference=np.sort(pooled))
    raise ValueError(f"unknown standardization method: {method!r}")


def apply_hi_standardizer(hi_trajs, fitted):
    """Apply a previously fitted :class:`HIStandardizer`."""
    if fitted.method == "none":
        return [np.asarray(h, float) for h in hi_trajs]
    if fitted.method == "iqr":
        return [np.asarray(h, float) / fitted.scale for h in hi_trajs]
    if fitted.method == "rank":
        sp = np.asarray(fitted.reference, float)
        n = len(sp)
        return [np.searchsorted(sp, np.asarray(h, float), side="right") / n for h in hi_trajs]
    raise ValueError(f"unknown fitted standardization method: {fitted.method!r}")


def iqr_standardize(hi_trajs):
    """PRIMARY (v3). Divide each fleet's HI by its within-fleet IQR: common cross-fleet scale,
    HI dynamic shape (end-of-life steep rise) preserved, outlier-robust."""
    return apply_hi_standardizer(hi_trajs, fit_hi_standardizer(hi_trajs, "iqr"))


def rank_normalize(hi_trajs):
    """REJECTED (v3) -- kept as a documented counter-example. Maps each fleet's HI to its
    within-fleet empirical CDF (Uniform[0,1]); this ABOLISHES end-of-life HI steepness and
    under-estimates the informative-censoring bias. Do NOT use as the frozen transform."""
    return apply_hi_standardizer(hi_trajs, fit_hi_standardizer(hi_trajs, "rank"))


def standardize_hi(hi_trajs, method="none"):
    """Dispatch. method in {'none','iqr','rank'}. FROZEN primary = 'iqr' (v3);
    'rank' is a rejected counter-example; 'none' = raw HI (no commitment)."""
    return apply_hi_standardizer(hi_trajs, fit_hi_standardizer(hi_trajs, method))
