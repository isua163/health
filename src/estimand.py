"""Estimand (protocol section 2): the NET / marginal RUL, from full trajectories.

Truth is evaluator-only. Arms never receive it -> estimator-blinded by construction.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class Truth:
    T: np.ndarray            # net failure time (all units, incl. those censored in the observed data)
    Z: np.ndarray            # latent health state (reference / diagnostics only)


def net_rmrl_truth(truth, t_L, H):
    """Per-unit net restricted mean residual life truth at landmark t_L.
    Returns (value, at_risk_mask). Only units with T > t_L are in the RUL population."""
    at_risk = truth.T > t_L
    value = np.clip(truth.T - t_L, 0.0, H)
    return value, at_risk


def closed_form_net_survival(t, net):
    """Marginal (net) survival of the synthetic generator: S_T(t) = (1 + theta t^m)^(-k).
    Used as a noise-free reference in tests."""
    k, theta, m = net["k"], net["theta"], net["m"]
    t = np.asarray(t, float)
    return (1.0 + theta * t**m) ** (-k)
