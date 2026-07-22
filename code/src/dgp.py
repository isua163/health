"""Data-generating process (protocol section 4).

Two layers:
  - net layer: latent health Z, net failure T (never simulated on real data).
  - censoring overlay: health-dependent hazard lambda_C = lambda0 * exp(beta*(HI - tau)),
    realized here in the shared-frailty form lambda_C(t|Z) = c0 * Z^beta.

The analyst observes only x_obs = Z * exp(sigma * eps)  (a NOISY health index).
The latent Z and net T (for censored units) live in Truth, seen ONLY by the
evaluator -- this makes the estimator-blinding constraint architectural.
"""
import numpy as np
from .arms.base import Dataset
from .estimand import Truth

# Frozen net-DGP parameters (mirror configs/grids.yaml -> synthetic).
NET = dict(k=2.0, theta=0.5, m=2.0)


def calibrate_c0(beta, c_target, rng, npilot=1_000_000, net=NET):
    """Solve lambda0-scale (c0) so the marginal censoring fraction hits c_target,
    decoupling censoring STRENGTH (c) from INFORMATIVENESS (beta)."""
    k, theta, m = net["k"], net["theta"], net["m"]
    Z = rng.gamma(k, theta, npilot)
    T = (rng.exponential(1.0, npilot) / Z) ** (1.0 / m)
    Ep = rng.exponential(1.0, npilot)
    return float(np.quantile(Ep / (T * Z**beta), c_target))


def sample_synthetic(n, beta, sigma, c0, rng, net=NET):
    """Return (observed Dataset, Truth, realized_censoring_fraction)."""
    k, theta, m = net["k"], net["theta"], net["m"]
    Z = rng.gamma(k, theta, n)
    T = (rng.exponential(1.0, n) / Z) ** (1.0 / m)          # NET failure (kept for all units)
    C = rng.exponential(1.0, n) / (c0 * Z**beta)            # health-dependent censoring
    Ttil = np.minimum(T, C)
    event = (T <= C).astype(float)                          # 1=failure, 0=censored (delta)
    x_obs = Z * np.exp(sigma * rng.standard_normal(n))      # observed noisy health index
    ds = Dataset(unit_id=np.arange(n), Ttil=Ttil, event=event, x_obs=x_obs)
    truth = Truth(T=T, Z=Z)
    return ds, truth, float(1.0 - event.mean())
