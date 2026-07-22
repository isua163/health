"""Arm interface and the OBSERVED-ONLY Dataset (protocol section 5).

Dataset carries only what a deployed analyst sees: observed time, event, and an
observed health-index summary x_obs. It deliberately does NOT expose the latent
health state Z or the net failure time of censored units -- so no arm can use
oracle information. Ground truth lives in estimand.Truth, held by the evaluator.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class Dataset:
    unit_id: np.ndarray      # unit key; time-window rows of one unit share a unit_id (no-leakage granularity)
    Ttil: np.ndarray         # observed time = min(T, C)
    event: np.ndarray        # 1 = failure observed, 0 = preventive replacement / censored (delta)
    x_obs: np.ndarray        # observed health-index summary (baseline covariate); NO latent Z
    hi_obs: object = None     # optional: list of observed (noised) HI trajectories, one per unit,
    #                           truncated at Ttil_i -- for time-varying censoring models


class Arm(ABC):
    """A RUL method. fit() sees observed training data only; predict_rmrl() returns
    a per-unit restricted mean residual life at landmark t_L over horizon H."""
    name: str = "arm"

    @abstractmethod
    def fit(self, ds: Dataset) -> "Arm":
        ...

    @abstractmethod
    def predict_rmrl(self, ds: Dataset, t_L: float, H: float) -> np.ndarray:
        """Return array of length len(ds.unit_id) (marginal arms broadcast a scalar)."""
        ...
