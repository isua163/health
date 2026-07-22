"""Arm (1): naive regression RUL. FROZEN label = observed time min(T, C).

Regresses the observed (possibly censored) label on the health index, IGNORING
censoring. Because censored units carry label C < T, the learned E[min(T,C)|x] is
below E[T|x] -> this arm is net-PESSIMISTIC (opposite sign to the survival arm).
Its bias sign is set by this label convention; reported by method class.

Point regressor: log-linear OLS  log(Ttil) ~ [1, log(x_obs)]. Swap in any
regressor / trajectory features for real data. RUL = clip(pred_lifetime - t_L, 0, H).
"""
import numpy as np
from .base import Arm, Dataset


class NaiveRegressionArm(Arm):
    name = "naive_regression"
    LABEL = "min(T,C)"          # FROZEN label convention

    def __init__(self, feature="logx"):
        self.feature = feature

    def _design(self, x):
        lx = np.log(np.maximum(x, 1e-8))
        return np.column_stack([np.ones_like(lx), lx])

    def fit(self, ds: Dataset) -> "NaiveRegressionArm":
        y = np.log(np.maximum(ds.Ttil, 1e-8))       # label = observed time (ignores censoring)
        self.coef_, *_ = np.linalg.lstsq(self._design(ds.x_obs), y, rcond=None)
        return self

    def predict_rmrl(self, ds: Dataset, t_L: float, H: float) -> np.ndarray:
        life = np.exp(self._design(ds.x_obs) @ self.coef_)
        return np.clip(life - t_L, 0.0, H)          # per-unit (covariate-conditional)
