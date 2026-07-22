"""Arm (2): naive survival RUL under the INDEPENDENT-censoring assumption.

Estimates net survival by plain Kaplan-Meier on (Ttil, event), ignoring the
health-dependent (informative) censoring. This is the arm the Proposition proves
optimistic under positive dependence.
"""
import numpy as np
from .base import Arm, Dataset
from .._survival import km, rmrl_from_survival, step_eval


class NaiveSurvivalArm(Arm):
    name = "naive_survival"

    def fit(self, ds: Dataset) -> "NaiveSurvivalArm":
        self.grid_, self.surv_, _ = km(ds.Ttil, ds.event)
        return self

    def predict_rmrl(self, ds: Dataset, t_L: float, H: float) -> np.ndarray:
        val = rmrl_from_survival(self.grid_, self.surv_, t_L, H)
        return np.full(len(ds.unit_id), val)

    def survival(self, grid):
        """Estimated net survival on an arbitrary grid (used by the Proposition test)."""
        return step_eval(np.asarray(grid), self.grid_, self.surv_)
