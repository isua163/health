import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "matr_ridge_selection_reaudit.py"
spec = importlib.util.spec_from_file_location("matr_ridge_selection_reaudit", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_current_weight_diagnostic_names_are_mapped():
    got = mod._normalise_weight_diagnostic({
        "max_weight": 3.0,
        "ess_over_n_at_risk": 0.75,
        "n_exp_clipped": 2,
    })
    assert got == {
        "max_weight": 3.0,
        "ess_over_risk_set": 0.75,
        "exp_clipping_count": 2.0,
    }


def test_legacy_names_remain_accepted():
    got = mod._normalise_weight_diagnostic({
        "ess_over_risk_set": 0.8,
        "exp_clipping_count": 0,
    })
    assert np.isnan(got["max_weight"])
    assert got["ess_over_risk_set"] == 0.8
    assert got["exp_clipping_count"] == 0.0
