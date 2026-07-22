import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "matr_bootstrap_summarize.py"
spec = importlib.util.spec_from_file_location("matr_bootstrap_summarize", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def _rows():
    rows = []
    for outer, truth in [(0, 10.0), (1, 20.0)]:
        for inner in [0, 1]:
            for arm, estimate in [
                ("naive", 1.2 * truth),
                ("oracle_tv_ipcw", truth),
                ("crossfit_tv_ipcw", 1.1 * truth),
            ]:
                rows.append({
                    "outer_b": outer,
                    "batch_label": "B",
                    "beta": 1.0,
                    "inner_r": inner,
                    "arm": arm,
                    "estimate": estimate,
                    "truth_net_rmst": truth,
                    "fit_success": True,
                    "max_weight": 1.0,
                    "min_ess_over_risk": 1.0,
                    "exp_clipping": 0.0,
                    "solver_fallback": 0.0,
                })
    return pd.DataFrame(rows)


def _points():
    return pd.DataFrame([
        {"beta": 1.0, "scope": "B", "arm": "naive", "mean_signed_gap_pct": 20.0},
        {"beta": 1.0, "scope": "B", "arm": "oracle_tv_ipcw", "mean_signed_gap_pct": 0.0},
        {"beta": 1.0, "scope": "B", "arm": "crossfit_tv_ipcw", "mean_signed_gap_pct": 10.0},
    ])


def test_resample_specific_truth_is_used():
    out = mod.summarize(_rows(), _points())
    naive = out[out.statistic == "naive_signed_gap_pct"].iloc[0]
    signed = out[out.statistic == "crossfit_minus_naive_signed_error_pp"].iloc[0]
    absolute = out[out.statistic == "crossfit_minus_naive_absolute_error_pp"].iloc[0]
    assert np.isclose(naive.outer_mean, 20.0)
    assert np.isclose(signed.outer_mean, -10.0)
    assert np.isclose(absolute.outer_mean, -10.0)


def test_missing_resample_truth_is_rejected():
    d = _rows().drop(columns=["truth_net_rmst"])
    with pytest.raises(ValueError, match="truth_net_rmst"):
        mod.prepare_bootstrap_data(d)


def test_inconsistent_truth_within_outer_batch_is_rejected():
    d = _rows()
    d.loc[d.index[0], "truth_net_rmst"] = 999.0
    with pytest.raises(ValueError, match="inconsistent resample truth"):
        mod.prepare_bootstrap_data(d)
