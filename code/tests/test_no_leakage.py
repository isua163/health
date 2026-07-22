"""Guard 1 (protocol section 7): no unit's rows cross the train/test split."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.arms.base import Dataset
from src.splits import unit_level_split, subset


def test_split_disjoint_and_complete():
    uids = np.arange(20)
    tr, te = unit_level_split(uids, seed=0)
    assert tr.isdisjoint(te), "train/test unit sets overlap"
    assert tr | te == set(uids.tolist()), "split does not cover all units"


def test_windows_stay_with_unit():
    # 10 units, 5 time-window rows each -> subset must keep every unit whole
    unit_id = np.repeat(np.arange(10), 5)
    ds = Dataset(unit_id=unit_id, Ttil=np.ones(50), event=np.ones(50), x_obs=np.ones(50))
    tr, te = unit_level_split(unit_id, seed=1)
    ds_tr, ds_te = subset(ds, tr), subset(ds, te)
    assert set(ds_tr.unit_id.tolist()).isdisjoint(set(ds_te.unit_id.tolist()))
    assert len(ds_tr.unit_id) + len(ds_te.unit_id) == 50


if __name__ == "__main__":
    test_split_disjoint_and_complete(); test_windows_stay_with_unit()
    print("PASS test_no_leakage")


def test_static_baseline_feature_does_not_depend_on_future_lifetime_or_censoring():
    from src.data_xjtu import impose_informative_censoring, static_baseline_summary

    # Same first record, deliberately different complete tails and lifetimes.
    hi = [
        np.array([2.0, 10.0, 20.0]),
        np.array([2.0, 3.0, 4.0, 5.0, 100.0]),
    ]
    ds, _ = impose_informative_censoring(
        hi, beta=0.0, tau=0.0, lam0=0.0, sigma=0.0,
        rng=np.random.default_rng(10), baseline_records=1,
    )
    assert np.allclose(ds.x_obs, [2.0, 2.0])

    # The helper rejects an unavailable fixed window rather than silently using
    # a censoring-dependent shorter window.
    try:
        static_baseline_summary(np.array([1.0]), n_records=2)
    except ValueError:
        pass
    else:
        raise AssertionError("unavailable fixed baseline window must raise")
