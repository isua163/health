import importlib.util
from pathlib import Path
import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "matr_bootstrap_design.py"
spec = importlib.util.spec_from_file_location("matr_bootstrap_design", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def test_grouped_fold_assignment_covers_active_sources():
    counts=np.array([3,0,2,1,0,4],int)
    assignment,load,groups=mod.grouped_folds(counts,np.random.default_rng(4),n_folds=3)
    assert set(assignment)==set(np.flatnonzero(counts))
    assert int(load.sum())==int(counts.sum())
    assert int(groups.sum())==int(np.sum(counts>0))

def test_grouped_fold_assignment_is_deterministic():
    counts=np.array([1,2,3,4,5],int)
    a=mod.grouped_folds(counts,np.random.default_rng(9),n_folds=3)
    b=mod.grouped_folds(counts,np.random.default_rng(9),n_folds=3)
    assert a[0]==b[0]
    assert np.array_equal(a[1],b[1])
    assert np.array_equal(a[2],b[2])

def test_condition_inference_fallback_is_three_groups_of_five():
    names=[f"originaldata/bearing{i}.mat" for i in range(1,16)]
    from src.data_xjtu import infer_xjtu_conditions
    labels=infer_xjtu_conditions(names)
    assert list(labels[:5])==["35Hz_12kN"]*5
    assert list(labels[5:10])==["37.5Hz_11kN"]*5
    assert list(labels[10:])==["40Hz_10kN"]*5
