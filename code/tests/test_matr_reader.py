"""Fixture test for the MATR/Severson reader, on a FABRICATED HDF5 .mat that replicates the
CONFIRMED real layout (from this project's h5py dump):
    f['batch'] group; per-cell fields shape (n,1) of HDF5 object refs;
    batch['summary'][i] -> group with QDischarge (1, n_cycles);  batch['cycle_life'][i] -> (1,1);
    the first summary row is an all-zero placeholder (QDischarge[0]==0).
Run this on the DATA MACHINE (needs h5py) BEFORE pointing the reader at the real GB-scale .mat.
It verifies the dereferencing, placeholder trim, D6-iv cross-check, and overlay integration.
"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from src.battery_data import (_load_matr_capacity, matr_crosscheck, load_battery_hi,
                              Q_NOM_SEVERSON)
from src.standardize import standardize_hi
from src.data_xjtu import calibrate_lambda0, impose_informative_censoring
from src.estimand import net_rmrl_truth
from src.metrics import bias_by_class
from src.arms.naive_survival import NaiveSurvivalArm


def _make_matr_mat(path, cycle_lives, q_nom=Q_NOM_SEVERSON, seed=0):
    """Fabricate one MATR batch .mat replicating the confirmed HDF5 layout (refs + placeholder)."""
    import h5py
    try:
        ref_dt = h5py.ref_dtype
    except AttributeError:
        ref_dt = h5py.special_dtype(ref=h5py.Reference)
    rng = np.random.default_rng(seed)
    n = len(cycle_lives)
    with h5py.File(path, "w") as f:
        batch = f.create_group("batch")
        summ = np.empty((n, 1), dtype=ref_dt); clife = np.empty((n, 1), dtype=ref_dt)
        for i, cl in enumerate(cycle_lives):
            # fade 1.07 -> ~0.878 over cl cycles (crosses 0.88 EOL near the end); leading 0 placeholder
            q = np.concatenate([[0.0], np.linspace(1.07, 0.878, cl) + 3e-4 * rng.standard_normal(cl)])
            g = f.create_group(f"summ_{i}")
            g.create_dataset("QDischarge", data=q.reshape(1, -1))
            g.create_dataset("cycle", data=(np.arange(len(q)) + 1.0).reshape(1, -1))
            summ[i, 0] = g.ref
            d = f.create_dataset(f"cl_{i}", data=np.array([[float(cl)]]))
            clife[i, 0] = d.ref
        batch.create_dataset("summary", data=summ)
        batch.create_dataset("cycle_life", data=clife)


def test_matr_reader_and_integration():
    try:
        import h5py  # noqa: F401
    except ImportError:
        print("SKIP test_matr_reader: h5py not installed  (conda install -c conda-forge h5py -y)")
        return
    cls = [200, 500, 900, 1400, 1190, 350]              # varied cycle lives -> real dispersion
    with tempfile.TemporaryDirectory() as d:
        _make_matr_mat(os.path.join(d, "2017-05-12_batchdata_updated_struct_errorcorrect.mat"), cls)

        caps, clives, names = _load_matr_capacity(d)                       # (1) parser
        assert len(caps) == len(cls), f"got {len(caps)} cells, expected {len(cls)}"
        assert all(c[0] > 0.5 for c in caps), "leading placeholder (Q~0) not trimmed"
        assert sorted(int(round(x)) for x in clives) == sorted(cls), "cycle_life mismatch"
        print(f"  [1] parser OK: {len(caps)} cells; leading-0 trimmed; cycle_lives recovered")

        rows = matr_crosscheck(d)                                          # (2) D6-iv cross-check
        rels = [r[3] for r in rows if not np.isnan(r[3])]
        assert np.median(rels) < 10, f"cross-check median rel err {np.median(rels):.1f}% too high"
        print(f"  [2] cross-check OK: our EOL vs published cycle_life, median rel err {np.median(rels):.1f}%")

        hi_trajs, nm = load_battery_hi(d, source="matr")                   # (3) HI trajectories
        assert len(hi_trajs) == len(cls)
        for hi in hi_trajs:
            assert hi[-1] > hi[0], "HI should increase over life"
        assert sorted(len(h) for h in hi_trajs) == sorted(cls), "T_i should equal published cycle_life"
        print(f"  [3] load_battery_hi OK: {len(hi_trajs)} increasing HI trajs (T_i=cycle_life)")

        hi_std = standardize_hi(hi_trajs, "iqr")                           # (4) overlay integration
        T = np.array([len(h) for h in hi_std], float); H = float(np.percentile(T, 90))
        tau = float(np.percentile(np.concatenate(hi_std), 70))
        rng = np.random.default_rng(1)
        lam0 = calibrate_lambda0(hi_std, 2.0, tau, 0.4)
        ds, truth = impose_informative_censoring(hi_std, 2.0, tau, lam0, 0.0, rng)
        val, at = net_rmrl_truth(truth, 0.0, H)
        bp = bias_by_class(NaiveSurvivalArm().fit(ds).predict_rmrl(ds, 0.0, H), val, at)["bias_pct"]
        print(f"  [4] overlay integration OK on MATR-format HI: naive-survival bias {bp:+.1f}%")


if __name__ == "__main__":
    test_matr_reader_and_integration()
    print("PASS test_matr_reader")
