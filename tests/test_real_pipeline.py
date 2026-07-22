"""Guard 4: the real-data path works end-to-end WITHOUT the real data downloaded.

Fabricates XJTU-format CSVs and HI trajectories to exercise every function:
HI computation, the (header-tolerant) CSV reader, directory loading, and the
informative-censoring overlay -> confirms naive-survival optimism is reproduced
on the real-data code path before pointing it at gigabytes of vibration.
"""
import os
import sys
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from src.hi import rms, kurtosis, compute_hi
from src.data_xjtu import (load_hi_trajectory_csv, load_all_hi, calibrate_lambda0,
                           impose_informative_censoring, _read_record)
from src.estimand import net_rmrl_truth
from src.metrics import bias_by_class
from src.arms.naive_survival import NaiveSurvivalArm
from src._survival import rmrl_from_survival


def test_hi_functions():
    assert abs(rms(np.full(1000, 2.0)) - 2.0) < 1e-9
    g = np.random.default_rng(0).standard_normal(50000)
    assert abs(kurtosis(g) - 3.0) < 0.3                      # Gaussian kurtosis ~ 3
    rec = np.column_stack([np.full(100, 3.0), np.zeros(100)])  # 2-channel record
    assert abs(compute_hi(rec, "rms", channel=0) - 3.0) < 1e-9


def test_csv_reader_and_directory():
    with tempfile.TemporaryDirectory() as d:
        bdir = os.path.join(d, "35Hz12kN", "Bearing1_1")
        os.makedirs(bdir)
        rng = np.random.default_rng(1)
        n_rec = 20
        for i in range(1, n_rec + 1):
            amp = 0.5 + 0.1 * i                              # amplitude grows -> RMS rises
            sig = amp * rng.standard_normal((256, 2))
            header = "Horizontal_vibration_signals,Vertical_vibration_signals\n" if i % 2 == 0 else ""
            with open(os.path.join(bdir, f"{i}.csv"), "w") as f:
                f.write(header)
                np.savetxt(f, sig, delimiter=",")
        # header-tolerant read
        assert _read_record(os.path.join(bdir, "2.csv")).shape == (256, 2)
        traj = load_hi_trajectory_csv(bdir, "rms", 0)
        assert len(traj) == n_rec
        assert traj[-1] > traj[0]                            # HI increases over life
        trajs, names = load_all_hi(d, "rms", 0, cache=False)
        assert len(trajs) == 1 and len(trajs[0]) == n_rec


def test_overlay_reproduces_optimism():
    """Fabricate informatively-censored bearings (frailer -> shorter life + higher HI);
    naive KM must overestimate net RMST."""
    rng = np.random.default_rng(2)
    hi_trajs = []
    for _ in range(400):
        z = rng.gamma(2.0, 0.5)                              # latent frailty
        life = max(20, int(120 / z))                         # frailer -> shorter
        base = 0.3 + 0.4 * z                                 # frailer -> higher HI level
        hi = base + 0.5 * (np.arange(life) / life) ** 2 * z + 0.03 * rng.standard_normal(life)
        hi_trajs.append(np.maximum(hi, 1e-3))
    T_net = np.array([len(h) for h in hi_trajs], float)
    H = float(np.percentile(T_net, 90))
    tau = float(np.percentile(np.concatenate(hi_trajs), 70))
    lam0 = calibrate_lambda0(hi_trajs, beta=2.0, tau=tau, c_target=0.4)
    ds, truth = impose_informative_censoring(hi_trajs, 2.0, tau, lam0, sigma=0.0, rng=rng)
    cens_frac = 1.0 - ds.event.mean()
    assert 0.25 < cens_frac < 0.55, f"calibration off: cens={cens_frac:.2f}"
    val, atrisk = net_rmrl_truth(truth, t_L=0.0, H=H)
    pred = NaiveSurvivalArm().fit(ds).predict_rmrl(ds, 0.0, H)
    bias_pct = 100 * (np.mean(pred[atrisk]) - np.mean(val[atrisk])) / np.mean(val[atrisk])
    assert bias_pct > 2.0, f"expected naive-survival optimism on real path, got {bias_pct:.1f}%"
    # assertion above is the test outcome; do not return a value from a pytest test.


def test_mat_reader():
    """Fabricate per-bearing .mat files (rawnet = samples x channels x records) and
    verify the .mat path loads rising HI trajectories of correct length."""
    import scipy.io as sio
    from src.data_xjtu import load_all_hi, _detect_format
    with tempfile.TemporaryDirectory() as d:
        orig = os.path.join(d, "originaldata")
        os.makedirs(orig)
        rng = np.random.default_rng(3)
        lifetimes = {1: 15, 2: 25}
        for bid, n_rec in lifetimes.items():
            amp = (0.5 + 0.08 * np.arange(n_rec))                 # amplitude rises over life
            rawnet = amp[None, None, :] * rng.standard_normal((512, 2, n_rec))
            sio.savemat(os.path.join(orig, f"bearing{bid}.mat"),
                        {"rawnet": rawnet, "raw": rawnet[:, :, 0]})
        assert _detect_format(d) == "mat"
        trajs, names = load_all_hi(d, "rms", 0, cache=False)
        assert len(trajs) == 2
        assert sorted(len(t) for t in trajs) == [15, 25]          # lifetimes recovered
        assert trajs[0][-1] > trajs[0][0]                          # HI rises over life


def test_tv_ipcw_recovers():
    """Guard 5: with adequate N and sigma=0 (correct time-varying model), time-varying
    IPCW recovers net RMST -- i.e., the correction demonstrably works (H3)."""
    from src.arms.ipcw_correction import TimeVaryingIPCWArm
    rng = np.random.default_rng(0)
    hi_trajs = []
    for _ in range(200):
        z = rng.gamma(2.0, 0.5); life = max(30, int(150 / z)); base = 0.3 + 0.5 * z
        h = base + 1.0 * (np.arange(life) / life) ** 1.5 * (0.5 + z) + 0.03 * rng.standard_normal(life)
        hi_trajs.append(np.maximum(h, 1e-3))
    T = np.array([len(h) for h in hi_trajs], float); H = float(np.percentile(T, 90))
    tau = float(np.percentile(np.concatenate(hi_trajs), 70))
    lam0 = calibrate_lambda0(hi_trajs, 1.0, tau, 0.4)
    rng2 = np.random.default_rng(1); nb, tb = [], []
    for _ in range(40):
        ds, truth = impose_informative_censoring(hi_trajs, 1.0, tau, lam0, 0.0, rng2)
        val, at = net_rmrl_truth(truth, 0.0, H)
        nb.append(bias_by_class(NaiveSurvivalArm().fit(ds).predict_rmrl(ds, 0.0, H), val, at)["bias_pct"])
        tb.append(bias_by_class(TimeVaryingIPCWArm().fit(ds).predict_rmrl(ds, 0.0, H), val, at)["bias_pct"])
    naive_bias, tv_bias = float(np.mean(nb)), float(np.mean(tb))
    assert naive_bias > 5, f"expected naive optimism, got {naive_bias:.1f}%"
    assert abs(tv_bias) < 5, f"TV-IPCW should recover net (|bias|<5%), got {tv_bias:.1f}%"
    assert abs(tv_bias) < naive_bias / 2, "TV-IPCW should more than halve the naive bias"
    # assertions above are the test outcome; do not return a value from a pytest test.


def test_femto_reader():
    """Fabricate FEMTO acc_*.csv (6 cols; vibration in the LAST two) plus a temp_ file to
    ignore, including a semicolon-delimited bearing; verify the FEMTO branch reads correctly
    and the overlay + naive arm run end-to-end."""
    from src.data_xjtu import _detect_format, load_all_hi, build_from_directory
    from src.arms.naive_survival import NaiveSurvivalArm
    with tempfile.TemporaryDirectory() as d:
        rng = np.random.default_rng(4)
        for bi, delim in [(1, ","), (2, ";"), (3, ",")]:
            bdir = os.path.join(d, "Learning_set", f"Bearing1_{bi}")
            os.makedirs(bdir)
            n_rec = 20 + 5 * bi
            for i in range(1, n_rec + 1):
                amp = 0.5 + 0.08 * i                              # RMS rises over life
                rec = np.column_stack([np.zeros((128, 4)),        # dummy timestamp cols
                                       amp * rng.standard_normal(128),   # horizontal (col 4)
                                       amp * rng.standard_normal(128)])  # vertical   (col 5)
                np.savetxt(os.path.join(bdir, f"acc_{i:05d}.csv"), rec, delimiter=delim)
            np.savetxt(os.path.join(bdir, "temp_00001.csv"),      # MUST be ignored (5 cols)
                       rng.standard_normal((60, 5)), delimiter=delim)
        assert _detect_format(d) == "femto"
        trajs, names = load_all_hi(d, "rms", 0, cache=False)
        assert len(trajs) == 3
        assert sorted(len(t) for t in trajs) == [25, 30, 35]      # #acc files (temp ignored)
        assert trajs[0][-1] > trajs[0][0]                          # HI rises over life
        # end-to-end: overlay + naive arm run without error
        ds, truth, meta = build_from_directory(d, beta=1.0, sigma=0.0, c_target=0.4,
                                               rng=np.random.default_rng(0), cache=False)
        assert meta["n_bearings"] == 3 and ds.hi_obs is not None


def test_cv_sweep_reproduces():
    """Guard 6: the controlled dispersion sweep (legacy synthetic backbone of the figure) is
    reproducible and monotone-rising in log(p90/p10). Monotonicity must hold under the
    frozen iqr transform (D1 v3) AND raw HI (the ordering-agreement D1 requires); the bias
    MAGNITUDE is HI-scale-dependent, so the double-digit check is raw-HI only."""
    from src.cv_sweep import run as cv_run
    from scipy.stats import spearmanr
    res = {}
    with tempfile.TemporaryDirectory(prefix="ress_test_cv_") as outdir:
        for method in ["none", "iqr"]:
            rows = cv_run(beta=1.0, sigma=0.5, N=40, seed=0, standardize=method,
                          outdir=outdir)   # deterministic, never writes formal results
            res[method] = rows
            disp = [r["log_p90_p10"] for r in rows]; bias = [r["bias"] for r in rows]
            assert disp[0] < 0.6 and disp[-1] > 3.0, f"{method}: dispersion should span low->high"
            rho, _ = spearmanr(disp, bias)
            assert rho > 0.8, f"{method}: sweep should be monotone in dispersion, rho={rho:.2f}"
    assert max(r["bias"] for r in res["none"]) > 15, "raw-HI bias should reach double digits"


if __name__ == "__main__":
    test_hi_functions()
    test_csv_reader_and_directory()
    test_mat_reader()
    test_femto_reader()
    test_overlay_reproduces_optimism()
    test_tv_ipcw_recovers()
    test_cv_sweep_reproduces()
    print("PASS test_real_pipeline  (mat/csv/femto readers OK; naive optimism confirmed; "
          "TV-IPCW recovery guard passed)")


def test_condition_inference_and_unit_equal_health_threshold():
    from src.data_xjtu import infer_xjtu_conditions, health_threshold

    names = [f"originaldata/bearing{i}.mat" for i in range(1, 16)]
    labels = infer_xjtu_conditions(names)
    assert len(set(labels[:5])) == 1 and len(set(labels[5:10])) == 1
    assert labels[0] != labels[5] != labels[10]

    # Record weighting lets the long trajectory dominate; unit weighting gives
    # each bearing equal total mass and therefore a different threshold.
    hi = [np.zeros(100), np.array([10.0, 11.0])]
    record_q = health_threshold(hi, 70, "record")
    unit_q = health_threshold(hi, 70, "unit")
    assert record_q == 0.0
    assert unit_q >= 10.0
