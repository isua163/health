"""XJTU-SY real-data path (auto-detects .mat or .csv), then imposes the SAME
health-dependent censoring overlay (protocol section 4) on the real trajectory.
Net truth T_i = lifetime in records (health-threshold endpoint; note in datasets.yaml).

Two on-disk formats are supported:
  * CSV (GitHub WangBiaoXJTU): <root>/<cond>/<Bearing_i_j>/1.csv,2.csv,...
        each CSV: col0=horizontal, col1=vertical.
  * MAT (Mendeley mpn45f4gxc): <root>/originaldata/bearing{i}.mat, ONE file per
        bearing, variable 'rawnet' of shape (n_samples, n_channels, n_records).
"""
import os
import re
import glob
import numpy as np

from .hi import HI_REGISTRY
from .arms.base import Dataset
from .estimand import Truth


# ---------- format detection ----------
def _detect_format(root):
    has_acc = has_mat = has_csv = False
    for _, _, fs in os.walk(root):
        low = [f.lower() for f in fs]
        has_acc = has_acc or any(f.startswith("acc") and f.endswith(".csv") for f in low)
        has_mat = has_mat or any(f.endswith(".mat") for f in low)
        has_csv = has_csv or any(f.endswith(".csv") for f in low)
    if has_acc:
        return "femto"          # PRONOSTIA/FEMTO: per-record acc_*.csv, vibration in last two columns
    if has_mat:
        return "mat"            # XJTU-SY Mendeley: one .mat per bearing (rawnet)
    if has_csv:
        return "csv"            # XJTU-SY GitHub: per-record CSV, vibration in first two columns
    return None


def _bearing_number(path):
    m = re.search(r"(\d+)", os.path.splitext(os.path.basename(path))[0])
    return int(m.group(1)) if m else path


# ---------- CSV format (one file per acquisition) ----------
def _read_record(path):
    try:
        return np.loadtxt(path, delimiter=",")
    except ValueError:
        return np.loadtxt(path, delimiter=",", skiprows=1)


def list_bearings_csv(root):
    out = [dp for dp, _, fs in os.walk(root) if any(f.lower().endswith(".csv") for f in fs)]
    return sorted(out)


def load_hi_trajectory_csv(bearing_dir, hi_name="rms", channel=0):
    files = sorted(glob.glob(os.path.join(bearing_dir, "*.csv")), key=_bearing_number)
    fn = HI_REGISTRY[hi_name]
    return np.array([fn(_read_record(f)[:, channel]) for f in files], float)


# ---------- FEMTO / PRONOSTIA format (per-record acc_*.csv; vibration = LAST two columns) ----------
def _read_femto_record(path):
    """Read one FEMTO acc CSV. Columns are [h, m, s, us, horizontal, vertical]; the two
    vibration channels are the LAST two columns. Delimiter may be ',' or ';'."""
    with open(path) as f:
        first = f.readline()
    delim = ";" if first.count(";") > first.count(",") else ","
    return np.loadtxt(path, delimiter=delim)


def list_bearings_femto(root):
    out = [dp for dp, _, fs in os.walk(root)
           if any(f.lower().startswith("acc") and f.lower().endswith(".csv") for f in fs)]
    return sorted(out)


def load_hi_trajectory_femto(bearing_dir, hi_name="rms", channel=0):
    """channel 0 = horizontal (second-to-last col), 1 = vertical (last col). Ignores temp_*.csv."""
    files = sorted(set(glob.glob(os.path.join(bearing_dir, "acc_*.csv")) +
                       glob.glob(os.path.join(bearing_dir, "acc*.csv"))), key=_bearing_number)
    fn = HI_REGISTRY[hi_name]
    out = []
    for f in files:
        rec = _read_femto_record(f)
        vib = rec[:, -2:]                      # last two columns = (horizontal, vertical)
        out.append(fn(vib[:, channel]))
    return np.array(out, float)


# ---------- MAT format (one file per bearing; trajectory in a 3-D array) ----------
def _canonical3d(arr):
    """Reorder a 3-D vibration array to (samples, channels, records)."""
    shp = arr.shape
    twos = [i for i, s in enumerate(shp) if s == 2]
    ch = twos[0] if twos else int(np.argmin(shp))
    samp = int(np.argmax(shp))
    rec = ({0, 1, 2} - {ch, samp}).pop()
    return np.transpose(arr, (samp, ch, rec))


def list_bearings_mat(root):
    mats = [os.path.join(dp, f) for dp, _, fs in os.walk(root)
            for f in fs if f.lower().endswith(".mat")]
    orig = [m for m in mats if os.path.basename(os.path.dirname(m)).lower() == "originaldata"]
    return sorted(orig if orig else mats, key=_bearing_number)


def load_hi_trajectory_mat(mat_path, hi_name="rms", channel=0, var="rawnet"):
    from scipy.io import loadmat
    d = loadmat(mat_path, variable_names=[var]) if var else loadmat(mat_path)
    arr = d.get(var)
    if arr is None or getattr(arr, "ndim", 0) != 3:               # fall back: largest 3-D array
        d = loadmat(mat_path)
        cands = [v for k, v in d.items() if not k.startswith("__") and getattr(v, "ndim", 0) == 3]
        arr = max(cands, key=lambda a: a.size) if cands else None
    if arr is None:
        raise ValueError(f"{mat_path}: no 3-D trajectory array (expected '{var}').")
    canon = _canonical3d(np.asarray(arr, float))                  # (samples, channels, records)
    fn = HI_REGISTRY[hi_name]
    return np.array([fn(canon[:, channel, k]) for k in range(canon.shape[2])], float)


# ---------- unified loading ----------
def load_all_hi(root, hi_name="rms", channel=0, cache=True):
    # Certified full rebuilds set this flag so every formal result is derived from
    # the hashed raw source files rather than a pre-existing source-root cache.
    if os.environ.get("RESS_DISABLE_XJTU_CACHE", "0") == "1":
        cache = False
    cache_path = os.path.join(root, f"_hi_cache_{hi_name}_ch{channel}.npz")
    if cache and os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=True)
        trajs, names = list(z["trajs"]), list(z["names"])
        if len(trajs) > 0:                          # ignore a stale/empty cache
            return trajs, names
    fmt = _detect_format(root)
    if fmt == "femto":
        srcs = list_bearings_femto(root)
        trajs = [load_hi_trajectory_femto(s, hi_name, channel) for s in srcs]
    elif fmt == "mat":
        srcs = list_bearings_mat(root)
        trajs = [load_hi_trajectory_mat(s, hi_name, channel) for s in srcs]
    elif fmt == "csv":
        srcs = list_bearings_csv(root)
        trajs = [load_hi_trajectory_csv(s, hi_name, channel) for s in srcs]
    else:
        raise FileNotFoundError(f"No FEMTO/.mat/.csv acquisition files found under {root}")
    names = [os.path.relpath(s, root) for s in srcs]
    if not trajs:
        raise FileNotFoundError(
            f"No bearings found under {root} (detected format={fmt}). "
            f"If a stale _hi_cache_*.npz exists there, delete it and retry.")
    if cache:                                        # only cache non-empty results
        np.savez(cache_path, trajs=np.array(trajs, dtype=object), names=np.array(names))
    return trajs, names


def infer_xjtu_conditions(names):
    """Infer XJTU-SY operating-condition labels from paths or canonical order.

    The CSV distribution normally embeds speed/load labels in the directory
    path.  The MAT distribution commonly exposes only bearing1--bearing15; its
    canonical ordering is five bearings per operating condition.
    """
    labels = []
    for name in names:
        text = str(name).replace("\\", "/").lower()
        match = re.search(
            r"(35(?:\.0)?hz[^/]*12kn|37\.5hz[^/]*11kn|40(?:\.0)?hz[^/]*10kn)",
            text,
        )
        labels.append(match.group(1) if match else "")
    if labels and all(labels):
        return np.asarray(labels, object)
    if len(names) == 15:
        canonical = ("35Hz_12kN", "37.5Hz_11kN", "40Hz_10kN")
        return np.asarray([canonical[i // 5] for i in range(15)], object)
    return np.asarray(["all"] * len(names), object)


def health_threshold(hi_trajs, percentile=70.0, weighting="record"):
    """Fleet health threshold with explicit record- or unit-equal weighting."""
    if weighting == "record":
        return float(np.percentile(np.concatenate(hi_trajs), percentile))
    if weighting != "unit":
        raise ValueError("weighting must be 'record' or 'unit'")
    values = np.concatenate([np.asarray(h, float) for h in hi_trajs])
    weights = np.concatenate([
        np.full(len(h), 1.0 / (len(hi_trajs) * len(h)), float) for h in hi_trajs
    ])
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cdf = np.cumsum(weights)
    q = float(percentile) / 100.0
    return float(values[min(np.searchsorted(cdf, q, side="left"), len(values) - 1)])


# ---------- informative-censoring overlay on real trajectories ----------
def static_baseline_summary(observed_hi, n_records=1):
    """Return a leakage-free static health summary available at baseline.

    ``n_records`` is a fixed, design-level window and must not depend on the
    complete failure time or the realised censoring time.  The primary XJTU
    analysis uses ``n_records=1`` so every unit has the covariate before the
    first replacement decision.  Longer fixed windows require an explicit
    landmark design and are therefore not silently shortened here.
    """
    obs = np.asarray(observed_hi, dtype=float)
    n_records = int(n_records)
    if n_records < 1:
        raise ValueError("n_records must be at least one")
    if len(obs) < n_records:
        raise ValueError(
            f"baseline window requires {n_records} records but only {len(obs)} are observed"
        )
    return float(np.mean(obs[:n_records]))


def _log_pre_failure_hazard_mass(hi, beta, tau):
    """Log of sum exp(beta * (HI-tau)) over times strictly before failure.

    The overlay declares a terminal-time record to be a failure, not a censoring
    event (failure precedes censoring at the same recorded time).  Calibration
    must therefore exclude the final record.
    """
    arr = np.asarray(hi, dtype=float)
    if arr.size <= 1:
        return -np.inf
    eta = beta * (arr[:-1] - tau)
    m = float(np.max(eta))
    if not np.isfinite(m):
        return m
    return m + float(np.log(np.exp(eta - m).sum()))


def _expected_cens_fraction(hi_trajs, beta, tau, lam0):
    """Expected fraction censored strictly before the terminal failure time.

    For per-record hazards p_j = 1-exp(-lambda_j), the probability of at least
    one pre-failure replacement is 1-exp(-sum_j lambda_j).  The log-sum-exp
    implementation remains stable for concentrated terminal health signals.
    """
    if lam0 <= 0:
        return 0.0
    log_lam0 = float(np.log(lam0))
    probs = []
    for traj in hi_trajs:
        log_mass = _log_pre_failure_hazard_mass(traj, beta, tau)
        z = log_lam0 + log_mass
        if z >= np.log(745.0):
            prob = 1.0
        elif z <= -745.0 or not np.isfinite(z):
            prob = 0.0 if z < 0 else 1.0
        else:
            prob = float(-np.expm1(-np.exp(z)))
        probs.append(prob)
    return float(np.mean(probs)) if probs else np.nan


def calibrate_lambda0(hi_trajs, beta, tau, c_target, lo=1e-300, hi=1e3, iters=100):
    """Calibrate lambda0 to the event definition used by the overlay.

    Censoring is only possible before the final record.  A log-scale bisection
    permits very small baseline rates when beta concentrates nearly all hazard
    near the failure endpoint.
    """
    if not 0.0 <= c_target < 1.0:
        raise ValueError(f"c_target must be in [0,1), got {c_target}")
    if c_target == 0.0:
        return 0.0
    if not hi_trajs:
        raise ValueError("hi_trajs must be non-empty")

    max_fraction = float(np.mean([len(np.asarray(x)) > 1 for x in hi_trajs]))
    if c_target >= max_fraction:
        raise ValueError(
            f"c_target={c_target} is not attainable; maximum is {max_fraction}"
        )

    log_lo = float(np.log(lo))
    log_hi = float(np.log(hi))
    f_lo = _expected_cens_fraction(hi_trajs, beta, tau, float(np.exp(log_lo)))
    f_hi = _expected_cens_fraction(hi_trajs, beta, tau, float(np.exp(log_hi)))

    while f_lo > c_target and log_lo > -745.0:
        log_lo = max(-745.0, log_lo - 20.0)
        f_lo = _expected_cens_fraction(hi_trajs, beta, tau, float(np.exp(log_lo)))
    while f_hi < c_target and log_hi < 700.0:
        log_hi = min(700.0, log_hi + 20.0)
        f_hi = _expected_cens_fraction(hi_trajs, beta, tau, float(np.exp(log_hi)))

    if f_lo > c_target or f_hi < c_target:
        raise RuntimeError(
            "Could not bracket the requested censoring fraction: "
            f"f(lo)={f_lo}, f(hi)={f_hi}, target={c_target}"
        )

    for _ in range(iters):
        log_mid = 0.5 * (log_lo + log_hi)
        mid = float(np.exp(log_mid))
        if _expected_cens_fraction(hi_trajs, beta, tau, mid) < c_target:
            log_lo = log_mid
        else:
            log_hi = log_mid
    return float(np.exp(0.5 * (log_lo + log_hi)))


def impose_informative_censoring(
    hi_trajs, beta, tau, lam0, sigma, rng, baseline_records=1
):
    Ttil, event, x_obs, Tnet, frail, hi_obs = [], [], [], [], [], []
    for hi in hi_trajs:
        n = len(hi)
        T_i = float(n)
        eta = np.clip(beta * (np.asarray(hi, dtype=float) - tau), -700.0, 700.0)
        hazard = lam0 * np.exp(eta)
        p = -np.expm1(-np.minimum(hazard, 745.0))
        fired = np.where(rng.random(n) < p)[0]
        if len(fired) and fired[0] < n - 1:
            Ttil.append(float(fired[0] + 1)); event.append(0.0)
        else:
            Ttil.append(T_i); event.append(1.0)
        L = int(Ttil[-1])
        obs = hi[:L] * np.exp(sigma * rng.standard_normal(L))   # observed noised HI up to obs time
        hi_obs.append(obs)
        x_obs.append(static_baseline_summary(obs, baseline_records))
        Tnet.append(T_i); frail.append(float(np.mean(hi)))
    ds = Dataset(unit_id=np.arange(len(hi_trajs)),
                 Ttil=np.array(Ttil), event=np.array(event), x_obs=np.array(x_obs),
                 hi_obs=hi_obs)
    return ds, Truth(T=np.array(Tnet), Z=np.array(frail))


def impose_informative_censoring_from_draws(
    hi_trajs, beta, tau, lam0, sigma, uniforms, noises, baseline_records=1
):
    """Overlay censoring using supplied random draws for paired comparisons.

    ``uniforms`` and ``noises`` contain one full-length vector per trajectory.
    Reusing them across beta values gives common-random-number estimates of the
    health-selection increment without changing the estimator information set.
    """
    if not (len(hi_trajs) == len(uniforms) == len(noises)):
        raise ValueError("hi_trajs, uniforms, and noises must have equal length")
    Ttil, event, x_obs, Tnet, frail, hi_obs = [], [], [], [], [], []
    for hi, uu, zz in zip(hi_trajs, uniforms, noises):
        h = np.asarray(hi, float)
        uu = np.asarray(uu, float)
        zz = np.asarray(zz, float)
        n = len(h)
        if len(uu) < n or len(zz) < n:
            raise ValueError("each supplied draw vector must cover the full trajectory")
        eta = np.clip(beta * (h - tau), -700.0, 700.0)
        hazard = lam0 * np.exp(eta)
        p = -np.expm1(-np.minimum(hazard, 745.0))
        fired = np.flatnonzero(uu[:max(n - 1, 0)] < p[:max(n - 1, 0)])
        if len(fired):
            L, ev = int(fired[0] + 1), 0.0
        else:
            L, ev = n, 1.0
        obs = h[:L] * np.exp(sigma * zz[:L])
        Ttil.append(float(L)); event.append(ev); hi_obs.append(obs)
        x_obs.append(static_baseline_summary(obs, baseline_records))
        Tnet.append(float(n)); frail.append(float(np.mean(h)))
    ds = Dataset(
        unit_id=np.arange(len(hi_trajs)), Ttil=np.asarray(Ttil),
        event=np.asarray(event), x_obs=np.asarray(x_obs), hi_obs=hi_obs,
    )
    return ds, Truth(T=np.asarray(Tnet), Z=np.asarray(frail))


def build_from_directory(root, beta, sigma, c_target, rng,
                         hi_name="rms", channel=0, tau_pct=70.0, cache=True,
                         baseline_records=1, threshold_weighting="record"):
    hi_trajs, names = load_all_hi(root, hi_name, channel, cache=cache)
    tau = health_threshold(hi_trajs, tau_pct, threshold_weighting)
    lam0 = calibrate_lambda0(hi_trajs, beta, tau, c_target)
    ds, truth = impose_informative_censoring(
        hi_trajs, beta, tau, lam0, sigma, rng,
        baseline_records=baseline_records,
    )
    meta = dict(tau=tau, lam0=lam0, n_bearings=len(hi_trajs), names=names,
                lifetimes=[len(h) for h in hi_trajs],
                static_covariate=f"mean of first {int(baseline_records)} observed HI record(s)",
                threshold_weighting=threshold_weighting,
                conditions=infer_xjtu_conditions(names).tolist())
    return ds, truth, meta
