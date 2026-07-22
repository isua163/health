"""Battery run-to-EOL path (pre-reg v2 addendum, D5): map per-cell capacity fade to an
INCREASING health indicator and impose the SAME censoring overlay as the bearing path.

FROZEN (D5 / D6-iv): HI = 1 - Q/Q_nom (increasing); net lifetime T_i = first cycle with
Q < eol_frac*Q_nom, eol_frac = 0.80 primary (0.70 / 0.85 as sensitivity). Q_nom = rated
capacity (Severson A123 LFP: 1.1 Ah). T_i = len(truncated HI trajectory), matching the
bearing convention. Downstream (calibrate_lambda0 / impose_informative_censoring / arms /
metrics) is HI-agnostic and unchanged; cross-fleet comparability via src.standardize (D1).

WHAT IS AND ISN'T VERIFIED HERE (honesty; see tests/test_battery.py and tests/test_matr_reader.py):
  * VERIFIED on FABRICATED data: HI mapping, EOL detection, incomplete-cell exclusion, and
    end-to-end overlay integration (test_battery.py); the MATR reader's dereferencing, leading-
    placeholder trim, and D6-iv cross-check against a fixture replicating the CONFIRMED real HDF5
    layout (test_matr_reader.py -- run on the data machine, needs h5py).
  * _load_matr_capacity is written against the CONFIRMED layout (this project's h5py dump), using
    the exact dereference pattern the dump proved. _load_nasa_capacity is still a SKELETON (NASA =
    D5 smoke test only) -- to be filled against the real NASA .mat.
"""
import numpy as np

Q_NOM_SEVERSON = 1.1        # A123 APR18650M1A rated capacity (Ah)


def capacity_to_hi(Q, Q_nom):
    """Increasing HI from per-cycle discharge capacity Q. FROZEN D5. (0 at full, 0.2 at 80%.)"""
    return 1.0 - np.asarray(Q, float) / Q_nom


def net_lifetime_from_capacity(Q, Q_nom, eol_frac=0.80, tol_frac=0.005):
    """EOL cycle count (1-based). First cycle with Q < eol_frac*Q_nom + tol_frac*Q_nom
    (the small tolerance absorbs recordings that stop right AT ~80%, e.g. min(Q)=0.881 vs
    eol=0.880 -- confirmed for MATR). If there is no crossing but the cell got essentially to
    EOL (min(Q) within 2% of the threshold, i.e. recorded-to-EOL), T_i = full recorded length;
    otherwise None (genuinely truncated / never-degraded -> dropped upstream)."""
    Q = np.asarray(Q, float)
    eol = eol_frac * Q_nom
    below = np.where(Q < eol + tol_frac * Q_nom)[0]
    if below.size:
        return int(below[0] + 1)
    if Q.size >= 2 and float(np.min(Q)) <= eol + 0.02 * Q_nom:     # recorded-to-EOL, no strict crossing
        return int(Q.size)
    return None


def build_battery_hi_trajectories(cell_capacities, Q_nom, eol_frac=0.80, monotone=False,
                                  cycle_lives=None, min_life_frac=0.7):
    """cell_capacities: list of per-cycle discharge-capacity arrays (one per cell).
    Returns (hi_trajs, kept_idx): each hi_traj is the INCREASING HI truncated at EOL
    (len = T_i). If `cycle_lives` is given, T_i = the published cycle life (authoritative,
    glitch-robust); cells whose recording is < min_life_frac of their published life are
    dropped (cross-batch-continued / truncated cells). Else T_i = first cycle with
    Q<eol_frac*Q_nom, and cells that never reach EOL are dropped (incomplete). `monotone`
    optionally enforces a non-decreasing HI envelope (task-B smoothing otherwise)."""
    hi_trajs, kept = [], []
    for i, Q in enumerate(cell_capacities):
        Q = np.asarray(Q, float)
        if cycle_lives is not None and cycle_lives[i] and not np.isnan(cycle_lives[i]):
            cl = int(round(cycle_lives[i]))
            if len(Q) < min_life_frac * cl:                      # truncated / continued-across-batch -> drop
                continue
            T = min(cl, len(Q))                                  # published cycle life
        else:
            T = net_lifetime_from_capacity(Q, Q_nom, eol_frac)
        if T is None or T < 2:
            continue
        hi = capacity_to_hi(Q[:T], Q_nom)
        if monotone:
            hi = np.maximum.accumulate(hi)
        hi_trajs.append(np.maximum(hi, 1e-6))
        kept.append(i)
    if not hi_trajs:
        raise ValueError("no cell reached EOL at the given threshold (check Q_nom / eol_frac)")
    return hi_trajs, kept


# ----------------------------- FILE PARSERS -----------------------------
# _load_matr_capacity is written against the CONFIRMED real layout (h5py dump, this project):
#   f['batch'] Group; each field shape (n_cells, 1) of HDF5 object references.
#   batch['summary'][i,0] -> Group;  group['QDischarge'] shape (1, n_cycles) = per-cycle
#       discharge capacity (Ah).  batch['cycle_life'][i,0] -> (1,1) published cycle life.
#   The FIRST summary row is an all-zero placeholder (QDischarge[0]==0) -> trimmed here.
def _load_matr_capacity(root, batch_files=None, q_nom=Q_NOM_SEVERSON, min_q_frac=0.5):
    """MATR/Severson batch .mat (MATLAB v7.3 / HDF5) -> (caps, cycle_lives, names).
    caps[i] = per-cycle discharge-capacity array (leading placeholder cycles dropped).
    Pass batch_files=[...] to restrict batches (e.g. the 3 Severson batches, excluding the
    2019-01-24 CLO batch, for the primary anchor); default = every *.mat under root."""
    import os, glob
    try:
        import h5py
    except ImportError:
        raise ImportError("MATR is MATLAB v7.3 -> needs h5py:  conda install -c conda-forge h5py -y")
    files = ([os.path.join(root, b) for b in batch_files] if batch_files
             else sorted(glob.glob(os.path.join(root, "*.mat"))))
    if not files:
        raise FileNotFoundError(f"no .mat under {root}")
    caps, clives, names = [], [], []
    thr = min_q_frac * q_nom
    for path in files:
        tag = os.path.basename(path)[:10]
        with h5py.File(path, "r") as f:
            batch = f["batch"]
            # same dereference pattern proven by the project's h5py dump:
            #   f[ np.asarray(batch['summary'][()]).ravel()[i] ]['QDischarge']
            summ_refs = np.asarray(batch["summary"][()]).ravel()
            cl_refs = np.asarray(batch["cycle_life"][()]).ravel()
            for i in range(summ_refs.shape[0]):
                qd = np.asarray(f[summ_refs[i]]["QDischarge"]).ravel().astype(float)
                good = np.where(qd > thr)[0]                      # drop leading placeholder(s) (Q~0)
                if good.size == 0:
                    continue
                caps.append(qd[good[0]:])
                try:
                    clives.append(float(np.asarray(f[cl_refs[i]]).ravel()[0]))
                except Exception:
                    clives.append(float("nan"))
                names.append(f"{tag}_cell{i}")
    return caps, clives, names


def matr_crosscheck(root, batch_files=None, q_nom=Q_NOM_SEVERSON, eol_frac=0.80):
    """D6-iv validation + glitch detector: our EOL detection (first cycle Q<eol_frac*q_nom)
    vs the published cycle_life, per cell. Large mismatches flag interior capacity glitches
    or recording-boundary effects (-> prefer cycle_lives as T_i, or add smoothing)."""
    caps, clives, names = _load_matr_capacity(root, batch_files, q_nom)
    rows, none_ct = [], 0
    for cap, cl, nm in zip(caps, clives, names):
        our_T = net_lifetime_from_capacity(cap, q_nom, eol_frac)
        if our_T is None:
            none_ct += 1
        rel = (abs(our_T - cl) / cl * 100.0) if (our_T and cl and not np.isnan(cl)) else float("nan")
        rows.append((nm, our_T, cl, rel))
    finite = [r[3] for r in rows if not np.isnan(r[3])]
    print(f"MATR cross-check: {len(rows)} cells; our T_i (first Q<{eol_frac}*{q_nom}) vs published cycle_life")
    if finite:
        print(f"  median |rel err| = {np.median(finite):.1f}%   within 5%: "
              f"{100*np.mean([e < 5 for e in finite]):.0f}%   within 10%: {100*np.mean([e < 10 for e in finite]):.0f}%")
    print(f"  cells where our EOL not reached (None): {none_ct}  "
          f"(these get dropped unless T_i=cycle_life is used)")
    worst = sorted([r for r in rows if not np.isnan(r[3])], key=lambda r: -r[3])[:5]
    if worst:
        print("  worst 5 (name, our_T, cycle_life, rel%):")
        for nm, t, cl, rel in worst:
            print(f"    {nm:22s} our={t}  pub={cl:.0f}  rel={rel:.0f}%")
    return rows


def _load_nasa_capacity(root):
    """SKELETON -- fill against the real NASA PCoE battery .mat (B0005/6/7/18).
    Each battery .mat holds a struct of cycle records; discharge cycles carry the measured
    'Capacity'. Return a list of per-cell discharge-capacity arrays. NASA is the D5 SMOKE
    TEST only (few cells, low dispersion) -- NOT a dose-response point."""
    raise NotImplementedError(
        "fill _load_nasa_capacity against the real .mat; return list of per-cell capacity arrays")


def load_battery_hi(root, source="matr", Q_nom=None, eol_frac=0.80, monotone=False,
                    batch_files=None, use_published_life=True):
    """Real capacity -> increasing HI trajectories. Mirrors data_xjtu.load_all_hi's
    (trajs, names) contract so the rest of the pipeline is reused. For MATR/Severson, T_i
    defaults to the published cycle_life (use_published_life=True; robust to recording-boundary
    effects); set False to use our own EOL crossing. batch_files restricts which batches (e.g.
    the 3 Severson batches for the primary anchor)."""
    if Q_nom is None:
        Q_nom = Q_NOM_SEVERSON
    if source in ("matr", "severson"):
        caps, clives, cell_names = _load_matr_capacity(root, batch_files, Q_nom)
        cl = clives if use_published_life else None
        hi_trajs, kept = build_battery_hi_trajectories(caps, Q_nom, eol_frac, monotone, cycle_lives=cl)
        return hi_trajs, [cell_names[i] for i in kept]
    if source == "nasa":
        caps = _load_nasa_capacity(root)
        hi_trajs, kept = build_battery_hi_trajectories(caps, Q_nom, eol_frac, monotone)
        return hi_trajs, [f"nasa_cell{i}" for i in kept]
    raise ValueError(f"unknown battery source: {source!r}")
