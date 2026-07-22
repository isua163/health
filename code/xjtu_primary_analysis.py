#!/usr/bin/env python3
"""Computationally tractable XJTU analyses used in the manuscript.

The workflow separates:

1. ``policy``: same-observed-signal versus latent-policy-state overlays,
   including same-sample and leave-one-unit-out cross-fitted TV-IPCW.
2. ``bootstrap``: condition-stratified complete-unit bootstrap for the selected
   primary arms.  By default cross-fitting is *not* nested in the bootstrap;
   it can be requested explicitly with ``--bootstrap-arms tv_crossfit``.

Both phases print progress, checkpoint CSV files, and can resume completed
seeds/bootstrap indices.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CANDIDATES = [HERE]
for candidate in CANDIDATES:
    if (candidate / "src").exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
        break

from src.arms.base import Dataset  # noqa: E402
from src.arms.naive_survival import NaiveSurvivalArm  # noqa: E402
from src.arms.ipcw_correction import (  # noqa: E402
    TimeVaryingIPCWArm,
    _cumhaz_before_times,
    _fit_cloglog,
    _person_period,
)
from src._survival import rmrl_from_survival  # noqa: E402
from src.data_xjtu import (  # noqa: E402
    calibrate_lambda0,
    health_threshold,
    infer_xjtu_conditions,
    load_all_hi,
    static_baseline_summary,
)

VALID_ARMS = ("naive", "tv_same_sample", "tv_crossfit")
_WORKER: dict = {}


def _normalise_arms(values: Iterable[str]) -> tuple[str, ...]:
    arms = tuple(dict.fromkeys(values))
    bad = sorted(set(arms) - set(VALID_ARMS))
    if bad:
        raise ValueError(f"unknown arms: {bad}; choose from {VALID_ARMS}")
    if not arms:
        raise ValueError("at least one arm is required")
    return arms


def overlay(trajs, beta, sigma, c, tau, rng, policy_signal):
    noises = [rng.standard_normal(len(h)) for h in trajs]
    uniforms = [rng.random(len(h)) for h in trajs]
    observed = [np.asarray(h, float) * np.exp(sigma * z) for h, z in zip(trajs, noises)]
    policy = observed if policy_signal == "observed" else [np.asarray(h, float) for h in trajs]
    lam0 = calibrate_lambda0(policy, beta, tau, c)

    tt, ev, hi_obs, x = [], [], [], []
    for hobs, hpol, u in zip(observed, policy, uniforms):
        n = len(hpol)
        haz = lam0 * np.exp(np.clip(beta * (hpol - tau), -700, 700))
        p = -np.expm1(-np.minimum(haz, 745.0))
        fired = np.flatnonzero(u[: max(n - 1, 0)] < p[: max(n - 1, 0)])
        L = int(fired[0] + 1) if len(fired) else n
        e = 0 if len(fired) else 1
        obs = np.asarray(hobs[:L], float)
        tt.append(L)
        ev.append(e)
        hi_obs.append(obs)
        x.append(static_baseline_summary(obs, 1))

    ds = Dataset(
        np.arange(len(trajs)),
        np.asarray(tt, float),
        np.asarray(ev, int),
        np.asarray(x, float),
        hi_obs,
    )
    return ds, lam0


def crossfit_tv_rmst(ds, H):
    n = len(ds.unit_id)
    cum = []
    for i in range(n):
        keep = np.arange(n) != i
        train = Dataset(
            ds.unit_id[keep],
            ds.Ttil[keep],
            ds.event[keep],
            ds.x_obs[keep],
            [ds.hi_obs[k] for k in np.flatnonzero(keep)],
        )
        x, y = _person_period(train.hi_obs, train.event)
        fit = _fit_cloglog(x, y)
        cum.append(_cumhaz_before_times(ds.hi_obs[i], fit.intercept, fit.slope))

    fails = np.unique(ds.Ttil[ds.event == 1])
    surv = 1.0
    times, vals = [], []
    for u in fails:
        idx = np.flatnonzero(ds.Ttil >= u)
        w = np.array(
            [
                np.exp(
                    np.clip(
                        cum[i][min(int(u), len(cum[i]) - 1)],
                        None,
                        30.0,
                    )
                )
                for i in idx
            ]
        )
        d = (ds.Ttil[idx] == u) & (ds.event[idx] == 1)
        if w.sum() > 0:
            surv *= max(0.0, 1.0 - w[d].sum() / w.sum())
            times.append(float(u))
            vals.append(float(surv))
    if not times:
        return float(H)
    return float(rmrl_from_survival(np.asarray(times), np.asarray(vals), 0.0, H))


def estimates(ds, H, arms):
    out = {}
    if "naive" in arms:
        out["naive"] = float(NaiveSurvivalArm().fit(ds).predict_rmrl(ds, 0, H)[0])
    if "tv_same_sample" in arms:
        try:
            out["tv_same_sample"] = float(
                TimeVaryingIPCWArm().fit(ds).predict_rmrl(ds, 0, H)[0]
            )
        except Exception:
            out["tv_same_sample"] = np.nan
    if "tv_crossfit" in arms:
        try:
            out["tv_crossfit"] = crossfit_tv_rmst(ds, H)
        except Exception:
            out["tv_crossfit"] = np.nan
    return out


def one_pair(trajs, H, tau, sigma, c, seed, policy_signal, arms):
    rows = []
    truth = float(np.mean(np.minimum([len(h) for h in trajs], H)))
    for beta in (0.0, 1.0):
        rng = np.random.default_rng(seed)  # common random numbers across beta
        ds, lam0 = overlay(trajs, beta, sigma, c, tau, rng, policy_signal)
        for arm, val in estimates(ds, H, arms).items():
            rows.append(
                dict(
                    seed=int(seed),
                    policy_signal=policy_signal,
                    beta=beta,
                    arm=arm,
                    estimate=val,
                    truth=truth,
                    bias_pct=100.0 * (val - truth) / truth,
                    censor_fraction=float(np.mean(ds.event == 0)),
                    lambda0=lam0,
                    fit_ok=bool(np.isfinite(val)),
                )
            )
    return rows


def summarize_increment(df, group_cols):
    wide = (
        df.pivot_table(
            index=group_cols + ["seed"],
            columns="beta",
            values="bias_pct",
            aggfunc="first",
        )
        .reset_index()
    )
    if 0.0 not in wide.columns:
        wide[0.0] = np.nan
    if 1.0 not in wide.columns:
        wide[1.0] = np.nan
    wide["increment_pct_points"] = wide[1.0] - wide[0.0]
    return wide


def _atomic_csv(df: pd.DataFrame, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _policy_init(trajs, H, tau, sigma, c, arms):
    _WORKER.clear()
    _WORKER.update(
        trajs=trajs,
        H=H,
        tau=tau,
        sigma=sigma,
        c=c,
        arms=arms,
    )


def _policy_task(task):
    policy, seed = task
    return one_pair(
        _WORKER["trajs"],
        _WORKER["H"],
        _WORKER["tau"],
        _WORKER["sigma"],
        _WORKER["c"],
        seed,
        policy,
        _WORKER["arms"],
    )


def _bootstrap_init(trajs, cond, H, tau_percentile, sigma, c, R_bootstrap, seed, arms):
    _WORKER.clear()
    groups = {g: np.flatnonzero(cond == g) for g in np.unique(cond)}
    _WORKER.update(
        trajs=trajs,
        groups=groups,
        H=H,
        tau_percentile=tau_percentile,
        sigma=sigma,
        c=c,
        R_bootstrap=R_bootstrap,
        seed=seed,
        arms=arms,
    )


def _bootstrap_task(b):
    rng = np.random.default_rng(_WORKER["seed"] + 900_000 + int(b))
    idx = np.concatenate(
        [rng.choice(ix, size=len(ix), replace=True) for ix in _WORKER["groups"].values()]
    )
    bt = [_WORKER["trajs"][i] for i in idx]
    bt_tau = float(health_threshold(bt, _WORKER["tau_percentile"], "unit"))
    local = []
    for r in range(_WORKER["R_bootstrap"]):
        local.extend(
            one_pair(
                bt,
                _WORKER["H"],
                bt_tau,
                _WORKER["sigma"],
                _WORKER["c"],
                _WORKER["seed"] + 10_000_000 + int(b) * _WORKER["R_bootstrap"] + r,
                "observed",
                _WORKER["arms"],
            )
        )
    d = pd.DataFrame(local)
    incb = summarize_increment(d, ["policy_signal", "arm"])
    rows = []
    for arm, g in incb.groupby("arm"):
        vals = g.increment_pct_points
        rows.append(
            dict(
                bootstrap=int(b),
                arm=arm,
                increment_pct_points=float(vals.mean()),
                successful_pairs=int(vals.notna().sum()),
                overlays_requested=int(_WORKER["R_bootstrap"]),
                source_indices=";".join(map(str, idx.tolist())),
                tau=float(bt_tau),
            )
        )
    return rows


def _run_tasks(tasks, worker, initializer, initargs, n_jobs):
    if n_jobs == 1:
        initializer(*initargs)
        for task in tasks:
            yield worker(task)
        return
    with ProcessPoolExecutor(
        max_workers=n_jobs,
        initializer=initializer,
        initargs=initargs,
    ) as ex:
        yield from ex.map(worker, tasks, chunksize=1)


def run_policy(a, trajs, H, tau):
    rep_path = a.out_dir / "policy_crossfit_replicates.csv"
    existing = pd.DataFrame()
    completed = set()
    if a.resume and rep_path.exists():
        existing = pd.read_csv(rep_path)
        if not existing.empty:
            completed = set(zip(existing.policy_signal.astype(str), existing.seed.astype(int)))

    tasks = [
        (policy, a.seed + r)
        for policy in a.policy_signals
        for r in range(a.R)
        if (policy, a.seed + r) not in completed
    ]
    rows = existing.to_dict("records") if not existing.empty else []
    start = time.time()
    print(
        f"policy phase: {len(tasks)} tasks pending, R={a.R}, "
        f"signals={','.join(a.policy_signals)}, arms={','.join(a.policy_arms)}, n_jobs={a.n_jobs}",
        flush=True,
    )
    for done, result in enumerate(
        _run_tasks(
            tasks,
            _policy_task,
            _policy_init,
            (trajs, H, tau, a.sigma, a.c, a.policy_arms),
            a.n_jobs,
        ),
        start=1,
    ):
        rows.extend(result)
        if done % a.checkpoint_every == 0 or done == len(tasks):
            _atomic_csv(pd.DataFrame(rows), rep_path)
            print(
                f"policy: {done}/{len(tasks)} new tasks; elapsed={(time.time()-start)/60:.1f} min",
                flush=True,
            )

    rep = pd.DataFrame(rows)
    if rep.empty:
        raise RuntimeError("policy phase produced no rows")
    _atomic_csv(rep, rep_path)
    inc = summarize_increment(rep, ["policy_signal", "arm"])
    summ = (
        inc.groupby(["policy_signal", "arm"])
        .increment_pct_points.agg(["count", "mean", "std"])
        .reset_index()
    )
    summ["mcse"] = summ["std"] / np.sqrt(summ["count"])
    fail = (
        (~rep.fit_ok)
        .groupby([rep.policy_signal, rep.arm])
        .mean()
        .rename("fit_failure_fraction")
        .reset_index()
    )
    summ = summ.merge(fail, on=["policy_signal", "arm"], how="left")
    _atomic_csv(summ, a.out_dir / "crossfit_summary.csv")
    print(f"wrote {rep_path}", flush=True)
    print(f"wrote {a.out_dir / 'crossfit_summary.csv'}", flush=True)


def run_bootstrap(a, trajs, cond, H):
    rep_path = a.out_dir / "fleet_bootstrap_replicates.csv"
    existing = pd.DataFrame()
    completed = set()
    if a.resume and rep_path.exists():
        existing = pd.read_csv(rep_path)
        if not existing.empty:
            # A bootstrap is complete only if every requested arm is present.
            counts = existing.groupby("bootstrap").arm.nunique()
            completed = set(counts[counts >= len(a.bootstrap_arms)].index.astype(int))

    tasks = [b for b in range(a.B) if b not in completed]
    rows = existing.to_dict("records") if not existing.empty else []
    start = time.time()
    print(
        f"bootstrap phase: {len(tasks)} resamples pending, B={a.B}, "
        f"R_bootstrap={a.R_bootstrap}, arms={','.join(a.bootstrap_arms)}, n_jobs={a.n_jobs}",
        flush=True,
    )
    for done, result in enumerate(
        _run_tasks(
            tasks,
            _bootstrap_task,
            _bootstrap_init,
            (
                trajs,
                cond,
                H,
                a.tau_percentile,
                a.sigma,
                a.c,
                a.R_bootstrap,
                a.seed,
                a.bootstrap_arms,
            ),
            a.n_jobs,
        ),
        start=1,
    ):
        rows.extend(result)
        if done % a.checkpoint_every == 0 or done == len(tasks):
            _atomic_csv(pd.DataFrame(rows), rep_path)
            print(
                f"bootstrap: {done}/{len(tasks)} new resamples; elapsed={(time.time()-start)/60:.1f} min",
                flush=True,
            )

    br = pd.DataFrame(rows)
    if br.empty:
        raise RuntimeError("bootstrap phase produced no rows")
    br = br.sort_values(["bootstrap", "arm"]).drop_duplicates(["bootstrap", "arm"], keep="last")
    _atomic_csv(br, rep_path)
    bs = (
        br.groupby("arm")
        .increment_pct_points.agg(
            B="count",
            mean="mean",
            median="median",
            q025=lambda x: np.nanquantile(x, 0.025),
            q975=lambda x: np.nanquantile(x, 0.975),
            sd="std",
        )
        .reset_index()
    )
    _atomic_csv(bs, a.out_dir / "fleet_bootstrap_summary.csv")
    print(f"wrote {rep_path}", flush=True)
    print(f"wrote {a.out_dir / 'fleet_bootstrap_summary.csv'}", flush=True)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xjtu", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("results/xjtu"))
    ap.add_argument("--phase", choices=["policy", "bootstrap", "all"], default="all")
    ap.add_argument("--R", type=int, default=1000)
    ap.add_argument("--B", type=int, default=5000)
    ap.add_argument("--R-bootstrap", type=int, default=30)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--c", type=float, default=0.4)
    ap.add_argument("--tau-percentile", type=float, default=70.0)
    ap.add_argument("--horizon-quantile", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=20260715)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--checkpoint-every", type=int, default=25)
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--policy-signals",
        nargs="+",
        choices=["latent", "observed"],
        default=["latent", "observed"],
    )
    ap.add_argument(
        "--policy-arms",
        nargs="+",
        choices=list(VALID_ARMS),
        default=list(VALID_ARMS),
    )
    ap.add_argument(
        "--bootstrap-arms",
        nargs="+",
        choices=list(VALID_ARMS),
        default=["naive", "tv_same_sample"],
        help="Keep tv_crossfit out of the B=5000 bootstrap unless explicitly required.",
    )
    return ap.parse_args()


def main():
    a = parse_args()
    if a.R < 1 or a.B < 1 or a.R_bootstrap < 1:
        raise ValueError("R, B, and R-bootstrap must be positive")
    if a.n_jobs < 1:
        raise ValueError("n-jobs must be at least 1")
    if a.checkpoint_every < 1:
        raise ValueError("checkpoint-every must be at least 1")
    if not (0.0 < a.horizon_quantile <= 1.0):
        raise ValueError("horizon-quantile must be in (0,1]")
    a.policy_arms = _normalise_arms(a.policy_arms)
    a.bootstrap_arms = _normalise_arms(a.bootstrap_arms)
    a.policy_signals = tuple(dict.fromkeys(a.policy_signals))
    a.out_dir.mkdir(parents=True, exist_ok=True)

    trajs, names = load_all_hi(str(a.xjtu), cache=False)
    cond = infer_xjtu_conditions(names)
    H = float(np.quantile([len(h) for h in trajs], a.horizon_quantile))
    tau = float(health_threshold(trajs, a.tau_percentile, "unit"))
    print(f"loaded {len(trajs)} trajectories; H={H:g}; tau={tau:g}", flush=True)

    if a.phase in ("policy", "all"):
        run_policy(a, trajs, H, tau)
    if a.phase in ("bootstrap", "all"):
        run_bootstrap(a, trajs, cond, H)


if __name__ == "__main__":
    main()
