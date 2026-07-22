#!/usr/bin/env python3
"""Support diagnostics for the observed-signal XJTU configuration.

This script requires the raw XJTU-SY source data.  It produces two analyses
that cannot be reconstructed from resample-level summary files alone:

1. Time-specific same-sample and leave-one-unit-out TV-IPCW weight diagnostics
   at p25, p50, and the fixed p50 RMST horizon H.
2. A direct nested-Monte-Carlo convergence audit for the final
   condition-stratified, within-resample-threshold, observed-signal bootstrap
   at R_inner = 30, 100, and 300 (configurable).

The calculations reuse the primary XJTU overlay and estimator implementation
so that event/censoring timing and common-random-number pairing are identical.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("submission_xjtu", HERE / "xjtu_primary_analysis.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load xjtu_primary_analysis.py")
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)

from src.arms.ipcw_correction import (  # noqa: E402
    TimeVaryingIPCWArm,
    _cumhaz_before_times,
    _fit_cloglog,
    _person_period,
)
from src.data_xjtu import health_threshold, infer_xjtu_conditions, load_all_hi  # noqa: E402


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def crossfit_cumhaz(ds):
    """Fit one censoring model per held-out unit and return cumulative hazards."""
    n = len(ds.unit_id)
    cum = []
    for i in range(n):
        keep = np.arange(n) != i
        train = M.Dataset(
            ds.unit_id[keep],
            ds.Ttil[keep],
            ds.event[keep],
            ds.x_obs[keep],
            [ds.hi_obs[k] for k in np.flatnonzero(keep)],
        )
        x, y = _person_period(train.hi_obs, train.event)
        fit = _fit_cloglog(x, y)
        cum.append(_cumhaz_before_times(ds.hi_obs[i], fit.intercept, fit.slope))
    return cum


def diagnostics_from_cumhaz(ds, cumhaz, times, exp_clip=30.0):
    rows = []
    for label, u0 in times:
        u = int(max(1, round(float(u0))))
        idx = np.flatnonzero(ds.Ttil >= u)
        if len(idx) == 0:
            rows.append(
                dict(
                    checkpoint=label,
                    time=u,
                    n_at_risk=0,
                    ess=0.0,
                    ess_over_n_at_risk=np.nan,
                    weight_median=np.nan,
                    weight_p95=np.nan,
                    weight_p99=np.nan,
                    max_weight=np.nan,
                    max_weight_unit=-1,
                    exponent_clip_fraction=np.nan,
                )
            )
            continue
        logw = np.asarray(
            [cumhaz[i][min(u, len(cumhaz[i]) - 1)] for i in idx], dtype=float
        )
        clipped = logw >= exp_clip
        w = np.exp(np.minimum(logw, exp_clip))
        ess = float(w.sum() ** 2 / np.sum(w**2))
        imax = int(np.argmax(w))
        rows.append(
            dict(
                checkpoint=label,
                time=u,
                n_at_risk=int(len(idx)),
                ess=ess,
                ess_over_n_at_risk=ess / len(idx),
                weight_median=float(np.median(w)),
                weight_p95=float(np.percentile(w, 95)),
                weight_p99=float(np.percentile(w, 99)),
                max_weight=float(w[imax]),
                max_weight_unit=int(idx[imax]),
                exponent_clip_fraction=float(clipped.mean()),
            )
        )
    return rows


def run_weight_diagnostics(args, trajs, conditions, H, tau):
    rows = []
    times = [("p25", 0.25 * H), ("p50", 0.50 * H), ("H", H)]
    start = time.time()
    for r in range(args.R_diagnostics):
        seed = args.seed + r
        for beta in (0.0, 1.0):
            rng = np.random.default_rng(seed)
            ds, lam0 = M.overlay(trajs, beta, args.sigma, args.c, tau, rng, "observed")

            same = TimeVaryingIPCWArm().fit(ds)
            same_rows = same.diagnostics_at_times([t for _, t in times])
            for (label, _), d in zip(times, same_rows):
                d = dict(d)
                idx, w = same._weights_at_time(d["time"])
                logw = np.log(np.maximum(w, 1e-300)) if len(w) else np.asarray([])
                max_unit = int(idx[int(np.argmax(w))]) if len(w) else -1
                rows.append(
                    dict(
                        replicate=r,
                        seed=seed,
                        beta=beta,
                        arm="tv_same_sample",
                        checkpoint=label,
                        lambda0=lam0,
                        censor_fraction=float(np.mean(ds.event == 0)),
                        max_weight_unit=max_unit,
                        max_weight_condition=(str(conditions[max_unit]) if max_unit >= 0 else ""),
                        exponent_clip_fraction=(float((logw >= same.exp_clip).mean()) if len(logw) else np.nan),
                        **d,
                    )
                )

            cum = crossfit_cumhaz(ds)
            for d in diagnostics_from_cumhaz(ds, cum, times):
                max_unit = int(d.pop("max_weight_unit"))
                rows.append(
                    dict(
                        replicate=r,
                        seed=seed,
                        beta=beta,
                        arm="tv_crossfit",
                        lambda0=lam0,
                        censor_fraction=float(np.mean(ds.event == 0)),
                        max_weight_unit=max_unit,
                        max_weight_condition=(str(conditions[max_unit]) if max_unit >= 0 else ""),
                        **d,
                    )
                )

        if (r + 1) % args.checkpoint_every == 0 or r + 1 == args.R_diagnostics:
            print(
                f"diagnostics: {r+1}/{args.R_diagnostics}; elapsed={(time.time()-start)/60:.1f} min",
                flush=True,
            )

    rep = pd.DataFrame(rows)
    atomic_csv(rep, args.out_dir / "weight_diagnostics.csv")

    group = ["beta", "arm", "checkpoint", "time"]
    summary = (
        rep.groupby(group, dropna=False)
        .agg(
            n_rows=("replicate", "count"),
            n_at_risk_median=("n_at_risk", "median"),
            n_at_risk_p10=("n_at_risk", lambda x: x.quantile(0.10)),
            ess_ratio_median=("ess_over_n_at_risk", "median"),
            ess_ratio_p10=("ess_over_n_at_risk", lambda x: x.quantile(0.10)),
            weight_median_median=("weight_median", "median"),
            weight_p95_median=("weight_p95", "median"),
            weight_p99_median=("weight_p99", "median"),
            max_weight_p95=("max_weight", lambda x: x.quantile(0.95)),
            max_weight_max=("max_weight", "max"),
            exponent_clip_fraction=("exponent_clip_fraction", "mean"),
        )
        .reset_index()
    )
    atomic_csv(summary, args.out_dir / "weight_diagnostics_summary.csv")

    dominance = (
        rep[rep.max_weight_unit >= 0]
        .groupby(["beta", "arm", "checkpoint", "max_weight_unit", "max_weight_condition"])
        .size()
        .rename("times_maximum_weight")
        .reset_index()
        .sort_values(["beta", "arm", "checkpoint", "times_maximum_weight"], ascending=[True, True, True, False])
    )
    atomic_csv(dominance, args.out_dir / "weight_unit_dominance.csv")


def one_bootstrap_convergence(b, args, trajs, groups, H):
    rng = np.random.default_rng(args.seed + 900_000 + int(b))
    idx = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in groups.values()])
    bt = [trajs[i] for i in idx]
    bt_tau = float(health_threshold(bt, args.tau_percentile, "unit"))

    increments = {"naive": [], "tv_same_sample": []}
    for r in range(args.R_max):
        rows = M.one_pair(
            bt,
            H,
            bt_tau,
            args.sigma,
            args.c,
            args.seed + 10_000_000 + int(b) * args.R_max + r,
            "observed",
            ("naive", "tv_same_sample"),
        )
        d = pd.DataFrame(rows)
        for arm in increments:
            q = d[d.arm == arm].set_index("beta").bias_pct
            increments[arm].append(float(q.loc[1.0] - q.loc[0.0]))

    increments["tv_minus_naive"] = (
        np.asarray(increments["tv_same_sample"]) - np.asarray(increments["naive"])
    ).tolist()

    out = []
    for R in args.R_checkpoints:
        for arm, values in increments.items():
            x = np.asarray(values[:R], dtype=float)
            out.append(
                dict(
                    bootstrap=int(b),
                    R_checkpoint=int(R),
                    arm=arm,
                    mean=float(np.mean(x)),
                    inner_sd=float(np.std(x, ddof=1)),
                    inner_mcse=float(np.std(x, ddof=1) / math.sqrt(R)),
                    overlays_successful=int(np.isfinite(x).sum()),
                    source_indices=";".join(map(str, idx.tolist())),
                    tau=bt_tau,
                )
            )
    return out


def run_convergence(args, trajs, conditions, H):
    groups = {g: np.flatnonzero(conditions == g) for g in np.unique(conditions)}
    rows = []
    start = time.time()
    for b in range(args.B_convergence):
        rows.extend(one_bootstrap_convergence(b, args, trajs, groups, H))
        if (b + 1) % args.checkpoint_every == 0 or b + 1 == args.B_convergence:
            print(
                f"convergence: {b+1}/{args.B_convergence}; elapsed={(time.time()-start)/60:.1f} min",
                flush=True,
            )
    rep = pd.DataFrame(rows)
    atomic_csv(rep, args.out_dir / "mc_convergence_replicates.csv")

    summary = (
        rep.groupby(["R_checkpoint", "arm"])
        .agg(
            n_bootstraps=("bootstrap", "count"),
            mean_across_bootstraps=("mean", "mean"),
            median_across_bootstraps=("mean", "median"),
            q025=("mean", lambda x: x.quantile(0.025)),
            q975=("mean", lambda x: x.quantile(0.975)),
            outer_sd=("mean", "std"),
            median_inner_mcse=("inner_mcse", "median"),
            p90_inner_mcse=("inner_mcse", lambda x: x.quantile(0.90)),
        )
        .reset_index()
    )
    summary["median_inner_mcse_over_outer_sd"] = summary.median_inner_mcse / summary.outer_sd
    summary["fraction_negative"] = np.nan
    for i, row in summary.iterrows():
        if row.arm == "tv_minus_naive":
            q = rep.loc[(rep["R_checkpoint"] == row.R_checkpoint) & (rep["arm"] == row.arm), "mean"]
            summary.loc[i, "fraction_negative"] = float((q < 0).mean())
    atomic_csv(summary, args.out_dir / "mc_convergence_summary.csv")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xjtu", type=Path, required=True)
    ap.add_argument("--phase", choices=["diagnostics", "convergence", "all"], default="all")
    ap.add_argument("--out-dir", type=Path, default=Path("results/xjtu"))
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--c", type=float, default=0.4)
    ap.add_argument("--tau-percentile", type=float, default=70.0)
    ap.add_argument("--horizon-quantile", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=20260715)
    ap.add_argument("--R-diagnostics", type=int, default=5000)
    ap.add_argument("--B-convergence", type=int, default=100)
    ap.add_argument("--R-checkpoints", type=int, nargs="+", default=[30, 100, 300])
    ap.add_argument("--checkpoint-every", type=int, default=10)
    return ap.parse_args()


def main():
    args = parse_args()
    args.R_checkpoints = sorted(set(args.R_checkpoints))
    if not args.R_checkpoints or min(args.R_checkpoints) < 2:
        raise ValueError("R-checkpoints must contain integers >= 2")
    args.R_max = max(args.R_checkpoints)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    trajs, names = load_all_hi(str(args.xjtu), cache=False)
    conditions = infer_xjtu_conditions(names)
    H = float(np.quantile([len(h) for h in trajs], args.horizon_quantile))
    tau = float(health_threshold(trajs, args.tau_percentile, "unit"))
    print(f"loaded {len(trajs)} trajectories; H={H:g}; tau={tau:g}", flush=True)

    if args.phase in ("diagnostics", "all"):
        run_weight_diagnostics(args, trajs, conditions, H, tau)
    if args.phase in ("convergence", "all"):
        run_convergence(args, trajs, conditions, H)


if __name__ == "__main__":
    main()
