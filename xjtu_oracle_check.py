#!/usr/bin/env python3
"""Standalone oracle-IPCW audit with no dependence on the main analysis pipeline.

This small audit intentionally reimplements the complete-data generator, overlay,
weighted product-limit estimator, and exact RMST integration using only NumPy and
the Python standard library.  It checks that the known censoring survival
recovers the net RMST under a separate known-DGP simulation and reduces the risk
that shared repository survival utilities make the main validation self-consistent.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

H = 90


def generate_complete(n: int, rng: np.random.Generator):
    z = rng.normal(size=n)
    log_t = math.log(60.0) - 0.24 * z + 0.34 * rng.normal(size=n)
    T = np.clip(np.rint(np.exp(log_t)), 12, 120).astype(int)
    X = []
    for zi, ti in zip(z, T):
        u = np.arange(1, ti + 1, dtype=float) / ti
        X.append(0.55 * zi + 0.65 * u + 1.25 * u**3)
    return T, X


def calibrate(seed: int = 991777, pilot_n: int = 10000, target: float = 0.4):
    rng = np.random.default_rng(seed)
    _, X = generate_complete(pilot_n, rng)
    tau = float(np.percentile(np.concatenate(X), 70))
    S = np.array([np.exp(np.clip(x[:-1] - tau, -30, 30)).sum() for x in X])
    lo, hi = 1e-8, 10.0
    for _ in range(70):
        mid = math.sqrt(lo * hi)
        frac = float(np.mean(1.0 - np.exp(-mid * S)))
        if frac < target:
            lo = mid
        else:
            hi = mid
    return tau, math.sqrt(lo * hi)


def overlay(T, X, tau, lam0, rng):
    n = len(T)
    obs = np.empty(n, float)
    event = np.ones(n, int)
    cum_before = []
    for i, (ti, x) in enumerate(zip(T, X)):
        mu = lam0 * np.exp(np.clip(x - tau, -30, 30))
        fired = np.flatnonzero(rng.random(max(ti - 1, 0)) < (1.0 - np.exp(-mu[: max(ti - 1, 0)])))
        if len(fired):
            obs[i] = int(fired[0] + 1)
            event[i] = 0
        else:
            obs[i] = int(ti)
        cb = np.zeros(ti + 1, float)
        if ti > 1:
            cb[2:] = np.cumsum(mu[:-1])
        cum_before.append(cb)
    return obs, event, cum_before


def weighted_product_limit(obs, event, cum_before):
    failures = np.unique(obs[event == 1])
    times, surv = [], []
    s = 1.0
    for uf in failures:
        u = int(round(float(uf)))
        risk = np.flatnonzero(obs >= uf)
        w = np.exp(np.array([cum_before[i][min(u, len(cum_before[i]) - 1)] for i in risk]))
        dead = (obs[risk] == uf) & (event[risk] == 1)
        s *= max(0.0, 1.0 - float(w[dead].sum()) / float(w.sum()))
        times.append(float(uf))
        surv.append(s)
    return np.asarray(times), np.asarray(surv)


def exact_rmst(times, surv, horizon):
    if len(times) == 0:
        return float(horizon)
    knots = times[(times > 0) & (times < horizon)]
    bounds = np.concatenate(([0.0], knots, [float(horizon)]))
    idx = np.searchsorted(times, bounds[:-1], side="right") - 1
    level = np.where(idx >= 0, surv[np.clip(idx, 0, len(surv) - 1)], 1.0)
    return float(np.sum(np.diff(bounds) * level))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--N", type=int, default=500)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--out-dir", type=Path, default=Path("results/xjtu"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tau, lam0 = calibrate()
    rng = np.random.default_rng(args.seed)
    rows = []
    for r in range(args.reps):
        T, X = generate_complete(args.N, rng)
        obs, event, cum_before = overlay(T, X, tau, lam0, rng)
        tt, ss = weighted_product_limit(obs, event, cum_before)
        estimate = exact_rmst(tt, ss, H)
        truth = float(np.mean(np.minimum(T, H)))
        rows.append({
            "replicate": r,
            "truth_rmst": truth,
            "oracle_rmst": estimate,
            "bias_pct": 100.0 * (estimate - truth) / truth,
            "realized_censor_fraction": float(np.mean(event == 0)),
        })

    raw = args.out_dir / "standalone_oracle_check_replicates.csv"
    with raw.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    b = np.array([r["bias_pct"] for r in rows], float)
    c = np.array([r["realized_censor_fraction"] for r in rows], float)
    summary = {
        "implementation": "standalone_numpy_only",
        "N": args.N,
        "reps": args.reps,
        "seed": args.seed,
        "H": H,
        "pilot_n": 10000,
        "tau": tau,
        "lambda0": lam0,
        "mean_bias_pct": float(b.mean()),
        "mcse_bias_pct": float(b.std(ddof=1) / math.sqrt(len(b))),
        "median_bias_pct": float(np.median(b)),
        "p10_bias_pct": float(np.percentile(b, 10)),
        "p90_bias_pct": float(np.percentile(b, 90)),
        "mean_censor_fraction": float(c.mean()),
    }
    out = args.out_dir / "standalone_oracle_check_summary.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary))
        w.writeheader(); w.writerow(summary)
    print(summary)
    print(f"wrote {raw}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
