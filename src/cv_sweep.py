"""Generator-specific synthetic dispersion reference for Figure 3.

Only the log-normal lifetime dispersion parameter is varied.  The resulting
curve is a descriptive reference for the selected generator and overlay; it is
not a causal calibration curve and is not transported to the real fleets.

The formal default uses three independent seeds and 100 fleet replicates per
seed at each dispersion point.  Replicate-level results are retained and the
summary reports Monte-Carlo standard errors and fleet-to-fleet quantiles.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Iterable

import numpy as np

from .data_xjtu import calibrate_lambda0, impose_informative_censoring
from .estimand import net_rmrl_truth
from .metrics import bias_by_class
from .arms.naive_survival import NaiveSurvivalArm
from .standardize import standardize_hi

_ROOT = Path(__file__).resolve().parents[1]
SIGMA_LOG_GRID = [0.15, 0.35, 0.55, 0.75, 0.95, 1.15, 1.35, 1.55, 1.75]
MEDIAN_LIFE = 400
DEFAULT_SEEDS = (20260731, 20260801, 20260802)


def log_p90_p10(L):
    a, b = np.percentile(L, [10, 90])
    return float(np.log(b / max(a, 1e-9)))


def logSD(L):
    return float(np.std(np.log(np.asarray(L, float))))


def gini(L):
    x = np.sort(np.asarray(L, float)); n = len(x); c = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(c) / c[-1]) / n)


def iqr_over_med(L):
    q1, q2, q3 = np.percentile(L, [25, 50, 75])
    return float((q3 - q1) / max(q2, 1e-9))


def mad_over_med(L):
    m = np.median(L)
    return float(np.median(np.abs(np.asarray(L) - m)) / max(m, 1e-9))


def cv(L):
    x = np.asarray(L, float)
    return float(x.std() / x.mean())


DISPERSION = {
    "log_p90_p10": log_p90_p10,
    "logSD": logSD,
    "Gini": gini,
    "IQR_med": iqr_over_med,
    "MAD_med": mad_over_med,
    "CV": cv,
}


def make_fleet(N, sigma_log, rng, med_life=MEDIAN_LIFE):
    L = np.clip(
        np.exp(np.log(med_life) + sigma_log * rng.standard_normal(N)),
        20,
        3000,
    ).astype(int)
    hi = []
    for Li in L:
        f = np.clip(med_life / Li, 0.3, 5.0)
        base = 0.3 + 0.4 * f
        u = np.arange(Li) / Li
        hi.append(np.maximum(base + 0.6 * u**1.5 * f + 0.03 * rng.standard_normal(Li), 1e-3))
    return hi, L


def _naive_bias(hi, beta, sigma, c, rng, reps):
    T = np.array([len(h) for h in hi], float)
    H = float(np.percentile(T, 90))
    tau = float(np.percentile(np.concatenate(hi), 70))
    lam0 = calibrate_lambda0(hi, beta, tau, c)
    b = []
    for _ in range(reps):
        ds, truth = impose_informative_censoring(hi, beta, tau, lam0, sigma, rng)
        val, at = net_rmrl_truth(truth, 0.0, H)
        pred = NaiveSurvivalArm().fit(ds).predict_rmrl(ds, 0.0, H)
        b.append(bias_by_class(pred, val, at)["bias_pct"])
    return float(np.mean(b))


def _parse_seeds(seeds: str | Iterable[int] | None, seed: int | None):
    if seed is not None:
        return (int(seed),)
    if seeds is None:
        return DEFAULT_SEEDS
    if isinstance(seeds, str):
        out = tuple(int(x.strip()) for x in seeds.split(",") if x.strip())
    else:
        out = tuple(int(x) for x in seeds)
    if not out:
        raise ValueError("at least one seed is required")
    return out


def run(
    beta=1.0,
    sigma=0.5,
    N=40,
    seed=None,
    standardize="iqr",
    *,
    seeds=None,
    fleet_reps=None,
    cens_reps=5,
    outdir=None,
):
    """Run the reference sweep and return summary rows.

    ``seed`` is retained for backward-compatible tests and creates a single-seed
    run.  Formal runs should use ``seeds`` and at least 100 fleet replicates per
    seed.
    """
    seeds = _parse_seeds(seeds, seed)
    if fleet_reps is None:
        fleet_reps = 8 if seed is not None else 100
    if fleet_reps < 2 or cens_reps < 1:
        raise ValueError("fleet_reps must be >=2 and cens_reps >=1")
    outdir = Path(outdir) if outdir else _ROOT / "results"
    outdir.mkdir(parents=True, exist_ok=True)

    replicates = []
    print(
        f"synthetic dispersion reference: N={N}, beta={beta}, sigma={sigma}, "
        f"standardize={standardize}, seeds={list(seeds)}, "
        f"fleet_reps/seed={fleet_reps}, censoring overlays/arm={cens_reps}",
        flush=True,
    )
    for master_seed in seeds:
        rng = np.random.default_rng(master_seed)
        for sl in SIGMA_LOG_GRID:
            for rep in range(fleet_reps):
                hi, L = make_fleet(N, sl, rng)
                hi = standardize_hi(hi, standardize)
                row = {
                    "seed": master_seed,
                    "replicate": rep,
                    "sigma_log": sl,
                    **{name: fn(L) for name, fn in DISPERSION.items()},
                }
                b1 = _naive_bias(hi, beta, sigma, 0.4, rng, cens_reps)
                b0 = _naive_bias(hi, 0.0, sigma, 0.4, rng, cens_reps)
                row["bias"] = b1 - b0
                replicates.append(row)

    summary = []
    for sl in SIGMA_LOG_GRID:
        rr = [r for r in replicates if r["sigma_log"] == sl]
        bias = np.asarray([r["bias"] for r in rr], float)
        seed_means = [
            float(np.mean([r["bias"] for r in rr if int(r["seed"]) == int(master_seed)]))
            for master_seed in seeds
        ]
        rec = {
            "sigma_log": sl,
            **{name: float(np.mean([r[name] for r in rr])) for name in DISPERSION},
            "bias": float(np.mean(bias)),
            "bias_mcse": float(np.std(bias, ddof=1) / np.sqrt(len(bias))),
            "bias_p10": float(np.percentile(bias, 10)),
            "bias_p90": float(np.percentile(bias, 90)),
            "bias_seed_mean_min": float(min(seed_means)),
            "bias_seed_mean_max": float(max(seed_means)),
            "n_fleets": len(bias),
            "n_seeds": len(seeds),
            "seeds": ";".join(map(str, seeds)),
            "beta": beta,
            "sigma": sigma,
            "N": N,
            "cens_reps": cens_reps,
            "standardize": standardize,
            "scope_note": "generator-specific descriptive reference; not a causal or field calibration curve",
        }
        summary.append(rec)
        print(
            f"sigma_log={sl:4.2f} x={rec['log_p90_p10']:5.2f} "
            f"excess={rec['bias']:+7.3f} +/- {rec['bias_mcse']:.3f} pp",
            flush=True,
        )

    rep_path = outdir / f"cv_sweep_synthetic_{standardize}_replicates.csv"
    sum_path = outdir / f"cv_sweep_synthetic_{standardize}.csv"
    rep_fields = list(replicates[0])
    with rep_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rep_fields); w.writeheader(); w.writerows(replicates)
    sum_fields = list(summary[0])
    with sum_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sum_fields); w.writeheader(); w.writerows(summary)
    print(f"wrote {sum_path}")
    print(f"wrote {rep_path}")
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--sigma", type=float, default=0.5)
    p.add_argument("--N", type=int, default=40)
    p.add_argument("--standardize", choices=["none", "iqr", "rank"], default="iqr")
    p.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    p.add_argument("--fleet-reps", type=int, default=100)
    p.add_argument("--cens-reps", type=int, default=5)
    p.add_argument("--outdir", default=str(_ROOT / "results"))
    a = p.parse_args(argv)
    run(
        beta=a.beta,
        sigma=a.sigma,
        N=a.N,
        standardize=a.standardize,
        seeds=a.seeds,
        fleet_reps=a.fleet_reps,
        cens_reps=a.cens_reps,
        outdir=a.outdir,
    )


if __name__ == "__main__":
    main()
