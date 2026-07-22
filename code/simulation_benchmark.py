#!/usr/bin/env python3
"""Separate known-DGP validation of fitted and oracle time-varying IPCW.

The simulated censoring mechanism is discrete-time and known.  It uses the same
complementary-log-log form as the fitted time-varying arm.  The script compares
naive KM, fitted static IPCW, fitted time-varying IPCW, and oracle time-varying
IPCW.  One-factor-at-a-time scenarios cover sample size, measurement noise,
censoring prevalence, positivity stress, coarser health updates, and model
misspecification.  Replicate rows and time-specific weight/risk-set diagnostics
are retained.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src._survival import km, rmrl_from_survival  # noqa: E402
from src.arms.base import Dataset  # noqa: E402
from src.arms.ipcw_correction import IPCWCorrectionArm, TimeVaryingIPCWArm  # noqa: E402
from src.data_xjtu import static_baseline_summary  # noqa: E402

DEFAULT_SEEDS = (20260811, 20260812, 20260813)
HORIZON = 90


@dataclass(frozen=True)
class Scenario:
    name: str
    N: int = 100
    beta: float = 1.0
    sigma: float = 0.0
    censor_target: float = 0.4
    update_every: int = 1
    quadratic: float = 0.0


SCENARIOS = {
    s.name: s for s in (
        Scenario("reference"),
        Scenario("small_n", N=15),
        Scenario("large_n", N=500),
        Scenario("measurement_noise", sigma=0.5),
        Scenario("light_censoring", censor_target=0.2),
        Scenario("heavy_censoring", censor_target=0.6),
        Scenario("strong_selection", beta=2.0),
        Scenario("coarse_updates", update_every=5),
        Scenario("quadratic_misspec", quadratic=0.35),
    )
}


def generate_complete(n, rng, update_every=1):
    z = rng.normal(0.0, 1.0, n)
    log_t = math.log(60.0) - 0.24 * z + 0.34 * rng.normal(size=n)
    lifetimes = np.clip(np.rint(np.exp(log_t)), 12, 120).astype(int)
    health = []
    for zi, Ti in zip(z, lifetimes):
        u = np.arange(1, Ti + 1, dtype=float) / Ti
        base = 0.55 * zi + 0.65 * u + 1.25 * u**3
        if update_every > 1:
            src = (np.arange(Ti) // update_every) * update_every
            base = base[src]
        health.append(base.astype(float))
    return lifetimes, health


def log_hazard(hi, beta, tau, lam0, quadratic):
    d = np.asarray(hi, float) - tau
    return math.log(lam0) + beta * d + quadratic * d * d


def expected_censor_fraction(health, beta, tau, lam0, quadratic):
    vals = []
    for hi in health:
        if len(hi) <= 1:
            vals.append(0.0)
            continue
        mu = np.exp(np.clip(log_hazard(hi[:-1], beta, tau, lam0, quadratic), -30, 30))
        vals.append(1.0 - math.exp(-float(mu.sum())))
    return float(np.mean(vals))


def calibrate_policy(scenario, seed):
    rng = np.random.default_rng(seed)
    _, health = generate_complete(10000, rng, scenario.update_every)
    pooled = np.concatenate([h for h in health])
    tau = float(np.percentile(pooled, 70))
    # Precompute each calibration unit's total relative cumulative hazard over records
    # at which censoring can occur.  This makes calibration vectorised.
    S = np.asarray([
        float(np.exp(np.clip(scenario.beta * (h[:-1] - tau)
                             + scenario.quadratic * (h[:-1] - tau) ** 2, -30, 30)).sum())
        if len(h) > 1 else 0.0
        for h in health
    ])
    lo, hi = 1e-8, 10.0
    for _ in range(70):
        mid = math.sqrt(lo * hi)
        frac = float(np.mean(1.0 - np.exp(-mid * S)))
        if frac < scenario.censor_target:
            lo = mid
        else:
            hi = mid
    return tau, math.sqrt(lo * hi)


def impose_overlay(lifetimes, health_true, scenario, tau, lam0, rng):
    n = len(lifetimes)
    obs_t = np.empty(n, float)
    event = np.ones(n, int)
    x_obs = np.empty(n, float)
    hi_obs = []
    oracle_cum_before = []

    for i, (Ti, htrue) in enumerate(zip(lifetimes, health_true)):
        eta = log_hazard(htrue, scenario.beta, tau, lam0, scenario.quadratic)
        mu = np.exp(np.clip(eta, -30, 30))
        p = 1.0 - np.exp(-mu)
        u = rng.random(max(Ti - 1, 0))
        fired = np.flatnonzero(u < p[: max(Ti - 1, 0)])
        if len(fired):
            L = int(fired[0] + 1)
            event[i] = 0
        else:
            L = int(Ti)
        obs_t[i] = L
        noisy = htrue[:L] * np.exp(scenario.sigma * rng.normal(size=L))
        hi_obs.append(noisy.astype(float))
        # Match the real-data analysis: a fixed first-record baseline covariate.
        # The old 20%-of-observed-length window changed with the censoring time.
        x_obs[i] = static_baseline_summary(noisy, n_records=1)

        # cumulative true censoring hazard before integer times 0..Ti
        before = np.zeros(Ti + 1, float)
        if Ti > 1:
            before[2:] = np.cumsum(mu[:-1])
        oracle_cum_before.append(before)

    ds = Dataset(np.arange(n), obs_t, event, x_obs, hi_obs)
    return ds, oracle_cum_before


def weighted_km_from_cumhaz(time, event, cum_before, exp_clip=30.0):
    time = np.asarray(time, float)
    event = np.asarray(event, int)
    fails = np.unique(time[event == 1])
    if len(fails) == 0:
        return np.array([float(np.max(time))]), np.array([1.0])
    s = 1.0
    tt, ss = [], []
    for u0 in fails:
        u = int(round(float(u0)))
        idx = np.flatnonzero(time >= u0)
        logw = np.array([cum_before[i][min(u, len(cum_before[i]) - 1)] for i in idx])
        w = np.exp(np.clip(logw, None, exp_clip))
        Yw = float(w.sum())
        dead = (time[idx] == u0) & (event[idx] == 1)
        dNw = float(w[dead].sum())
        s *= max(0.0, 1.0 - dNw / Yw)
        tt.append(float(u0)); ss.append(float(s))
    return np.asarray(tt), np.asarray(ss)


def weight_diag_from_cumhaz(time, cum_before, times, exp_clip=30.0):
    time = np.asarray(time, float)
    out = []
    for u in times:
        ui = int(round(float(u)))
        idx = np.flatnonzero(time >= ui)
        if len(idx) == 0:
            out.append(dict(time=ui, n_at_risk=0, ess=0.0, ess_over_n_at_risk=np.nan,
                            weight_median=np.nan, weight_p95=np.nan,
                            weight_p99=np.nan, max_weight=np.nan))
            continue
        logw = np.array([cum_before[i][min(ui, len(cum_before[i]) - 1)] for i in idx])
        w = np.exp(np.clip(logw, None, exp_clip))
        ess = float(w.sum() ** 2 / np.sum(w**2))
        out.append(dict(time=ui, n_at_risk=len(idx), ess=ess, ess_over_n_at_risk=ess/len(idx),
                        weight_median=float(np.median(w)), weight_p95=float(np.percentile(w,95)),
                        weight_p99=float(np.percentile(w,99)), max_weight=float(np.max(w))))
    return out


def pct_bias(est, truth):
    return 100.0 * (float(est) - float(truth)) / float(truth)


def summarize(rows, keys):
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = []
    for key, rr in sorted(groups.items()):
        vals = np.array([float(r["bias_pct"]) for r in rr if r["bias_pct"] != ""], float)
        rec = dict(zip(keys, key))
        rec["n_attempted"] = len(rr)
        rec["n_successful"] = len(vals)
        rec["fit_failure_fraction"] = 1.0 - len(vals) / len(rr)
        if len(vals):
            rec.update(mean_bias_pct=float(vals.mean()),
                       mcse_bias_pct=float(vals.std(ddof=1)/math.sqrt(len(vals))) if len(vals)>1 else np.nan,
                       median_bias_pct=float(np.median(vals)),
                       p10_bias_pct=float(np.percentile(vals,10)),
                       p90_bias_pct=float(np.percentile(vals,90)))
        else:
            rec.update(mean_bias_pct=np.nan, mcse_bias_pct=np.nan,
                       median_bias_pct=np.nan, p10_bias_pct=np.nan,
                       p90_bias_pct=np.nan)
        out.append(rec)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reps", type=int, default=200)
    p.add_argument(
        "--seeds", nargs="+", default=[",".join(map(str, DEFAULT_SEEDS))],
        help="random seeds; accepts either comma-separated or space-separated values",
    )
    p.add_argument("--scenarios", nargs="*", choices=sorted(SCENARIOS), default=sorted(SCENARIOS))
    p.add_argument("--out-dir", type=Path, default=HERE.parent / "results" / "simulation")
    p.add_argument("--quick-check", action="store_true")
    a = p.parse_args()
    seed_tokens = []
    for token in a.seeds:
        seed_tokens.extend(part.strip() for part in str(token).split(","))
    seeds = [int(x) for x in seed_tokens if x]
    if not seeds:
        p.error("--seeds must contain at least one integer seed")
    if a.quick_check:
        a.reps = min(a.reps, 3)
        a.scenarios = ["reference", "small_n"]
    a.out_dir.mkdir(parents=True, exist_ok=True)

    policies = {name: calibrate_policy(SCENARIOS[name], 991000 + sorted(SCENARIOS).index(name))
                for name in a.scenarios}
    rows, diagnostics = [], []
    diag_times = [int(round(HORIZON*q)) for q in (0.25, 0.50, 0.75, 0.90, 1.0)]

    for name in a.scenarios:
        sc = SCENARIOS[name]
        tau, lam0 = policies[name]
        print(f"{name}: {asdict(sc)} tau={tau:.4f} lambda0={lam0:.6g}")
        for seed in seeds:
            rng = np.random.default_rng(seed + 1009 * (list(SCENARIOS).index(name)+1))
            for rep in range(a.reps):
                T, hi = generate_complete(sc.N, rng, sc.update_every)
                ds, oracle_before = impose_overlay(T, hi, sc, tau, lam0, rng)
                truth = float(np.mean(np.minimum(T, HORIZON)))
                realized_c = float(np.mean(ds.event == 0))

                # Naive KM
                tg, sg, _ = km(ds.Ttil, ds.event)
                est = rmrl_from_survival(tg, sg, 0.0, HORIZON)
                rows.append(dict(scenario=name, seed=seed, replicate=rep, arm="naive_km",
                                 bias_pct=pct_bias(est, truth), realized_censor_fraction=realized_c,
                                 fit_success=1, fit_status=0, fit_method="not_applicable",
                                 fit_message="", coef_intercept="", coef_slope=""))

                # Static fitted IPCW. Small samples can exhibit separation in the
                # censoring model, so a finite unpenalized MLE need not exist. Retain
                # that event as a failed replicate instead of aborting the full
                # validation run or silently accepting diverging coefficients.
                try:
                    static = IPCWCorrectionArm("linear").fit(ds)
                    est = float(static.predict_rmrl(ds, 0.0, HORIZON)[0])
                    rows.append(dict(
                        scenario=name, seed=seed, replicate=rep, arm="ipcw_static_fitted",
                        bias_pct=pct_bias(est, truth), realized_censor_fraction=realized_c,
                        fit_success=int(static.fit_success_), fit_status=static.fit_status_,
                        fit_method=static.fit_method_, fit_message=static.fit_message_,
                        coef_intercept=static.coef_[0], coef_slope=static.coef_[1],
                    ))
                except Exception as exc:
                    rows.append(dict(
                        scenario=name, seed=seed, replicate=rep, arm="ipcw_static_fitted",
                        bias_pct="", realized_censor_fraction=realized_c,
                        fit_success=0, fit_status=-1, fit_method="failed",
                        fit_message=str(exc), coef_intercept="", coef_slope="",
                    ))

                # Fitted time-varying IPCW.  Failed fits are retained explicitly.
                try:
                    tv = TimeVaryingIPCWArm().fit(ds)
                    est = float(tv.predict_rmrl(ds, 0.0, HORIZON)[0])
                    rows.append(dict(scenario=name, seed=seed, replicate=rep, arm="ipcw_tv_fitted",
                                     bias_pct=pct_bias(est, truth), realized_censor_fraction=realized_c,
                                     fit_success=int(tv.fit_success_), fit_status=tv.fit_status_,
                                     fit_method=tv.fit_method_, fit_message=tv.fit_message_,
                                     coef_intercept=tv.coef_[0], coef_slope=tv.coef_[1]))
                    for d in tv.diagnostics_at_times(diag_times):
                        diagnostics.append(dict(scenario=name, seed=seed, replicate=rep,
                                                arm="ipcw_tv_fitted", **d))
                except Exception as exc:
                    rows.append(dict(scenario=name, seed=seed, replicate=rep, arm="ipcw_tv_fitted",
                                     bias_pct="", realized_censor_fraction=realized_c,
                                     fit_success=0, fit_status=-1, fit_method="failed",
                                     fit_message=str(exc), coef_intercept="", coef_slope=""))

                # Oracle time-varying IPCW
                ot, os = weighted_km_from_cumhaz(ds.Ttil, ds.event, oracle_before)
                est = rmrl_from_survival(ot, os, 0.0, HORIZON)
                rows.append(dict(scenario=name, seed=seed, replicate=rep, arm="ipcw_tv_oracle",
                                 bias_pct=pct_bias(est, truth), realized_censor_fraction=realized_c,
                                 fit_success=1, fit_status=0, fit_method="known_DGP",
                                 fit_message="", coef_intercept=math.log(lam0), coef_slope=sc.beta))
                for d in weight_diag_from_cumhaz(ds.Ttil, oracle_before, diag_times):
                    diagnostics.append(dict(scenario=name, seed=seed, replicate=rep,
                                            arm="ipcw_tv_oracle", **d))

    raw_path = a.out_dir / "replicates.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    summary = summarize(rows, ["scenario", "arm"])
    summary_path = a.out_dir / "estimator_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader(); w.writerows(summary)

    diag_path = a.out_dir / "weight_diagnostics.csv"
    with diag_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(diagnostics[0]))
        w.writeheader(); w.writerows(diagnostics)

    design_path = a.out_dir / "design.csv"
    with design_path.open("w", newline="", encoding="utf-8") as f:
        fields = list(asdict(next(iter(SCENARIOS.values())))) + ["noise_model", "tau", "lambda0", "H", "calibration_n", "threshold_percentile", "calibration_seed", "replicate_seed_offset", "n_seeds", "reps_per_seed"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for name in a.scenarios:
            row = asdict(SCENARIOS[name]); row.update(noise_model="multiplicative_lognormal", tau=policies[name][0], lambda0=policies[name][1],
                                                       H=HORIZON, calibration_n=10000, threshold_percentile=70,
                                                       calibration_seed=991000 + sorted(SCENARIOS).index(name),
                                                       replicate_seed_offset=1009 * (list(SCENARIOS).index(name) + 1),
                                                       n_seeds=len(seeds), reps_per_seed=a.reps)
            w.writerow(row)

    print(f"wrote {raw_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {diag_path}")
    print(f"wrote {design_path}")


if __name__ == "__main__":
    main()
