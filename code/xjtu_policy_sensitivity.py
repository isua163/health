#!/usr/bin/env python3
"""Sensitivity of the XJTU selection increment to policy and measurement noise.

The grid varies health-measurement noise sigma, target preventive-replacement
fraction c, and health-selection strength beta.  Common uniforms and standard
normal draws are reused across beta and sigma within each paired replicate.
Results include paired increment MCSEs and the TV-IPCW reduction relative to
naive survival, avoiding an unsupported single-configuration claim.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.arms.ipcw_correction import IPCWCorrectionArm, TimeVaryingIPCWArm  # noqa: E402
from src.arms.naive_survival import NaiveSurvivalArm  # noqa: E402
from src.data_xjtu import (  # noqa: E402
    calibrate_lambda0,
    health_threshold,
    impose_informative_censoring_from_draws,
    load_all_hi,
)

ARMS = (
    ("naive_survival", NaiveSurvivalArm),
    ("ipcw_static", lambda: IPCWCorrectionArm("log")),
    ("ipcw_tv", TimeVaryingIPCWArm),
)


def pct_bias(est, truth):
    return 100.0 * (float(est) - truth) / truth


def summarize(values):
    v = np.asarray([x for x in values if np.isfinite(x)], float)
    if len(v) == 0:
        return dict(n=0, mean=np.nan, sd=np.nan, mcse=np.nan, p10=np.nan, p90=np.nan)
    sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    return dict(n=len(v), mean=float(v.mean()), sd=sd,
                mcse=float(sd / math.sqrt(len(v))),
                p10=float(np.percentile(v, 10)), p90=float(np.percentile(v, 90)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xjtu", required=True)
    p.add_argument("--R", type=int, default=1000)
    p.add_argument("--sigmas", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75])
    p.add_argument("--c-values", nargs="+", type=float, default=[0.2, 0.4, 0.6])
    p.add_argument("--betas", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0])
    p.add_argument("--horizon-quantile", type=float, default=0.50)
    p.add_argument("--threshold-weighting", choices=["record", "unit"], default="record")
    p.add_argument("--baseline-records", type=int, default=1)
    p.add_argument("--seed", type=int, default=20260903)
    p.add_argument("--out-dir", type=Path, default=REPO / "results")
    p.add_argument("--quick-check", action="store_true")
    a = p.parse_args()
    if a.quick_check:
        a.R = min(a.R, 3)
        a.sigmas = a.sigmas[:2]
        a.c_values = a.c_values[:1]
        a.betas = a.betas[:3]
    if 0.0 not in a.betas:
        raise ValueError("betas must include 0.0 as the selection-null reference")
    a.out_dir.mkdir(parents=True, exist_ok=True)

    hi, _ = load_all_hi(a.xjtu, "rms", 0, cache=False)
    T = np.asarray([len(h) for h in hi], float)
    H = float(np.quantile(T, a.horizon_quantile))
    truth = float(np.mean(np.minimum(T, H)))
    tau = health_threshold(hi, 70.0, a.threshold_weighting)
    policies = {
        (c, beta): calibrate_lambda0(hi, beta, tau, c)
        for c in a.c_values for beta in a.betas
    }
    rng = np.random.default_rng(a.seed)

    rows = []
    for c in a.c_values:
        for rep in range(a.R):
            uniforms = [rng.random(len(h)) for h in hi]
            noises = [rng.standard_normal(len(h)) for h in hi]
            for sigma in a.sigmas:
                cell = {}
                realized = {}
                failures = {}
                for beta in a.betas:
                    ds, _ = impose_informative_censoring_from_draws(
                        hi, beta, tau, policies[(c, beta)], sigma, uniforms, noises,
                        baseline_records=a.baseline_records,
                    )
                    realized[beta] = float(np.mean(ds.event == 0))
                    for arm_name, factory in ARMS:
                        try:
                            arm = factory().fit(ds)
                            cell[(beta, arm_name)] = pct_bias(
                                float(arm.predict_rmrl(ds, 0.0, H)[0]), truth
                            )
                        except Exception as exc:
                            failures[(beta, arm_name)] = str(exc)

                for beta in a.betas:
                    for arm_name, _ in ARMS:
                        ok = (beta, arm_name) in cell and (0.0, arm_name) in cell
                        rows.append(dict(
                            replicate=rep, c=c, sigma=sigma, beta=beta, arm=arm_name,
                            bias_pct=cell.get((beta, arm_name), np.nan),
                            beta0_bias_pct=cell.get((0.0, arm_name), np.nan),
                            increment_vs_beta0=(
                                cell[(beta, arm_name)] - cell[(0.0, arm_name)] if ok else np.nan
                            ),
                            realized_censor_fraction=realized[beta],
                            lambda0=policies[(c, beta)], tau=tau, horizon_quantile=a.horizon_quantile, H=H, truth_rmst=truth,
                            fit_success=int(ok),
                            fit_message=failures.get((beta, arm_name), ""),
                            threshold_weighting=a.threshold_weighting,
                            baseline_records=a.baseline_records,
                        ))
        print(f"c={c:g}: completed {a.R} paired overlays across sigma/beta grid")

    rep_path = a.out_dir / "xjtu_policy_sensitivity_replicates.csv"
    with rep_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    summary_rows = []
    for c in a.c_values:
        for sigma in a.sigmas:
            for beta in a.betas:
                for arm_name, _ in ARMS:
                    rr = [r for r in rows if r["c"] == c and r["sigma"] == sigma
                          and r["beta"] == beta and r["arm"] == arm_name]
                    ok = [r for r in rr if r["fit_success"] == 1]
                    summary_rows.append(dict(
                        c=c, sigma=sigma, beta=beta, arm=arm_name,
                        attempted=len(rr), successful=len(ok),
                        failure_fraction=1.0 - len(ok) / len(rr),
                        endpoint="increment_vs_beta0_pct_points", horizon_quantile=a.horizon_quantile, H=H,
                        **summarize([r["increment_vs_beta0"] for r in ok]),
                    ))

    summary_path = a.out_dir / "xjtu_policy_sensitivity_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0])); w.writeheader(); w.writerows(summary_rows)

    reduction_rows = []
    lookup = {(r["replicate"], r["c"], r["sigma"], r["beta"], r["arm"]): r for r in rows}
    for c in a.c_values:
        for sigma in a.sigmas:
            for beta in [b for b in a.betas if b != 0.0]:
                reductions = []
                absolute_differences = []
                for rep in range(a.R):
                    nrow = lookup[(rep, c, sigma, beta, "naive_survival")]
                    trow = lookup[(rep, c, sigma, beta, "ipcw_tv")]
                    ni, ti = nrow["increment_vs_beta0"], trow["increment_vs_beta0"]
                    if np.isfinite(ni) and np.isfinite(ti):
                        absolute_differences.append(ti - ni)
                        if abs(ni) > 1e-8:
                            reductions.append(1.0 - ti / ni)
                abs_s = summarize(absolute_differences)
                rel_s = summarize(reductions)
                reduction_rows.append(dict(
                    c=c, sigma=sigma, beta=beta, horizon_quantile=a.horizon_quantile, H=H,
                    paired_n=abs_s["n"], tv_minus_naive_increment_mean=abs_s["mean"],
                    tv_minus_naive_increment_mcse=abs_s["mcse"],
                    relative_reduction_n=rel_s["n"], relative_reduction_mean=rel_s["mean"],
                    relative_reduction_sd=rel_s["sd"], relative_reduction_mcse=rel_s["mcse"],
                    relative_reduction_p10=rel_s["p10"], relative_reduction_p90=rel_s["p90"],
                ))

    reduction_path = a.out_dir / "xjtu_policy_sensitivity_reductions.csv"
    with reduction_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(reduction_rows[0])); w.writeheader(); w.writerows(reduction_rows)

    print(f"wrote {rep_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {reduction_path}")


if __name__ == "__main__":
    main()
