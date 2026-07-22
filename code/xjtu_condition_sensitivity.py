#!/usr/bin/env python3
"""XJTU operating-condition and health-threshold weighting sensitivity.

The script reports the pooled 15-bearing analysis and separate analyses within
each of the three speed/load conditions.  It also compares the original
record-weighted pooled health percentile with a unit-equal percentile, preventing
long trajectories from receiving disproportionate weight in the policy threshold.
All beta contrasts use common uniforms and health-noise draws.
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

from src.arms.competing_risks import CompetingRisksArm  # noqa: E402
from src.arms.ipcw_correction import IPCWCorrectionArm, TimeVaryingIPCWArm  # noqa: E402
from src.arms.naive_regression import NaiveRegressionArm  # noqa: E402
from src.arms.naive_survival import NaiveSurvivalArm  # noqa: E402
from src.data_xjtu import (  # noqa: E402
    calibrate_lambda0,
    health_threshold,
    impose_informative_censoring_from_draws,
    infer_xjtu_conditions,
    load_all_hi,
)

ARMS = (
    ("naive_regression", NaiveRegressionArm),
    ("naive_survival", NaiveSurvivalArm),
    ("competing_risks", CompetingRisksArm),
    ("ipcw_static", lambda: IPCWCorrectionArm("log")),
    ("ipcw_tv", TimeVaryingIPCWArm),
)


def pct_bias(est, truth):
    return 100.0 * (float(est) - truth) / truth


def summarize(values):
    v = np.asarray(values, float)
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
    p.add_argument("--sigma", type=float, default=0.5)
    p.add_argument("--c", type=float, default=0.4)
    p.add_argument("--betas", nargs=2, type=float, default=[0.0, 1.0])
    p.add_argument("--horizon-quantile", type=float, default=0.50)
    p.add_argument("--threshold-weightings", nargs="+", choices=["record", "unit"],
                   default=["record", "unit"])
    p.add_argument("--baseline-records", type=int, default=1)
    p.add_argument("--seed", type=int, default=20260902)
    p.add_argument("--out-dir", type=Path, default=REPO / "results")
    p.add_argument("--quick-check", action="store_true")
    a = p.parse_args()
    if a.quick_check:
        a.R = min(a.R, 3)
    a.out_dir.mkdir(parents=True, exist_ok=True)

    full_hi, names = load_all_hi(a.xjtu, "rms", 0, cache=False)
    labels = infer_xjtu_conditions(names)
    unique_conditions = list(dict.fromkeys(labels.tolist()))
    groups = [("pooled", np.arange(len(full_hi), dtype=int))]
    groups += [(str(label), np.flatnonzero(labels == label)) for label in unique_conditions]

    rng = np.random.default_rng(a.seed)
    rows = []
    for group_name, idx in groups:
        hi = [np.asarray(full_hi[i], float) for i in idx]
        T = np.asarray([len(h) for h in hi], float)
        H = float(np.quantile(T, a.horizon_quantile))
        truth = float(np.mean(np.minimum(T, H)))

        for weighting in a.threshold_weightings:
            tau = health_threshold(hi, 70.0, weighting)
            lam = {b: calibrate_lambda0(hi, b, tau, a.c) for b in a.betas}
            for rep in range(a.R):
                uniforms = [rng.random(len(h)) for h in hi]
                noises = [rng.standard_normal(len(h)) for h in hi]
                values = {}
                failures = {}
                realized_c = {}
                for beta in a.betas:
                    ds, _ = impose_informative_censoring_from_draws(
                        hi, beta, tau, lam[beta], a.sigma, uniforms, noises,
                        baseline_records=a.baseline_records,
                    )
                    realized_c[beta] = float(np.mean(ds.event == 0))
                    for arm_name, factory in ARMS:
                        try:
                            arm = factory().fit(ds)
                            est = float(arm.predict_rmrl(ds, 0.0, H)[0])
                            values[(beta, arm_name)] = pct_bias(est, truth)
                        except Exception as exc:
                            failures[(beta, arm_name)] = str(exc)

                for arm_name, _ in ARMS:
                    b0, b1 = a.betas
                    ok = (b0, arm_name) in values and (b1, arm_name) in values
                    rows.append(dict(
                        group=group_name,
                        analysis_scope="pooled" if group_name == "pooled" else "within_condition",
                        n_units=len(hi), source_indices=";".join(map(str, idx.tolist())),
                        threshold_weighting=weighting, tau=tau,
                        horizon_quantile=a.horizon_quantile, H=H, truth_rmst=truth,
                        replicate=rep, arm=arm_name, beta0=b0, beta1=b1,
                        beta0_bias_pct=values.get((b0, arm_name), np.nan),
                        beta1_bias_pct=values.get((b1, arm_name), np.nan),
                        increment_beta1_minus_beta0=(
                            values[(b1, arm_name)] - values[(b0, arm_name)] if ok else np.nan
                        ),
                        realized_censor_beta0=realized_c[b0],
                        realized_censor_beta1=realized_c[b1],
                        fit_success=int(ok),
                        fit_message=" | ".join(
                            f"beta={b}: {failures[(b, arm_name)]}"
                            for b in a.betas if (b, arm_name) in failures
                        ),
                        sigma=a.sigma, c=a.c, baseline_records=a.baseline_records,
                    ))

            print(f"{group_name} weighting={weighting}: completed {a.R} paired overlays")

    rep_path = a.out_dir / "xjtu_condition_sensitivity_replicates.csv"
    with rep_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    summary_rows = []
    for group_name, _ in groups:
        for weighting in a.threshold_weightings:
            for arm_name, _ in ARMS:
                rr = [r for r in rows if r["group"] == group_name
                      and r["threshold_weighting"] == weighting and r["arm"] == arm_name]
                ok = [r for r in rr if r["fit_success"] == 1]
                rec = dict(
                    group=group_name,
                    analysis_scope=rr[0]["analysis_scope"],
                    n_units=rr[0]["n_units"], threshold_weighting=weighting,
                    arm=arm_name, attempted=len(rr), successful=len(ok),
                    failure_fraction=1.0 - len(ok) / len(rr),
                    horizon_quantile=a.horizon_quantile, H=rr[0]["H"], tau=rr[0]["tau"],
                    **summarize([r["increment_beta1_minus_beta0"] for r in ok]),
                )
                summary_rows.append(rec)

    summary_path = a.out_dir / "xjtu_condition_sensitivity_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0])); w.writeheader(); w.writerows(summary_rows)

    print(f"wrote {rep_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
