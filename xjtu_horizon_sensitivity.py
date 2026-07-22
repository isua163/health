#!/usr/bin/env python3
"""XJTU RMST-horizon and tail-support sensitivity analysis.

For each paired preventive-replacement overlay, the script evaluates empirical
net-lifetime horizons (default p50/p70/p80/p90), reports absolute risk-set sizes
and IPCW weight diagnostics at fractions of each horizon, and contrasts:

1. the fixed target horizon requested by the scientific estimand; and
2. a conservative paired common-support horizon capped where both beta cells
   retain a minimum observed risk set and an observed failure.

The fixed-horizon estimate is retained even when unsupported, but is explicitly
flagged so summaries can report both unconditional and supported-only results.
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

from src._survival import maximum_supported_time, risk_set_diagnostics  # noqa: E402
from src.arms.competing_risks import CompetingRisksArm  # noqa: E402
from src.arms.ipcw_correction import IPCWCorrectionArm, TimeVaryingIPCWArm  # noqa: E402
from src.arms.naive_regression import NaiveRegressionArm  # noqa: E402
from src.arms.naive_survival import NaiveSurvivalArm  # noqa: E402
from src.data_xjtu import (  # noqa: E402
    calibrate_lambda0,
    impose_informative_censoring_from_draws,
    load_all_hi,
)

ARMS = (
    ("naive_regression", NaiveRegressionArm),
    ("naive_survival", NaiveSurvivalArm),
    ("competing_risks", CompetingRisksArm),
    ("ipcw_static", lambda: IPCWCorrectionArm("log")),
    ("ipcw_tv", TimeVaryingIPCWArm),
)


def pct_bias(estimate, truth_mean):
    return 100.0 * (float(estimate) - float(truth_mean)) / float(truth_mean)


def finite_summary(values):
    v = np.asarray([x for x in values if np.isfinite(x)], float)
    if len(v) == 0:
        return dict(n=0, mean=np.nan, sd=np.nan, mcse=np.nan, p10=np.nan, p90=np.nan)
    sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    return dict(
        n=int(len(v)), mean=float(v.mean()), sd=sd,
        mcse=float(sd / math.sqrt(len(v))),
        p10=float(np.percentile(v, 10)), p90=float(np.percentile(v, 90)),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xjtu", required=True)
    p.add_argument("--R", type=int, default=1000)
    p.add_argument("--sigma", type=float, default=0.5)
    p.add_argument("--c", type=float, default=0.4)
    p.add_argument("--betas", nargs=2, type=float, default=[0.0, 1.0])
    p.add_argument("--horizon-quantiles", nargs="+", type=float,
                   default=[0.50, 0.70, 0.80, 0.90])
    p.add_argument("--min-at-risk", type=int, default=5)
    p.add_argument("--baseline-records", type=int, default=1)
    p.add_argument("--seed", type=int, default=20260901)
    p.add_argument("--out-dir", type=Path, default=REPO / "results")
    p.add_argument("--quick-check", action="store_true")
    a = p.parse_args()
    if a.quick_check:
        a.R = min(a.R, 3)
    a.out_dir.mkdir(parents=True, exist_ok=True)

    hi, names = load_all_hi(a.xjtu, "rms", 0, cache=False)
    T = np.asarray([len(h) for h in hi], float)
    horizons = {q: float(np.quantile(T, q)) for q in a.horizon_quantiles}
    tau = float(np.percentile(np.concatenate(hi), 70))
    lam = {beta: calibrate_lambda0(hi, beta, tau, a.c) for beta in a.betas}
    rng = np.random.default_rng(a.seed)

    estimate_rows, diagnostic_rows = [], []
    for rep in range(a.R):
        uniforms = [rng.random(len(h)) for h in hi]
        noises = [rng.standard_normal(len(h)) for h in hi]
        datasets, fitted = {}, {}

        for beta in a.betas:
            ds, _ = impose_informative_censoring_from_draws(
                hi, beta, tau, lam[beta], a.sigma, uniforms, noises,
                baseline_records=a.baseline_records,
            )
            datasets[beta] = ds
            fitted[beta] = {}
            for arm_name, factory in ARMS:
                try:
                    fitted[beta][arm_name] = factory().fit(ds)
                except Exception as exc:
                    fitted[beta][arm_name] = exc

        for hq, H in horizons.items():
            support = {
                beta: maximum_supported_time(
                    datasets[beta].Ttil, datasets[beta].event,
                    min_at_risk=a.min_at_risk, require_observed_failure=True,
                )
                for beta in a.betas
            }
            common_H = min([H] + [support[b] for b in a.betas])
            fixed_supported = int(common_H >= H - 1e-12)

            for beta in a.betas:
                ds = datasets[beta]
                checkpoints = sorted({float(max(1, int(round(H * f))))
                                      for f in (0.50, 0.75, 0.90, 1.00)})
                base_diag = risk_set_diagnostics(ds.Ttil, ds.event, checkpoints)
                weight_by_arm = {}
                for arm_name in ("ipcw_static", "ipcw_tv"):
                    arm = fitted[beta][arm_name]
                    if not isinstance(arm, Exception):
                        weight_by_arm[arm_name] = {
                            float(r["time"]): r for r in arm.diagnostics_at_times(checkpoints)
                        }
                last_failure = float(np.max(ds.Ttil[ds.event == 1])) if np.any(ds.event == 1) else 0.0
                for d in base_diag:
                    for arm_name in ("unweighted", "ipcw_static", "ipcw_tv"):
                        row = dict(
                            replicate=rep, beta=beta, sigma=a.sigma,
                            horizon_quantile=hq, H=H,
                            checkpoint_fraction=float(d["time"] / H) if H else np.nan,
                            last_observed_failure=last_failure,
                            max_supported_min3=maximum_supported_time(
                                ds.Ttil, ds.event, 3, require_observed_failure=True),
                            max_supported_min5=maximum_supported_time(
                                ds.Ttil, ds.event, 5, require_observed_failure=True),
                            arm=arm_name,
                            **d,
                        )
                        wd = weight_by_arm.get(arm_name, {}).get(float(d["time"]))
                        if wd:
                            row.update(
                                ess=wd["ess"], ess_over_n_at_risk=wd["ess_over_n_at_risk"],
                                weight_median=wd["weight_median"], weight_p95=wd["weight_p95"],
                                weight_p99=wd["weight_p99"], max_weight=wd["max_weight"],
                            )
                        diagnostic_rows.append(row)

            for arm_name, _ in ARMS:
                estimates_fixed, estimates_common = {}, {}
                failed = False
                failure_messages = []
                for beta in a.betas:
                    arm = fitted[beta][arm_name]
                    if isinstance(arm, Exception):
                        failed = True
                        failure_messages.append(f"beta={beta}: {arm}")
                        continue
                    estimates_fixed[beta] = float(arm.predict_rmrl(datasets[beta], 0.0, H)[0])
                    estimates_common[beta] = float(arm.predict_rmrl(datasets[beta], 0.0, common_H)[0])

                if failed:
                    estimate_rows.append(dict(
                        replicate=rep, arm=arm_name, horizon_quantile=hq, H=H,
                        target_mode="fit_failed", common_support_H=common_H,
                        fixed_horizon_supported_min_at_risk=fixed_supported,
                        fit_success=0, fit_message=" | ".join(failure_messages),
                    ))
                    continue

                for mode, target_H, estimates in (
                    ("fixed_H", H, estimates_fixed),
                    (f"paired_common_support_min{a.min_at_risk}", common_H, estimates_common),
                ):
                    truth_mean = float(np.mean(np.minimum(T, target_H))) if target_H > 0 else np.nan
                    b0, b1 = a.betas
                    bias0 = pct_bias(estimates[b0], truth_mean) if target_H > 0 else np.nan
                    bias1 = pct_bias(estimates[b1], truth_mean) if target_H > 0 else np.nan
                    estimate_rows.append(dict(
                        replicate=rep, arm=arm_name, horizon_quantile=hq, H=H,
                        target_mode=mode, evaluated_H=target_H, common_support_H=common_H,
                        fixed_horizon_supported_min_at_risk=fixed_supported,
                        beta0=b0, beta1=b1, beta0_bias_pct=bias0, beta1_bias_pct=bias1,
                        increment_beta1_minus_beta0=bias1 - bias0,
                        truth_rmst=truth_mean, fit_success=1, fit_message="",
                        min_at_risk=a.min_at_risk, sigma=a.sigma, c=a.c,
                        baseline_records=a.baseline_records,
                    ))

        if (rep + 1) % max(1, a.R // 10) == 0:
            print(f"{rep + 1}/{a.R} paired overlays")

    rep_path = a.out_dir / "xjtu_horizon_support_replicates.csv"
    with rep_path.open("w", newline="", encoding="utf-8") as f:
        fields = sorted({k for r in estimate_rows for k in r})
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(estimate_rows)

    diag_path = a.out_dir / "xjtu_horizon_support_diagnostics.csv"
    with diag_path.open("w", newline="", encoding="utf-8") as f:
        fields = sorted({k for r in diagnostic_rows for k in r})
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(diagnostic_rows)

    summary_rows = []
    for hq in a.horizon_quantiles:
        for arm_name, _ in ARMS:
            for mode in ("fixed_H", f"paired_common_support_min{a.min_at_risk}"):
                rr = [r for r in estimate_rows if r.get("fit_success") == 1
                      and r.get("horizon_quantile") == hq and r.get("arm") == arm_name
                      and r.get("target_mode") == mode]
                vals = [r["increment_beta1_minus_beta0"] for r in rr]
                rec = dict(horizon_quantile=hq, arm=arm_name, target_mode=mode,
                           support_fraction=float(np.mean([
                               r["fixed_horizon_supported_min_at_risk"] for r in rr
                           ])) if rr else np.nan,
                           median_evaluated_H=float(np.median([r["evaluated_H"] for r in rr])) if rr else np.nan,
                           **finite_summary(vals))
                if mode == "fixed_H":
                    supported = [r["increment_beta1_minus_beta0"] for r in rr
                                 if r["fixed_horizon_supported_min_at_risk"] == 1]
                    s = finite_summary(supported)
                    rec.update(supported_n=s["n"], supported_mean=s["mean"],
                               supported_sd=s["sd"], supported_mcse=s["mcse"])
                summary_rows.append(rec)

    summary_path = a.out_dir / "xjtu_horizon_support_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0])); w.writeheader(); w.writerows(summary_rows)

    print(f"wrote {rep_path}")
    print(f"wrote {diag_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
