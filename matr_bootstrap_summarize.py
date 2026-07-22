#!/usr/bin/env python3
"""Summarize the MATR joint cell-and-policy redesign replicates.

Each outer resample defines its own finite-cohort endpoint benchmark and its own
recalibrated synthetic policy. Bias statistics therefore use the
``truth_net_rmst`` stored for that outer replicate and batch. Reported
percentiles are empirical redesign ranges, not fixed-policy confidence intervals.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ARMS = ("naive", "oracle_tv_ipcw", "crossfit_tv_ipcw")
STAT_MAP = {
    "naive": "naive_signed_gap_pct",
    "oracle_tv_ipcw": "oracle_tv_ipcw_signed_gap_pct",
    "crossfit_tv_ipcw": "crossfit_tv_ipcw_signed_gap_pct",
}


def corrected_summary(values: np.ndarray, inner_var: np.ndarray, r_inner: int):
    sd = float(np.std(values, ddof=1))
    mean_inner = float(np.mean(inner_var))
    residual = mean_inner / max(r_inner, 1)
    variance = max(sd * sd - residual, 0.0)
    se = math.sqrt(variance)
    return sd, mean_inner, residual, variance, se


def prepare_bootstrap_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate and add truth-relative signed and absolute errors."""
    required = {
        "outer_b", "batch_label", "beta", "inner_r", "arm", "estimate",
        "truth_net_rmst", "fit_success",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"missing columns: {missing}")

    d = raw[(raw.beta.astype(float) == 1.0) & raw.arm.isin(ARMS)].copy()
    d = d[d.fit_success.astype(bool)].copy()
    if d.empty:
        raise ValueError("no successful beta=1 bootstrap rows")

    for col in ("estimate", "truth_net_rmst"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    if not np.isfinite(d[["estimate", "truth_net_rmst"]].to_numpy(float)).all():
        raise ValueError("estimate and truth_net_rmst must be finite")
    if (d.truth_net_rmst <= 0).any():
        raise ValueError("truth_net_rmst must be positive")

    # The finite-cohort truth must be identical across arms and inner overlays
    # within an outer resample/batch, but may differ across outer resamples.
    truth_nunique = d.groupby(["outer_b", "batch_label"])["truth_net_rmst"].nunique()
    if int(truth_nunique.max()) != 1:
        bad = truth_nunique[truth_nunique != 1]
        raise ValueError(f"inconsistent resample truth within outer/batch: {bad.index.tolist()[:5]}")

    d["signed_gap_pct"] = 100.0 * (d.estimate - d.truth_net_rmst) / d.truth_net_rmst
    d["absolute_gap_pct"] = d.signed_gap_pct.abs()
    return d


def _row(
    *, batch: str, statistic: str, vals: np.ndarray, inner_var: np.ndarray,
    r_inner: int, b_outer: int, point: float,
) -> dict[str, object]:
    sd, mean_inner, residual, variance, se = corrected_summary(vals, inner_var, r_inner)
    lo = float(np.percentile(vals, 2.5))
    hi = float(np.percentile(vals, 97.5))
    return {
        "prefix_B": b_outer,
        "R_inner": r_inner,
        "batch_label": batch,
        "beta": 1.0,
        "statistic": statistic,
        "outer_mean": float(vals.mean()),
        "raw_outer_sd": sd,
        "mean_inner_variance": mean_inner,
        "residual_inner_mc_variance_of_mean": residual,
        "descriptive_percentile_lo": lo,
        "descriptive_percentile_hi": hi,
        "full_cohort_reference_point": point,
        "redesign_center_minus_reference": float(vals.mean() - point),
        "n_outer": int(len(vals)),
    }


def summarize(raw: pd.DataFrame, point_table: pd.DataFrame) -> pd.DataFrame:
    d = prepare_bootstrap_data(raw)
    r_inner = int(d.inner_r.nunique())
    b_outer = int(d.outer_b.nunique())
    batches = list(dict.fromkeys(d.batch_label.astype(str)))

    point = point_table[
        (point_table.beta.astype(float) == 1.0) & point_table.arm.isin(ARMS)
    ].set_index(["scope", "arm"])["mean_signed_gap_pct"].astype(float).to_dict()

    grouped = d.groupby(["outer_b", "batch_label", "arm"])["signed_gap_pct"].agg(
        ["mean", "var", "count"]
    ).reset_index()
    rows: list[dict[str, object]] = []

    for batch in batches:
        gb = grouped[grouped.batch_label == batch]
        for arm in ARMS:
            g = gb[gb.arm == arm].sort_values("outer_b")
            vals = g["mean"].to_numpy(float)
            inner_var = g["var"].fillna(0.0).to_numpy(float)
            rows.append(_row(
                batch=batch,
                statistic=STAT_MAP[arm],
                vals=vals,
                inner_var=inner_var,
                r_inner=r_inner,
                b_outer=b_outer,
                point=float(point[(batch, arm)]),
            ))

        piv = d[d.batch_label == batch].pivot_table(
            index=["outer_b", "inner_r"], columns="arm", values="signed_gap_pct"
        ).reset_index()
        required_arms = {"naive", "crossfit_tv_ipcw"}
        if not required_arms.issubset(piv.columns):
            raise ValueError(f"paired arms missing for {batch}")
        piv["signed_error_diff_pp"] = piv.crossfit_tv_ipcw - piv.naive
        piv["absolute_error_diff_pp"] = piv.crossfit_tv_ipcw.abs() - piv.naive.abs()

        paired_points = {
            "crossfit_minus_naive_signed_error_pp": float(
                point[(batch, "crossfit_tv_ipcw")] - point[(batch, "naive")]
            ),
            "crossfit_minus_naive_absolute_error_pp": float(
                abs(point[(batch, "crossfit_tv_ipcw")]) - abs(point[(batch, "naive")])
            ),
        }
        for statistic, column in (
            ("crossfit_minus_naive_signed_error_pp", "signed_error_diff_pp"),
            ("crossfit_minus_naive_absolute_error_pp", "absolute_error_diff_pp"),
        ):
            outer = piv.groupby("outer_b")[column].agg(["mean", "var"]).reset_index().sort_values("outer_b")
            rows.append(_row(
                batch=batch,
                statistic=statistic,
                vals=outer["mean"].to_numpy(float),
                inner_var=outer["var"].fillna(0.0).to_numpy(float),
                r_inner=r_inner,
                b_outer=b_outer,
                point=paired_points[statistic],
            ))

    return pd.DataFrame(rows).sort_values(["batch_label", "statistic"]).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    out = (args.out_dir or root / "results" / "matr_bootstrap").resolve()
    out.mkdir(parents=True, exist_ok=True)
    inp = (args.input or out / "replicate_results.csv").resolve()

    raw = pd.read_csv(inp)
    point_table = pd.read_csv(root / "results" / "matr_primary" / "estimator_summary.csv")
    intervals = summarize(raw, point_table)
    intervals.to_csv(out / "redesign_summary.csv", index=False)

    d = prepare_bootstrap_data(raw)
    b_outer = int(d.outer_b.nunique())
    r_inner = int(d.inner_r.nunique())
    support = d.groupby(["batch_label", "beta", "arm"]).agg(
        n_rows=("estimate", "size"),
        fit_failure_fraction=("fit_success", lambda x: 1.0 - float(np.mean(x))),
        p95_max_weight=("max_weight", lambda x: float(np.nanpercentile(x, 95))),
        p99_max_weight=("max_weight", lambda x: float(np.nanpercentile(x, 99))),
        max_weight=("max_weight", "max"),
        p10_min_ess=("min_ess_over_risk", lambda x: float(np.nanpercentile(x, 10))),
        p01_min_ess=("min_ess_over_risk", lambda x: float(np.nanpercentile(x, 1))),
        min_ess=("min_ess_over_risk", "min"),
        exp_clipping_fraction=("exp_clipping", "mean"),
        solver_fallback_fraction=("solver_fallback", "mean"),
    ).reset_index()
    support.insert(0, "prefix_B", b_outer)
    support.to_csv(out / "support_summary.csv", index=False)

    truth_by_outer = d.groupby(["outer_b", "batch_label"])["truth_net_rmst"].first().reset_index()
    truth_by_outer.to_csv(out / "resample_truth_audit.csv", index=False)

    report = {
        "analysis": "MATR joint cell-and-policy redesign with resample-specific endpoint truth",
        "status": "PASS",
        "B": b_outer,
        "R_inner": r_inner,
        "n_input_rows": int(len(raw)),
        "n_interval_rows": int(len(intervals)),
        "truth_source": "truth_net_rmst stored for each outer_b and batch_label",
        "policy_resampling_mode": "adaptive_redesign",
        "range_interpretation": "descriptive cell-and-policy redesign distribution; not a confidence interval and not evaluated by a zero threshold",
        "paired_statistics": [
            "crossfit_minus_naive_signed_error_pp",
            "crossfit_minus_naive_absolute_error_pp",
        ],
    }
    (out / "summary_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("MATR BOOTSTRAP SUMMARY COMPLETED")
    print("status=PASS")
    print(f"B={b_outer}; R_inner={r_inner}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
