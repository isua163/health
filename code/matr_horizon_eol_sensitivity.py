#!/usr/bin/env python3
"""matr_horizon_eol restriction-horizon and end-of-life threshold sensitivity analyses.

Version 1.4 records structurally unattainable EOL-by-batch policy designs
instead of forcing a different replacement target.  The frozen 40% policy is
run only where that target is mathematically attainable after endpoint
perturbation and the 50-cycle run-in.
"""
from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

H_MULT = (0.6, 0.8, 1.0)
EOL_FRAC = (0.78, 0.80, 0.82)
HORIZON_MODES = ("adaptive_batch_median", "fixed_original_batch_median")
ESTIMATORS = ("naive", "oracle_product_limit", "oracle_HT_RMST", "crossfit_TV_IPCW")


def first_crossing(q: np.ndarray, threshold: float) -> float:
    idx = np.flatnonzero(np.isfinite(q) & (q < float(threshold)))
    return float(idx[0] + 1) if idx.size else float("nan")


def reconstruct_life(row: dict[str, str], cell: Any, threshold: float) -> tuple[int, str, float, float]:
    q = np.asarray(cell.arrays.get("QDischarge", []), float)
    finite = q[np.isfinite(q)]
    last = float(finite[-1]) if finite.size else float("nan")
    crossing = first_crossing(q, threshold)
    if np.isfinite(last) and last < threshold and np.isfinite(crossing):
        return int(round(crossing)), "first_crossing", last, crossing
    return int(round(float(row["official_cycle_record_count"]))) + 1, "cycle_record_count_plus_1", last, crossing


def build_threshold_ir_path(core: Any, ir: Any, lifetime: int,
                            baseline_cycles: int, smooth_window: int) -> np.ndarray:
    """Build endpoint-aligned IR without using records after the perturbed EOL.

    A stricter EOL threshold can move an endpoint inside the policy run-in.
    Such a unit has no replacement opportunity.  If it also has no valid IR
    before that endpoint, a finite zero path is used only for cohort
    bookkeeping; it is excluded from policy-scale, threshold and calibration
    calculations.  A unit extending beyond the run-in still raises on missing
    baseline IR.
    """
    raw = np.asarray(ir, float).ravel()
    T = int(lifetime)
    if T < 1:
        raise ValueError("reconstructed lifetime must be positive")
    truncated = raw[:min(T, len(raw))]
    try:
        return core.build_ir_signal(truncated, T, baseline_cycles, smooth_window)
    except ValueError as exc:
        if (T <= int(baseline_cycles) + 1
                and "no positive finite IR value in the baseline window" in str(exc)):
            return np.zeros(T, dtype=float)
        raise


def standardize_policy_paths_allow_prerunin_endpoints(
        core: Any, paths: list[np.ndarray], policy_start: int
) -> tuple[list[np.ndarray], float, float, int]:
    """Standardize on eligible records while retaining early endpoints."""
    start = max(0, int(policy_start))
    arr = [np.asarray(x, float).ravel() for x in paths]
    if any(len(x) < 1 or not np.all(np.isfinite(x)) for x in arr):
        raise ValueError("all endpoint-aligned paths must be non-empty and finite")
    reference = [x[start:-1] for x in arr if len(x) > start + 1]
    n_ineligible = len(arr) - len(reference)
    if not reference:
        raise ValueError("no unit extends beyond the policy run-in under this EOL threshold")
    scale = core.unit_equal_iqr_scale(reference)
    z = [x / scale for x in arr]
    eligible_standardized = [x[start:-1] for x in z if len(x) > start + 1]
    tau = core.unit_equal_quantile(eligible_standardized, 0.70)
    return z, float(scale), float(tau), int(n_ineligible)


def assess_batch_attainability(paths_z: list[np.ndarray], idx: np.ndarray,
                                policy_start: int, target: float) -> dict[str, Any]:
    """Return the maximum possible batch replacement fraction.

    Units ending at or before ``policy_start + 1`` can never be replaced under
    the frozen policy.  Therefore the eligible-unit fraction is an upper bound
    on the batch-level replacement fraction for every finite baseline hazard.
    """
    n_units = int(len(idx))
    n_eligible = int(sum(len(paths_z[int(i)]) > int(policy_start) + 1 for i in idx))
    eligible_fraction = float(n_eligible / n_units) if n_units else 0.0
    attainable = bool(eligible_fraction + 1e-12 >= float(target))
    return {
        "n_units": n_units,
        "n_policy_eligible": n_eligible,
        "n_pre_runin_or_terminal_at_runin": n_units - n_eligible,
        "eligible_fraction": eligible_fraction,
        "theoretical_max_replacement_fraction": eligible_fraction,
        "nominal_target_replacement_fraction": float(target),
        "nominal_target_attainable": attainable,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--matr", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--R", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--nominal-capacity-Ah", type=float, default=1.1)
    ap.add_argument("--target-censor", type=float, default=0.40)
    ap.add_argument("--baseline-cycles", type=int, default=50)
    ap.add_argument("--smooth-window", type=int, default=5)
    ap.add_argument("--ridge-slope", type=float, default=16.0)
    ap.add_argument("--checkpoint-every", type=int, default=5)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    root = a.root.resolve()
    matr = a.matr.resolve()
    out = (a.out_dir or root / "results" / "matr_horizon_eol").resolve()
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root / "code" / "src"))
    import sensitivity_common as c

    data, audit, finalizer, core, surv = c.load_primary_data(
        root, matr, a.baseline_cycles, a.smooth_window
    )
    idx_by_batch = {
        b: np.flatnonzero(data.batches == b) for b in c.PRIMARY_BATCHES
    }

    original_H = {
        b: float(np.median(data.life[idx])) for b, idx in idx_by_batch.items()
    }
    horizon_audit: list[dict[str, Any]] = []
    for b, idx in idx_by_batch.items():
        life = data.life[idx]
        horizon_audit.append({
            "batch": b,
            "n_units": len(idx),
            "H_definition": "batch median complete endpoint time",
            "H": original_H[b],
            "q25_T": float(np.percentile(life, 25)),
            "median_T": float(np.median(life)),
            "q75_T": float(np.percentile(life, 75)),
            "q90_T": float(np.percentile(life, 90)),
            "max_T": float(np.max(life)),
            "H_over_q90": original_H[b] / float(np.percentile(life, 90)),
            "H_over_max": original_H[b] / float(np.max(life)),
        })
    c.write_csv(out / "matr_horizon_eol_horizon_definition_audit.csv", horizon_audit)

    # Part 1: fixed-cohort horizon sensitivity under the frozen primary policy.
    ir_z, scale, tau = core.standardize_policy_paths(
        data.ir_raw, policy_start=a.baseline_cycles
    )
    lambda0 = {
        b: core.calibrate_lambda0(
            [ir_z[i] for i in idx], 1.0, tau, a.target_censor,
            policy_start=a.baseline_cycles
        )
        for b, idx in idx_by_batch.items()
    }
    horizon_gap: list[dict[str, Any]] = []
    for b, idx in idx_by_batch.items():
        for m in H_MULT:
            H = m * original_H[b]
            truth = float(np.mean(np.minimum(data.life[idx], H)))
            crude, _ = core.exact_crude_rmst(
                [ir_z[i] for i in idx], 1.0, tau, lambda0[b], H,
                policy_start=a.baseline_cycles
            )
            horizon_gap.append({
                "batch": b,
                "horizon_multiplier": m,
                "H": H,
                "net_rmst": truth,
                "exact_crude_functional": crude,
                "estimand_gap_cycles": crude - truth,
                "estimand_gap_pct_net": 100 * (crude - truth) / truth,
            })
    c.write_csv(out / "matr_horizon_eol_horizon_exact_estimand_gaps.csv", horizon_gap)

    hrp = out / "matr_horizon_eol_horizon_replicates.csv"
    if a.resume:
        hrows = c.read_csv(hrp)
        counts: dict[int, int] = {}
        for r in hrows:
            key = int(float(r["replicate"]))
            counts[key] = counts.get(key, 0) + 1
        expected = len(c.PRIMARY_BATCHES) * len(H_MULT) * len(ESTIMATORS)
        hcomplete = {r for r, n in counts.items() if n == expected}
        hrows = [r for r in hrows if int(float(r["replicate"])) in hcomplete]
    else:
        hrows = []
        hcomplete = set()

    for rep in range(a.R):
        if rep in hcomplete:
            continue
        rng = np.random.default_rng(a.seed + 10000019 * rep)
        uniforms = [rng.random(max(len(x) - 1, 0)) for x in ir_z]
        for b, idx in idx_by_batch.items():
            tt, ee, obs, oracle = core.overlay_from_uniforms(
                [ir_z[i] for i in idx], 1.0, tau, lambda0[b],
                [uniforms[i] for i in idx], policy_start=a.baseline_cycles
            )
            fitted, fits = core.fit_crossfit_cumhaz(
                obs, ee, data.folds[idx], policy_start=a.baseline_cycles,
                ridge_slope=a.ridge_slope
            )
            nt, ns, _ = surv.km(tt, ee)
            ot, os = core.weighted_product_limit(tt, ee, oracle)
            ft, fs = core.weighted_product_limit(tt, ee, fitted)
            for m in H_MULT:
                H = m * original_H[b]
                truth = float(np.mean(np.minimum(data.life[idx], H)))
                vals = [
                    ("naive", float(surv.rmrl_from_survival(nt, ns, 0, H))),
                    ("oracle_product_limit", float(surv.rmrl_from_survival(ot, os, 0, H))),
                    ("oracle_HT_RMST", float(core.ht_ipcw_rmst(tt, ee, oracle, H))),
                    ("crossfit_TV_IPCW", float(surv.rmrl_from_survival(ft, fs, 0, H))),
                ]
                for est, val in vals:
                    hrows.append({
                        "replicate": rep,
                        "batch": b,
                        "horizon_multiplier": m,
                        "H": H,
                        "truth": truth,
                        "realized_replacement": float(np.mean(ee == 0)),
                        "estimator": est,
                        "estimate": val,
                        "bias_cycles": val - truth,
                        "bias_pct_net": 100 * (val - truth) / truth,
                    })
        if (rep + 1) % a.checkpoint_every == 0 or rep == a.R - 1:
            c.write_csv(hrp, hrows)
        print(f"horizon replicate {rep + 1}/{a.R} completed", flush=True)
    hsummary = c.summarize(
        hrows, ["batch", "horizon_multiplier", "estimator"], "bias_pct_net"
    )
    c.write_csv(out / "matr_horizon_eol_horizon_summary.csv", hsummary)

    # Part 2: endpoint sensitivity on the same frozen 124 IDs.
    thresholds = [round(a.nominal_capacity_Ah * f, 3) for f in EOL_FRAC]
    variants: dict[float, dict[str, Any]] = {}
    endpoint_rows: list[dict[str, Any]] = []
    estimability_rows: list[dict[str, Any]] = []
    calibration_failures: list[tuple[float, str, str]] = []

    for frac, thr in zip(EOL_FRAC, thresholds):
        life: list[float] = []
        paths: list[np.ndarray] = []
        for uid, row in zip(data.ids, data.endpoint_rows):
            T, source, last, cross = reconstruct_life(
                row, data.cells_by_id[uid], thr
            )
            life.append(float(T))
            paths.append(build_threshold_ir_path(
                core, data.cells_by_id[uid].arrays.get("IR", []), T,
                a.baseline_cycles, a.smooth_window
            ))
            endpoint_rows.append({
                "unit_id": uid,
                "batch": row["batch_label"],
                "nominal_capacity_Ah": a.nominal_capacity_Ah,
                "EOL_fraction": frac,
                "EOL_threshold_Ah": thr,
                "reconstructed_lifetime": T,
                "endpoint_source": source,
                "q_last_finite": last,
                "first_crossing": cross,
                "frozen_lifetime_80pct": float(row["author_reconstructed_lifetime"]),
                "difference_from_frozen": T - float(row["author_reconstructed_lifetime"]),
                "policy_eligible_records": max(T - a.baseline_cycles - 1, 0),
                "endpoint_before_or_at_policy_runin": bool(T <= a.baseline_cycles + 1),
            })

        z, sc, ta, n_ineligible = standardize_policy_paths_allow_prerunin_endpoints(
            core, paths, policy_start=a.baseline_cycles
        )
        life_arr = np.asarray(life, float)
        v: dict[str, Any] = {
            "life": life_arr,
            "paths": paths,
            "z": z,
            "scale": sc,
            "tau": ta,
            "n_pre_runin_endpoints": n_ineligible,
            "H_adaptive": {
                b: float(np.median(life_arr[idx])) for b, idx in idx_by_batch.items()
            },
            "lambda0": {},
            "estimable": {},
        }

        for b, idx in idx_by_batch.items():
            audit_row = assess_batch_attainability(
                z, idx, a.baseline_cycles, a.target_censor
            )
            status = "not_estimable_policy_incompatible"
            lam: float | None = None
            if audit_row["nominal_target_attainable"]:
                try:
                    lam = float(core.calibrate_lambda0(
                        [z[i] for i in idx], 1.0, ta, a.target_censor,
                        policy_start=a.baseline_cycles
                    ))
                    status = "estimable"
                    v["lambda0"][b] = lam
                    v["estimable"][b] = True
                except RuntimeError as exc:
                    status = "calibration_failed_despite_eligibility"
                    v["estimable"][b] = False
                    calibration_failures.append((frac, b, str(exc)))
            else:
                v["estimable"][b] = False

            estimability_rows.append({
                "EOL_fraction": frac,
                "EOL_threshold_Ah": thr,
                "batch": b,
                **audit_row,
                "analysis_status": status,
                "lambda0": "" if lam is None else lam,
                "min_lifetime": float(np.min(life_arr[idx])),
                "median_lifetime": float(np.median(life_arr[idx])),
                "max_lifetime": float(np.max(life_arr[idx])),
                "signal_scale": sc,
                "tau": ta,
            })
        variants[frac] = v

    c.write_csv(out / "matr_horizon_eol_EOL_endpoint_reconstruction.csv", endpoint_rows)
    c.write_csv(out / "matr_horizon_eol_EOL_policy_estimability.csv", estimability_rows)

    estimable_pairs = [
        (frac, b) for frac in EOL_FRAC for b in c.PRIMARY_BATCHES
        if variants[frac]["estimable"].get(b, False)
    ]
    nonestimable_pairs = [
        (frac, b) for frac in EOL_FRAC for b in c.PRIMARY_BATCHES
        if not variants[frac]["estimable"].get(b, False)
    ]

    eol_gap: list[dict[str, Any]] = []
    for frac, v in variants.items():
        for b, idx in idx_by_batch.items():
            for mode, H in [
                ("adaptive_batch_median", v["H_adaptive"][b]),
                ("fixed_original_batch_median", original_H[b]),
            ]:
                truth = float(np.mean(np.minimum(v["life"][idx], H)))
                base = {
                    "EOL_fraction": frac,
                    "EOL_threshold_Ah": round(a.nominal_capacity_Ah * frac, 3),
                    "batch": b,
                    "horizon_mode": mode,
                    "H": H,
                    "net_rmst": truth,
                    "signal_scale": v["scale"],
                    "tau": v["tau"],
                    "n_pre_runin_endpoints_all_batches": v["n_pre_runin_endpoints"],
                    "n_pre_runin_endpoints_batch": int(sum(
                        len(v["z"][i]) <= a.baseline_cycles + 1 for i in idx
                    )),
                }
                if v["estimable"].get(b, False):
                    lam = v["lambda0"][b]
                    crude, _ = core.exact_crude_rmst(
                        [v["z"][i] for i in idx], 1.0, v["tau"], lam, H,
                        policy_start=a.baseline_cycles
                    )
                    eol_gap.append({
                        **base,
                        "analysis_status": "estimable",
                        "exact_crude_functional": crude,
                        "estimand_gap_cycles": crude - truth,
                        "estimand_gap_pct_net": 100 * (crude - truth) / truth,
                        "lambda0": lam,
                    })
                else:
                    eol_gap.append({
                        **base,
                        "analysis_status": "not_estimable_policy_incompatible",
                        "exact_crude_functional": "",
                        "estimand_gap_cycles": "",
                        "estimand_gap_pct_net": "",
                        "lambda0": "",
                    })
    c.write_csv(out / "matr_horizon_eol_EOL_exact_estimand_gaps.csv", eol_gap)

    erp = out / "matr_horizon_eol_EOL_replicates.csv"
    expected_eol_rows_per_rep = len(estimable_pairs) * len(HORIZON_MODES) * len(ESTIMATORS)
    if a.resume:
        erows = c.read_csv(erp)
        counts: dict[int, int] = {}
        for r in erows:
            key = int(float(r["replicate"]))
            counts[key] = counts.get(key, 0) + 1
        ecomplete = {
            r for r, n in counts.items() if n == expected_eol_rows_per_rep
        }
        erows = [
            r for r in erows if int(float(r["replicate"])) in ecomplete
        ]
    else:
        erows = []
        ecomplete = set()

    maxlen = [
        max(len(variants[f]["z"][i]) for f in EOL_FRAC)
        for i in range(len(data.ids))
    ]
    for rep in range(a.R):
        if rep in ecomplete:
            continue
        rng = np.random.default_rng(a.seed + 500000003 + 10000019 * rep)
        uniforms = [rng.random(max(n - 1, 0)) for n in maxlen]
        for frac, b in estimable_pairs:
            v = variants[frac]
            idx = idx_by_batch[b]
            tt, ee, obs, oracle = core.overlay_from_uniforms(
                [v["z"][i] for i in idx], 1.0, v["tau"], v["lambda0"][b],
                [uniforms[i][:max(len(v["z"][i]) - 1, 0)] for i in idx],
                policy_start=a.baseline_cycles
            )
            fitted, fits = core.fit_crossfit_cumhaz(
                obs, ee, data.folds[idx], policy_start=a.baseline_cycles,
                ridge_slope=a.ridge_slope
            )
            nt, ns, _ = surv.km(tt, ee)
            ot, os = core.weighted_product_limit(tt, ee, oracle)
            ft, fs = core.weighted_product_limit(tt, ee, fitted)
            for mode, H in [
                ("adaptive_batch_median", v["H_adaptive"][b]),
                ("fixed_original_batch_median", original_H[b]),
            ]:
                truth = float(np.mean(np.minimum(v["life"][idx], H)))
                vals = [
                    ("naive", float(surv.rmrl_from_survival(nt, ns, 0, H))),
                    ("oracle_product_limit", float(surv.rmrl_from_survival(ot, os, 0, H))),
                    ("oracle_HT_RMST", float(core.ht_ipcw_rmst(tt, ee, oracle, H))),
                    ("crossfit_TV_IPCW", float(surv.rmrl_from_survival(ft, fs, 0, H))),
                ]
                for est, val in vals:
                    erows.append({
                        "replicate": rep,
                        "EOL_fraction": frac,
                        "EOL_threshold_Ah": round(a.nominal_capacity_Ah * frac, 3),
                        "batch": b,
                        "horizon_mode": mode,
                        "H": H,
                        "truth": truth,
                        "realized_replacement": float(np.mean(ee == 0)),
                        "estimator": est,
                        "estimate": val,
                        "bias_cycles": val - truth,
                        "bias_pct_net": 100 * (val - truth) / truth,
                    })
        if (rep + 1) % a.checkpoint_every == 0 or rep == a.R - 1:
            c.write_csv(erp, erows)
        print(f"EOL replicate {rep + 1}/{a.R} completed", flush=True)

    esummary = c.summarize(
        erows, ["EOL_fraction", "batch", "horizon_mode", "estimator"],
        "bias_pct_net"
    )
    c.write_csv(out / "matr_horizon_eol_EOL_summary.csv", esummary)

    match80 = all(
        abs(float(r["difference_from_frozen"])) < 1e-12
        for r in endpoint_rows if abs(float(r["EOL_fraction"]) - 0.8) < 1e-9
    )
    all_80_estimable = all(
        variants[0.80]["estimable"].get(b, False) for b in c.PRIMARY_BATCHES
    )
    explicit_nonestimable = all(
        any(abs(float(r["EOL_fraction"]) - f) < 1e-9
            and r["batch"] == b
            and r["analysis_status"] == "not_estimable_policy_incompatible"
            for r in estimability_rows)
        for f, b in nonestimable_pairs
    )

    checks = {
        "original_H_equals_batch_median": all(
            abs(original_H[b] - float(np.median(data.life[idx_by_batch[b]]))) < 1e-12
            for b in c.PRIMARY_BATCHES
        ),
        "EOL_80_reconstructs_frozen_endpoints": match80,
        "all_80pct_batches_attain_nominal_target": all_80_estimable,
        "horizon_rows_complete": len(hrows) == a.R * len(c.PRIMARY_BATCHES) * len(H_MULT) * len(ESTIMATORS),
        "EOL_rows_complete_for_estimable_designs": len(erows) == a.R * expected_eol_rows_per_rep,
        "all_estimates_finite": all(
            np.isfinite(float(r["estimate"])) for r in hrows + erows
        ),
        "same_124_units_all_EOL": len(endpoint_rows) == 124 * len(EOL_FRAC),
        "pre_runin_endpoints_retained_without_policy_eligibility": all(
            (int(float(r["policy_eligible_records"])) == 0)
            == (str(r["endpoint_before_or_at_policy_runin"]).lower() == "true")
            for r in endpoint_rows
        ),
        "unattainable_designs_explicitly_recorded": explicit_nonestimable,
        "no_calibration_failure_among_attainable_designs": len(calibration_failures) == 0,
        "at_least_one_structural_nonestimability_boundary": len(nonestimable_pairs) >= 1,
    }
    report = {
        "analysis": "matr_horizon_eol_horizon_EOL_sensitivity",
        "status": "PASS" if all(checks.values()) else "REVIEW_REQUIRED",
        "python": platform.python_version(),
        "design": {
            "R": a.R,
            "seed": a.seed,
            "horizon_multipliers": H_MULT,
            "nominal_capacity_Ah": a.nominal_capacity_Ah,
            "EOL_fractions": EOL_FRAC,
            "EOL_threshold_Ah": thresholds,
            "target_replacement": a.target_censor,
            "ridge_slope": a.ridge_slope,
            "same_frozen_unit_IDs": True,
            "EOL_horizon_modes": HORIZON_MODES,
            "unattainable_policy_rule": "record as structurally not estimable; do not recalibrate to a different target",
        },
        "checks": checks,
        "original_H": original_H,
        "original_policy": {"scale": scale, "tau": tau, "lambda0": lambda0},
        "EOL_pre_runin_endpoint_counts": {
            str(frac): int(variants[frac]["n_pre_runin_endpoints"])
            for frac in EOL_FRAC
        },
        "estimable_EOL_batch_pairs": [
            {"EOL_fraction": f, "batch": b} for f, b in estimable_pairs
        ],
        "nonestimable_EOL_batch_pairs": [
            {"EOL_fraction": f, "batch": b,
             "reason": "nominal 40% replacement target exceeds the eligible-unit fraction after endpoint perturbation"}
            for f, b in nonestimable_pairs
        ],
        "calibration_failures": [
            {"EOL_fraction": f, "batch": b, "message": msg}
            for f, b, msg in calibration_failures
        ],
        "EOL_policy_handling": (
            "Units ending at or before the 50-cycle run-in remain in the net-RMST cohort. "
            "If a batch cannot attain the frozen 40% replacement target, its policy-overlay EOL sensitivity is marked not estimable rather than being forced to a different policy."
        ),
    }
    c.json_dump(out / "matr_horizon_eol_horizon_EOL_report.json", report)

    print("matr_horizon_eol HORIZON AND EOL SENSITIVITY COMPLETED")
    print(f"status={report['status']}")
    print(f"80pct_endpoint_exact_match={match80}")
    print("pre_runin_endpoint_counts=" + str({
        str(frac): int(variants[frac]["n_pre_runin_endpoints"])
        for frac in EOL_FRAC
    }))
    print("nonestimable_EOL_batch_pairs=" + str(nonestimable_pairs))
    for r in horizon_audit:
        print(
            f"{r['batch']:12s} H={r['H']:.1f}=median(T); "
            f"H/q90={r['H_over_q90']:.3f}; H/max={r['H_over_max']:.3f}"
        )
    print(f"out_dir={out}")
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
