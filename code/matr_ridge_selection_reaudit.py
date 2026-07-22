#!/usr/bin/env python3
"""Re-audit MATR censoring-ridge candidates using beta=0 overlays only.

The script records the full evidence chain requested for the censoring-model
ridge: negative-control bias, fit failures, held-out weights, ESS/risk-set
ratios, clipping and fold-seed stability.  It never evaluates beta=1.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matr_primary_analysis as primary
from src.matr_ipcw import (
    calibrate_lambda0,
    expected_censor_fraction,
    make_stratified_folds,
    standardize_policy_paths,
    weight_diagnostics,
)


def _normalise_weight_diagnostic(diag: dict[str, object]) -> dict[str, float]:
    """Map the core diagnostic contract to stable ridge-audit column names.

    ``weight_diagnostics`` exposes ``ess_over_n_at_risk`` and
    ``n_exp_clipped``.  Older ridge-audit code looked for different names,
    silently producing missing ESS and clipping values.  The aliases below
    also accept legacy outputs so existing external callers remain usable.
    """
    return {
        "max_weight": float(diag.get("max_weight", np.nan)),
        "ess_over_risk_set": float(
            diag.get("ess_over_n_at_risk", diag.get("ess_over_risk_set", np.nan))
        ),
        "exp_clipping_count": float(
            diag.get("n_exp_clipped", diag.get("exp_clipping_count", np.nan))
        ),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--matr", required=True, type=Path)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--ridges", type=float, nargs="+", default=[4.0, 16.0, 64.0])
    p.add_argument("--fold-seeds", type=int, nargs="+", default=[20261001, 20261003, 20261005])
    p.add_argument("--overlay-seeds", type=int, nargs="+", default=[20261011, 20261021, 20261031])
    p.add_argument("--R", type=int, default=200)
    p.add_argument("--target-censor", type=float, default=0.40)
    p.add_argument("--baseline-cycles", type=int, default=50)
    p.add_argument("--smooth-window", type=int, default=5)
    p.add_argument("--exp-clip", type=float, default=30.0)
    p.add_argument("--selected-ridge", type=float, default=16.0)
    p.add_argument("--quick-check", action="store_true",
                   help="Run a structural smoke test without overwriting the full ridge-selection lock.")
    a = p.parse_args()

    root = a.root.resolve()
    out = (a.out_dir or root / "results" / "matr_ridge_audit").resolve()
    out.mkdir(parents=True, exist_ok=True)
    unit_ids, batches, lifetimes, raw_paths = primary._load_primary_cohort(
        root, a.matr.resolve(), a.baseline_cycles, a.smooth_window
    )
    paths, scale, tau = standardize_policy_paths(raw_paths, policy_start=a.baseline_cycles)
    idx_map = primary._batch_indices(batches)
    scopes = primary._batch_scopes(batches, lifetimes)

    beta = 0.0
    lambda0 = {}
    for batch in primary.PRIMARY_BATCHES:
        bp = primary._subset(paths, idx_map[batch])
        lambda0[batch] = calibrate_lambda0(
            bp, beta, tau, a.target_censor, policy_start=a.baseline_cycles
        )
        attained = expected_censor_fraction(
            bp, beta, tau, lambda0[batch], policy_start=a.baseline_cycles
        )
        if abs(attained - a.target_censor) > 1e-8:
            raise RuntimeError(f"calibration failed for {batch}")

    replicate_rows = []
    fit_rows = []
    for ridge in a.ridges:
        for fold_seed in a.fold_seeds:
            folds = make_stratified_folds(batches, n_folds=5, seed=fold_seed)
            for overlay_seed in a.overlay_seeds:
                rng = np.random.default_rng(overlay_seed)
                for rep in range(a.R):
                    uniforms = [rng.random(max(len(path) - 1, 0)) for path in paths]
                    times, events, observed, _ = primary._overlay_by_batch(
                        paths, uniforms, batches, beta, tau, lambda0, a.baseline_cycles
                    )
                    try:
                        cross, fits, _, _ = primary._fit_by_batch(
                            observed, events, folds, batches, a.baseline_cycles,
                            ridge_slope=float(ridge),
                        )
                        fit_success = True
                        error = ""
                    except Exception as exc:
                        cross = []
                        fits = []
                        fit_success = False
                        error = f"{type(exc).__name__}: {exc}"

                    for batch in primary.PRIMARY_BATCHES:
                        idx = idx_map[batch]
                        scope = next(s for s in scopes if s["scope"] == batch)
                        H = float(scope["H"])
                        truth = float(scope["truth"])
                        if fit_success:
                            estimate = primary._estimate_weighted(
                                times[idx], events[idx], primary._subset(cross, idx), H
                            )
                            diag = _normalise_weight_diagnostic(weight_diagnostics(
                                times[idx], primary._subset(cross, idx), [H], a.exp_clip
                            )[0])
                            signed = 100.0 * (estimate - truth) / truth
                        else:
                            estimate = np.nan
                            signed = np.nan
                            diag = _normalise_weight_diagnostic({})
                        replicate_rows.append({
                            "ridge_slope": ridge,
                            "fold_seed": fold_seed,
                            "overlay_seed": overlay_seed,
                            "replicate": rep,
                            "batch": batch,
                            "fit_success": fit_success,
                            "estimate": estimate,
                            "truth_net_rmst": truth,
                            "signed_gap_pct": signed,
                            "max_weight": diag.get("max_weight", np.nan),
                            "ess_over_risk_set": diag.get("ess_over_risk_set", np.nan),
                            "exp_clipping_count": diag.get("exp_clipping_count", np.nan),
                            "error": error,
                        })
                    for item in fits:
                        fit = item["fit"]
                        fit_rows.append({
                            "ridge_slope": ridge,
                            "fold_seed": fold_seed,
                            "overlay_seed": overlay_seed,
                            "replicate": rep,
                            "batch": item["batch"],
                            "fold": item["fold"],
                            "success": bool(fit.success),
                            "slope": float(fit.slope),
                            "max_abs_slope": abs(float(fit.slope)),
                            "grad_norm": float(fit.grad_norm),
                            "n_iter": int(fit.n_iter),
                            "method": str(getattr(fit, "method", "newton")),
                            "message": str(fit.message),
                        })
                    if (rep + 1) % max(1, min(25, a.R)) == 0:
                        print(
                            f"ridge={ridge:g} fold_seed={fold_seed} overlay_seed={overlay_seed} "
                            f"rep={rep + 1}/{a.R}", flush=True
                        )

    reps = pd.DataFrame(replicate_rows)
    fits = pd.DataFrame(fit_rows)
    reps.to_csv(out / "replicates.csv", index=False)
    fits.to_csv(out / "fit_diagnostics.csv", index=False)

    summary = reps.groupby(["ridge_slope", "batch"], dropna=False).agg(
        n_rows=("replicate", "size"),
        fit_failure_fraction=("fit_success", lambda x: 1.0 - float(np.mean(x))),
        mean_beta0_signed_gap_pct=("signed_gap_pct", "mean"),
        mean_abs_beta0_gap_pct=("signed_gap_pct", lambda x: float(np.nanmean(np.abs(x)))),
        p95_abs_beta0_gap_pct=("signed_gap_pct", lambda x: float(np.nanpercentile(np.abs(x), 95))),
        p95_max_weight=("max_weight", lambda x: float(np.nanpercentile(x, 95))),
        p99_max_weight=("max_weight", lambda x: float(np.nanpercentile(x, 99))),
        max_weight=("max_weight", "max"),
        p10_ess_over_risk=("ess_over_risk_set", lambda x: float(np.nanpercentile(x, 10))),
        min_ess_over_risk=("ess_over_risk_set", "min"),
        clipping_fraction=("exp_clipping_count", lambda x: float(np.nanmean(np.asarray(x) > 0))),
    ).reset_index()
    summary.to_csv(out / "candidate_summary.csv", index=False)

    fold_stability = reps.groupby(["ridge_slope", "batch", "fold_seed"]).agg(
        mean_beta0_signed_gap_pct=("signed_gap_pct", "mean"),
        p99_max_weight=("max_weight", lambda x: float(np.nanpercentile(x, 99))),
        p10_ess_over_risk=("ess_over_risk_set", lambda x: float(np.nanpercentile(x, 10))),
        fit_failure_fraction=("fit_success", lambda x: 1.0 - float(np.mean(x))),
    ).reset_index()
    fold_stability.to_csv(out / "fold_seed_stability.csv", index=False)

    selected = summary[np.isclose(summary.ridge_slope, a.selected_ridge)]
    checks = {
        "beta1_results_used": False,
        "all_candidates_reported": set(map(float, a.ridges)) == set(summary.ridge_slope.astype(float)),
        "selected_has_no_fit_failures": bool((selected.fit_failure_fraction == 0).all()),
        "selected_has_no_clipping": bool((selected.clipping_fraction == 0).all()),
        "selected_p99_weight_below_25": bool((selected.p99_max_weight < 25).all()),
        "selected_p10_ess_over_risk_above_0p5": bool((selected.p10_ess_over_risk > 0.5).all()),
    }
    full_status = "PASS" if all(v for k, v in checks.items() if k != "beta1_results_used") else "REVIEW_REQUIRED"
    lock = {
        "analysis": "MATR censoring-model ridge selection audit",
        "status": "PASS_QUICK_CHECK" if a.quick_check else full_status,
        "candidate_ridges": list(map(float, a.ridges)),
        "selected_ridge_slope": float(a.selected_ridge),
        "audit_role": (
            "descriptive beta=0 negative-control stability re-audit; ridge 16 remains "
            "the frozen specification and is not re-selected from beta=1 results"
        ),
        "selection_criterion": (
            "ridge 16 was frozen before the formal beta=1 analyses; the beta=0 audit "
            "reports fit, clipping, weight, ESS and seed-stability diagnostics"
        ),
        "beta1_results_used": False,
        "R_per_overlay_seed": int(a.R),
        "fold_seeds": list(map(int, a.fold_seeds)),
        "overlay_seeds": list(map(int, a.overlay_seeds)),
        "checks": checks,
        "candidate_summary": str((out / "candidate_summary.csv").relative_to(root)),
        "fold_seed_stability": str((out / "fold_seed_stability.csv").relative_to(root)),
        "policy_scale": scale,
        "tau": tau,
    }
    lock_path = (
        out / "quick_validation.json"
        if a.quick_check
        else root / "validation" / "ridge_selection_audit.json"
    )
    lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print(f"RIDGE RE-AUDIT {lock['status']} -> {lock_path}")
    return 0 if a.quick_check or lock["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
