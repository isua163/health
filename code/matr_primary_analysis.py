#!/usr/bin/env python3
"""MATR batch-stratified endpoint-benchmark TV-IPCW analysis.

Design features:
- the three protocol batches retain separate RMST horizons and are never pooled
  through one product-limit curve;
- a common global IR scale/threshold is used, but the baseline replacement
  hazard is calibrated within each batch to 40% expected replacement;
- nuisance censoring models are fitted separately within batch, with unit-level
  five-fold cross-fitting;
- the policy-dependent crude RMST is computed exactly from the known overlay
  probabilities, rather than estimated from one realised overlay;
- event-time weighted hazard increments are recorded because checkpoint ESS can
  miss instability created by tied failures.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.util
import json
import hashlib
import math
import platform
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src._survival import km, rmrl_from_survival  # noqa: E402
from src.arms.competing_risks import aalen_johansen_cif1  # noqa: E402
from src.matr_ipcw import (  # noqa: E402
    build_ir_signal,
    calibrate_lambda0,
    exact_crude_rmst,
    exact_any_exit_rmst,
    expected_censor_fraction,
    fit_crossfit_cumhaz,
    fit_same_sample_cumhaz,
    ht_ipcw_rmst,
    make_stratified_folds,
    overlay_from_uniforms,
    standardize_policy_paths,
    weight_diagnostics,
    weighted_event_diagnostics,
    weighted_product_limit,
)

PRIMARY_BATCHES = ("MATR-05-12", "MATR-06-30", "MATR-04-12")
BETA_VALUES = (0.0, 1.0)
NET_ARMS = ("naive", "oracle_tv_ipcw", "oracle_ht_rmst", "crossfit_tv_ipcw", "same_sample_tv_ipcw")
WEIGHTED_ARMS = ("oracle_tv_ipcw", "crossfit_tv_ipcw", "same_sample_tv_ipcw")


def _load_audit_module(root: Path):
    path = root / "code" / "matr_data.py"
    spec = importlib.util.spec_from_file_location("matr_data_audit_primary", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import MATR data module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _subset(values: Sequence[Any], idx: np.ndarray) -> list[Any]:
    return [values[int(i)] for i in idx]


def _estimate_naive(times: np.ndarray, events: np.ndarray, horizon: float) -> float:
    grid, surv, _ = km(times, events)
    return float(rmrl_from_survival(grid, surv, 0.0, float(horizon)))


def _estimate_weighted(times: np.ndarray, events: np.ndarray,
                       cumhaz: list[np.ndarray], horizon: float) -> float:
    grid, surv = weighted_product_limit(times, events, cumhaz)
    return float(rmrl_from_survival(grid, surv, 0.0, float(horizon)))


def _estimate_empirical_crude(times: np.ndarray, events: np.ndarray, horizon: float) -> float:
    grid, cif1 = aalen_johansen_cif1(times, (events == 1).astype(int))
    return float(rmrl_from_survival(grid, np.clip(1.0 - cif1, 0.0, 1.0), 0.0, float(horizon)))


def _summarize(values: Iterable[float]) -> dict[str, float | int]:
    x = np.asarray(list(values), float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0, "mean": np.nan, "sd": np.nan, "mcse": np.nan,
                "p2p5": np.nan, "p10": np.nan, "median": np.nan,
                "p90": np.nan, "p95": np.nan, "p97p5": np.nan}
    sd = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    return {
        "n": int(len(x)), "mean": float(np.mean(x)), "sd": sd,
        "mcse": float(sd / math.sqrt(len(x))),
        "p2p5": float(np.percentile(x, 2.5)), "p10": float(np.percentile(x, 10)),
        "median": float(np.median(x)), "p90": float(np.percentile(x, 90)),
        "p95": float(np.percentile(x, 95)), "p97p5": float(np.percentile(x, 97.5)),
    }


def _load_primary_cohort(root: Path, matr: Path, baseline_cycles: int,
                         smooth_window: int) -> tuple[list[str], list[str], np.ndarray, list[np.ndarray]]:
    cohort_report_path = root / "validation" / "matr_cohort_validation.json"
    endpoint_path = root / "results" / "matr_cohort" / "endpoint_review.csv"
    if not cohort_report_path.exists() or not endpoint_path.exists():
        raise FileNotFoundError("cohort outputs are missing; run matr_cohort_reconstruction.py first")
    cohort_report = json.loads(cohort_report_path.read_text(encoding="utf-8"))
    status_value = cohort_report.get(
        "status", cohort_report.get("cohort audit_final_pass", False)
    )
    status_pass = status_value is True or str(status_value).strip().upper() == "PASS"
    if not status_pass:
        raise RuntimeError("cohort validation is not PASS; primary analysis is blocked")

    endpoint_rows = [r for r in _read_csv(endpoint_path) if _as_bool(r["primary_IR_cohort"])]
    endpoint_by_id = {r["unit_id"]: r for r in endpoint_rows}
    if len(endpoint_by_id) != 124:
        raise RuntimeError(f"expected 124 primary units, found {len(endpoint_by_id)}")

    audit = _load_audit_module(root)
    schema: list[str] = []
    cells = audit.harmonize(audit.read_raw_cells(matr, schema))
    cell_by_id = {c.unit_id: c for c in cells}

    unit_ids: list[str] = []
    batches: list[str] = []
    lifetimes: list[int] = []
    signals: list[np.ndarray] = []
    for unit_id in sorted(endpoint_by_id):
        row = endpoint_by_id[unit_id]
        cell = cell_by_id.get(unit_id)
        if cell is None:
            raise RuntimeError(f"cohort unit is missing from raw harmonization: {unit_id}")
        T = int(float(row["author_reconstructed_lifetime"]))
        ir = np.asarray(cell.arrays.get("IR", []), float)
        path = build_ir_signal(ir, T, baseline_cycles=baseline_cycles,
                               smooth_window=smooth_window)
        if len(path) != T:
            raise RuntimeError(f"signal length mismatch for {unit_id}: {len(path)} != {T}")
        unit_ids.append(unit_id)
        batches.append(str(row["batch_label"]))
        lifetimes.append(T)
        signals.append(path)

    counts = {batch: batches.count(batch) for batch in sorted(set(batches))}
    expected = {"MATR-04-12": 40, "MATR-05-12": 41, "MATR-06-30": 43}
    if counts != expected:
        raise RuntimeError(f"primary cohort batch counts differ from the frozen cohort: {counts}")
    return unit_ids, batches, np.asarray(lifetimes, float), signals


def _batch_indices(batches: Sequence[str]) -> dict[str, np.ndarray]:
    labels = np.asarray(batches, object)
    return {batch: np.flatnonzero(labels == batch) for batch in PRIMARY_BATCHES}


def _batch_scopes(batches: Sequence[str], lifetimes: np.ndarray) -> list[dict[str, Any]]:
    idx_map = _batch_indices(batches)
    scopes: list[dict[str, Any]] = []
    for batch in PRIMARY_BATCHES:
        idx = idx_map[batch]
        H = float(np.median(lifetimes[idx]))
        scopes.append({
            "scope": batch,
            "indices": idx,
            "H": H,
            "truth": float(np.mean(np.minimum(lifetimes[idx], H))),
        })
    return scopes


def _overlay_by_batch(paths: list[np.ndarray], uniforms: list[np.ndarray], batches: Sequence[str],
                      beta: float, tau: float, lambda_by_batch: dict[str, float],
                      policy_start: int) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    n = len(paths)
    times = np.full(n, np.nan, float)
    events = np.full(n, -1, int)
    observed: list[np.ndarray | None] = [None] * n
    cumhaz: list[np.ndarray | None] = [None] * n
    for batch, idx in _batch_indices(batches).items():
        tt, ee, oo, cc = overlay_from_uniforms(
            _subset(paths, idx), beta, tau, lambda_by_batch[batch],
            _subset(uniforms, idx), policy_start=policy_start,
        )
        times[idx] = tt
        events[idx] = ee
        for local, global_i in enumerate(idx):
            observed[int(global_i)] = oo[local]
            cumhaz[int(global_i)] = cc[local]
    if np.any(~np.isfinite(times)) or np.any(events < 0) or any(x is None for x in observed + cumhaz):
        raise RuntimeError("batch overlay mapping is incomplete")
    return times, events, [np.asarray(x, float) for x in observed], [np.asarray(x, float) for x in cumhaz]  # type: ignore[arg-type]


def _fit_by_batch(observed: list[np.ndarray], events: np.ndarray, folds: np.ndarray,
                  batches: Sequence[str], policy_start: int, ridge_slope: float = 0.0) -> tuple[
                      list[np.ndarray], list[dict[str, Any]], list[np.ndarray], list[dict[str, Any]]
                  ]:
    n = len(observed)
    cross_all: list[np.ndarray | None] = [None] * n
    same_all: list[np.ndarray | None] = [None] * n
    cross_fit_rows: list[dict[str, Any]] = []
    same_fit_rows: list[dict[str, Any]] = []
    for batch, idx in _batch_indices(batches).items():
        obs_b = _subset(observed, idx)
        ev_b = events[idx]
        fold_b = folds[idx]
        cross_b, fits_b = fit_crossfit_cumhaz(obs_b, ev_b, fold_b, policy_start=policy_start, ridge_slope=ridge_slope)
        same_b, same_fit = fit_same_sample_cumhaz(obs_b, ev_b, policy_start=policy_start, ridge_slope=ridge_slope)
        for local, global_i in enumerate(idx):
            cross_all[int(global_i)] = cross_b[local]
            same_all[int(global_i)] = same_b[local]
        for fold_value, fit in zip(sorted(np.unique(fold_b)), fits_b):
            cross_fit_rows.append({"batch": batch, "fold": int(fold_value), "fit": fit})
        same_fit_rows.append({"batch": batch, "fit": same_fit})
    if any(x is None for x in cross_all + same_all):
        raise RuntimeError("batch-stratified fitted cumulative hazards are incomplete")
    return (
        [np.asarray(x, float) for x in cross_all], cross_fit_rows,
        [np.asarray(x, float) for x in same_all], same_fit_rows,
    )  # type: ignore[arg-type]


def _event_summary(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    if not rows:
        return {
            "n_unique_failure_times": 0, "min_n_at_risk_at_failure": 0,
            "max_weighted_hazard_increment": np.nan,
            "p95_weighted_hazard_increment": np.nan,
            "max_failures_tied": 0,
        }
    return {
        "n_unique_failure_times": int(len(rows)),
        "min_n_at_risk_at_failure": int(min(int(r["n_at_risk"]) for r in rows)),
        "max_weighted_hazard_increment": float(max(float(r["weighted_hazard_increment"]) for r in rows)),
        "p95_weighted_hazard_increment": float(np.percentile(
            [float(r["weighted_hazard_increment"]) for r in rows], 95
        )),
        "max_failures_tied": int(max(int(r["n_failures"]) for r in rows)),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matr", required=True, type=Path)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--R", type=int, default=100)
    p.add_argument("--target-censor", type=float, default=0.40)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--fold-seed", type=int, default=20261001)
    p.add_argument("--seed", type=int, default=20261002)
    p.add_argument("--baseline-cycles", type=int, default=50)
    p.add_argument("--smooth-window", type=int, default=5)
    p.add_argument("--exp-clip", type=float, default=30.0)
    p.add_argument("--ridge-slope", type=float, default=0.0)
    p.add_argument("--verify-ridge-selection", action="store_true")
    p.add_argument("--analysis-label", default=None)
    p.add_argument("--report-name", default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--solver-check-replicate", type=int, default=None,
                   help="replay one zero-based overlay and check nuisance solvers only")
    p.add_argument("--quick-check", action="store_true")
    a = p.parse_args()

    root = a.root.resolve()
    matr = a.matr.resolve()
    if not matr.is_dir():
        raise FileNotFoundError(f"MATR directory not found: {matr}")
    if a.quick_check:
        a.R = min(int(a.R), 3)
    if a.R < 1:
        raise ValueError("R must be positive")
    if a.solver_check_replicate is not None and int(a.solver_check_replicate) < 0:
        raise ValueError("solver-check-replicate must be non-negative")
    if not 0.0 < float(a.target_censor) < 1.0:
        raise ValueError("target-censor must lie in (0,1)")
    if not np.isfinite(float(a.ridge_slope)) or float(a.ridge_slope) < 0.0:
        raise ValueError("ridge-slope must be finite and non-negative")

    ridge_lock_required = bool(a.verify_ridge_selection)
    selection_audit: dict[str, Any] = {"required": ridge_lock_required}
    if ridge_lock_required:
        lock_path = root / "validation" / "ridge_selection_audit.json"
        if not lock_path.exists():
            raise FileNotFoundError(lock_path)
        lock_bytes = lock_path.read_bytes()
        lock_report = json.loads(lock_bytes.decode("utf-8"))
        selected = lock_report.get("selected_ridge_slope")
        lock_pass = str(lock_report.get("status", "")).upper() == "PASS"
        beta1_unused = not bool(lock_report.get("beta1_results_used", True))
        selection_audit.update({
            "path": str(lock_path.relative_to(root)),
            "sha256": hashlib.sha256(lock_bytes).hexdigest(),
            "status_pass": lock_pass,
            "selected_ridge_slope": selected,
            "beta1_results_used": not beta1_unused,
        })
        if not lock_pass or selected is None:
            raise RuntimeError("ridge-selection audit did not pass")
        if abs(float(selected) - float(a.ridge_slope)) > 1e-12:
            raise RuntimeError(
                f"ridge-slope {a.ridge_slope} does not match audited ridge {selected}"
            )
        if not beta1_unused:
            raise RuntimeError("ridge selection used beta=1 results")

    out_dir = (root / "validation" / "matr_primary_quick_check") if a.quick_check else (
        a.out_dir.resolve() if a.out_dir is not None else root / "results" / "matr_primary"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    validation = root / "validation"
    validation.mkdir(parents=True, exist_ok=True)

    unit_ids, batches, lifetimes, raw_paths = _load_primary_cohort(
        root, matr, a.baseline_cycles, a.smooth_window
    )
    z_paths, policy_scale, tau = standardize_policy_paths(raw_paths, policy_start=a.baseline_cycles)
    folds = make_stratified_folds(batches, n_folds=a.folds, seed=a.fold_seed)
    scopes = _batch_scopes(batches, lifetimes)
    idx_map = _batch_indices(batches)

    lambda0: dict[float, dict[str, float]] = {}
    expected_c: dict[float, dict[str, float]] = {}
    for beta in BETA_VALUES:
        lambda0[beta] = {}
        expected_c[beta] = {}
        for batch in PRIMARY_BATCHES:
            batch_paths = _subset(z_paths, idx_map[batch])
            lam = calibrate_lambda0(
                batch_paths, beta, tau, a.target_censor,
                policy_start=a.baseline_cycles,
            )
            lambda0[beta][batch] = lam
            expected_c[beta][batch] = expected_censor_fraction(
                batch_paths, beta, tau, lam, policy_start=a.baseline_cycles
            )

    if a.solver_check_replicate is not None:
        target_rep = int(a.solver_check_replicate)
        rng_check = np.random.default_rng(int(a.seed))
        uniforms: list[np.ndarray] | None = None
        for _ in range(target_rep + 1):
            uniforms = [rng_check.random(max(len(path) - 1, 0)) for path in z_paths]
        assert uniforms is not None
        solver_rows: list[dict[str, Any]] = []
        realized_rows: list[dict[str, Any]] = []
        for beta in BETA_VALUES:
            times, events, observed, _ = _overlay_by_batch(
                z_paths, uniforms, batches, beta, tau, lambda0[beta], a.baseline_cycles
            )
            try:
                _, cross_fits, _, same_fits = _fit_by_batch(
                    observed, events, folds, batches, a.baseline_cycles,
                    ridge_slope=a.ridge_slope,
                )
            except Exception as exc:
                failure = {
                    "analysis": "primary locked-ridge exact solver replay",
                    "replicate_zero_based": target_rep,
                    "beta": beta,
                    "ridge_slope": float(a.ridge_slope),
                    "seed": int(a.seed),
                    "fold_seed": int(a.fold_seed),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "events_by_batch": {
                        batch: {
                            "failures": int(np.sum(events[idx_map[batch]] == 1)),
                            "replacements": int(np.sum(events[idx_map[batch]] == 0)),
                        }
                        for batch in PRIMARY_BATCHES
                    },
                }
                failure_path = validation / "solver_check_failure.json"
                failure_path.write_text(json.dumps(_jsonable(failure), indent=2), encoding="utf-8")
                raise
            for item in cross_fits:
                fit = item["fit"]
                solver_rows.append({
                    "beta": beta, "batch": str(item["batch"]),
                    "fit_scope": f"crossfit_fold_{item['fold']}",
                    "success": bool(fit.success), "method": str(getattr(fit, "method", "newton")),
                    "message": str(fit.message), "intercept": float(fit.intercept),
                    "slope": float(fit.slope), "n_iter": int(fit.n_iter),
                    "objective": float(fit.objective), "grad_norm": float(fit.grad_norm),
                })
            for item in same_fits:
                fit = item["fit"]
                solver_rows.append({
                    "beta": beta, "batch": str(item["batch"]),
                    "fit_scope": "same_sample", "success": bool(fit.success),
                    "method": str(getattr(fit, "method", "newton")),
                    "message": str(fit.message), "intercept": float(fit.intercept),
                    "slope": float(fit.slope), "n_iter": int(fit.n_iter),
                    "objective": float(fit.objective), "grad_norm": float(fit.grad_norm),
                })
            for batch in PRIMARY_BATCHES:
                idx = idx_map[batch]
                realized_rows.append({
                    "beta": beta, "batch": batch,
                    "failures": int(np.sum(events[idx] == 1)),
                    "replacements": int(np.sum(events[idx] == 0)),
                    "realized_censor_fraction": float(np.mean(events[idx] == 0)),
                })
        fallback_rows = [r for r in solver_rows if str(r["method"]).startswith("fallback_")]
        report = {
            "analysis": "primary locked-ridge exact solver replay",
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "replicate_zero_based": target_rep,
            "seed": int(a.seed), "fold_seed": int(a.fold_seed),
            "ridge_slope": float(a.ridge_slope),
            "diagnostic_ridge_lock": selection_audit,
            "n_fits": len(solver_rows),
            "n_fallback_fits": len(fallback_rows),
            "all_fits_successful": bool(all(r["success"] for r in solver_rows)),
            "fallback_fraction": float(len(fallback_rows) / len(solver_rows)) if solver_rows else None,
            "fits": solver_rows,
            "realized_overlay": realized_rows,
            "computational_pass": bool(all(r["success"] for r in solver_rows)),
        }
        report_path = validation / "solver_check_report.json"
        report_path.write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
        print("primary SOLVER REPLAY SUMMARY")
        print(f"  replicate_zero_based={target_rep}; ridge={a.ridge_slope:g}")
        print(f"  fits={len(solver_rows)}; fallbacks={len(fallback_rows)}")
        for beta in BETA_VALUES:
            rows_beta = [r for r in solver_rows if r["beta"] == beta]
            fb_beta = [r for r in rows_beta if str(r["method"]).startswith("fallback_")]
            methods = sorted(set(str(r["method"]) for r in rows_beta))
            print(f"  beta={beta:g}: methods={methods}; fallbacks={len(fb_beta)}")
        print(f"  computational_pass={report['computational_pass']}")
        print(f"Wrote {report_path}")
        print("primary SOLVER REPLAY COMPLETED")
        return 0 if report["computational_pass"] else 2

    fold_rows = [
        {"unit_id": unit_ids[i], "batch_label": batches[i], "fold": int(folds[i]),
         "lifetime": int(lifetimes[i])}
        for i in range(len(unit_ids))
    ]
    fold_path = out_dir / "fold_assignment.csv"
    _write_csv(fold_path, fold_rows)

    # Exact policy-defined crude quantities: deterministic for the frozen fleet.
    crude_rows: list[dict[str, Any]] = []
    for beta in BETA_VALUES:
        for scope in scopes:
            batch = str(scope["scope"])
            idx = np.asarray(scope["indices"], int)
            H = float(scope["H"])
            truth = float(scope["truth"])
            crude, g_terminal = exact_crude_rmst(
                _subset(z_paths, idx), beta, tau, lambda0[beta][batch], H,
                policy_start=a.baseline_cycles,
            )
            any_exit = exact_any_exit_rmst(
                _subset(z_paths, idx), beta, tau, lambda0[beta][batch], H,
                policy_start=a.baseline_cycles,
            )
            crude_rows.append({
                "beta": beta, "scope": batch, "n_units": int(len(idx)), "H": H,
                "truth_net_rmst": truth, "exact_crude_rmst": crude,
                "exact_any_exit_rmst": any_exit,
                "estimand_gap": crude - truth,
                "estimand_gap_pct_of_net": 100.0 * (crude - truth) / truth,
                "any_exit_minus_net": any_exit - truth,
                "any_exit_minus_net_pct": 100.0 * (any_exit - truth) / truth,
                "crude_minus_any_exit": crude - any_exit,
                "mean_G_failure_minus": float(np.mean(g_terminal)),
                "min_G_failure_minus": float(np.min(g_terminal)),
                "p10_G_failure_minus": float(np.percentile(g_terminal, 10)),
            })
    crude_path = out_dir / "estimand_gap.csv"
    _write_csv(crude_path, crude_rows)

    policy_design = {
        "analysis": "primaryb_MATR_batch_stratified_known_truth_TV_IPCW",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "mode": "quick_check" if a.quick_check else "analysis",
        "n_units": len(unit_ids),
        "batch_counts": {b: batches.count(b) for b in PRIMARY_BATCHES},
        "endpoint": "frozen public MATR endpoint reconstruction",
        "signal": "log(causal trailing-median IR / first-50-cycle positive-IR median)",
        "smooth_window": int(a.smooth_window),
        "baseline_cycles": int(a.baseline_cycles),
        "policy_scale": "global unit-equal IQR of policy-eligible pre-failure records",
        "policy_scale_value": policy_scale,
        "threshold": "global unit-equal p70 after IQR scaling",
        "tau": tau,
        "betas": list(BETA_VALUES),
        "target_censor_fraction_within_each_batch": float(a.target_censor),
        "lambda0_by_beta_batch": {
            str(beta): lambda0[beta] for beta in BETA_VALUES
        },
        "expected_censor_fraction_by_beta_batch": {
            str(beta): expected_c[beta] for beta in BETA_VALUES
        },
        "calibration_rationale": (
            "batch-specific baseline hazards hold expected replacement burden fixed at 40%; "
            "the common beta and global signal scale retain a shared health-selection contrast"
        ),
        "estimation": (
            "batch-stratified product-limit and batch-stratified nuisance fitting; "
            "no unstratified pooled survival curve"
        ),
        "crossfit": {"n_folds": int(a.folds), "fold_seed": int(a.fold_seed),
                     "unit_level": True, "fitted_separately_within_batch": True},
        "overlay_seed": int(a.seed), "R": int(a.R),
        "policy_run_in": {"cycles": int(a.baseline_cycles),
                          "first_eligible_replacement_record": int(a.baseline_cycles + 1)},
        "event_timing": "failure before censoring at a shared terminal record",
        "horizons": {s["scope"]: s["H"] for s in scopes},
        "truth_net_rmst": {s["scope"]: s["truth"] for s in scopes},
        "crude_status": "exact finite-fleet policy estimand; empirical AJ retained only as validation diagnostic",
        "same_sample_status": "diagnostic only; cross-fitted TV-IPCW is primary fitted estimator",
        "ridge_slope": float(a.ridge_slope),
        "ridge_status": (
            "locked from beta=0 negative-control diagnostics"
            if ridge_lock_required else "user-specified/unlocked"
        ),
        "diagnostic_ridge_lock": selection_audit,
        "oracle_ht_status": (
            "estimand-aligned implementation diagnostic; not the primary product-limit estimator"
        ),
        "analysis_label": a.analysis_label or ("quick_check" if a.quick_check else "analysis"),
    }
    design_path = out_dir / "policy_design.json"
    design_path.write_text(json.dumps(_jsonable(policy_design), indent=2), encoding="utf-8")

    rng = np.random.default_rng(int(a.seed))
    replicate_rows: list[dict[str, Any]] = []
    empirical_crude_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    event_summary_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []

    for rep in range(int(a.R)):
        uniforms = [rng.random(max(len(path) - 1, 0)) for path in z_paths]
        for beta in BETA_VALUES:
            times, events, observed, oracle_cumhaz = _overlay_by_batch(
                z_paths, uniforms, batches, beta, tau, lambda0[beta], a.baseline_cycles
            )
            try:
                cross_cumhaz, cross_fits, same_cumhaz, same_fits = _fit_by_batch(
                    observed, events, folds, batches, a.baseline_cycles, ridge_slope=a.ridge_slope
                )
            except Exception as exc:
                failure = {
                    "analysis_label": a.analysis_label or ("quick_check" if a.quick_check else "analysis"),
                    "replicate_zero_based": rep, "beta": beta,
                    "seed": int(a.seed), "fold_seed": int(a.fold_seed),
                    "ridge_slope": float(a.ridge_slope),
                    "error_type": type(exc).__name__, "error": str(exc),
                    "events_by_batch": {
                        batch: {
                            "failures": int(np.sum(events[idx_map[batch]] == 1)),
                            "replacements": int(np.sum(events[idx_map[batch]] == 0)),
                        }
                        for batch in PRIMARY_BATCHES
                    },
                }
                failure_path = validation / "fit_failure.json"
                failure_path.write_text(json.dumps(_jsonable(failure), indent=2), encoding="utf-8")
                raise

            for item in cross_fits:
                batch = str(item["batch"])
                fit = item["fit"]
                true_intercept = float(np.log(lambda0[beta][batch]) - beta * tau)
                fit_rows.append({
                    "replicate": rep, "beta": beta, "batch": batch,
                    "fit_scope": f"crossfit_fold_{item['fold']}",
                    "success": fit.success, "intercept": fit.intercept, "slope": fit.slope,
                    "true_intercept": true_intercept, "true_slope": beta,
                    "ridge_slope": float(a.ridge_slope),
                    "intercept_error": fit.intercept - true_intercept,
                    "slope_error": fit.slope - beta, "n_iter": fit.n_iter,
                    "objective": fit.objective, "grad_norm": fit.grad_norm,
                    "message": fit.message, "solver_method": getattr(fit, "method", "newton"),
                })
            for item in same_fits:
                batch = str(item["batch"])
                fit = item["fit"]
                true_intercept = float(np.log(lambda0[beta][batch]) - beta * tau)
                fit_rows.append({
                    "replicate": rep, "beta": beta, "batch": batch,
                    "fit_scope": "same_sample", "success": fit.success,
                    "intercept": fit.intercept, "slope": fit.slope,
                    "true_intercept": true_intercept, "true_slope": beta,
                    "ridge_slope": float(a.ridge_slope),
                    "intercept_error": fit.intercept - true_intercept,
                    "slope_error": fit.slope - beta, "n_iter": fit.n_iter,
                    "objective": fit.objective, "grad_norm": fit.grad_norm,
                    "message": fit.message, "solver_method": getattr(fit, "method", "newton"),
                })

            for scope in scopes:
                batch = str(scope["scope"])
                idx = np.asarray(scope["indices"], int)
                H = float(scope["H"])
                truth = float(scope["truth"])
                tt = times[idx]
                ee = events[idx]
                oracle_sub = _subset(oracle_cumhaz, idx)
                cross_sub = _subset(cross_cumhaz, idx)
                same_sub = _subset(same_cumhaz, idx)
                estimates = {
                    "naive": _estimate_naive(tt, ee, H),
                    "oracle_tv_ipcw": _estimate_weighted(tt, ee, oracle_sub, H),
                    "oracle_ht_rmst": ht_ipcw_rmst(tt, ee, oracle_sub, H, exp_clip=None),
                    "crossfit_tv_ipcw": _estimate_weighted(tt, ee, cross_sub, H),
                    "same_sample_tv_ipcw": _estimate_weighted(tt, ee, same_sub, H),
                }
                realized_censor = float(np.mean(ee == 0))
                for arm, estimate in estimates.items():
                    gap_pct = 100.0 * (estimate - truth) / truth
                    replicate_rows.append({
                        "replicate": rep, "beta": beta, "scope": batch,
                        "n_units": int(len(idx)), "H": H, "truth_net_rmst": truth,
                        "arm": arm, "estimate": estimate,
                        "signed_gap_pct": gap_pct, "absolute_gap_pct": abs(gap_pct),
                        "realized_censor_fraction_scope": realized_censor,
                        "observed_failures_scope": int(np.sum(ee == 1)),
                        "preventive_replacements_scope": int(np.sum(ee == 0)),
                    })

                empirical_crude = _estimate_empirical_crude(tt, ee, H)
                exact_crude = next(
                    float(r["exact_crude_rmst"]) for r in crude_rows
                    if r["beta"] == beta and r["scope"] == batch
                )
                empirical_crude_rows.append({
                    "replicate": rep, "beta": beta, "scope": batch,
                    "n_units": int(len(idx)), "H": H,
                    "empirical_AJ_crude_rmst": empirical_crude,
                    "exact_crude_rmst": exact_crude,
                    "empirical_minus_exact": empirical_crude - exact_crude,
                    "empirical_minus_exact_pct_of_exact": 100.0 * (empirical_crude - exact_crude) / exact_crude,
                })

                checkpoints = [0.25 * H, 0.50 * H, 0.75 * H, H]
                for arm, hazards in (
                    ("oracle_tv_ipcw", oracle_sub),
                    ("crossfit_tv_ipcw", cross_sub),
                    ("same_sample_tv_ipcw", same_sub),
                ):
                    for frac, diag in zip(
                        (0.25, 0.50, 0.75, 1.00),
                        weight_diagnostics(tt, hazards, checkpoints, a.exp_clip),
                    ):
                        weight_rows.append({
                            "replicate": rep, "beta": beta, "scope": batch,
                            "arm": arm, "checkpoint_fraction_H": frac, "H": H,
                            **diag,
                        })
                    erows = weighted_event_diagnostics(
                        tt, ee, hazards, horizon=H, exp_clip=a.exp_clip
                    )
                    for erow in erows:
                        event_rows.append({
                            "replicate": rep, "beta": beta, "scope": batch,
                            "arm": arm, "H": H, **erow,
                        })
                    event_summary_rows.append({
                        "replicate": rep, "beta": beta, "scope": batch,
                        "arm": arm, "H": H, **_event_summary(erows),
                    })
        print(f"completed overlay {rep + 1}/{a.R}", flush=True)

    replicate_path = out_dir / "replicates.csv"
    empirical_crude_path = out_dir / "empirical_crude_diagnostic.csv"
    weight_path = out_dir / "weight_diagnostics.csv"
    event_path = out_dir / "event_time_diagnostics.csv"
    event_summary_path = out_dir / "event_time_summary.csv"
    fit_path = out_dir / "fit_diagnostics.csv"
    _write_csv(replicate_path, replicate_rows)
    _write_csv(empirical_crude_path, empirical_crude_rows)
    _write_csv(weight_path, weight_rows)
    _write_csv(event_path, event_rows)
    _write_csv(event_summary_path, event_summary_rows)
    _write_csv(fit_path, fit_rows)

    summary_rows: list[dict[str, Any]] = []
    for beta in BETA_VALUES:
        for batch in PRIMARY_BATCHES:
            for arm in NET_ARMS:
                rr = [r for r in replicate_rows if r["beta"] == beta and r["scope"] == batch and r["arm"] == arm]
                est = _summarize(float(r["estimate"]) for r in rr)
                gap = _summarize(float(r["signed_gap_pct"]) for r in rr)
                abs_gap = _summarize(float(r["absolute_gap_pct"]) for r in rr)
                summary_rows.append({
                    "beta": beta, "scope": batch, "arm": arm,
                    "n_units": rr[0]["n_units"], "H": rr[0]["H"],
                    "truth_net_rmst": rr[0]["truth_net_rmst"], "R": len(rr),
                    "mean_estimate": est["mean"], "sd_estimate": est["sd"],
                    "mean_signed_gap_pct": gap["mean"], "mcse_signed_gap_pct": gap["mcse"],
                    "p2p5_signed_gap_pct": gap["p2p5"], "p97p5_signed_gap_pct": gap["p97p5"],
                    "absolute_mean_signed_gap_pct": abs(float(gap["mean"])),
                    "mean_absolute_gap_pct": abs_gap["mean"],
                    "mean_realized_censor_fraction": float(np.mean([
                        r["realized_censor_fraction_scope"] for r in rr
                    ])),
                    "mean_observed_failures_scope": float(np.mean([
                        r["observed_failures_scope"] for r in rr
                    ])),
                })

    # Descriptive standardisation only: average the three batch-specific percent
    # gaps after estimation.  This is not called a pooled RMST estimand.
    standardized_rows: list[dict[str, Any]] = []
    for rep in range(int(a.R)):
        for beta in BETA_VALUES:
            for arm in NET_ARMS:
                rr = [r for r in replicate_rows if r["replicate"] == rep and r["beta"] == beta and r["arm"] == arm]
                if len(rr) != len(PRIMARY_BATCHES):
                    raise RuntimeError("standardized summary is missing a batch")
                standardized_rows.append({
                    "replicate": rep, "beta": beta, "arm": arm,
                    "equal_batch_mean_signed_gap_pct": float(np.mean([r["signed_gap_pct"] for r in rr])),
                    "n_weighted_mean_signed_gap_pct": float(np.average(
                        [r["signed_gap_pct"] for r in rr], weights=[r["n_units"] for r in rr]
                    )),
                })
    standardized_path = out_dir / "equal_batch_replicates.csv"
    _write_csv(standardized_path, standardized_rows)

    standardized_summary_rows: list[dict[str, Any]] = []
    for beta in BETA_VALUES:
        for arm in NET_ARMS:
            rr = [r for r in standardized_rows if r["beta"] == beta and r["arm"] == arm]
            eq = _summarize(float(r["equal_batch_mean_signed_gap_pct"]) for r in rr)
            nw = _summarize(float(r["n_weighted_mean_signed_gap_pct"]) for r in rr)
            standardized_summary_rows.append({
                "beta": beta, "arm": arm, "R": len(rr),
                "mean_equal_batch_gap_pct": eq["mean"], "mcse_equal_batch_gap_pct": eq["mcse"],
                "p2p5_equal_batch_gap_pct": eq["p2p5"], "p97p5_equal_batch_gap_pct": eq["p97p5"],
                "mean_n_weighted_gap_pct": nw["mean"], "mcse_n_weighted_gap_pct": nw["mcse"],
            })
    standardized_summary_path = out_dir / "equal_batch_summary.csv"
    _write_csv(standardized_summary_path, standardized_summary_rows)

    summary_path = out_dir / "estimator_summary.csv"
    _write_csv(summary_path, summary_rows)

    contrast_rows: list[dict[str, Any]] = []
    for rep in range(int(a.R)):
        for batch in PRIMARY_BATCHES:
            for arm in NET_ARMS:
                rr = [r for r in replicate_rows if r["replicate"] == rep and r["scope"] == batch and r["arm"] == arm]
                by_beta = {float(r["beta"]): r for r in rr}
                contrast_rows.append({
                    "replicate": rep, "scope": batch, "arm": arm,
                    "beta0_signed_gap_pct": by_beta[0.0]["signed_gap_pct"],
                    "beta1_signed_gap_pct": by_beta[1.0]["signed_gap_pct"],
                    "contrast_beta1_minus_beta0_pct": (
                        float(by_beta[1.0]["signed_gap_pct"]) - float(by_beta[0.0]["signed_gap_pct"])
                    ),
                })
    contrast_path = out_dir / "paired_contrasts.csv"
    _write_csv(contrast_path, contrast_rows)

    contrast_summary_rows: list[dict[str, Any]] = []
    for batch in PRIMARY_BATCHES:
        for arm in NET_ARMS:
            rr = [r for r in contrast_rows if r["scope"] == batch and r["arm"] == arm]
            stats = _summarize(float(r["contrast_beta1_minus_beta0_pct"]) for r in rr)
            contrast_summary_rows.append({
                "scope": batch, "arm": arm, "R": len(rr),
                "mean_contrast_pct": stats["mean"], "mcse_contrast_pct": stats["mcse"],
                "p2p5_contrast_pct": stats["p2p5"], "p97p5_contrast_pct": stats["p97p5"],
            })
    contrast_summary_path = out_dir / "paired_correction_summary.csv"
    _write_csv(contrast_summary_path, contrast_summary_rows)

    weight_summary_rows: list[dict[str, Any]] = []
    for beta in BETA_VALUES:
        for batch in PRIMARY_BATCHES:
            for arm in WEIGHTED_ARMS:
                for frac in (0.25, 0.50, 0.75, 1.00):
                    rr = [r for r in weight_rows if r["beta"] == beta and r["scope"] == batch
                          and r["arm"] == arm and r["checkpoint_fraction_H"] == frac]
                    ess = _summarize(float(r["ess_over_n_at_risk"]) for r in rr)
                    weight_summary_rows.append({
                        "beta": beta, "scope": batch, "arm": arm,
                        "checkpoint_fraction_H": frac, "R": len(rr),
                        "median_n_at_risk": float(np.median([r["n_at_risk"] for r in rr])),
                        "p10_ess_over_n_at_risk": ess["p10"],
                        "median_ess_over_n_at_risk": ess["median"],
                        "p95_max_weight": float(np.percentile([r["max_weight"] for r in rr], 95)),
                        "max_max_weight": float(np.max([r["max_weight"] for r in rr])),
                        "fraction_replicates_with_exp_clipping": float(np.mean([
                            r["n_exp_clipped"] > 0 for r in rr
                        ])),
                    })
    weight_summary_path = out_dir / "support_summary.csv"
    _write_csv(weight_summary_path, weight_summary_rows)

    event_aggregate_rows: list[dict[str, Any]] = []
    for beta in BETA_VALUES:
        for batch in PRIMARY_BATCHES:
            for arm in WEIGHTED_ARMS:
                rr = [r for r in event_summary_rows if r["beta"] == beta and r["scope"] == batch and r["arm"] == arm]
                event_aggregate_rows.append({
                    "beta": beta, "scope": batch, "arm": arm, "R": len(rr),
                    "median_min_n_at_risk_at_failure": float(np.median([
                        r["min_n_at_risk_at_failure"] for r in rr
                    ])),
                    "p10_min_n_at_risk_at_failure": float(np.percentile([
                        r["min_n_at_risk_at_failure"] for r in rr
                    ], 10)),
                    "p95_max_weighted_hazard_increment": float(np.percentile([
                        r["max_weighted_hazard_increment"] for r in rr
                    ], 95)),
                    "max_max_weighted_hazard_increment": float(np.max([
                        r["max_weighted_hazard_increment"] for r in rr
                    ])),
                    "max_tied_failures": int(np.max([r["max_failures_tied"] for r in rr])),
                })
    event_aggregate_path = out_dir / "event_support.csv"
    _write_csv(event_aggregate_path, event_aggregate_rows)

    fit_success = [bool(r["success"]) for r in fit_rows]
    expected_ok = all(
        abs(expected_c[beta][batch] - a.target_censor) <= 1e-8
        for beta in BETA_VALUES for batch in PRIMARY_BATCHES
    )
    beta1_summary = {
        batch: {
            arm: next(
                r for r in summary_rows
                if r["beta"] == 1.0 and r["scope"] == batch and r["arm"] == arm
            )
            for arm in NET_ARMS
        }
        for batch in PRIMARY_BATCHES
    }
    beta1_standardized = {
        r["arm"]: r for r in standardized_summary_rows if r["beta"] == 1.0
    }
    beta1_crude = {
        r["scope"]: r for r in crude_rows if r["beta"] == 1.0
    }
    crossfit_H_by_beta = {
        beta: {
            batch: next(
                r for r in weight_summary_rows
                if r["beta"] == beta and r["scope"] == batch
                and r["arm"] == "crossfit_tv_ipcw" and r["checkpoint_fraction_H"] == 1.0
            )
            for batch in PRIMARY_BATCHES
        }
        for beta in BETA_VALUES
    }
    crossfit_event_by_beta = {
        beta: {
            batch: next(
                r for r in event_aggregate_rows
                if r["beta"] == beta and r["scope"] == batch and r["arm"] == "crossfit_tv_ipcw"
            )
            for batch in PRIMARY_BATCHES
        }
        for beta in BETA_VALUES
    }
    beta1_crossfit_H = crossfit_H_by_beta[1.0]
    beta1_crossfit_event = crossfit_event_by_beta[1.0]

    smoke_only = bool(a.quick_check or a.R < 20)
    beta0_pl_differences = []
    for rep in range(int(a.R)):
        for batch in PRIMARY_BATCHES:
            lookup = {
                r["arm"]: float(r["estimate"])
                for r in replicate_rows
                if r["replicate"] == rep and r["beta"] == 0.0 and r["scope"] == batch
            }
            beta0_pl_differences.append(abs(lookup["naive"] - lookup["oracle_tv_ipcw"]))
    solver_methods = [str(r.get("solver_method", "newton")) for r in fit_rows]
    fallback_fit_count = int(sum(method.startswith("fallback_") for method in solver_methods))
    precision_fit_count = int(sum(method == "newton_precision" for method in solver_methods))
    total_fit_count = int(len(solver_methods))
    fallback_fit_fraction = float(fallback_fit_count / total_fit_count) if total_fit_count else 0.0
    solver_method_counts = {method: solver_methods.count(method) for method in sorted(set(solver_methods))}

    core_checks = {
        "cohort_validation_passed": True,
        "cohort_exactly_124": len(unit_ids) == 124,
        "all_fits_successful": bool(all(fit_success)),
        "solver_fallback_fraction_at_most_0p01": bool(fallback_fit_fraction <= 0.01),
        "expected_censor_exactly_calibrated_within_batch": bool(expected_ok),
        "exact_crude_not_below_net": bool(all(r["estimand_gap"] >= -1e-10 for r in crude_rows)),
        "beta0_naive_equals_oracle_product_limit": bool(max(beta0_pl_differences, default=0.0) <= 1e-10),
        "diagnostic_selected_ridge_locked": bool(
            (not ridge_lock_required)
            or (
                selection_audit.get("status_pass")
                and not selection_audit.get("beta1_results_used", True)
                and True
                and abs(
                    float(selection_audit.get("selected_ridge_slope"))
                    - float(a.ridge_slope)
                )
                <= 1e-12
            )
        ),
    }
    stability_checks: dict[str, bool] = {}
    for beta in BETA_VALUES:
        for batch in PRIMARY_BATCHES:
            w = crossfit_H_by_beta[beta][batch]
            e = crossfit_event_by_beta[beta][batch]
            prefix = f"beta{beta:g}_{batch}"
            stability_checks[f"{prefix}_crossfit_H_median_risk_at_least_5"] = bool(w["median_n_at_risk"] >= 5)
            stability_checks[f"{prefix}_crossfit_H_p10_ESS_at_least_0p20"] = bool(
                np.isfinite(w["p10_ess_over_n_at_risk"]) and w["p10_ess_over_n_at_risk"] >= 0.20
            )
            stability_checks[f"{prefix}_crossfit_clipping_at_most_0p01"] = bool(
                w["fraction_replicates_with_exp_clipping"] <= 0.01
            )
            stability_checks[f"{prefix}_event_increment_below_0p80"] = bool(
                np.isfinite(e["max_max_weighted_hazard_increment"])
                and e["max_max_weighted_hazard_increment"] < 0.80
            )
    computational_checks = {**core_checks, **stability_checks}
    computational_pass = bool(all(core_checks.values()) if smoke_only else all(computational_checks.values()))

    gate = {
        "analysis": a.analysis_label or "primaryb_MATR_batch_stratified_analysis_gate",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python": sys.version, "platform": platform.platform(),
        "mode": a.analysis_label or ("quick_check" if a.quick_check else "analysis"), "smoke_test_only": smoke_only,
        "R": int(a.R), "ridge_slope": float(a.ridge_slope),
        "diagnostic_ridge_lock": selection_audit,
        "solver_diagnostics": {
            "total_fits": total_fit_count,
            "fallback_fit_count": fallback_fit_count,
            "fallback_fit_fraction": fallback_fit_fraction,
            "newton_precision_fit_count": precision_fit_count,
            "method_counts": solver_method_counts,
            "fallback_optimizes_identical_penalized_likelihood": True,
        },
        "computational_checks": computational_checks,
        "computational_pass": computational_pass,
        "stability_checks_provisional_in_quick_check": smoke_only,
        "scientific_results_are_not_gates": True,
        "primary_beta1_by_batch": {
            batch: {
                arm: {
                    "mean_signed_gap_pct": row["mean_signed_gap_pct"],
                    "mcse_signed_gap_pct": row["mcse_signed_gap_pct"],
                    "absolute_mean_signed_gap_pct": row["absolute_mean_signed_gap_pct"],
                }
                for arm, row in arms.items()
            }
            for batch, arms in beta1_summary.items()
        },
        "beta1_standardized_descriptive": beta1_standardized,
        "beta1_exact_crude_by_batch": beta1_crude,
        "beta1_crossfit_weight_at_H_by_batch": beta1_crossfit_H,
        "beta1_crossfit_event_support_by_batch": beta1_crossfit_event,
        "outputs": {
            "policy_design": str(design_path.relative_to(root)),
            "fold_assignment": str(fold_path.relative_to(root)),
            "exact_crude": str(crude_path.relative_to(root)),
            "replicates": str(replicate_path.relative_to(root)),
            "summary": str(summary_path.relative_to(root)),
            "standardized_replicates": str(standardized_path.relative_to(root)),
            "standardized_summary": str(standardized_summary_path.relative_to(root)),
            "contrasts": str(contrast_path.relative_to(root)),
            "contrast_summary": str(contrast_summary_path.relative_to(root)),
            "empirical_crude_diagnostic": str(empirical_crude_path.relative_to(root)),
            "weight_diagnostics": str(weight_path.relative_to(root)),
            "weight_summary": str(weight_summary_path.relative_to(root)),
            "event_time_diagnostics": str(event_path.relative_to(root)),
            "event_time_summary": str(event_summary_path.relative_to(root)),
            "event_time_aggregate": str(event_aggregate_path.relative_to(root)),
            "fit_diagnostics": str(fit_path.relative_to(root)),
        },
    }
    report_name = a.report_name or ("quick_check_report.json" if a.quick_check else "analysis_report.json")
    report_path = validation / report_name
    report_path.write_text(json.dumps(_jsonable(gate), indent=2), encoding="utf-8")

    print("primary MATR BATCH-STRATIFIED IPCW SUMMARY")
    print(f"  mode={a.analysis_label or ('quick_check' if a.quick_check else 'analysis')}; R={a.R}; n=124")
    print(f"  ridge_slope={a.ridge_slope:g}; selection_audit={ridge_lock_required}")
    print(
        f"  solver fits={total_fit_count}; fallbacks={fallback_fit_count} "
        f"({fallback_fit_fraction:.4%}); methods={solver_method_counts}"
    )
    print(f"  global policy scale={policy_scale:.6g}; tau={tau:.6g}")
    for beta in BETA_VALUES:
        print(f"  beta={beta:g} batch calibration:")
        for batch in PRIMARY_BATCHES:
            print(
                f"    {batch}: lambda0={lambda0[beta][batch]:.8g}; "
                f"expected censor={expected_c[beta][batch]:.4f}"
            )
    print("  beta=1 mean signed gaps by batch (% of batch net RMST):")
    for batch in PRIMARY_BATCHES:
        values = beta1_summary[batch]
        print(
            f"    {batch}: naive={values['naive']['mean_signed_gap_pct']:+.3f}; "
            f"oraclePL={values['oracle_tv_ipcw']['mean_signed_gap_pct']:+.3f}; "
            f"oracleHT={values['oracle_ht_rmst']['mean_signed_gap_pct']:+.3f}; "
            f"crossfit={values['crossfit_tv_ipcw']['mean_signed_gap_pct']:+.3f}; "
            f"same={values['same_sample_tv_ipcw']['mean_signed_gap_pct']:+.3f}"
        )
    print("  beta=1 exact crude estimand gaps (% of batch net RMST):")
    for batch in PRIMARY_BATCHES:
        print(f"    {batch}: {beta1_crude[batch]['estimand_gap_pct_of_net']:+.3f}")
    print("  beta=1 crossfit support at H:")
    for batch in PRIMARY_BATCHES:
        w = beta1_crossfit_H[batch]
        e = beta1_crossfit_event[batch]
        print(
            f"    {batch}: risk={w['median_n_at_risk']:.1f}; "
            f"p10 ESS/risk={w['p10_ess_over_n_at_risk']:.3f}; "
            f"p95 max weight={w['p95_max_weight']:.3g}; "
            f"max event increment={e['max_max_weighted_hazard_increment']:.3f}"
        )
    print(f"  computational_pass={gate['computational_pass']}")
    print(f"Wrote {report_path}")
    print("primary MATR IPCW QUICK CHECK COMPLETED" if a.quick_check else "primary MATR IPCW ANALYSIS COMPLETED")
    return 0 if gate["computational_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
