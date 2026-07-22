#!/usr/bin/env python3
"""Fully nested nuisance-preprocessing audit for the MATR benchmark.

The imposed replacement policy remains the fixed, externally declared synthetic
policy used by the primary benchmark.  In contrast, every nuisance-model fold
estimates its signal scale and centering threshold from *observed training-unit
prefixes only* and applies those values to held-out units.  This separates the
policy definition from the out-of-fold nuisance-prediction audit and removes the
benchmark-wide preprocessing leakage identified during review.

Required upstream outputs
-------------------------
Run ``matr_data.py`` and ``matr_endpoint_reconstruction.py`` first.  Raw MATR
files are not redistributed and must be supplied through ``--matr``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from matr_primary_analysis import (  # noqa: E402
    PRIMARY_BATCHES,
    _batch_indices,
    _batch_scopes,
    _estimate_naive,
    _estimate_weighted,
    _fit_by_batch,
    _load_primary_cohort,
    _overlay_by_batch,
    _subset,
    _summarize,
    _write_csv,
)
from src.matr_ipcw import (  # noqa: E402
    calibrate_lambda0,
    expected_censor_fraction,
    fit_cloglog_fast,
    fitted_cumhaz_before,
    make_stratified_folds,
    person_period,
    standardize_policy_paths,
    unit_equal_iqr_scale,
    unit_equal_quantile,
    weight_diagnostics,
    weighted_event_diagnostics,
)


def _observed_raw_prefixes(raw_paths: Sequence[np.ndarray], times: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for path, time in zip(raw_paths, times):
        length = int(round(float(time)))
        prefix = np.asarray(path, float)[:length].copy()
        if len(prefix) != length:
            raise RuntimeError(f"Observed time {length} exceeds raw path length {len(path)}")
        out.append(prefix)
    return out


def _eligible_preprocessing_path(path: np.ndarray, event: int, policy_start: int) -> np.ndarray:
    """Rows available to the censoring likelihood for one observed training unit."""
    z = np.asarray(path, float)
    start = max(0, int(policy_start))
    if int(event) == 0:  # triggering censoring record is observed
        values = z[start:]
    else:  # terminal failure record is excluded from the censoring likelihood
        values = z[start:-1]
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("Training unit contributes no eligible observed preprocessing rows")
    return values


def _fit_fold_preprocessing(
    observed_raw: Sequence[np.ndarray],
    events: np.ndarray,
    train: np.ndarray,
    policy_start: int,
) -> tuple[float, float]:
    reference = [
        _eligible_preprocessing_path(observed_raw[int(i)], int(events[int(i)]), policy_start)
        for i in train
    ]
    scale = unit_equal_iqr_scale(reference)
    scaled = [x / scale for x in reference]
    tau = unit_equal_quantile(scaled, 0.70)
    return float(scale), float(tau)


def _unit_person_period(path: np.ndarray, event: int, policy_start: int) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(path, float)
    start = max(0, int(policy_start))
    if int(event) == 0:
        x = z[start:]
        y = np.zeros(len(x), float)
        if len(y):
            y[-1] = 1.0
    else:
        x = z[start:-1]
        y = np.zeros(len(x), float)
    return x, y


def _calibration_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    if len(y) == 0:
        return {
            "n_rows": 0,
            "n_censor_events": 0,
            "observed_censor_rate": float("nan"),
            "mean_predicted_censor_probability": float("nan"),
            "brier_score": float("nan"),
            "log_loss": float("nan"),
        }
    p = np.clip(np.asarray(p, float), 1e-12, 1.0 - 1e-12)
    y = np.asarray(y, float)
    return {
        "n_rows": int(len(y)),
        "n_censor_events": int(y.sum()),
        "observed_censor_rate": float(y.mean()),
        "mean_predicted_censor_probability": float(p.mean()),
        "brier_score": float(np.mean((y - p) ** 2)),
        "log_loss": float(-np.mean(y * np.log(p) + (1.0 - y) * np.log1p(-p))),
    }


def _nested_fit_by_batch(
    observed_raw: Sequence[np.ndarray],
    events: np.ndarray,
    folds: np.ndarray,
    batches: Sequence[str],
    policy_start: int,
    ridge_slope: float,
    replicate: int,
) -> tuple[list[np.ndarray], list[dict[str, Any]], list[dict[str, Any]]]:
    """Fit fold-specific preprocessing and cloglog models within each batch."""
    n = len(observed_raw)
    predictions: list[np.ndarray | None] = [None] * n
    fit_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []

    for batch, batch_idx in _batch_indices(batches).items():
        fold_values = sorted(np.unique(folds[batch_idx]))
        for fold_value in fold_values:
            train = batch_idx[folds[batch_idx] != fold_value]
            test = batch_idx[folds[batch_idx] == fold_value]
            scale, tau = _fit_fold_preprocessing(observed_raw, events, train, policy_start)
            transformed = [np.asarray(path, float) / scale - tau for path in observed_raw]
            x_train, y_train = person_period(
                transformed, events, indices=train, policy_start=policy_start
            )
            fit = fit_cloglog_fast(x_train, y_train, ridge_slope=ridge_slope)
            if not fit.success:
                raise RuntimeError(
                    f"Nested fit failed: replicate={replicate}, batch={batch}, "
                    f"fold={fold_value}, message={fit.message}"
                )

            held_y: list[np.ndarray] = []
            held_p: list[np.ndarray] = []
            for i in test:
                i = int(i)
                x_path = transformed[i]
                predictions[i] = fitted_cumhaz_before(
                    x_path, fit.intercept, fit.slope, policy_start=policy_start
                )
                x_rows, y_rows = _unit_person_period(x_path, int(events[i]), policy_start)
                if len(x_rows):
                    eta = np.clip(fit.intercept + fit.slope * x_rows, -25.0, 25.0)
                    p_rows = -np.expm1(-np.exp(eta))
                    held_y.append(y_rows)
                    held_p.append(p_rows)

            fit_rows.append({
                "replicate": replicate,
                "batch": batch,
                "fold": int(fold_value),
                "n_train_units": int(len(train)),
                "n_test_units": int(len(test)),
                "training_prefix_scale": scale,
                "training_prefix_tau70": tau,
                "ridge_slope": float(ridge_slope),
                "intercept": float(fit.intercept),
                "slope": float(fit.slope),
                "n_iter": int(fit.n_iter),
                "objective": float(fit.objective),
                "grad_norm": float(fit.grad_norm),
                "solver_method": str(getattr(fit, "method", "newton")),
                "fit_message": str(fit.message),
                "n_train_person_period_rows": int(len(x_train)),
                "n_train_censor_events": int(y_train.sum()),
            })
            yy = np.concatenate(held_y) if held_y else np.array([], float)
            pp = np.concatenate(held_p) if held_p else np.array([], float)
            calibration_rows.append({
                "replicate": replicate,
                "batch": batch,
                "fold": int(fold_value),
                **_calibration_metrics(yy, pp),
            })

    if any(value is None for value in predictions):
        raise RuntimeError("Nested out-of-fold cumulative-hazard predictions are incomplete")
    return [np.asarray(x, float) for x in predictions], fit_rows, calibration_rows  # type: ignore[arg-type]


def _unit_dominance_rows(
    times: np.ndarray,
    events: np.ndarray,
    cumhaz: Sequence[np.ndarray],
    horizon: float,
    exp_clip: float,
    arm: str,
    batch: str,
    replicate: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    failures = np.unique(times[(events == 1) & (times <= float(horizon))])
    for time_value in failures:
        u = int(round(float(time_value)))
        risk_idx = np.flatnonzero(times >= time_value)
        logw = np.asarray([
            cumhaz[int(i)][min(u, len(cumhaz[int(i)]) - 1)] for i in risk_idx
        ], float)
        weights = np.exp(np.minimum(logw, float(exp_clip)))
        total = float(weights.sum())
        dead = (times[risk_idx] == time_value) & (events[risk_idx] == 1)
        dead_weights = weights[dead]
        dead_total = float(dead_weights.sum())
        rows.append({
            "replicate": replicate,
            "batch": batch,
            "arm": arm,
            "time": float(time_value),
            "n_at_risk": int(len(risk_idx)),
            "n_failures": int(dead.sum()),
            "largest_unit_risk_weight_share": float(weights.max() / total) if total > 0 else float("nan"),
            "largest_failure_unit_share_of_weighted_failures": (
                float(dead_weights.max() / dead_total) if dead_total > 0 else float("nan")
            ),
            "n_exp_clipped": int(np.sum(logw > float(exp_clip))),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matr", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--R", type=int, default=200)
    parser.add_argument("--target-censor", type=float, default=0.40)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20261001)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--baseline-cycles", type=int, default=50)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--ridge-slope", type=float, default=16.0)
    parser.add_argument("--exp-clip", type=float, default=30.0)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--quick-check", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    matr = args.matr.resolve()
    if not matr.is_dir():
        raise FileNotFoundError(matr)
    if args.quick_check:
        args.R = min(args.R, 3)
    if args.R < 1:
        raise ValueError("R must be positive")
    out_dir = (args.out_dir or root / "results" / "matr_nested_preprocessing").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    unit_ids, batches, lifetimes, raw_paths = _load_primary_cohort(
        root, matr, args.baseline_cycles, args.smooth_window
    )
    policy_paths, policy_scale, policy_tau = standardize_policy_paths(
        raw_paths, policy_start=args.baseline_cycles
    )
    folds = make_stratified_folds(batches, n_folds=args.folds, seed=args.fold_seed)
    idx_map = _batch_indices(batches)
    scopes = _batch_scopes(batches, lifetimes)

    lambda_by_batch: dict[str, float] = {}
    expected_by_batch: dict[str, float] = {}
    for batch in PRIMARY_BATCHES:
        batch_paths = _subset(policy_paths, idx_map[batch])
        lam = calibrate_lambda0(
            batch_paths, 1.0, policy_tau, args.target_censor,
            policy_start=args.baseline_cycles,
        )
        lambda_by_batch[batch] = lam
        expected_by_batch[batch] = expected_censor_fraction(
            batch_paths, 1.0, policy_tau, lam, policy_start=args.baseline_cycles
        )

    rng = np.random.default_rng(args.seed)
    replicate_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    dominance_rows: list[dict[str, Any]] = []

    for replicate in range(args.R):
        uniforms = [rng.random(max(len(path) - 1, 0)) for path in policy_paths]
        times, events, observed_policy, oracle = _overlay_by_batch(
            policy_paths, uniforms, batches, 1.0, policy_tau, lambda_by_batch,
            args.baseline_cycles,
        )
        observed_raw = _observed_raw_prefixes(raw_paths, times)

        global_cross, _, _, _ = _fit_by_batch(
            observed_policy, events, folds, batches, args.baseline_cycles,
            ridge_slope=args.ridge_slope,
        )
        nested_cross, nested_fits, nested_calibration = _nested_fit_by_batch(
            observed_raw, events, folds, batches, args.baseline_cycles,
            args.ridge_slope, replicate,
        )
        fit_rows.extend(nested_fits)
        calibration_rows.extend(nested_calibration)

        for scope in scopes:
            batch = str(scope["scope"])
            idx = np.asarray(scope["indices"], int)
            horizon = float(scope["H"])
            truth = float(scope["truth"])
            tt = times[idx]
            ee = events[idx]
            hazards = {
                "oracle_tv_ipcw": _subset(oracle, idx),
                "global_preprocessing_crossfit": _subset(global_cross, idx),
                "nested_preprocessing_crossfit": _subset(nested_cross, idx),
            }
            estimates = {"naive": _estimate_naive(tt, ee, horizon)}
            estimates.update({
                arm: _estimate_weighted(tt, ee, hh, horizon) for arm, hh in hazards.items()
            })
            for arm, estimate in estimates.items():
                signed = 100.0 * (estimate - truth) / truth
                replicate_rows.append({
                    "replicate": replicate,
                    "scope": batch,
                    "n_units": int(len(idx)),
                    "H": horizon,
                    "truth_net_rmst": truth,
                    "arm": arm,
                    "estimate": estimate,
                    "signed_gap_pct": signed,
                    "absolute_gap_pct": abs(signed),
                    "realized_censor_fraction": float(np.mean(ee == 0)),
                })

            checkpoints = [0.25 * horizon, 0.50 * horizon, 0.75 * horizon, horizon]
            for arm, hh in hazards.items():
                for fraction, diagnostic in zip(
                    (0.25, 0.50, 0.75, 1.00),
                    weight_diagnostics(tt, hh, checkpoints, exp_clip=args.exp_clip),
                ):
                    weight_rows.append({
                        "replicate": replicate,
                        "scope": batch,
                        "arm": arm,
                        "checkpoint_fraction_H": fraction,
                        "H": horizon,
                        **diagnostic,
                    })
                for row in weighted_event_diagnostics(
                    tt, ee, hh, horizon=horizon, exp_clip=args.exp_clip
                ):
                    event_rows.append({
                        "replicate": replicate,
                        "scope": batch,
                        "arm": arm,
                        "H": horizon,
                        **row,
                    })
                dominance_rows.extend(_unit_dominance_rows(
                    tt, ee, hh, horizon, args.exp_clip, arm, batch, replicate
                ))
        print(f"completed nested overlay {replicate + 1}/{args.R}", flush=True)

    summary_rows: list[dict[str, Any]] = []
    rdata = pd.DataFrame(replicate_rows)
    for batch in PRIMARY_BATCHES:
        for arm in sorted(rdata["arm"].unique()):
            values = rdata[(rdata["scope"] == batch) & (rdata["arm"] == arm)]
            signed = _summarize(values["signed_gap_pct"])
            absolute = _summarize(values["absolute_gap_pct"])
            summary_rows.append({
                "scope": batch,
                "arm": arm,
                "R": int(len(values)),
                "n_units": int(values["n_units"].iloc[0]),
                "H": float(values["H"].iloc[0]),
                "truth_net_rmst": float(values["truth_net_rmst"].iloc[0]),
                "mean_signed_gap_pct": signed["mean"],
                "mcse_signed_gap_pct": signed["mcse"],
                "mean_absolute_gap_pct": absolute["mean"],
                "mcse_absolute_gap_pct": absolute["mcse"],
                "mean_realized_censor_fraction": float(values["realized_censor_fraction"].mean()),
            })

    paired_rows: list[dict[str, Any]] = []
    pivot = rdata.pivot_table(
        index=["replicate", "scope", "truth_net_rmst"],
        columns="arm", values="signed_gap_pct", aggfunc="first"
    ).reset_index()
    for batch in PRIMARY_BATCHES:
        values = pivot[pivot["scope"] == batch].copy()
        for comparator in ["global_preprocessing_crossfit", "nested_preprocessing_crossfit"]:
            values[f"{comparator}_movement"] = values[comparator] - values["naive"]
            values[f"{comparator}_abs_improvement"] = (
                values["naive"].abs() - values[comparator].abs()
            )
        values["nested_minus_global_signed_error_pp"] = (
            values["nested_preprocessing_crossfit"] - values["global_preprocessing_crossfit"]
        )
        values["nested_minus_global_absolute_error_pp"] = (
            values["nested_preprocessing_crossfit"].abs()
            - values["global_preprocessing_crossfit"].abs()
        )
        for metric in [
            "global_preprocessing_crossfit_movement",
            "global_preprocessing_crossfit_abs_improvement",
            "nested_preprocessing_crossfit_movement",
            "nested_preprocessing_crossfit_abs_improvement",
            "nested_minus_global_signed_error_pp",
            "nested_minus_global_absolute_error_pp",
        ]:
            stat = _summarize(values[metric])
            paired_rows.append({
                "scope": batch,
                "metric": metric,
                "R": int(len(values)),
                "mean": stat["mean"],
                "mcse": stat["mcse"],
                "p2p5": stat["p2p5"],
                "p97p5": stat["p97p5"],
            })

    _write_csv(out_dir / "replicates.csv", replicate_rows)
    _write_csv(out_dir / "estimator_summary.csv", summary_rows)
    _write_csv(out_dir / "paired_summary.csv", paired_rows)
    _write_csv(out_dir / "fold_preprocessing_and_fit.csv", fit_rows)
    _write_csv(out_dir / "heldout_calibration.csv", calibration_rows)
    _write_csv(out_dir / "weight_diagnostics.csv", weight_rows)
    _write_csv(out_dir / "event_time_diagnostics.csv", event_rows)
    _write_csv(out_dir / "unit_dominance.csv", dominance_rows)

    design = {
        "analysis": "MATR fully nested nuisance-preprocessing audit",
        "R": int(args.R),
        "policy": {
            "status": "fixed external synthetic benchmark policy",
            "global_complete_path_scale": policy_scale,
            "global_complete_path_tau70": policy_tau,
            "beta": 1.0,
            "target_replacement_fraction": float(args.target_censor),
            "lambda0_by_batch": lambda_by_batch,
            "expected_replacement_fraction_by_batch": expected_by_batch,
        },
        "nuisance_preprocessing": (
            "Within every batch/fold/overlay, scale and 70th-percentile centering are "
            "estimated from observed training-unit censoring-likelihood rows only; "
            "held-out units do not contribute."
        ),
        "folds": int(args.folds),
        "fold_seed": int(args.fold_seed),
        "overlay_seed": int(args.seed),
        "ridge_slope": float(args.ridge_slope),
        "numerical_exp_clip": float(args.exp_clip),
        "weight_cap": None,
        "unit_ids": unit_ids,
    }
    (out_dir / "design.json").write_text(json.dumps(design, indent=2), encoding="utf-8")
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
