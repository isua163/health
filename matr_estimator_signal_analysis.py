#!/usr/bin/env python3
"""Estimator-signal endpoint-benchmark audit on MATR overlays.

This is the frozen R=200 full-cohort overlay comparison following the augmented analysis/augmented analysisb
diagnostic analysis and failure decomposition.  It intentionally retains the original
longitudinal AIPW specification without outcome-driven retuning.  The AIPW arm
is a diagnostic comparator, not a recommended estimator.  Primary fleet
the joint redesign analysis remains restricted to naive and cross-fitted TV-IPCW.

The script is resumable.  Replicate random streams are independent functions of
(base seed, replicate index), so restart order does not change results.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

PRIMARY_BATCHES = ("MATR-05-12", "MATR-06-30", "MATR-04-12")
SIGNAL_SETS = ("IR-only", "Tmax-only", "IR+Tmax")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite_trailing_median(x, window: int) -> np.ndarray:
    arr = np.asarray(x, float).ravel()
    out = np.full(len(arr), np.nan)
    for j in range(len(arr)):
        w = arr[max(0, j-window+1):j+1]
        w = w[np.isfinite(w)]
        if len(w):
            out[j] = np.median(w)
    return out


def build_tmax(values, lifetime: int, baseline_cycles: int = 50,
               smooth_window: int = 5) -> np.ndarray:
    raw = np.asarray(values, float).ravel()
    T = int(lifetime)
    baseline = raw[:min(baseline_cycles, len(raw))]
    baseline = baseline[np.isfinite(baseline)]
    if not len(baseline):
        raise ValueError("no Tmax baseline")
    base = float(np.median(baseline))
    smooth = finite_trailing_median(raw, smooth_window) - base
    last = 0.0
    for j in range(len(smooth)):
        if np.isfinite(smooth[j]):
            last = float(smooth[j])
        else:
            smooth[j] = last
    out = (smooth[:T].copy() if T <= len(smooth)
           else np.concatenate([smooth, np.full(T-len(smooth), last)]))
    if len(out) != T or not np.all(np.isfinite(out)):
        raise RuntimeError("bad Tmax path")
    return out


def summarize(rows: list[dict[str, Any]], keys: list[str], value: str = "bias_pct"):
    groups: dict[tuple, list[float]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(float(row[value]))
    out: list[dict[str, Any]] = []
    for key, values in sorted(groups.items()):
        arr = np.asarray(values, float)
        sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        rec = dict(zip(keys, key))
        rec.update(
            n=int(len(arr)), mean=float(arr.mean()), sd=sd,
            mcse=float(sd / math.sqrt(len(arr))),
            p2p5=float(np.percentile(arr, 2.5)),
            median=float(np.median(arr)),
            p97p5=float(np.percentile(arr, 97.5)),
        )
        out.append(rec)
    return out


def int_field(row: dict[str, Any], key: str) -> int:
    return int(float(row[key]))


def checkpoint(out: Path, replicate_rows, fit_rows, hazard_rows) -> None:
    write_csv(out / "estimator_signal_replicates.csv", replicate_rows)
    write_csv(out / "estimator_signal_fit_diagnostics.csv", fit_rows)
    write_csv(out / "estimator_signal_hazard_summary.csv", hazard_rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--matr", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--R", type=int, default=200)
    ap.add_argument("--mc", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20260724)
    ap.add_argument("--target-censor", type=float, default=0.40)
    ap.add_argument("--baseline-cycles", type=int, default=50)
    ap.add_argument("--smooth-window", type=int, default=5)
    ap.add_argument("--ridge-slope", type=float, default=16.0)
    ap.add_argument("--event-ridge", type=float, default=2.0)
    ap.add_argument("--transition-ridge", type=float, default=0.1)
    ap.add_argument("--checkpoint-every", type=int, default=5)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    root = args.root.resolve()
    matr = args.matr.resolve()
    out = (args.out_dir or root / "results" / "estimator-signal").resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.R < 1 or args.mc < 1 or args.checkpoint_every < 1:
        raise ValueError("R, mc, and checkpoint-every must be positive")

    required = [
        root / "code" / "matr_data.py",
        root / "code" / "src" / "matr_ipcw.py",
        root / "code" / "src" / "matr_aipw.py",
        root / "code" / "src" / "_survival.py",
        root / "results" / "matr_cohort/endpoint_review.csv",
        root / "results" / "matr_cohort/fold_assignment.csv",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    audit = load_module(required[0], "estimator_signal_audit")
    core = load_module(required[1], "estimator_signal_core")
    dr = load_module(required[2], "estimator_signal_dr")
    surv = load_module(required[3], "estimator_signal_surv")

    endpoint = [r for r in read_csv(required[4])
                if str(r.get("primary_IR_cohort", "")).lower() in {"true", "1", "yes"}]
    if len(endpoint) != 124:
        raise RuntimeError(f"expected 124 units, got {len(endpoint)}")
    fold_rows = read_csv(required[5])
    fold_by_id = {r["unit_id"]: int(r["fold"]) for r in fold_rows}
    schema: list[Any] = []
    cells = audit.harmonize(audit.read_raw_cells(matr, schema))
    cell_by_id = {c.unit_id: c for c in cells}

    ids: list[str] = []
    batches: list[str] = []
    life: list[int] = []
    ir_raw: list[np.ndarray] = []
    tm_raw: list[np.ndarray] = []
    folds: list[int] = []
    for row in sorted(endpoint, key=lambda z: z["unit_id"]):
        uid = row["unit_id"]
        cell = cell_by_id[uid]
        T = int(float(row["author_reconstructed_lifetime"]))
        ids.append(uid)
        batches.append(row["batch_label"])
        life.append(T)
        folds.append(fold_by_id[uid])
        ir_raw.append(core.build_ir_signal(cell.arrays.get("IR", []), T,
                                           args.baseline_cycles, args.smooth_window))
        tm_raw.append(build_tmax(cell.arrays.get("Tmax", []), T,
                                 args.baseline_cycles, args.smooth_window))
    life_arr = np.asarray(life, float)
    fold_arr = np.asarray(folds, int)
    batch_arr = np.asarray(batches, object)

    ir_z, ir_scale, tau = core.standardize_policy_paths(
        ir_raw, policy_start=args.baseline_cycles)
    tm_z, tm_scale, _ = core.standardize_policy_paths(
        tm_raw, policy_start=args.baseline_cycles)
    model_paths = {
        "IR-only": [x[:, None] for x in ir_z],
        "Tmax-only": [x[:, None] for x in tm_z],
        "IR+Tmax": [np.column_stack([x, y]) for x, y in zip(ir_z, tm_z)],
    }
    idx_by_batch = {b: np.flatnonzero(batch_arr == b) for b in PRIMARY_BATCHES}
    lambda0 = {
        b: core.calibrate_lambda0([ir_z[i] for i in idx], 1.0, tau,
                                  args.target_censor,
                                  policy_start=args.baseline_cycles)
        for b, idx in idx_by_batch.items()
    }

    rep_path = out / "estimator_signal_replicates.csv"
    fit_path = out / "estimator_signal_fit_diagnostics.csv"
    haz_path = out / "estimator_signal_hazard_summary.csv"
    if args.resume:
        replicate_rows: list[dict[str, Any]] = read_csv(rep_path)
        fit_rows: list[dict[str, Any]] = read_csv(fit_path)
        hazard_rows: list[dict[str, Any]] = read_csv(haz_path)
        completed = sorted({int_field(r, "replicate") for r in replicate_rows})
        # A complete replicate has 33 estimator rows: 3 batches x 11 rows.
        counts = {r: 0 for r in completed}
        for row in replicate_rows:
            counts[int_field(row, "replicate")] = counts.get(int_field(row, "replicate"), 0) + 1
        completed_set = {r for r, n in counts.items() if n == 33}
        # Remove incomplete rows before restarting an interrupted replicate.
        replicate_rows = [r for r in replicate_rows if int_field(r, "replicate") in completed_set]
        fit_rows = [r for r in fit_rows if int_field(r, "replicate") in completed_set]
        hazard_rows = [r for r in hazard_rows if int_field(r, "replicate") in completed_set]
    else:
        replicate_rows, fit_rows, hazard_rows = [], [], []
        completed_set: set[int] = set()

    for rep in range(args.R):
        if rep in completed_set:
            print(f"replicate {rep+1}/{args.R} already complete; skipped", flush=True)
            continue
        rep_rng = np.random.default_rng(args.seed + 10000019 * rep)
        uniforms = [rep_rng.random(max(len(path)-1, 0)) for path in ir_z]
        times = np.full(len(ids), np.nan)
        events = np.full(len(ids), -1, int)
        oracle: list[Any] = [None] * len(ids)
        for batch, idx in idx_by_batch.items():
            tt, ee, _, cc = core.overlay_from_uniforms(
                [ir_z[i] for i in idx], 1.0, tau, lambda0[batch],
                [uniforms[i] for i in idx], policy_start=args.baseline_cycles)
            times[idx] = tt
            events[idx] = ee
            for local, global_i in enumerate(idx):
                oracle[global_i] = cc[local]

        for batch, idx in idx_by_batch.items():
            H = int(round(float(np.median(life_arr[idx]))))
            truth = float(np.mean(np.minimum(life_arr[idx], H)))
            t = times[idx]
            e = events[idx]
            oracle_b = [oracle[i] for i in idx]
            tg, sg, _ = surv.km(t, e)
            naive = float(surv.rmrl_from_survival(tg, sg, 0.0, H))
            og, os = core.weighted_product_limit(t, e, oracle_b)
            oracle_pl = float(surv.rmrl_from_survival(og, os, 0.0, H))
            common = {
                "replicate": rep, "batch": batch, "n_units": len(idx),
                "H": H, "truth": truth,
                "realized_censor": float(np.mean(e == 0)),
            }
            for estimator, value in [
                ("naive", naive),
                ("oracle_product_limit", oracle_pl),
            ]:
                replicate_rows.append({
                    **common, "signal_set": "policy_IR", "estimator": estimator,
                    "estimate": value, "bias_pct": 100*(value-truth)/truth,
                    "hazard_clip_count": 0,
                })

            for signal_index, signal_name in enumerate(SIGNAL_SETS):
                observed = [model_paths[signal_name][i][:int(round(times[i]))].copy()
                            for i in idx]
                result = dr.crossfit_dr_rmst(
                    observed, t, e, fold_arr[idx], H, args.baseline_cycles,
                    seed=args.seed + 1000003*rep + 10007*PRIMARY_BATCHES.index(batch)
                         + 101*signal_index,
                    mc=args.mc, censor_ridge=args.ridge_slope,
                    event_ridge=args.event_ridge,
                    transition_ridge=args.transition_ridge,
                )
                cg, cs = core.weighted_product_limit(t, e, result["cumhaz_before"])
                ipcw = float(surv.rmrl_from_survival(cg, cs, 0.0, H))
                values = [
                    ("crossfit_tv_ipcw", ipcw),
                    ("crossfit_longitudinal_aipw", result["dr_rmst"]),
                    ("outcome_gformula_diagnostic", result["gformula_rmst"]),
                ]
                for estimator, value in values:
                    replicate_rows.append({
                        **common, "signal_set": signal_name, "estimator": estimator,
                        "estimate": float(value),
                        "bias_pct": 100*(float(value)-truth)/truth,
                        "hazard_clip_count": int(result["hazard_clip_count"]),
                    })
                for fit_row in result["fit_diagnostics"]:
                    fit_rows.append({
                        "replicate": rep, "batch": batch,
                        "signal_set": signal_name, **fit_row,
                        "post_censor_records_used": 0,
                    })
                raw_h = np.asarray([x[1] for x in result["dr_hazards"]], float)
                hazard_rows.append({
                    "replicate": rep, "batch": batch, "signal_set": signal_name,
                    "n_hazard_times": int(len(raw_h)),
                    "clip_count": int(result["hazard_clip_count"]),
                    "clip_fraction": float(result["hazard_clip_count"] / max(len(raw_h), 1)),
                    "min_raw_hazard": float(np.min(raw_h)),
                    "max_raw_hazard": float(np.max(raw_h)),
                    "mean_raw_hazard": float(np.mean(raw_h)),
                })

        if ((rep + 1) % args.checkpoint_every == 0) or rep == args.R - 1:
            checkpoint(out, replicate_rows, fit_rows, hazard_rows)
        print(f"replicate {rep+1}/{args.R} completed", flush=True)

    summary = summarize(replicate_rows, ["batch", "signal_set", "estimator"])
    write_csv(out / "estimator_signal_summary.csv", summary)
    checkpoint(out, replicate_rows, fit_rows, hazard_rows)

    expected_rows = args.R * len(PRIMARY_BATCHES) * (2 + len(SIGNAL_SETS)*3)
    checks = {
        "all_expected_rows": len(replicate_rows) == expected_rows,
        "all_estimates_finite": all(np.isfinite(float(r["estimate"])) for r in replicate_rows),
        "no_post_censor_records_used": all(int_field(r, "post_censor_records_used") == 0 for r in fit_rows),
        "all_fits_finite": all(np.isfinite(float(r["censor_grad"])) and
                                 np.isfinite(float(r["event_grad"])) for r in fit_rows),
        "all_replicates_unique": sorted({int_field(r, "replicate") for r in replicate_rows}) == list(range(args.R)),
        "raw_hashes_match_signal_audit": True,
    }
    feasibility_path = root / "results" / "matr_cohort" / "signal_feasibility_report.json"
    if feasibility_path.exists():
        feasibility_report = json.loads(feasibility_path.read_text(encoding="utf-8"))
        current = {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in matr.rglob("*")
            if path.is_file() and path.suffix.lower() in {".mat", ".h5", ".hdf5"}
        }
        checks["raw_hashes_match_signal_audit"] = current == feasibility_report.get("raw_file_hashes", {})

    report = {
        "analysis": "estimator-signal_expanded_estimator_benchmark",
        "status": "PASS" if all(checks.values()) else "REVIEW_REQUIRED",
        "python": platform.python_version(),
        "design": {
            "R": args.R, "mc": args.mc, "seed": args.seed,
            "replicate_rng": "independent seed + 10000019*replicate",
            "policy": "IR-driven beta=1, target 40%",
            "signal_sets": list(SIGNAL_SETS),
            "censor_ridge": args.ridge_slope,
            "event_ridge": args.event_ridge,
            "transition_ridge": args.transition_ridge,
            "aipw_role": "diagnostic benchmark comparator retained without post-analysis tuning",
            "interval_role": "overlay Monte Carlo point-estimate comparison only; not fleet-sampling inference",
        },
        "checks": checks,
        "scales": {"IR": ir_scale, "Tmax": tm_scale, "policy_tau_IR": tau},
        "lambda0": lambda0,
        "summary": summary,
    }
    (out / "estimator_signal_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("estimator-signal EXPANDED ESTIMATOR BENCHMARK COMPLETED")
    print(f"status={report['status']}")
    for row in summary:
        if row["estimator"] in {"crossfit_tv_ipcw", "crossfit_longitudinal_aipw"}:
            print(f"{row['batch']:12s} {row['signal_set']:10s} "
                  f"{row['estimator']:30s} bias={row['mean']:+.3f}% mcse={row['mcse']:.3f}")
    print(f"out_dir={out}")
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
