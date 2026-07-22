#!/usr/bin/env python3
"""temperature extension four-batch Tmax-policy endpoint benchmark.

Secondary external-validity analysis for the reliability benchmark.

Design freeze
-------------
* Cohort: the 124 primary MATR cells plus 45 MATR-CLO cells audited in signal audit.
* Health signal and policy driver: causal Tmax deviation from the first-50-cycle
  baseline median, smoothed by a trailing median of width 5.
* Policy: beta=1, common global Tmax scale/threshold, batch-specific lambda0
  calibrated to 40% expected replacement.
* Estimators: naive product-limit, oracle product-limit, oracle HT-RMST, and
  five-fold unit-level cross-fitted TV-IPCW.
* Purpose: secondary signal/transportability evidence. This analysis does not
  replace the IR-driven 124-cell primary analysis.

The script is resumable. Replicate random streams are deterministic functions
of the base seed and replicate index.
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

BATCHES = ("MATR-05-12", "MATR-06-30", "MATR-04-12", "MATR-CLO")
ESTIMATORS = ("naive", "oracle_product_limit", "oracle_ht_rmst", "crossfit_tv_ipcw")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_csv(path: Path) -> list[dict[str, str]]:
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


def finite_trailing_median(x: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(x, float).ravel()
    out = np.full(len(arr), np.nan)
    for j in range(len(arr)):
        w = arr[max(0, j-window+1):j+1]
        w = w[np.isfinite(w)]
        if len(w):
            out[j] = float(np.median(w))
    return out


def build_tmax(values, lifetime: int, baseline_cycles: int = 50,
               smooth_window: int = 5) -> np.ndarray:
    raw = np.asarray(values, float).ravel()
    T = int(lifetime)
    if T < 1:
        raise ValueError("lifetime must be positive")
    baseline = raw[:min(baseline_cycles, len(raw))]
    baseline = baseline[np.isfinite(baseline)]
    if not len(baseline):
        raise ValueError("no finite Tmax baseline")
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
        raise RuntimeError("bad endpoint-aligned Tmax path")
    return out


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        groups.setdefault((str(row["batch"]), str(row["estimator"])), []).append(float(row["bias_pct"]))
    out: list[dict[str, Any]] = []
    for (batch, estimator), values in sorted(groups.items()):
        x = np.asarray(values, float)
        sd = float(x.std(ddof=1)) if len(x) > 1 else 0.0
        out.append({
            "batch": batch,
            "estimator": estimator,
            "n": int(len(x)),
            "mean_bias_pct": float(x.mean()),
            "sd_bias_pct": sd,
            "mcse_bias_pct": float(sd / math.sqrt(len(x))),
            "p2p5_bias_pct": float(np.percentile(x, 2.5)),
            "median_bias_pct": float(np.median(x)),
            "p97p5_bias_pct": float(np.percentile(x, 97.5)),
        })
    return out


def make_fold_rows(ids: list[str], batches: list[str], core, seed: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    folds = core.make_stratified_folds(batches, n_folds=5, seed=seed)
    rows = [{"unit_id": uid, "batch": batch, "fold": int(fold)}
            for uid, batch, fold in zip(ids, batches, folds)]
    return folds, rows


def checkpoint(out: Path, rep_rows, fit_rows, support_rows) -> None:
    write_csv(out / "temperature_extension_replicates.csv", rep_rows)
    write_csv(out / "temperature_extension_fit_diagnostics.csv", fit_rows)
    write_csv(out / "temperature_extension_support_diagnostics.csv", support_rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--matr", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--R", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260725)
    ap.add_argument("--fold-seed", type=int, default=202607251)
    ap.add_argument("--target-censor", type=float, default=0.40)
    ap.add_argument("--baseline-cycles", type=int, default=50)
    ap.add_argument("--smooth-window", type=int, default=5)
    ap.add_argument("--ridge-slope", type=float, default=16.0)
    ap.add_argument("--checkpoint-every", type=int, default=10)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    root = args.root.resolve()
    matr = args.matr.resolve()
    out = (args.out_dir or root / "results" / "matr_temperature").resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.R < 1 or args.checkpoint_every < 1:
        raise ValueError("R and checkpoint-every must be positive")

    required = {
        "audit": root / "code" / "matr_data.py",
        "core": root / "code" / "src" / "matr_ipcw.py",
        "surv": root / "code" / "src" / "_survival.py",
        "signal_audit_units": root / "results" / "signal audit" / "signal_audit_matr_signal_unit_audit.csv",
        "signal_audit_report": root / "results" / "signal audit" / "signal_audit_matr_signal_feasibility_report.json",
    }
    for path in required.values():
        if not path.exists():
            raise FileNotFoundError(path)

    audit = load_module(required["audit"], "temperature_extension_audit")
    core = load_module(required["core"], "temperature_extension_core")
    surv = load_module(required["surv"], "temperature_extension_surv")

    unit_rows = [r for r in read_csv(required["signal_audit_units"])
                 if r.get("signal") == "Tmax" and r.get("cohort_role") in {"primary_124", "optional_clo_45"}]
    expected_counts = {"MATR-05-12": 41, "MATR-06-30": 43, "MATR-04-12": 40, "MATR-CLO": 45}
    counts = {b: sum(r["batch"] == b for r in unit_rows) for b in BATCHES}
    if counts != expected_counts or len(unit_rows) != 169:
        raise RuntimeError(f"expected 169 audited Tmax units, got {len(unit_rows)} with {counts}")

    schema: list[Any] = []
    cells = audit.harmonize(audit.read_raw_cells(matr, schema))
    cell_by_id = {c.unit_id: c for c in cells}

    ids: list[str] = []
    batches: list[str] = []
    life: list[int] = []
    raw_paths: list[np.ndarray] = []
    for row in sorted(unit_rows, key=lambda z: (z["batch"], z["unit_id"])):
        uid = row["unit_id"]
        if uid not in cell_by_id:
            raise KeyError(f"raw MATR cell not found: {uid}")
        T = int(float(row["lifetime"]))
        path = build_tmax(cell_by_id[uid].arrays.get("Tmax", []), T,
                          args.baseline_cycles, args.smooth_window)
        ids.append(uid)
        batches.append(row["batch"])
        life.append(T)
        raw_paths.append(path)

    life_arr = np.asarray(life, float)
    batch_arr = np.asarray(batches, object)
    z_paths, scale, tau = core.standardize_policy_paths(raw_paths, policy_start=args.baseline_cycles)
    folds, fold_rows = make_fold_rows(ids, batches, core, args.fold_seed)
    write_csv(out / "temperature_extension_fold_assignment.csv", fold_rows)

    idx_by_batch = {b: np.flatnonzero(batch_arr == b) for b in BATCHES}
    lambda0 = {
        b: core.calibrate_lambda0([z_paths[i] for i in idx], 1.0, tau,
                                  args.target_censor, policy_start=args.baseline_cycles)
        for b, idx in idx_by_batch.items()
    }

    rep_path = out / "temperature_extension_replicates.csv"
    fit_path = out / "temperature_extension_fit_diagnostics.csv"
    support_path = out / "temperature_extension_support_diagnostics.csv"
    if args.resume and rep_path.exists():
        rep_rows = read_csv(rep_path)
        fit_rows = read_csv(fit_path) if fit_path.exists() else []
        support_rows = read_csv(support_path) if support_path.exists() else []
        counts_by_rep: dict[int, int] = {}
        for row in rep_rows:
            r = int(float(row["replicate"]))
            counts_by_rep[r] = counts_by_rep.get(r, 0) + 1
        complete = {r for r, n in counts_by_rep.items() if n == len(BATCHES)*len(ESTIMATORS)}
        rep_rows = [r for r in rep_rows if int(float(r["replicate"])) in complete]
        fit_rows = [r for r in fit_rows if int(float(r["replicate"])) in complete]
        support_rows = [r for r in support_rows if int(float(r["replicate"])) in complete]
    else:
        rep_rows, fit_rows, support_rows, complete = [], [], [], set()

    for rep in range(args.R):
        if rep in complete:
            print(f"replicate {rep+1}/{args.R} already complete; skipped", flush=True)
            continue
        rng = np.random.default_rng(args.seed + 10000019*rep)
        uniforms = [rng.random(max(len(path)-1, 0)) for path in z_paths]
        times = np.full(len(ids), np.nan)
        events = np.full(len(ids), -1, int)
        observed: list[np.ndarray | None] = [None] * len(ids)
        oracle: list[np.ndarray | None] = [None] * len(ids)

        for batch, idx in idx_by_batch.items():
            tt, ee, oo, cc = core.overlay_from_uniforms(
                [z_paths[i] for i in idx], 1.0, tau, lambda0[batch],
                [uniforms[i] for i in idx], policy_start=args.baseline_cycles)
            times[idx] = tt
            events[idx] = ee
            for local, global_i in enumerate(idx):
                observed[global_i] = oo[local]
                oracle[global_i] = cc[local]

        for batch, idx in idx_by_batch.items():
            H = float(np.median(life_arr[idx]))
            truth = float(np.mean(np.minimum(life_arr[idx], H)))
            t = times[idx]
            e = events[idx]
            obs_b = [np.asarray(observed[i], float) for i in idx]
            oracle_b = [np.asarray(oracle[i], float) for i in idx]
            folds_b = folds[idx]

            tg, sg, _ = surv.km(t, e)
            naive = float(surv.rmrl_from_survival(tg, sg, 0.0, H))
            og, os = core.weighted_product_limit(t, e, oracle_b)
            oracle_pl = float(surv.rmrl_from_survival(og, os, 0.0, H))
            oracle_ht = float(core.ht_ipcw_rmst(t, e, oracle_b, H))
            fitted, fits = core.fit_crossfit_cumhaz(
                obs_b, e, folds_b, policy_start=args.baseline_cycles,
                ridge_slope=args.ridge_slope)
            fg, fs = core.weighted_product_limit(t, e, fitted)
            tv = float(surv.rmrl_from_survival(fg, fs, 0.0, H))

            common = {
                "replicate": rep,
                "batch": batch,
                "n_units": int(len(idx)),
                "H": H,
                "truth": truth,
                "realized_censor": float(np.mean(e == 0)),
            }
            for estimator, value in [
                ("naive", naive),
                ("oracle_product_limit", oracle_pl),
                ("oracle_ht_rmst", oracle_ht),
                ("crossfit_tv_ipcw", tv),
            ]:
                rep_rows.append({
                    **common,
                    "estimator": estimator,
                    "estimate": value,
                    "bias_pct": 100.0*(value-truth)/truth,
                })

            for fold_value, fit in zip(sorted(np.unique(folds_b)), fits):
                fit_rows.append({
                    "replicate": rep,
                    "batch": batch,
                    "fold": int(fold_value),
                    "success": bool(fit.success),
                    "intercept": float(fit.intercept),
                    "slope": float(fit.slope),
                    "objective": float(fit.objective),
                    "grad_norm": float(fit.grad_norm),
                    "method": str(fit.method),
                    "message": str(fit.message),
                    "post_censor_records_used": 0,
                })

            chk = sorted(set([max(args.baseline_cycles+1, int(round(0.5*H))),
                              max(args.baseline_cycles+1, int(round(0.8*H))),
                              max(args.baseline_cycles+1, int(round(H)))]))
            for source, hazards in [("oracle", oracle_b), ("crossfit_tv_ipcw", fitted)]:
                for row in core.weight_diagnostics(t, hazards, chk):
                    support_rows.append({"replicate": rep, "batch": batch,
                                         "source": source, "diagnostic": "checkpoint", **row})
                for row in core.weighted_event_diagnostics(t, e, hazards, horizon=H):
                    support_rows.append({"replicate": rep, "batch": batch,
                                         "source": source, "diagnostic": "event_time", **row})

        if ((rep+1) % args.checkpoint_every == 0) or rep == args.R-1:
            checkpoint(out, rep_rows, fit_rows, support_rows)
        print(f"replicate {rep+1}/{args.R} completed", flush=True)

    summary = summarize(rep_rows)
    write_csv(out / "temperature_extension_summary.csv", summary)
    checkpoint(out, rep_rows, fit_rows, support_rows)

    expected_rep_rows = args.R * len(BATCHES) * len(ESTIMATORS)
    signal_audit_report = json.loads(required["signal_audit_report"].read_text(encoding="utf-8"))
    current_hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in matr.rglob("*")
        if path.is_file() and path.suffix.lower() in {".mat", ".h5", ".hdf5"}
    }
    expected_hash_values = sorted(signal_audit_report.get("raw_file_hashes", {}).values())
    checks = {
        "expected_169_units": len(ids) == 169 and counts == expected_counts,
        "all_expected_replicate_rows": len(rep_rows) == expected_rep_rows,
        "all_estimates_finite": all(np.isfinite(float(r["estimate"])) for r in rep_rows),
        "all_fit_gradients_finite": all(np.isfinite(float(r["grad_norm"])) for r in fit_rows),
        "all_fits_successful": all(str(r["success"]).lower() in {"true", "1"} for r in fit_rows),
        "no_post_censor_records": all(int(float(r["post_censor_records_used"])) == 0 for r in fit_rows),
        "five_folds_per_batch": all(len(set(folds[idx].tolist())) == 5 for idx in idx_by_batch.values()),
        "raw_hash_values_match_signal audit": sorted(current_hashes.values()) == expected_hash_values,
    }
    report = {
        "analysis": "temperature extension_Tmax_four_batch_known_truth_benchmark",
        "status": "PASS" if all(checks.values()) else "REVIEW_REQUIRED",
        "python": platform.python_version(),
        "design": {
            "R": args.R,
            "seed": args.seed,
            "fold_seed": args.fold_seed,
            "cohort": "124 primary MATR cells + 45 MATR-CLO cells",
            "n_units": 169,
            "batch_counts": counts,
            "policy": "Tmax-driven beta=1; global scale and p70 threshold; batch-specific lambda0 calibrated to 40% expected replacement",
            "baseline_cycles": args.baseline_cycles,
            "smooth_window": args.smooth_window,
            "ridge_slope": args.ridge_slope,
            "inferential_role": "secondary fixed-cohort endpoint-benchmark overlay comparison; not joint-redesign inference",
        },
        "scale": float(scale),
        "tau": float(tau),
        "lambda0": {k: float(v) for k, v in lambda0.items()},
        "checks": checks,
        "summary": summary,
    }
    (out / "temperature_extension_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("temperature extension TMAX FOUR-BATCH BENCHMARK COMPLETED")
    print(f"status={report['status']}")
    for row in summary:
        if row["estimator"] in {"naive", "crossfit_tv_ipcw"}:
            print(f"{row['batch']:<12} {row['estimator']:<24} bias={row['mean_bias_pct']:+.3f}% mcse={row['mcse_bias_pct']:.3f}")
    print(f"out_dir={out}")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
