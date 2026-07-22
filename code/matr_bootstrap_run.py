#!/usr/bin/env python3
"""Run the deterministic MATR joint cell-and-policy redesign analysis.

Each outer cell resample re-applies the prespecified benchmark construction:
the global signal scale and threshold and each batch-specific policy baseline
are recalibrated, while the restriction horizons remain fixed. The output is a
sample-adaptive redesign sensitivity, not a fixed-policy or fleet-superpopulation
bootstrap confidence interval. Failed fits remain explicit rows.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

PRIMARY_BATCHES = ("MATR-05-12", "MATR-06-30", "MATR-04-12")
PRIMARY_ARMS = ("naive", "oracle_tv_ipcw", "crossfit_tv_ipcw")
EXPECTED_BATCH_SIZES = {"MATR-05-12": 41, "MATR-06-30": 43, "MATR-04-12": 40}
REQUIRED_OUTER_COLUMNS = {
    "outer_b", "batch_label", "source_unit_id", "source_lifetime",
    "multiplicity", "crossfit_fold", "fold_total_positions",
    "fold_unique_source_units",
}
REQUIRED_SEED_COLUMNS = {
    "outer_b", "batch_label", "beta", "inner_r", "overlay_seed_uint64",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    if not fields:
        raise ValueError("refusing to write an empty CSV")
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def _load_primary_engine_module(root: Path):
    path = root / "code" / "matr_primary_analysis.py"
    if not path.exists():
        raise FileNotFoundError(f"primary engine script is missing: {path}")
    name = "matr_primary_engine_ipcw_engine_for_bootstrap"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import primary engine: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_core(root: Path):
    repo = root / "code"
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from src.matr_ipcw import (  # type: ignore
        calibrate_lambda0,
        exact_crude_rmst,
        expected_censor_fraction,
        fit_crossfit_cumhaz,
        overlay_from_uniforms,
        standardize_policy_paths,
        weight_diagnostics,
        weighted_event_diagnostics,
        weighted_product_limit,
    )
    return {
        "calibrate_lambda0": calibrate_lambda0,
        "exact_crude_rmst": exact_crude_rmst,
        "expected_censor_fraction": expected_censor_fraction,
        "fit_crossfit_cumhaz": fit_crossfit_cumhaz,
        "overlay_from_uniforms": overlay_from_uniforms,
        "standardize_policy_paths": standardize_policy_paths,
        "weight_diagnostics": weight_diagnostics,
        "weighted_event_diagnostics": weighted_event_diagnostics,
        "weighted_product_limit": weighted_product_limit,
    }


def _validate_manifest_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def _validate_manifests(outer: pd.DataFrame, seeds: pd.DataFrame) -> None:
    _validate_manifest_columns(outer, REQUIRED_OUTER_COLUMNS, "outer manifest")
    _validate_manifest_columns(seeds, REQUIRED_SEED_COLUMNS, "seed manifest")
    if outer.duplicated(["outer_b", "batch_label", "source_unit_id"]).any():
        raise ValueError("outer manifest contains duplicated outer/batch/source rows")
    if seeds.duplicated(["outer_b", "batch_label", "beta", "inner_r"]).any():
        raise ValueError("seed manifest contains duplicated outer/batch/beta/inner rows")
    if (pd.to_numeric(outer["multiplicity"], errors="raise") < 1).any():
        raise ValueError("outer multiplicities must be positive")
    if (pd.to_numeric(outer["crossfit_fold"], errors="raise") < 0).any():
        raise ValueError("crossfit folds must be non-negative")
    for outer_b, group in outer.groupby("outer_b", sort=True):
        labels = set(group["batch_label"].astype(str))
        if labels != set(PRIMARY_BATCHES):
            raise ValueError(f"outer {outer_b} has batch labels {sorted(labels)}")
        totals = group.groupby("batch_label")["multiplicity"].sum().astype(int).to_dict()
        if totals != EXPECTED_BATCH_SIZES:
            raise ValueError(f"outer {outer_b} position totals differ: {totals}")
        for batch, gb in group.groupby("batch_label"):
            folds = sorted(pd.to_numeric(gb["crossfit_fold"], errors="raise").astype(int).unique())
            if folds != [0, 1, 2, 3, 4]:
                raise ValueError(f"outer {outer_b}, {batch}: folds are {folds}, expected 0..4")
    seed_values = pd.to_numeric(seeds["overlay_seed_uint64"], errors="raise")
    if seed_values.duplicated().any():
        raise ValueError("overlay seeds are not globally unique")


def _finite_min(values: Iterable[float]) -> float:
    x = np.asarray(list(values), float)
    x = x[np.isfinite(x)]
    return float(np.min(x)) if len(x) else float("nan")


def _finite_max(values: Iterable[float]) -> float:
    x = np.asarray(list(values), float)
    x = x[np.isfinite(x)]
    return float(np.max(x)) if len(x) else float("nan")


def _naive_rmst(primary_engine: Any, times: np.ndarray, events: np.ndarray, horizon: float) -> float:
    return float(primary_engine._estimate_naive(times, events, horizon))


def _weighted_rmst(primary_engine: Any, times: np.ndarray, events: np.ndarray,
                   cumhaz: Sequence[np.ndarray], horizon: float) -> float:
    return float(primary_engine._estimate_weighted(times, events, list(cumhaz), horizon))


def _support_diagnostics(core: dict[str, Any], times: np.ndarray, events: np.ndarray,
                         cumhaz: list[np.ndarray], horizon: float, exp_clip: float) -> dict[str, Any]:
    checkpoints = [0.25 * horizon, 0.50 * horizon, 0.75 * horizon, horizon]
    wd = core["weight_diagnostics"](times, cumhaz, checkpoints, exp_clip=exp_clip)
    ed = core["weighted_event_diagnostics"](
        times, events, cumhaz, horizon=horizon, exp_clip=exp_clip
    )
    max_weight = _finite_max(
        [float(r.get("max_weight", np.nan)) for r in wd]
        + [float(r.get("max_weight", np.nan)) for r in ed]
    )
    min_ess = _finite_min(
        [float(r.get("ess_over_n_at_risk", np.nan)) for r in wd]
        + [float(r.get("ess_over_n_at_risk", np.nan)) for r in ed]
    )
    return {
        "max_weight": max_weight,
        "min_ess_over_risk": min_ess,
        "max_weighted_hazard_increment": _finite_max(
            float(r.get("weighted_hazard_increment", np.nan)) for r in ed
        ),
        "min_n_at_risk_at_failure": int(min(
            [int(r["n_at_risk"]) for r in ed], default=0
        )),
        "exp_clipping": bool(any(
            int(r.get("n_exp_clipped", 0)) > 0 for r in wd + ed
        )),
    }


def _failure_rows(outer_b: int, batches: Sequence[str], betas: Sequence[float],
                  inner_values: Sequence[int], error: BaseException,
                  step: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for batch in batches:
        for beta in betas:
            for inner_r in inner_values:
                for arm in PRIMARY_ARMS:
                    rows.append({
                        "outer_b": int(outer_b), "batch_label": str(batch),
                        "beta": float(beta), "inner_r": int(inner_r), "arm": arm,
                        "estimate": np.nan, "truth_net_rmst": np.nan,
                        "fit_success": False, "error_type": type(error).__name__,
                        "error_message": str(error), "failure_step": step,
                    })
    return rows


def _existing_keys(path: Path) -> tuple[list[dict[str, Any]], set[tuple[int, str, float, int, str]]]:
    if not path.exists():
        return [], set()
    df = pd.read_csv(path)
    required = {"outer_b", "batch_label", "beta", "inner_r", "arm"}
    if not required.issubset(df.columns):
        raise ValueError(f"existing output has an incompatible schema: {path}")
    rows = df.to_dict(orient="records")
    keys = {
        (int(r["outer_b"]), str(r["batch_label"]), float(r["beta"]), int(r["inner_r"]), str(r["arm"]))
        for r in rows
    }
    return rows, keys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--matr", required=True, type=Path,
                        help="Directory containing the four MATR .mat files directly")
    parser.add_argument("--outer-manifest", type=Path, default=None)
    parser.add_argument("--seed-manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--outer-start", type=int, default=0)
    parser.add_argument("--outer-stop", type=int, default=2,
                        help="Exclusive outer replicate stop")
    parser.add_argument("--inner-start", type=int, default=1)
    parser.add_argument("--inner-stop", type=int, default=2,
                        help="Inclusive inner overlay stop")
    parser.add_argument("--beta", type=float, nargs="+", default=[0.0, 1.0])
    parser.add_argument("--exp-clip", type=float, default=30.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=10,
                        help="Atomically save after this many completed outer replicates")
    parser.add_argument("--input-check-only", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    root = args.root.expanduser().resolve()
    matr = args.matr.expanduser().resolve()
    results = root / "results" / "matr_bootstrap"
    validation = root / "validation"
    outer_path = (args.outer_manifest or results / "outer_manifest.csv").expanduser().resolve()
    seed_path = (args.seed_manifest or results / "overlay_seed_manifest.csv").expanduser().resolve()
    output = (args.output or results / "replicate_results.csv").expanduser().resolve()
    report_path = (args.report or validation / "matr_bootstrap_run_report.json").expanduser().resolve()
    design_path = root / "results" / "matr_primary" / "analysis_design.json"

    for path, label in ((root, "root"), (matr, "MATR directory")):
        if not path.is_dir():
            raise FileNotFoundError(f"{label} not found: {path}")
    for path in (outer_path, seed_path, design_path):
        if not path.exists():
            raise FileNotFoundError(path)
    mat_files = sorted(matr.glob("*.mat"))
    if len(mat_files) < 3:
        raise FileNotFoundError(f"expected MATR .mat files directly under {matr}; found {len(mat_files)}")
    if args.outer_start < 0 or args.outer_stop <= args.outer_start:
        raise ValueError("outer range is invalid")
    if args.inner_start < 1 or args.inner_stop < args.inner_start:
        raise ValueError("inner range is invalid")
    if args.checkpoint_every < 1:
        raise ValueError("checkpoint-every must be positive")

    design = json.loads(design_path.read_text(encoding="utf-8"))
    if str(design.get("status", "")) != "frozen submission design":
        raise RuntimeError("primary analysis design is not frozen")
    baseline_cycles = int(design["baseline_cycles"])
    smooth_window = int(design["smooth_window"])
    ridge_slope = float(design["ridge_slope"])
    target_censor = float(design["target_censor_fraction_within_each_batch"])
    horizons = {str(k): float(v) for k, v in design["horizons"].items()}
    if ridge_slope != 16.0 or baseline_cycles != 50 or smooth_window != 5:
        raise RuntimeError("frozen primary design no longer matches ridge/run-in/smoothing contract")
    betas = tuple(float(x) for x in args.beta)
    if any(beta not in (0.0, 1.0) for beta in betas):
        raise ValueError("bootstrap runner accepts only beta values 0 and 1")

    outer = pd.read_csv(outer_path)
    seeds = pd.read_csv(seed_path, dtype={"overlay_seed_uint64": "uint64"})
    _validate_manifests(outer, seeds)
    selected_outer = list(range(int(args.outer_start), int(args.outer_stop)))
    selected_inner = list(range(int(args.inner_start), int(args.inner_stop) + 1))
    available_outer = set(pd.to_numeric(outer["outer_b"], errors="raise").astype(int))
    if not set(selected_outer).issubset(available_outer):
        raise ValueError("requested outer range is not available in the manifest")
    seed_subset = seeds[
        seeds["outer_b"].astype(int).isin(selected_outer)
        & seeds["inner_r"].astype(int).isin(selected_inner)
        & seeds["beta"].astype(float).isin(betas)
    ]
    expected_seed_rows = len(selected_outer) * len(PRIMARY_BATCHES) * len(betas) * len(selected_inner)
    if len(seed_subset) != expected_seed_rows:
        raise ValueError(f"seed subset has {len(seed_subset)} rows; expected {expected_seed_rows}")

    primary_engine = _load_primary_engine_module(root)
    core = _load_core(root)
    unit_ids, base_batches, base_lifetimes, base_raw_paths = primary_engine._load_primary_cohort(
        root, matr, baseline_cycles, smooth_window
    )
    base_index = {str(unit_id): i for i, unit_id in enumerate(unit_ids)}
    if len(base_index) != 124:
        raise RuntimeError(f"base cohort has {len(base_index)} unique units, expected 124")
    unknown = sorted(set(outer["source_unit_id"].astype(str)) - set(base_index))
    if unknown:
        raise RuntimeError(f"outer manifest references unknown source units: {unknown[:10]}")

    input_check = {
        "analysis": "MATR joint cell-and-policy redesign input check",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "root": str(root), "matr": str(matr),
        "mat_files": [{"name": p.name, "size_bytes": p.stat().st_size} for p in mat_files],
        "outer_range": [selected_outer[0], selected_outer[-1]],
        "inner_range": [selected_inner[0], selected_inner[-1]],
        "betas": list(betas), "ridge_slope": ridge_slope,
        "baseline_cycles": baseline_cycles, "smooth_window": smooth_window,
        "target_censor": target_censor, "horizons": horizons,
        "input_sha256": {
            str(design_path.relative_to(root)): _sha256(design_path),
            str(outer_path.relative_to(root)): _sha256(outer_path),
            str(seed_path.relative_to(root)): _sha256(seed_path),
        },
        "input_check_pass": True,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if args.input_check_only:
        input_check["status"] = "input_check_complete_heavy_compute_not_started"
        input_check["runtime_seconds"] = time.perf_counter() - started
        report_path.write_text(json.dumps(_jsonable(input_check), indent=2, ensure_ascii=False), encoding="utf-8")
        print("bootstrap B4000 RUNNER INPUT CHECK COMPLETED")
        print("status=PASS")
        print(f"  base cohort: {len(base_index)} units; MATR files: {len(mat_files)}")
        print(f"  report: {report_path}")
        return 0

    if output.exists() and args.overwrite:
        output.unlink()
    all_rows, done_keys = _existing_keys(output)
    if output.exists() and not args.overwrite:
        print(f"Resuming existing output with {len(all_rows)} rows: {output}")

    seed_lookup = {
        (int(r.outer_b), str(r.batch_label), float(r.beta), int(r.inner_r)): int(r.overlay_seed_uint64)
        for r in seed_subset.itertuples(index=False)
    }
    completed_outer: list[int] = []
    outer_failures: list[dict[str, Any]] = []
    completed_since_checkpoint = 0

    for outer_b in selected_outer:
        outer_started = time.perf_counter()
        expected_outer_keys = {
            (outer_b, batch, beta, inner_r, arm)
            for batch in PRIMARY_BATCHES for beta in betas
            for inner_r in selected_inner for arm in PRIMARY_ARMS
        }
        if expected_outer_keys.issubset(done_keys):
            print(f"outer_b={outer_b}: already complete; skipped")
            completed_outer.append(outer_b)
            continue
        try:
            ob = outer[outer["outer_b"].astype(int) == outer_b].copy()
            ob["multiplicity"] = pd.to_numeric(ob["multiplicity"], errors="raise").astype(int)
            ob["crossfit_fold"] = pd.to_numeric(ob["crossfit_fold"], errors="raise").astype(int)
            ob = ob.sort_values(["batch_label", "source_unit_id"], kind="mergesort")

            expanded_ids: list[str] = []
            expanded_sources: list[str] = []
            expanded_batches: list[str] = []
            expanded_lifetimes: list[float] = []
            expanded_raw_paths: list[np.ndarray] = []
            expanded_folds: list[int] = []
            for row in ob.itertuples(index=False):
                source = str(row.source_unit_id)
                base_i = base_index[source]
                lifetime = float(base_lifetimes[base_i])
                if abs(lifetime - float(row.source_lifetime)) > 1e-8:
                    raise RuntimeError(
                        f"lifetime mismatch for {source}: base={lifetime}, manifest={row.source_lifetime}"
                    )
                if str(base_batches[base_i]) != str(row.batch_label):
                    raise RuntimeError(f"batch mismatch for {source}")
                for copy_i in range(int(row.multiplicity)):
                    expanded_ids.append(f"b{outer_b}:{source}:copy{copy_i + 1}")
                    expanded_sources.append(source)
                    expanded_batches.append(str(row.batch_label))
                    expanded_lifetimes.append(lifetime)
                    expanded_raw_paths.append(np.asarray(base_raw_paths[base_i], float).copy())
                    expanded_folds.append(int(row.crossfit_fold))
            if len(expanded_ids) != 124:
                raise RuntimeError(f"outer {outer_b} expanded to {len(expanded_ids)} positions, expected 124")

            z_paths, signal_scale, tau = core["standardize_policy_paths"](
                expanded_raw_paths, policy_start=baseline_cycles
            )
            labels = np.asarray(expanded_batches, object)
            life = np.asarray(expanded_lifetimes, float)
            folds = np.asarray(expanded_folds, int)
            source_arr = np.asarray(expanded_sources, object)

            batch_context: dict[str, dict[str, Any]] = {}
            for batch in PRIMARY_BATCHES:
                idx = np.flatnonzero(labels == batch)
                if len(idx) != EXPECTED_BATCH_SIZES[batch]:
                    raise RuntimeError(f"outer {outer_b}, {batch}: expanded n={len(idx)}")
                unique_by_fold = [len(set(source_arr[idx][folds[idx] == f].tolist())) for f in range(5)]
                batch_rows = ob[ob["batch_label"].astype(str) == batch]
                batch_context[batch] = {
                    "idx": idx,
                    "paths": [z_paths[int(i)] for i in idx],
                    "folds": folds[idx],
                    "lifetimes": life[idx],
                    "H": horizons[batch],
                    "truth": float(np.mean(np.minimum(life[idx], horizons[batch]))),
                    "n_unique": int(batch_rows["source_unit_id"].nunique()),
                    "max_multiplicity": int(batch_rows["multiplicity"].max()),
                    "min_unique_per_fold": int(min(unique_by_fold)),
                }

            policy: dict[tuple[str, float], dict[str, float]] = {}
            for beta in betas:
                for batch in PRIMARY_BATCHES:
                    ctx = batch_context[batch]
                    lam = float(core["calibrate_lambda0"](
                        ctx["paths"], beta, tau, target_censor,
                        policy_start=baseline_cycles,
                    ))
                    expected_c = float(core["expected_censor_fraction"](
                        ctx["paths"], beta, tau, lam, policy_start=baseline_cycles
                    ))
                    crude, _ = core["exact_crude_rmst"](
                        ctx["paths"], beta, tau, lam, ctx["H"],
                        policy_start=baseline_cycles,
                    )
                    policy[(batch, beta)] = {
                        "lambda0": lam, "expected_censor_fraction": expected_c,
                        "exact_crude_rmst": float(crude),
                    }

            new_rows: list[dict[str, Any]] = []
            for batch in PRIMARY_BATCHES:
                ctx = batch_context[batch]
                for beta in betas:
                    pol = policy[(batch, beta)]
                    for inner_r in selected_inner:
                        loop_started = time.perf_counter()
                        keys = [(outer_b, batch, beta, inner_r, arm) for arm in PRIMARY_ARMS]
                        if all(key in done_keys for key in keys):
                            continue
                        seed = seed_lookup[(outer_b, batch, beta, inner_r)]
                        rng = np.random.default_rng(seed)
                        uniforms = [rng.random(max(len(path) - 1, 0)) for path in ctx["paths"]]
                        base_fields = {
                            "outer_b": outer_b, "batch_label": batch, "beta": beta,
                            "inner_r": inner_r, "overlay_seed_uint64": str(seed),
                            "truth_net_rmst": ctx["truth"],
                            "exact_crude_rmst": pol["exact_crude_rmst"],
                            "signal_scale": signal_scale, "tau": tau,
                            "lambda0": pol["lambda0"],
                            "expected_censor_fraction": pol["expected_censor_fraction"],
                            "n_bootstrap_positions": len(ctx["idx"]),
                            "n_unique_source_units": ctx["n_unique"],
                            "max_multiplicity": ctx["max_multiplicity"],
                            "min_unique_source_units_per_fold": ctx["min_unique_per_fold"],
                            "ridge_slope": ridge_slope,
                        }
                        try:
                            times, events, observed, oracle_cumhaz = core["overlay_from_uniforms"](
                                ctx["paths"], beta, tau, pol["lambda0"], uniforms,
                                policy_start=baseline_cycles,
                            )
                            realized = float(np.mean(events == 0))
                            common = {
                                **base_fields,
                                "realized_censor_fraction": realized,
                                "observed_failures": int(np.sum(events == 1)),
                                "preventive_replacements": int(np.sum(events == 0)),
                            }
                            naive = _naive_rmst(primary_engine, times, events, ctx["H"])
                            oracle = _weighted_rmst(primary_engine, times, events, oracle_cumhaz, ctx["H"])
                            oracle_diag = _support_diagnostics(
                                core, times, events, oracle_cumhaz, ctx["H"], args.exp_clip
                            )
                            new_rows.extend([
                                {
                                    **common, "arm": "naive", "estimate": naive,
                                    "fit_success": True, "max_weight": 1.0,
                                    "min_ess_over_risk": 1.0,
                                    "max_weighted_hazard_increment": np.nan,
                                    "min_n_at_risk_at_failure": np.nan,
                                    "exp_clipping": False, "solver_fallback": False,
                                    "fit_method": "not_applicable", "fit_n_iter": 0,
                                    "error_type": "", "error_message": "",
                                    "failure_step": "",
                                },
                                {
                                    **common, "arm": "oracle_tv_ipcw", "estimate": oracle,
                                    "fit_success": True, **oracle_diag,
                                    "solver_fallback": False, "fit_method": "oracle_known_hazard",
                                    "fit_n_iter": 0, "error_type": "", "error_message": "",
                                    "failure_step": "",
                                },
                            ])
                            try:
                                cross_cumhaz, fits = core["fit_crossfit_cumhaz"](
                                    observed, events, ctx["folds"],
                                    policy_start=baseline_cycles, ridge_slope=ridge_slope,
                                )
                                cross = _weighted_rmst(primary_engine, times, events, cross_cumhaz, ctx["H"])
                                cross_diag = _support_diagnostics(
                                    core, times, events, cross_cumhaz, ctx["H"], args.exp_clip
                                )
                                methods = sorted({str(getattr(f, "method", "newton")) for f in fits})
                                new_rows.append({
                                    **common, "arm": "crossfit_tv_ipcw", "estimate": cross,
                                    "fit_success": bool(all(bool(f.success) for f in fits)),
                                    **cross_diag,
                                    "solver_fallback": bool(any(
                                        str(getattr(f, "method", "")).startswith("fallback") for f in fits
                                    )),
                                    "fit_method": ";".join(methods),
                                    "fit_n_iter": int(max([int(f.n_iter) for f in fits], default=0)),
                                    "fit_total_iterations": int(sum(int(f.n_iter) for f in fits)),
                                    "fit_max_grad_norm": _finite_max(float(f.grad_norm) for f in fits),
                                    "error_type": "", "error_message": "", "failure_step": "",
                                })
                            except Exception as exc:
                                new_rows.append({
                                    **common, "arm": "crossfit_tv_ipcw", "estimate": np.nan,
                                    "fit_success": False, "max_weight": np.nan,
                                    "min_ess_over_risk": np.nan,
                                    "max_weighted_hazard_increment": np.nan,
                                    "min_n_at_risk_at_failure": np.nan,
                                    "exp_clipping": False, "solver_fallback": False,
                                    "fit_method": "failed", "fit_n_iter": np.nan,
                                    "error_type": type(exc).__name__, "error_message": str(exc),
                                    "failure_step": "crossfit_fit_or_estimation",
                                })
                        except Exception as exc:
                            for arm in PRIMARY_ARMS:
                                new_rows.append({
                                    **base_fields, "arm": arm, "estimate": np.nan,
                                    "fit_success": False,
                                    "realized_censor_fraction": np.nan,
                                    "observed_failures": np.nan,
                                    "preventive_replacements": np.nan,
                                    "max_weight": np.nan, "min_ess_over_risk": np.nan,
                                    "max_weighted_hazard_increment": np.nan,
                                    "min_n_at_risk_at_failure": np.nan,
                                    "exp_clipping": False, "solver_fallback": False,
                                    "fit_method": "failed", "fit_n_iter": np.nan,
                                    "error_type": type(exc).__name__, "error_message": str(exc),
                                    "failure_step": "overlay_or_primary_estimation",
                                })
                        elapsed = time.perf_counter() - loop_started
                        for row in new_rows[-3:]:
                            if (int(row["outer_b"]), str(row["batch_label"]), float(row["beta"]),
                                    int(row["inner_r"])) == (outer_b, batch, beta, inner_r):
                                row["runtime_seconds"] = elapsed

            for row in new_rows:
                key = (int(row["outer_b"]), str(row["batch_label"]), float(row["beta"]),
                       int(row["inner_r"]), str(row["arm"]))
                if key not in done_keys:
                    all_rows.append(row)
                    done_keys.add(key)
            completed_outer.append(outer_b)
            completed_since_checkpoint += 1
            if completed_since_checkpoint >= args.checkpoint_every or outer_b == selected_outer[-1]:
                _write_csv_atomic(output, all_rows)
                completed_since_checkpoint = 0
            print(
                f"outer_b={outer_b} completed; rows={len(new_rows)}; "
                f"elapsed={time.perf_counter() - outer_started:.1f}s"
            )
        except Exception as exc:
            failure = {
                "outer_b": outer_b, "error_type": type(exc).__name__,
                "error_message": str(exc), "failure_step": "outer_preparation",
            }
            outer_failures.append(failure)
            rows = _failure_rows(outer_b, PRIMARY_BATCHES, betas, selected_inner, exc, "outer_preparation")
            for row in rows:
                key = (int(row["outer_b"]), str(row["batch_label"]), float(row["beta"]),
                       int(row["inner_r"]), str(row["arm"]))
                if key not in done_keys:
                    all_rows.append(row)
                    done_keys.add(key)
            _write_csv_atomic(output, all_rows)
            print(f"outer_b={outer_b} FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)

    requested_keys = {
        (outer_b, batch, beta, inner_r, arm)
        for outer_b in selected_outer for batch in PRIMARY_BATCHES for beta in betas
        for inner_r in selected_inner for arm in PRIMARY_ARMS
    }
    complete = requested_keys.issubset(done_keys)
    requested_rows = [
        r for r in all_rows
        if (int(r["outer_b"]), str(r["batch_label"]), float(r["beta"]),
            int(r["inner_r"]), str(r["arm"])) in requested_keys
    ]
    expected_rows = len(requested_keys)
    fit_failures = sum(not _as_bool(r.get("fit_success", False)) for r in requested_rows)
    report = {
        **input_check,
        "analysis": "MATR joint cell-and-policy redesign runner",
        "status": "complete" if complete and not outer_failures else "complete_with_recorded_failures",
        "output": str(output),
        "output_sha256": _sha256(output) if output.exists() else None,
        "expected_requested_rows": expected_rows,
        "actual_requested_rows": len(requested_rows),
        "requested_key_complete": complete,
        "fit_failure_rows": fit_failures,
        "fit_failure_fraction": fit_failures / expected_rows if expected_rows else None,
        "completed_outer": completed_outer,
        "outer_failures": outer_failures,
        "runtime_seconds": time.perf_counter() - started,
        "checkpoint_every": args.checkpoint_every,
        "resampling_target": "sample-adaptive finite-cohort benchmark with policy recalibration",
        "policy_resampling_mode": "adaptive_redesign",
        "inference_scope": "empirical redesign sensitivity; not a fixed-policy or fleet-superpopulation confidence interval",
        "guardrails": {
            "multiplicity_expanded": True,
            "same_source_copies_share_manifest_fold": True,
            "global_signal_scale_recomputed_each_outer": True,
            "tau_recomputed_each_outer": True,
            "batch_lambda0_recalibrated_each_outer_beta": True,
            "fixed_horizons": True,
            "same_sample_arm_excluded": True,
            "oracle_ht_arm_excluded": True,
            "silent_failure_drop_forbidden": True,
        },
    }
    report_path.write_text(json.dumps(_jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8")
    print("MATR JOINT REDESIGN RUN COMPLETED")
    print(f"status={'PASS' if complete and not outer_failures else 'REVIEW_REQUIRED'}")
    print(f"  requested rows: {len(requested_rows)}/{expected_rows}")
    print(f"  failed rows: {fit_failures}")
    print(f"  output: {output}")
    print(f"  report: {report_path}")
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
