"""Shared utilities for policy, regularisation, horizon, and endpoint sensitivity analyses.

The module reuses the frozen cohort definition and primary estimator implementation.
It adds policy-schedule helpers and transparent output utilities. No result-driven
tuning is performed here.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

PRIMARY_BATCHES = ("MATR-05-12", "MATR-06-30", "MATR-04-12")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def finite_trailing_median(x: Sequence[float], window: int) -> np.ndarray:
    arr = np.asarray(x, float).ravel()
    out = np.full(len(arr), np.nan, float)
    for j in range(len(arr)):
        w = arr[max(0, j - int(window) + 1):j + 1]
        w = w[np.isfinite(w)]
        if w.size:
            out[j] = float(np.median(w))
    return out


def build_tmax(values: Sequence[float], lifetime: int, baseline_cycles: int = 50,
               smooth_window: int = 5) -> np.ndarray:
    raw = np.asarray(values, float).ravel()
    T = int(lifetime)
    if T < 1:
        raise ValueError("lifetime must be positive")
    baseline = raw[:min(int(baseline_cycles), len(raw))]
    baseline = baseline[np.isfinite(baseline)]
    if baseline.size == 0:
        raise ValueError("no finite Tmax baseline")
    base = float(np.median(baseline))
    smooth = finite_trailing_median(raw, int(smooth_window)) - base
    last = 0.0
    for j in range(len(smooth)):
        if np.isfinite(smooth[j]):
            last = float(smooth[j])
        else:
            smooth[j] = last
    if T <= len(smooth):
        out = smooth[:T].copy()
    else:
        out = np.concatenate([smooth, np.full(T - len(smooth), last, float)])
    if len(out) != T or not np.all(np.isfinite(out)):
        raise RuntimeError("Tmax construction failed")
    return out


@dataclass
class PrimaryData:
    ids: list[str]
    batches: np.ndarray
    life: np.ndarray
    folds: np.ndarray
    ir_raw: list[np.ndarray]
    tmax_raw: list[np.ndarray]
    cells_by_id: dict[str, Any]
    endpoint_rows: list[dict[str, str]]


def load_primary_data(root: Path, matr: Path, baseline_cycles: int = 50,
                      smooth_window: int = 5) -> tuple[PrimaryData, Any, Any, Any, Any]:
    root = Path(root).resolve()
    matr = Path(matr).resolve()
    required = [
        root / "code" / "matr_data.py",
        root / "code" / "matr_endpoint_reconstruction.py",
        root / "code" / "src" / "matr_ipcw.py",
        root / "code" / "src" / "_survival.py",
        root / "results" / "matr_cohort" / "endpoint_review.csv",
        root / "results" / "matr_cohort" / "fold_assignment.csv",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    audit = load_module(required[0], "sensitivity_audit")
    finalizer = load_module(required[1], "sensitivity_endpoint")
    core = load_module(required[2], "sensitivity_core")
    surv = load_module(required[3], "sensitivity_survival")

    endpoint = [r for r in read_csv(required[4])
                if str(r.get("primary_IR_cohort", "")).lower() in {"true", "1", "yes"}]
    if len(endpoint) != 124:
        raise RuntimeError(f"expected frozen primary cohort n=124, got {len(endpoint)}")
    fold_rows = read_csv(required[5])
    fold_by_id = {r["unit_id"]: int(float(r["fold"])) for r in fold_rows}

    schema: list[Any] = []
    raw = audit.read_raw_cells(matr, schema)
    cells = audit.harmonize(raw)
    cells_by_id = {c.unit_id: c for c in cells}

    ids: list[str] = []
    batches: list[str] = []
    life: list[float] = []
    folds: list[int] = []
    ir_raw: list[np.ndarray] = []
    tm_raw: list[np.ndarray] = []
    ordered_endpoint = sorted(endpoint, key=lambda z: z["unit_id"])
    for row in ordered_endpoint:
        uid = row["unit_id"]
        if uid not in cells_by_id or uid not in fold_by_id:
            raise KeyError(uid)
        cell = cells_by_id[uid]
        T = int(round(float(row["author_reconstructed_lifetime"])))
        ids.append(uid)
        batches.append(row["batch_label"])
        life.append(float(T))
        folds.append(fold_by_id[uid])
        ir_raw.append(core.build_ir_signal(cell.arrays.get("IR", []), T,
                                           baseline_cycles, smooth_window))
        tm_raw.append(build_tmax(cell.arrays.get("Tmax", []), T,
                                 baseline_cycles, smooth_window))
    data = PrimaryData(
        ids=ids,
        batches=np.asarray(batches, object),
        life=np.asarray(life, float),
        folds=np.asarray(folds, int),
        ir_raw=ir_raw,
        tmax_raw=tm_raw,
        cells_by_id=cells_by_id,
        endpoint_rows=ordered_endpoint,
    )
    return data, audit, finalizer, core, surv


def summarize(rows: Sequence[dict[str, Any]], keys: Sequence[str], value: str) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[float]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(float(row[value]))
    out: list[dict[str, Any]] = []
    for key, vals in sorted(groups.items(), key=lambda z: tuple(str(x) for x in z[0])):
        a = np.asarray(vals, float)
        sd = float(a.std(ddof=1)) if len(a) > 1 else 0.0
        rec = dict(zip(keys, key))
        rec.update({
            "n": int(len(a)),
            "mean": float(a.mean()),
            "sd": sd,
            "mcse": float(sd / math.sqrt(len(a))) if len(a) else float("nan"),
            "p2p5": float(np.percentile(a, 2.5)),
            "median": float(np.median(a)),
            "p97p5": float(np.percentile(a, 97.5)),
        })
        out.append(rec)
    return out


def eligible_indices(length: int, policy_start: int, inspection_interval: int) -> np.ndarray:
    """Eligible record indices before the terminal record.

    ``inspection_interval=1`` means continuous checking.  Larger values mean
    evaluation at fixed inspection occasions beginning at ``policy_start``.
    """
    T = int(length)
    start = max(0, int(policy_start))
    interval = max(1, int(inspection_interval))
    if T <= start + 1:
        return np.array([], int)
    return np.arange(start, T - 1, interval, dtype=int)


def scheduled_tau(core: Any, paths_z: Sequence[np.ndarray], q: float,
                  policy_start: int, inspection_interval: int) -> float:
    refs: list[np.ndarray] = []
    for path in paths_z:
        idx = eligible_indices(len(path), policy_start, inspection_interval)
        if idx.size == 0:
            raise ValueError("unit has no eligible inspection record")
        refs.append(np.asarray(path, float)[idx])
    return float(core.unit_equal_quantile(refs, q))


def expected_replacement_fraction(paths: Sequence[np.ndarray], beta: float, tau: float,
                                  lambda0: float, policy_start: int,
                                  inspection_interval: int) -> float:
    if lambda0 <= 0:
        return 0.0
    probs: list[float] = []
    for path in paths:
        z = np.asarray(path, float)
        idx = eligible_indices(len(z), policy_start, inspection_interval)
        if idx.size == 0:
            probs.append(0.0)
            continue
        eta = np.clip(float(beta) * (z[idx] - float(tau)), -700.0, 700.0)
        total = float(lambda0) * float(np.exp(eta).sum())
        probs.append(float(-np.expm1(-min(total, 745.0))))
    return float(np.mean(probs))


def calibrate_lambda0_scheduled(paths: Sequence[np.ndarray], beta: float, tau: float,
                                 target: float, policy_start: int,
                                 inspection_interval: int, iterations: int = 100) -> float:
    target = float(target)
    if not 0.0 <= target < 1.0:
        raise ValueError("target must lie in [0,1)")
    if target == 0:
        return 0.0
    lo, hi = -745.0, math.log(1e3)
    for _ in range(int(iterations)):
        mid = 0.5 * (lo + hi)
        got = expected_replacement_fraction(paths, beta, tau, math.exp(mid),
                                            policy_start, inspection_interval)
        if got < target:
            lo = mid
        else:
            hi = mid
    ans = float(math.exp(0.5 * (lo + hi)))
    got = expected_replacement_fraction(paths, beta, tau, ans,
                                        policy_start, inspection_interval)
    if not np.isfinite(got) or abs(got - target) > 1e-8:
        raise RuntimeError(f"calibration failed: target={target}, attained={got}")
    return ans


def overlay_scheduled(paths: Sequence[np.ndarray], beta: float, tau: float,
                      lambda0: float, uniforms: Sequence[np.ndarray],
                      policy_start: int, inspection_interval: int):
    if len(paths) != len(uniforms):
        raise ValueError("paths/uniforms mismatch")
    times: list[float] = []
    events: list[int] = []
    observed: list[np.ndarray] = []
    oracle_before: list[np.ndarray] = []
    for path, uu in zip(paths, uniforms):
        z = np.asarray(path, float)
        u = np.asarray(uu, float)
        T = len(z)
        if len(u) < max(T - 1, 0):
            raise ValueError("uniform vector is too short")
        idx = eligible_indices(T, policy_start, inspection_interval)
        mu = np.zeros(T, float)
        if idx.size:
            mu[idx] = float(lambda0) * np.exp(np.clip(float(beta) * (z[idx] - float(tau)), -700, 700))
            p = -np.expm1(-np.minimum(mu[idx], 745.0))
            fired = np.flatnonzero(u[idx] < p)
        else:
            fired = np.array([], int)
        if fired.size:
            fired_index = int(idx[int(fired[0])])
            L = fired_index + 1
            event = 0
        else:
            L = T
            event = 1
        prefix = z[:L].copy()
        before = np.zeros(L + 1, float)
        if L > 1:
            before[2:] = np.cumsum(mu[:L - 1])
        times.append(float(L))
        events.append(event)
        observed.append(prefix)
        oracle_before.append(before)
    return np.asarray(times, float), np.asarray(events, int), observed, oracle_before


def person_period_scheduled(paths: Sequence[np.ndarray], events: Sequence[int],
                            indices: Iterable[int] | None,
                            policy_start: int, inspection_interval: int):
    ev = np.asarray(events, int)
    use = range(len(paths)) if indices is None else list(indices)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for i in use:
        z = np.asarray(paths[i], float)
        if ev[i] == 0:
            idx = eligible_indices(len(z) + 1, policy_start, inspection_interval)
            idx = idx[idx < len(z)]
            if idx.size == 0 or idx[-1] != len(z) - 1:
                raise RuntimeError("censored prefix does not end at an eligible inspection")
            y = np.zeros(idx.size, float)
            y[-1] = 1.0
        else:
            idx = eligible_indices(len(z), policy_start, inspection_interval)
            y = np.zeros(idx.size, float)
        if idx.size:
            xs.append(z[idx])
            ys.append(y)
    if not xs:
        raise ValueError("no scheduled person-period rows")
    return np.concatenate(xs), np.concatenate(ys)


def fitted_cumhaz_scheduled(path: np.ndarray, intercept: float, slope: float,
                            policy_start: int, inspection_interval: int) -> np.ndarray:
    z = np.asarray(path, float)
    L = len(z)
    mu = np.zeros(L, float)
    idx = eligible_indices(L + 1, policy_start, inspection_interval)
    idx = idx[idx < L]
    if idx.size:
        mu[idx] = np.exp(np.clip(float(intercept) + float(slope) * z[idx], -25.0, 25.0))
    out = np.zeros(L + 1, float)
    if L > 1:
        out[2:] = np.cumsum(mu[:L - 1])
    return out


def fit_crossfit_scheduled(core: Any, observed_paths: Sequence[np.ndarray], events: Sequence[int],
                           folds: Sequence[int], policy_start: int,
                           inspection_interval: int, ridge_slope: float):
    fold = np.asarray(folds, int)
    ev = np.asarray(events, int)
    pred: list[np.ndarray | None] = [None] * len(observed_paths)
    fits: list[Any] = []
    for f in sorted(np.unique(fold)):
        train = np.flatnonzero(fold != f)
        test = np.flatnonzero(fold == f)
        x, y = person_period_scheduled(observed_paths, ev, train,
                                       policy_start, inspection_interval)
        fit = core.fit_cloglog_fast(x, y, ridge_slope=float(ridge_slope))
        if not fit.success:
            raise RuntimeError(f"scheduled censor model failed fold={f}: {fit.message}")
        fits.append(fit)
        for i in test:
            pred[i] = fitted_cumhaz_scheduled(np.asarray(observed_paths[i], float),
                                              fit.intercept, fit.slope,
                                              policy_start, inspection_interval)
    if any(x is None for x in pred):
        raise RuntimeError("incomplete scheduled cross-fit predictions")
    return [np.asarray(x, float) for x in pred], fits


def exact_crude_scheduled(paths: Sequence[np.ndarray], beta: float, tau: float,
                          lambda0: float, horizon: float, policy_start: int,
                          inspection_interval: int) -> tuple[float, np.ndarray]:
    H = float(horizon)
    gs: list[float] = []
    contributions: list[float] = []
    for path in paths:
        z = np.asarray(path, float)
        idx = eligible_indices(len(z), policy_start, inspection_interval)
        if idx.size and lambda0 > 0:
            eta = np.clip(float(beta) * (z[idx] - float(tau)), -700.0, 700.0)
            ch = float(lambda0) * float(np.exp(eta).sum())
        else:
            ch = 0.0
        g = float(np.exp(-min(ch, 745.0)))
        gs.append(g)
        contributions.append(g * max(H - float(len(z)), 0.0))
    crude = H - float(np.mean(contributions))
    return float(crude), np.asarray(gs, float)



def exact_any_exit_scheduled(paths: Sequence[np.ndarray], beta: float, tau: float,
                             lambda0: float, horizon: float, policy_start: int,
                             inspection_interval: int) -> float:
    """Exact ``E[min(T,C,H)]`` for the scheduled policy variants."""
    H = float(horizon)
    if not np.isfinite(H) or H < 0:
        raise ValueError("horizon must be finite and non-negative")
    vals: list[float] = []
    for path in paths:
        z = np.asarray(path, float)
        T = len(z)
        survival = 1.0
        expected = 0.0
        idx = eligible_indices(T, policy_start, inspection_interval)
        for j in idx:
            exit_time = float(int(j) + 1)
            if exit_time >= H:
                expected += survival * H
                survival = 0.0
                break
            mu = float(lambda0) * float(np.exp(np.clip(float(beta) * (z[int(j)] - float(tau)), -700.0, 700.0)))
            p = float(-np.expm1(-min(mu, 745.0)))
            expected += survival * p * exit_time
            survival *= 1.0 - p
        if survival > 0.0:
            expected += survival * min(float(T), H)
        vals.append(expected)
    return float(np.mean(vals))

def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
                    encoding="utf-8")


def _json_default(x: Any):
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x).__name__)
