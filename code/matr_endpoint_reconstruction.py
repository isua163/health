#!/usr/bin/env python3
"""Finalize cohort audit for the MATR/Severson cohort (strict v2.1).

This script performs the manual-review calculations required after
``matr_data.py``:

* reconstruct the author-style 0.88-Ah endpoint after the official continuation
  merges and exclusions;
* verify the exact primary cohort supported by internal resistance (IR);
* quantify the stale ``cycle_life`` descriptors for the five continuation cells;
* review an online IR signal based only on current/past observations;
* emit final cohort audit gate decisions, QC tables, and review figures.

No censoring model or IPCW estimator is fitted here.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import math
import os
import platform
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

EOL_ABS = 0.88
PRIMARY_BATCHES = ("2017-05-12", "2017-06-30", "2018-04-12")
CONTINUATION_MAP_1BASED = {1: 8, 2: 9, 3: 10, 4: 16, 5: 17}


def _load_audit_module(root: Path):
    path = root / "code" / "matr_data.py"
    if not path.exists():
        raise FileNotFoundError(f"Required cohort audit audit script not found: {path}")
    spec = importlib.util.spec_from_file_location("matr_audit_cohort", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _date_token(name: str) -> str:
    m = re.search(r"20\d{2}-\d{2}-\d{2}", name)
    return m.group(0) if m else Path(name).stem[:10]


def _count_cycle_refs(
    matr_root: Path,
) -> tuple[dict[tuple[str, int], int], dict[tuple[str, int], dict[str, Any]], list[str]]:
    """Count MATLAB cycle-struct records from the fields inside each HDF5 Group.

    In these v7.3 files, ``batch.cycles`` points to a Group whose keys are the
    cycle-struct fields (I, Qc, Qd, ...).  The number of keys is therefore *not*
    the number of cycles.  Each field contains one object reference per cycle;
    the modal field reference count is the validated record count.
    """
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required") from exc
    from collections import Counter

    counts: dict[tuple[str, int], int] = {}
    details: dict[tuple[str, int], dict[str, Any]] = {}
    notes: list[str] = []
    for path in sorted(matr_root.glob("*.mat")):
        token = _date_token(path.name)
        with h5py.File(path, "r") as h5:
            batch = h5["batch"]
            if "cycles" not in batch:
                notes.append(f"{path.name}: no /batch/cycles field")
                continue
            refs = np.asarray(batch["cycles"][()]).ravel()
            for i, ref in enumerate(refs, start=1):
                key = (token, i)
                try:
                    obj = h5[ref]
                    field_counts: dict[str, int] = {}
                    if isinstance(obj, h5py.Group):
                        for field in obj.keys():
                            child = obj[field]
                            try:
                                arr = np.asarray(child[()])
                                field_counts[str(field)] = int(arr.size)
                            except Exception:
                                shape = tuple(int(x) for x in getattr(child, "shape", ()))
                                field_counts[str(field)] = int(np.prod(shape)) if shape else 0
                        candidates = [n for n in field_counts.values() if n > 0]
                        if not candidates:
                            raise ValueError("cycles Group contains no countable fields")
                        freq = Counter(candidates)
                        max_freq = max(freq.values())
                        modal_values = [n for n, f in freq.items() if f == max_freq]
                        n = int(max(modal_values))
                        consistent = bool(all(v == n for v in candidates))
                    else:
                        shape = tuple(int(x) for x in getattr(obj, "shape", ()))
                        n = int(np.prod(shape)) if shape else int(np.asarray(obj[()]).size)
                        consistent = n > 0
                    counts[key] = n
                    details[key] = {
                        "object_type": type(obj).__name__,
                        "field_counts": field_counts,
                        "inferred_count": n,
                        "all_positive_fields_equal": consistent,
                    }
                    if i == 1:
                        notes.append(
                            f"{path.name}: cycles object type={type(obj).__name__}, "
                            f"inferred_count={n}, field_counts={field_counts}"
                        )
                except Exception as exc:
                    counts[key] = 0
                    details[key] = {
                        "object_type": "ERROR",
                        "field_counts": {},
                        "inferred_count": 0,
                        "all_positive_fields_equal": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    notes.append(f"{path.name} cell {i}: cycle-count error {type(exc).__name__}: {exc}")
    return counts, details, notes


def _first_crossing(q: np.ndarray, threshold: float) -> float:
    idx = np.flatnonzero(np.isfinite(q) & (q < threshold))
    return float(idx[0] + 1) if idx.size else float("nan")


def _rank_corr(y: np.ndarray) -> float:
    y = np.asarray(y, float)
    x = np.arange(1, len(y) + 1, dtype=float)
    mask = np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    x = x[mask]
    y = y[mask]
    # Average ranks, stable for ties.
    def ranks(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a, kind="mergesort")
        out = np.empty(len(a), float)
        i = 0
        while i < len(a):
            j = i + 1
            while j < len(a) and a[order[j]] == a[order[i]]:
                j += 1
            out[order[i:j]] = 0.5 * (i + j - 1) + 1.0
            i = j
        return out
    rx, ry = ranks(x), ranks(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _trailing_median(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan, float)
    for i in range(len(x)):
        lo = max(0, i - window + 1)
        z = x[lo : i + 1]
        z = z[np.isfinite(z)]
        if z.size:
            out[i] = float(np.median(z))
    return out


def _finite_median(x: np.ndarray) -> float:
    z = np.asarray(x, float)
    z = z[np.isfinite(z)]
    return float(np.median(z)) if z.size else float("nan")


def _finite_quantile(x: np.ndarray, p: float) -> float:
    z = np.asarray(x, float)
    z = z[np.isfinite(z)]
    return float(np.quantile(z, p)) if z.size else float("nan")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _json_value(x: Any) -> Any:
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {str(k): _json_value(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_value(v) for v in x]
    return x


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matr", required=True, type=Path)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--baseline-cycles", type=int, default=50)
    p.add_argument("--smooth-window", type=int, default=5)
    a = p.parse_args()

    root = a.root.resolve()
    matr = a.matr.resolve()
    results = root / "results"
    figures = root / "figures"
    validation = root / "validation"
    for d in (results, figures, validation):
        d.mkdir(parents=True, exist_ok=True)

    audit = _load_audit_module(root)
    schema_lines: list[str] = []
    raw = audit.read_raw_cells(matr, schema_lines)
    cells = audit.harmonize(raw)
    cycle_counts, cycle_details, cycle_notes = _count_cycle_refs(matr)
    raw_lookup = {(c.batch_token, c.raw_index_1based): c for c in raw}

    b2_counts = {
        idx: cycle_counts.get(("2017-06-30", idx), 0)
        for idx in CONTINUATION_MAP_1BASED.values()
    }

    endpoint_rows: list[dict[str, Any]] = []
    ir_rows: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []

    for cell in cells:
        official_eligible = not bool(cell.official_exclusion_reason) and not cell.continuation_duplicate
        q = np.asarray(cell.arrays.get("QDischarge", []), float)
        ir = np.asarray(cell.arrays.get("IR", []), float)
        n_q = int(len(q))
        n_ir = int(len(ir))

        base_cycles = cycle_counts.get((cell.batch_token, cell.raw_index_1based), 0)
        official_cycle_count = int(base_cycles)
        if cell.batch_token == "2017-05-12" and cell.raw_index_1based in CONTINUATION_MAP_1BASED:
            official_cycle_count += int(b2_counts.get(CONTINUATION_MAP_1BASED[cell.raw_index_1based], 0))
        if official_cycle_count <= 0:
            official_cycle_count = n_q

        source_trim_total = int(raw_lookup.get((cell.batch_token, cell.raw_index_1based)).trim_start) if (cell.batch_token, cell.raw_index_1based) in raw_lookup else -1
        source_cycle_field_consistent = bool(cycle_details.get((cell.batch_token, cell.raw_index_1based), {}).get("all_positive_fields_equal", False))
        if cell.batch_token == "2017-05-12" and cell.raw_index_1based in CONTINUATION_MAP_1BASED:
            seg_key = ("2017-06-30", CONTINUATION_MAP_1BASED[cell.raw_index_1based])
            seg_raw = raw_lookup.get(seg_key)
            source_trim_total += int(seg_raw.trim_start) if seg_raw is not None else -999
            source_cycle_field_consistent = source_cycle_field_consistent and bool(cycle_details.get(seg_key, {}).get("all_positive_fields_equal", False))

        crossing = _first_crossing(q, EOL_ABS)
        qf = q[np.isfinite(q)]
        q_last = float(qf[-1]) if qf.size else float("nan")
        # Exact logic of the public LoadData.m after harmonization:
        # use first crossing only when terminal capacity is below 0.88 Ah;
        # otherwise assign number of cycle records + 1.
        if np.isfinite(q_last) and q_last < EOL_ABS and np.isfinite(crossing):
            author_label = int(round(crossing))
            endpoint_source = "first_QDischarge_lt_0p88"
        else:
            author_label = int(official_cycle_count + 1)
            endpoint_source = "cycle_record_count_plus_1"

        published = float(cell.published_cycle_life)
        descriptor_abs_gap = float(abs(author_label - published)) if np.isfinite(published) else float("nan")
        descriptor_rel_gap = (
            float(descriptor_abs_gap / published) if np.isfinite(published) and published > 0 else float("nan")
        )
        signal_gap = int(author_label - n_ir)
        if signal_gap > 0:
            alignment = "pad_last_observed_IR"
        elif signal_gap < 0:
            alignment = "truncate_IR_at_endpoint"
        else:
            alignment = "exact"

        primary = bool(
            official_eligible
            and cell.batch_token in PRIMARY_BATCHES
            and n_ir >= 20
            and np.mean(np.isfinite(ir)) >= 0.90
            and np.mean(ir[np.isfinite(ir)] > 0) >= 0.95 if np.any(np.isfinite(ir)) else False
        )

        endpoint_rows.append(
            {
                "unit_id": cell.unit_id,
                "batch_token": cell.batch_token,
                "batch_label": cell.batch_label,
                "raw_cell_index": cell.raw_index_1based,
                "official_eligible": official_eligible,
                "primary_IR_cohort": primary,
                "continuation_appended": cell.continuation_appended,
                "continuation_duplicate": cell.continuation_duplicate,
                "official_exclusion_reason": cell.official_exclusion_reason,
                "published_cycle_life_descriptor": published,
                "n_summary_QDischarge": n_q,
                "n_summary_IR": n_ir,
                "official_cycle_record_count": official_cycle_count,
                "cycle_count_minus_summary_Q_length": int(official_cycle_count - n_q),
                "source_trim_start_total": source_trim_total,
                "cycle_struct_fields_consistent": source_cycle_field_consistent,
                "q_last_finite": q_last,
                "first_QDischarge_lt_0p88": crossing,
                "author_reconstructed_lifetime": author_label,
                "endpoint_source": endpoint_source,
                "descriptor_abs_gap": descriptor_abs_gap,
                "descriptor_rel_gap": descriptor_rel_gap,
                "author_lifetime_minus_IR_length": signal_gap,
                "signal_endpoint_alignment": alignment,
            }
        )

        if not primary:
            continue

        # Candidate policy signal: within-cell relative IR, using only current/past values.
        pos = np.where(np.isfinite(ir) & (ir > 0), ir, np.nan)
        base_n = min(int(a.baseline_cycles), len(pos))
        baseline = _finite_median(pos[:base_n])
        smooth = _trailing_median(pos, int(a.smooth_window))
        log_ratio = np.log(smooth / baseline) if np.isfinite(baseline) and baseline > 0 else np.full(len(ir), np.nan)

        # Align signal to the reconstructed endpoint. Padding is last-observation-carried-forward
        # and only applies after the final available cycle; no future observation is introduced.
        T = int(author_label)
        if T <= len(log_ratio):
            sig = log_ratio[:T]
        else:
            finite = log_ratio[np.isfinite(log_ratio)]
            last = float(finite[-1]) if finite.size else float("nan")
            sig = np.concatenate([log_ratio, np.full(T - len(log_ratio), last)])

        early_n = min(10, len(sig))
        late_n = min(10, len(sig))
        early = _finite_median(sig[:early_n])
        late = _finite_median(sig[-late_n:])
        rho = _rank_corr(sig)
        finite_ir = ir[np.isfinite(ir)]
        positive_fraction = float(np.mean(finite_ir > 0)) if finite_ir.size else float("nan")
        ir_iqr = _finite_quantile(finite_ir, 0.75) - _finite_quantile(finite_ir, 0.25)
        flags: list[str] = []
        if not np.isfinite(baseline) or baseline <= 0:
            flags.append("invalid_baseline")
        if positive_fraction < 0.95:
            flags.append("nonpositive_IR")
        if not np.isfinite(ir_iqr) or ir_iqr <= 1e-12:
            flags.append("constant_IR")
        if np.isfinite(rho) and rho < 0:
            flags.append("negative_rank_trend_review_only")
        if T > n_ir + 1:
            flags.append("endpoint_beyond_IR_support")

        ir_rows.append(
            {
                "unit_id": cell.unit_id,
                "batch_token": cell.batch_token,
                "batch_label": cell.batch_label,
                "author_reconstructed_lifetime": T,
                "baseline_cycles": base_n,
                "IR_baseline_median": baseline,
                "IR_positive_fraction": positive_fraction,
                "IR_iqr_raw": ir_iqr,
                "log_IR_ratio_early_median": early,
                "log_IR_ratio_late_median": late,
                "log_IR_ratio_late_minus_early": late - early if np.isfinite(early) and np.isfinite(late) else float("nan"),
                "log_IR_ratio_spearman_cycle": rho,
                "endpoint_signal_length_gap": signal_gap,
                "qc_flags": ";".join(flags),
                "qc_pass_for_primary": not any(x in flags for x in ("invalid_baseline", "nonpositive_IR", "constant_IR")),
            }
        )
        trajectories.append(
            {
                "unit_id": cell.unit_id,
                "batch_label": cell.batch_label,
                "x": np.arange(1, len(sig) + 1, dtype=float) / float(len(sig)),
                "y": sig,
            }
        )

    primary_end = [r for r in endpoint_rows if r["primary_IR_cohort"]]
    primary_ir = ir_rows
    by_batch: dict[str, int] = {}
    for row in primary_end:
        by_batch[row["batch_label"]] = by_batch.get(row["batch_label"], 0) + 1

    n_primary = len(primary_end)
    batches_ge20 = sorted([b for b, n in by_batch.items() if n >= 20])
    continuation_rows = [r for r in primary_end if r["continuation_appended"]]
    noncontinuation = [r for r in primary_end if not r["continuation_appended"]]
    descriptor_rel_noncont = np.asarray([r["descriptor_rel_gap"] for r in noncontinuation], float)
    signal_gaps = np.asarray([r["author_lifetime_minus_IR_length"] for r in primary_end], int)
    rhos = np.asarray([r["log_IR_ratio_spearman_cycle"] for r in primary_ir], float)
    deltas = np.asarray([r["log_IR_ratio_late_minus_early"] for r in primary_ir], float)
    cycle_q_gaps = np.asarray([r["cycle_count_minus_summary_Q_length"] for r in primary_end], int)
    cycle_ir_gaps = np.asarray([r["official_cycle_record_count"] - r["n_summary_IR"] for r in primary_end], int)
    trim_totals = np.asarray([r["source_trim_start_total"] for r in primary_end], int)
    reconstructed = np.asarray([r["author_reconstructed_lifetime"] for r in primary_end], int)
    cycle_totals = np.asarray([r["official_cycle_record_count"] for r in primary_end], int)
    noncont_abs_gaps = np.asarray([r["descriptor_abs_gap"] for r in noncontinuation], float)
    continuation_expected = {8: 662, 9: 981, 10: 1060, 16: 208, 17: 482}
    continuation_count_checks = {
        str(idx): int(cycle_counts.get(("2017-06-30", idx), 0)) == expected
        for idx, expected in continuation_expected.items()
    }
    expected_batch_counts = {"MATR-04-12": 40, "MATR-05-12": 41, "MATR-06-30": 43}
    endpoint_integrity = {
        "primary_cohort_exactly_124": n_primary == 124,
        "primary_batch_counts_match_public_harmonization": by_batch == expected_batch_counts,
        "all_primary_cycle_struct_fields_consistent": all(bool(r["cycle_struct_fields_consistent"]) for r in primary_end),
        "all_primary_cycle_counts_equal_summary_Q_lengths": bool(cycle_q_gaps.size and np.all(cycle_q_gaps == 0)),
        "all_primary_cycle_counts_equal_summary_IR_lengths": bool(cycle_ir_gaps.size and np.all(cycle_ir_gaps == 0)),
        # Prefix anomalies are recorded but do not alter the published cycle index.
        # The cohort audit loader retains all summary rows, matching public LoadData.m.
        "all_primary_prefix_diagnostics_nonnegative": bool(trim_totals.size and np.all(trim_totals >= 0)),
        "all_reconstructed_lifetimes_at_least_50": bool(reconstructed.size and np.all(reconstructed >= 50)),
        "all_reconstructed_lifetimes_within_cycle_support": bool(reconstructed.size and np.all(reconstructed <= cycle_totals + 1)),
        "noncontinuation_labels_within_one_cycle": bool(noncont_abs_gaps.size and np.all(noncont_abs_gaps <= 1.0)),
        "continuation_segment_counts_match_LoadData": bool(all(continuation_count_checks.values())),
        "all_primary_units_pass_basic_IR_QC": bool(primary_ir and all(bool(r["qc_pass_for_primary"]) for r in primary_ir)),
    }

    gates = {
        "G1_at_least_80_complete_independent_units": n_primary >= 80,
        "G2_at_least_80_units_with_noncapacity_IR_signal": n_primary >= 80,
        "G3_at_least_two_batches_with_20_IR_units": len(batches_ge20) >= 2,
    }
    final_pass = bool(all(gates.values()) and all(endpoint_integrity.values()))

    cohort_rows = []
    for batch in sorted(by_batch):
        batch_ir = [r for r in primary_ir if r["batch_label"] == batch]
        brho = np.asarray([r["log_IR_ratio_spearman_cycle"] for r in batch_ir], float)
        bdelta = np.asarray([r["log_IR_ratio_late_minus_early"] for r in batch_ir], float)
        cohort_rows.append(
            {
                "batch_label": batch,
                "n_primary_IR_units": by_batch[batch],
                "median_IR_rank_trend": _finite_median(brho),
                "n_negative_IR_rank_trend_review_only": int(np.sum(brho < 0)),
                "median_late_minus_early_log_IR": _finite_median(bdelta),
                "n_nonpositive_late_minus_early": int(np.sum(bdelta <= 0)),
            }
        )

    endpoint_path = results / "matr_cohort" / "endpoint_review.csv"
    ir_path = results / "matr_cohort" / "internal_resistance_qc.csv"
    cohort_path = results / "matr_cohort" / "cohort_decision.csv"
    _write_csv(endpoint_path, endpoint_rows)
    _write_csv(ir_path, ir_rows)
    _write_csv(cohort_path, cohort_rows)

    # Review figures. These are diagnostics, not manuscript figures.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        for tr in trajectories:
            # Light downsampling only for drawing efficiency.
            idx = np.linspace(0, len(tr["x"]) - 1, min(300, len(tr["x"]))).astype(int)
            ax.plot(tr["x"][idx], tr["y"][idx], alpha=0.18, linewidth=0.7)
        ax.axhline(0.0, linewidth=0.8)
        ax.set_xlabel("Fraction of reconstructed lifetime")
        ax.set_ylabel("log(IR / early-life baseline), trailing median")
        ax.set_title(f"MATR primary IR cohort review (n={n_primary})")
        fig.tight_layout()
        fig.savefig(figures / "matr_cohort_ir_trajectory_review.png", dpi=180)
        fig.savefig(figures / "matr_cohort_ir_trajectory_review.pdf")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.2, 5.4))
        x = np.asarray([r["published_cycle_life_descriptor"] for r in primary_end], float)
        y = np.asarray([r["author_reconstructed_lifetime"] for r in primary_end], float)
        cont = np.asarray([r["continuation_appended"] for r in primary_end], bool)
        ax.scatter(x[~cont], y[~cont], s=16, alpha=0.65, label="Other primary cells")
        ax.scatter(x[cont], y[cont], s=42, marker="x", label="Merged continuation cells")
        lo = float(np.nanmin(np.concatenate([x, y])))
        hi = float(np.nanmax(np.concatenate([x, y])))
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=0.9)
        ax.set_xlabel("Raw cycle_life descriptor")
        ax.set_ylabel("Author-style reconstructed lifetime")
        ax.set_title("Endpoint alignment after official harmonization")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(figures / "matr_cohort_endpoint_alignment.png", dpi=180)
        fig.savefig(figures / "matr_cohort_endpoint_alignment.pdf")
        plt.close(fig)
    except Exception as exc:
        cycle_notes.append(f"Figure generation warning: {type(exc).__name__}: {exc}")

    report = {
        "status": "PASS" if final_pass else "FAIL",
        "audit": "cohort audit_MATR_final_review_strict_v2_1",
        "script_version": "2.1.0",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "package_root": str(root),
        "matr_root": str(matr),
        "cohort audit_final_pass": final_pass,
        "gates": gates,
        "endpoint_integrity": endpoint_integrity,
        "continuation_count_checks": continuation_count_checks,
        "primary_cohort_decision": {
            "dataset": "Severson MATR 2017-05-12, 2017-06-30, and 2018-04-12 batches",
            "n_units": n_primary,
            "batch_counts": by_batch,
            "batches_with_at_least_20_units": batches_ge20,
            "signal": "internal resistance",
            "CLO_primary_status": "excluded from the primary cohort because IR is identically zero/unavailable; not treated as an independent quality failure",
            "exclusion_rule": "public LoadData.m continuation merge and cell exclusions only; no exclusion based on signal-outcome association",
        },
        "endpoint_decision": {
            "definition": "public LoadData.m logic after continuation merge: first QDischarge < 0.88 Ah when terminal QDischarge < 0.88 Ah; otherwise cycle-record count + 1",
            "continuation_cells_with_stale_raw_descriptor": len(continuation_rows),
            "continuation_unit_ids": [r["unit_id"] for r in continuation_rows],
            "cycle_count_minus_summary_Q_length_min": int(np.min(cycle_q_gaps)) if cycle_q_gaps.size else None,
            "cycle_count_minus_summary_Q_length_max": int(np.max(cycle_q_gaps)) if cycle_q_gaps.size else None,
            "cycle_count_minus_summary_IR_length_min": int(np.min(cycle_ir_gaps)) if cycle_ir_gaps.size else None,
            "cycle_count_minus_summary_IR_length_max": int(np.max(cycle_ir_gaps)) if cycle_ir_gaps.size else None,
            "cycle_index_policy": "retain complete published summary arrays; leading-invalid-prefix count is diagnostic only",
            "n_primary_with_nonzero_prefix_diagnostic": int(np.sum(trim_totals > 0)),
            "source_trim_start_total_max": int(np.max(trim_totals)) if trim_totals.size else None,
            "noncontinuation_descriptor_abs_gap_max": _finite_quantile(noncont_abs_gaps, 1.0),
            "noncontinuation_descriptor_relative_gap_median": _finite_median(descriptor_rel_noncont),
            "noncontinuation_descriptor_relative_gap_p95": _finite_quantile(descriptor_rel_noncont, 0.95),
            "signal_length_gap_median_cycles": _finite_median(signal_gaps),
            "signal_length_gap_min_cycles": int(np.min(signal_gaps)) if signal_gaps.size else None,
            "signal_length_gap_max_cycles": int(np.max(signal_gaps)) if signal_gaps.size else None,
            "policy_signal_alignment": "truncate at reconstructed failure time; when the endpoint is beyond the final IR record, carry the last observed IR forward only to the endpoint",
        },
        "IR_review": {
            "baseline_cycles": int(a.baseline_cycles),
            "causal_smoothing": f"trailing median window={int(a.smooth_window)} cycles",
            "transform": "log(smoothed IR / median positive IR in the early baseline window)",
            "median_rank_correlation_with_cycle": _finite_median(rhos),
            "n_negative_rank_correlations_review_only": int(np.sum(rhos < 0)),
            "median_late_minus_early_log_IR": _finite_median(deltas),
            "n_nonpositive_late_minus_early": int(np.sum(deltas <= 0)),
            "units_passing_basic_IR_QC": int(sum(bool(r["qc_pass_for_primary"]) for r in primary_ir)),
            "selection_note": "Trend direction is diagnostic only and is not used to remove cells.",
        },
        "prespecified_analysis_starting_point": {
            "primary_n": n_primary,
            "primary_signal": "within-cell log IR ratio",
            "baseline_cycles": int(a.baseline_cycles),
            "smoothing": f"causal trailing median {int(a.smooth_window)}",
            "fold_handling": "all pooled centering/scaling and censoring-model fitting must be estimated in the training folds",
            "main_policy": "beta=1 with approximately 40% replacement; beta=0 noninformative reference",
            "estimators": ["naive", "oracle TV-IPCW", "5-fold cross-fitted TV-IPCW"],
        },
        "outputs": {
            "endpoint_review": str(endpoint_path.relative_to(root)),
            "IR_QC": str(ir_path.relative_to(root)),
            "cohort_decision": str(cohort_path.relative_to(root)),
            "IR_trajectory_figure": "figures/matr_cohort_ir_trajectory_review.pdf",
            "endpoint_alignment_figure": "figures/matr_cohort_endpoint_alignment.pdf",
        },
        "cycle_object_notes": cycle_notes,
    }
    report_path = validation / "matr_cohort_validation.json"
    report_path.write_text(json.dumps(_json_value(report), indent=2, ensure_ascii=False), encoding="utf-8")

    print("cohort audit FINAL REVIEW SUMMARY")
    print(f"  primary IR cohort: {n_primary} units; batches={by_batch}")
    print(f"  continuation cells with stale raw cycle_life descriptor: {len(continuation_rows)}")
    print(f"  median IR rank trend: {_finite_median(rhos):.3f}; negative-review-only={int(np.sum(rhos < 0))}")
    print(f"  leading-prefix diagnostics: n_nonzero={int(np.sum(trim_totals > 0))}, max={int(np.max(trim_totals)) if trim_totals.size else 0} (retained in cycle index)")
    for key, value in gates.items():
        print(f"  {key}: {'PASS' if value else 'FAIL'}")
    print("  endpoint integrity:")
    for key, value in endpoint_integrity.items():
        print(f"    {key}: {'PASS' if value else 'FAIL'}")
    print(f"  cohort audit_final_pass: {'PASS' if final_pass else 'FAIL'}")
    print(f"Wrote {report_path}")
    return 0 if final_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
