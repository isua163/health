#!/usr/bin/env python3
"""MATR data audit for MATR/Severson battery run-to-EOL data.

Reads only per-cell descriptors and per-cycle ``summary`` fields from MATLAB v7.3
(HDF5) files. It does not read the much larger within-cycle records. Outputs are
intended to decide the preregistered cohort audit gates before any IPCW model is fitted.

Primary outputs
---------------
results/matr_cell_inventory.csv
results/matr_channel_missingness.csv
results/matr_eol_comparison.csv
results/matr_batch_summary.csv
validation/matr_audit_report.json
validation/matr_hdf5_schema.txt
validation/matr_raw_file_hashes.csv

The audit preserves raw cell indices, flags the continuation/exclusion rules in the
original Severson loading code, and creates a harmonized inventory. No raw data are
modified.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

Q_NOM = 1.1
EOL_ABS = 0.88
EOL_TOL = 0.005 * Q_NOM
MIN_Q_VALID = 0.5 * Q_NOM
CHANNELS = {
    "cycle": ("cycle",),
    "QDischarge": ("QDischarge", "qdischarge", "discharge_capacity"),
    "QCharge": ("QCharge", "qcharge", "charge_capacity"),
    "IR": ("IR", "ir", "internal_resistance"),
    "Tmax": ("Tmax", "tmax", "temperature_max"),
    "Tavg": ("Tavg", "tavg", "temperature_avg"),
    "Tmin": ("Tmin", "tmin", "temperature_min"),
    "chargetime": ("chargetime", "charge_time", "chargeTime"),
}
NONCAPACITY_CHANNELS = ("IR", "Tmax", "Tavg", "Tmin", "chargetime")
BATCH_LABELS = {
    "2017-05-12": "MATR-05-12",
    "2017-06-30": "MATR-06-30",
    "2018-04-12": "MATR-04-12",
    "2019-01-24": "MATR-CLO",
}
# Original Severson LoadData.m: batch2 cells 8:10,16:17 continue batch1 cells 1:5.
CONTINUATION_MAP_1BASED = {1: 8, 2: 9, 3: 10, 4: 16, 5: 17}
# LoadData.m optional removal: batteries not finished in Batch 1.
BATCH1_OPTIONAL_INCOMPLETE_1BASED = {9, 11, 13, 14, 23}


@dataclass
class RawCell:
    file_path: Path
    batch_token: str
    batch_label: str
    raw_index_1based: int
    source_tag: str
    published_cycle_life: float
    policy: str
    barcode: str
    channel_id: str
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    trim_start: int = 0
    schema_notes: list[str] = field(default_factory=list)


@dataclass
class HarmonizedCell:
    unit_id: str
    batch_token: str
    batch_label: str
    raw_index_1based: int
    source_files: str
    source_cells: str
    continuation_appended: bool
    continuation_duplicate: bool
    official_exclusion_reason: str
    optional_incomplete_flag: bool
    published_cycle_life: float
    policy: str
    barcode: str
    channel_id: str
    arrays: dict[str, np.ndarray]


def _date_token(name: str) -> str:
    m = re.search(r"20\d{2}-\d{2}-\d{2}", name)
    return m.group(0) if m else Path(name).stem[:10]


def _flatten_numeric(value: Any) -> np.ndarray:
    try:
        arr = np.asarray(value)
    except Exception:
        return np.asarray([], dtype=float)
    if arr.dtype.kind in "biufc":
        return arr.ravel().astype(float, copy=False)
    return np.asarray([], dtype=float)


def _decode_matlab_char(arr: np.ndarray) -> str:
    arr = np.asarray(arr)
    if arr.dtype.kind in "SU":
        return "".join(str(x) for x in arr.ravel()).strip()
    if arr.dtype.kind in "iu" and arr.size:
        vals = [int(x) for x in arr.ravel() if int(x) > 0]
        if vals and all(0 <= x <= 0x10FFFF for x in vals):
            try:
                return "".join(chr(x) for x in vals).strip()
            except Exception:
                return ""
    return ""


def _read_h5_item(h5: Any, dataset: Any, i: int) -> Any:
    """Read one MATLAB cell/struct field robustly whether stored as refs or numeric."""
    data = dataset[()]
    flat = np.asarray(data).ravel()
    if i >= len(flat):
        return None
    item = flat[i]
    try:
        import h5py
        if isinstance(item, h5py.Reference):
            if not item:
                return None
            obj = h5[item]
            if isinstance(obj, h5py.Group):
                return obj
            return obj[()]
    except Exception:
        pass
    return item


def _read_descriptor(h5: Any, batch: Any, names: Iterable[str], i: int) -> str:
    key_map = {k.lower(): k for k in batch.keys()}
    for candidate in names:
        key = key_map.get(candidate.lower())
        if key is None:
            continue
        try:
            value = _read_h5_item(h5, batch[key], i)
            if value is None:
                continue
            if hasattr(value, "keys"):
                continue
            arr = np.asarray(value)
            text = _decode_matlab_char(arr)
            if text:
                return text
            nums = _flatten_numeric(arr)
            if nums.size == 1 and np.isfinite(nums[0]):
                return str(nums[0])
        except Exception:
            continue
    return ""


def _read_scalar(h5: Any, batch: Any, names: Iterable[str], i: int) -> float:
    key_map = {k.lower(): k for k in batch.keys()}
    for candidate in names:
        key = key_map.get(candidate.lower())
        if key is None:
            continue
        try:
            value = _read_h5_item(h5, batch[key], i)
            if value is None or hasattr(value, "keys"):
                continue
            nums = _flatten_numeric(value)
            nums = nums[np.isfinite(nums)]
            if nums.size:
                return float(nums[0])
        except Exception:
            continue
    return float("nan")


def _group_array(group: Any, aliases: Iterable[str]) -> tuple[np.ndarray, str]:
    key_map = {k.lower(): k for k in group.keys()}
    for alias in aliases:
        key = key_map.get(alias.lower())
        if key is None:
            continue
        try:
            arr = _flatten_numeric(group[key][()])
            return arr, key
        except Exception:
            continue
    return np.asarray([], dtype=float), ""


def _trim_arrays(arrays: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], int]:
    q = np.asarray(arrays.get("QDischarge", []), float)
    valid = np.flatnonzero(np.isfinite(q) & (q > MIN_Q_VALID))
    start = int(valid[0]) if valid.size else 0
    out: dict[str, np.ndarray] = {}
    for key, arr in arrays.items():
        x = np.asarray(arr, float).ravel()
        out[key] = x[start:] if start < len(x) else np.asarray([], dtype=float)
    return out, start


def _sha256(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _first_crossing(q: np.ndarray, threshold: float) -> float:
    idx = np.flatnonzero(np.isfinite(q) & (q < threshold))
    return float(idx[0] + 1) if idx.size else float("nan")


def _safe_percentile(x: np.ndarray, p: float) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, p)) if x.size else float("nan")


def _rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    if len(x) < 3:
        return float("nan")
    # Stable average ranks without scipy.
    def rankdata(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty(len(a), float)
        i = 0
        while i < len(a):
            j = i + 1
            while j < len(a) and a[order[j]] == a[order[i]]:
                j += 1
            ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
            i = j
        return ranks
    rx, ry = rankdata(x), rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _channel_metrics(arr: np.ndarray, n_window: int) -> dict[str, Any]:
    x = np.asarray(arr, float)[: max(0, int(n_window))]
    finite = np.isfinite(x)
    xf = x[finite]
    n = int(len(x))
    nfin = int(finite.sum())
    iqr = (_safe_percentile(xf, 75) - _safe_percentile(xf, 25)) if nfin else float("nan")
    median = _safe_percentile(xf, 50)
    cycle = np.arange(1, n + 1, dtype=float)
    corr = _rank_correlation(cycle, x) if n else float("nan")
    positive_fraction = float(np.mean(xf > 0)) if nfin else float("nan")
    return {
        "n_window": n,
        "n_finite": nfin,
        "coverage_fraction": float(nfin / n) if n else 0.0,
        "missing_fraction": float(1.0 - nfin / n) if n else 1.0,
        "positive_fraction": positive_fraction,
        "zero_fraction_among_finite": float(np.mean(xf == 0)) if nfin else float("nan"),
        "median": median,
        "iqr": float(iqr),
        "min": float(np.min(xf)) if nfin else float("nan"),
        "max": float(np.max(xf)) if nfin else float("nan"),
        "first_finite": float(xf[0]) if nfin else float("nan"),
        "last_finite": float(xf[-1]) if nfin else float("nan"),
        "spearman_cycle": corr,
        "nonconstant": bool(nfin >= 3 and np.isfinite(iqr) and iqr > 1e-12),
    }


def _concat_arrays(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key in set(a) | set(b):
        x = np.asarray(a.get(key, []), float)
        y = np.asarray(b.get(key, []), float)
        out[key] = np.concatenate([x, y])
    return out


def read_raw_cells(matr_root: Path, schema_lines: list[str]) -> list[RawCell]:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required: conda install -c conda-forge h5py -y") from exc

    files = sorted(matr_root.glob("*.mat"))
    if not files:
        raise FileNotFoundError(f"No .mat files found directly under {matr_root}")
    cells: list[RawCell] = []
    for path in files:
        token = _date_token(path.name)
        label = BATCH_LABELS.get(token, token)
        with h5py.File(path, "r") as h5:
            if "batch" not in h5:
                schema_lines.append(f"{path.name}: ERROR no /batch group; root keys={list(h5.keys())}")
                continue
            batch = h5["batch"]
            if "summary" not in batch:
                schema_lines.append(f"{path.name}: ERROR no /batch/summary; batch keys={list(batch.keys())}")
                continue
            refs = np.asarray(batch["summary"][()]).ravel()
            schema_lines.append(f"FILE {path.name}")
            schema_lines.append(f"  root_keys={list(h5.keys())}")
            schema_lines.append(f"  batch_keys={list(batch.keys())}")
            schema_lines.append(f"  n_summary_refs={len(refs)}")
            seen_summary_keys: set[str] = set()
            for i, ref in enumerate(refs):
                try:
                    summary = h5[ref]
                    if not hasattr(summary, "keys"):
                        raise TypeError("summary reference is not a group")
                    seen_summary_keys.update(summary.keys())
                    arrays: dict[str, np.ndarray] = {}
                    notes: list[str] = []
                    for canonical, aliases in CHANNELS.items():
                        arr, actual = _group_array(summary, aliases)
                        arrays[canonical] = arr
                        if not actual:
                            notes.append(f"missing:{canonical}")
                    # Preserve the complete published cycle index.  The public
                    # LoadData.m appends and labels the full summary arrays; it does
                    # not remove an initial prefix.  ``trim`` is retained only as a
                    # QC diagnostic for leading invalid QDischarge values.
                    _, trim = _trim_arrays(arrays)
                    if trim:
                        notes.append(f"leading_invalid_QDischarge_prefix:{trim}")
                    cl = _read_scalar(h5, batch, ("cycle_life", "cyclelife", "life"), i)
                    policy = _read_descriptor(h5, batch, ("policy_readable", "policy", "charge_policy"), i)
                    barcode = _read_descriptor(h5, batch, ("barcode",), i)
                    channel_id = _read_descriptor(h5, batch, ("channel", "channel_id"), i)
                    cells.append(
                        RawCell(
                            file_path=path,
                            batch_token=token,
                            batch_label=label,
                            raw_index_1based=i + 1,
                            source_tag=f"{token}_cell{i+1}",
                            published_cycle_life=cl,
                            policy=policy,
                            barcode=barcode,
                            channel_id=channel_id,
                            arrays=arrays,
                            trim_start=trim,
                            schema_notes=notes,
                        )
                    )
                except Exception as exc:
                    schema_lines.append(f"  cell {i+1}: READ ERROR {type(exc).__name__}: {exc}")
            schema_lines.append(f"  observed_summary_keys={sorted(seen_summary_keys)}")
            schema_lines.append("")
    if not cells:
        raise RuntimeError("No cells could be read from MATR files; inspect matr_hdf5_schema.txt")
    return cells


def official_batch3_exclusions(raw_batch3: list[RawCell]) -> dict[int, str]:
    """Reproduce the ordering of exclusions in the authors' LoadData.m."""
    reasons: dict[int, str] = {}
    working = list(raw_batch3)
    # MATLAB batch3(38)=[]: channel 46 data collection problem.
    if len(working) >= 38:
        removed = working.pop(37)
        reasons[removed.raw_index_1based] = "author_exclusion_channel46_data_collection"
    # rind = find(endcap3 > 0.885); remove incomplete cells.
    kept: list[RawCell] = []
    for cell in working:
        q = np.asarray(cell.arrays.get("QDischarge", []), float)
        last = q[np.isfinite(q)][-1] if np.any(np.isfinite(q)) else float("nan")
        if np.isfinite(last) and last > 0.885:
            reasons[cell.raw_index_1based] = "author_exclusion_end_capacity_gt_0p885"
        else:
            kept.append(cell)
    working = kept
    # nind=[3,40:41] after prior removals: noisy Batch 8 batteries.
    for position in sorted((3, 40, 41), reverse=True):
        if len(working) >= position:
            removed = working.pop(position - 1)
            reasons[removed.raw_index_1based] = "author_exclusion_noisy_batch8"
    return reasons


def harmonize(raw: list[RawCell]) -> list[HarmonizedCell]:
    by_batch: dict[str, list[RawCell]] = {}
    for c in raw:
        by_batch.setdefault(c.batch_token, []).append(c)
    for x in by_batch.values():
        x.sort(key=lambda c: c.raw_index_1based)

    continuation_sources = set(CONTINUATION_MAP_1BASED.values())
    b1_lookup = {c.raw_index_1based: c for c in by_batch.get("2017-05-12", [])}
    b2_lookup = {c.raw_index_1based: c for c in by_batch.get("2017-06-30", [])}
    b3_exclusions = official_batch3_exclusions(by_batch.get("2018-04-12", []))

    out: list[HarmonizedCell] = []
    for c in raw:
        continuation_duplicate = c.batch_token == "2017-06-30" and c.raw_index_1based in continuation_sources
        reason = "continuation_segment_merged_into_2017_05_12" if continuation_duplicate else ""
        arrays = c.arrays
        source_files = c.file_path.name
        source_cells = c.source_tag
        appended = False
        published = c.published_cycle_life
        if c.batch_token == "2017-05-12" and c.raw_index_1based in CONTINUATION_MAP_1BASED:
            source_idx = CONTINUATION_MAP_1BASED[c.raw_index_1based]
            seg = b2_lookup.get(source_idx)
            if seg is not None:
                arrays = _concat_arrays(c.arrays, seg.arrays)
                source_files = f"{c.file_path.name};{seg.file_path.name}"
                source_cells = f"{c.source_tag};{seg.source_tag}"
                appended = True
                # The parent cycle_life is authoritative when present; otherwise use segment.
                if not np.isfinite(published) and np.isfinite(seg.published_cycle_life):
                    published = seg.published_cycle_life
        if c.batch_token == "2018-04-12":
            reason = b3_exclusions.get(c.raw_index_1based, reason)
        optional = c.batch_token == "2017-05-12" and c.raw_index_1based in BATCH1_OPTIONAL_INCOMPLETE_1BASED
        if optional and not reason:
            reason = "author_optional_unfinished_batch1"
        out.append(
            HarmonizedCell(
                unit_id=f"{c.batch_token}_cell{c.raw_index_1based}",
                batch_token=c.batch_token,
                batch_label=c.batch_label,
                raw_index_1based=c.raw_index_1based,
                source_files=source_files,
                source_cells=source_cells,
                continuation_appended=appended,
                continuation_duplicate=continuation_duplicate,
                official_exclusion_reason=reason,
                optional_incomplete_flag=optional,
                published_cycle_life=published,
                policy=c.policy,
                barcode=c.barcode,
                channel_id=c.channel_id,
                arrays=arrays,
            )
        )
    return out


def _csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def build_outputs(cells: list[HarmonizedCell], results_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    missingness: list[dict[str, Any]] = []
    eol_rows: list[dict[str, Any]] = []

    for cell in cells:
        q = np.asarray(cell.arrays.get("QDischarge", []), float)
        qf = q[np.isfinite(q)]
        n_q = int(len(q))
        cl = cell.published_cycle_life
        cl_round = int(round(cl)) if np.isfinite(cl) and cl > 0 else 0
        n_window = min(n_q, cl_round) if cl_round else n_q
        recording_ratio = float(n_q / cl) if np.isfinite(cl) and cl > 0 else float("nan")
        initial_pool = qf[: min(5, len(qf))]
        q_initial = float(np.median(initial_pool)) if len(initial_pool) else float("nan")
        eol_author = _first_crossing(q, EOL_ABS)
        eol_rated_tol = _first_crossing(q, EOL_ABS + EOL_TOL)
        eol_initial = _first_crossing(q, 0.80 * q_initial) if np.isfinite(q_initial) else float("nan")
        official_eligible = not bool(cell.official_exclusion_reason) and not cell.continuation_duplicate
        complete_95 = bool(
            official_eligible
            and n_window >= 2
            and (
                (np.isfinite(cl) and cl > 0 and recording_ratio >= 0.95)
                or np.isfinite(eol_rated_tol)
            )
        )
        complete_70 = bool(
            official_eligible
            and n_window >= 2
            and (
                (np.isfinite(cl) and cl > 0 and recording_ratio >= 0.70)
                or np.isfinite(eol_rated_tol)
            )
        )

        per_channel: dict[str, dict[str, Any]] = {}
        for channel in CHANNELS:
            arr = np.asarray(cell.arrays.get(channel, []), float)
            metrics = _channel_metrics(arr, n_window)
            per_channel[channel] = metrics
            viable = bool(
                channel in NONCAPACITY_CHANNELS
                and metrics["coverage_fraction"] >= 0.90
                and metrics["n_finite"] >= 20
                and metrics["nonconstant"]
                and (channel != "IR" or metrics["positive_fraction"] >= 0.95)
            )
            missingness.append(
                {
                    "unit_id": cell.unit_id,
                    "batch_token": cell.batch_token,
                    "batch_label": cell.batch_label,
                    "raw_cell_index": cell.raw_index_1based,
                    "channel": channel,
                    **metrics,
                    "viable_online_signal": viable,
                    "complete_95": complete_95,
                    "official_eligible": official_eligible,
                }
            )

        ir = per_channel["IR"]
        signal_viability = {
            ch: bool(
                per_channel[ch]["coverage_fraction"] >= 0.90
                and per_channel[ch]["n_finite"] >= 20
                and per_channel[ch]["nonconstant"]
                and (ch != "IR" or per_channel[ch]["positive_fraction"] >= 0.95)
            )
            for ch in NONCAPACITY_CHANNELS
        }
        row = {
            "unit_id": cell.unit_id,
            "batch_token": cell.batch_token,
            "batch_label": cell.batch_label,
            "raw_cell_index": cell.raw_index_1based,
            "source_files": cell.source_files,
            "source_cells": cell.source_cells,
            "continuation_appended": cell.continuation_appended,
            "continuation_duplicate": cell.continuation_duplicate,
            "official_exclusion_reason": cell.official_exclusion_reason,
            "optional_incomplete_flag": cell.optional_incomplete_flag,
            "official_eligible": official_eligible,
            "policy": cell.policy,
            "barcode": cell.barcode,
            "channel_id": cell.channel_id,
            "published_cycle_life": cl,
            "n_capacity_records": n_q,
            "analysis_window_cycles": n_window,
            "recording_to_published_ratio": recording_ratio,
            "q_initial_median_first5": q_initial,
            "q_last_finite": float(qf[-1]) if len(qf) else float("nan"),
            "q_min": float(np.min(qf)) if len(qf) else float("nan"),
            "eol_author_q_lt_0p88": eol_author,
            "eol_rated80_with_tolerance": eol_rated_tol,
            "eol_initial80": eol_initial,
            "complete_95": complete_95,
            "complete_70": complete_70,
            "IR_coverage": ir["coverage_fraction"],
            "IR_iqr": ir["iqr"],
            "IR_spearman_cycle": ir["spearman_cycle"],
            "IR_viable": signal_viability["IR"],
            "Tmax_viable": signal_viability["Tmax"],
            "Tavg_viable": signal_viability["Tavg"],
            "Tmin_viable": signal_viability["Tmin"],
            "chargetime_viable": signal_viability["chargetime"],
            "primary_IR_candidate": bool(complete_95 and signal_viability["IR"]),
        }
        inventory.append(row)

        def relerr(est: float) -> float:
            return float(abs(est - cl) / cl) if np.isfinite(est) and np.isfinite(cl) and cl > 0 else float("nan")
        eol_rows.append(
            {
                "unit_id": cell.unit_id,
                "batch_token": cell.batch_token,
                "batch_label": cell.batch_label,
                "published_cycle_life": cl,
                "n_capacity_records": n_q,
                "q_initial_median_first5": q_initial,
                "threshold_author_absolute": EOL_ABS,
                "threshold_rated80_tolerant": EOL_ABS + EOL_TOL,
                "threshold_initial80": 0.80 * q_initial if np.isfinite(q_initial) else float("nan"),
                "eol_author_q_lt_0p88": eol_author,
                "eol_rated80_with_tolerance": eol_rated_tol,
                "eol_initial80": eol_initial,
                "rel_error_author_vs_published": relerr(eol_author),
                "rel_error_rated_tolerant_vs_published": relerr(eol_rated_tol),
                "rel_error_initial80_vs_published": relerr(eol_initial),
                "official_eligible": official_eligible,
                "complete_95": complete_95,
                "official_exclusion_reason": cell.official_exclusion_reason,
            }
        )

    batch_summary: list[dict[str, Any]] = []
    batches = sorted({r["batch_label"] for r in inventory})
    for batch in batches:
        rows = [r for r in inventory if r["batch_label"] == batch]
        eligible = [r for r in rows if r["official_eligible"]]
        complete = [r for r in rows if r["complete_95"]]
        summary: dict[str, Any] = {
            "batch_label": batch,
            "batch_token": rows[0]["batch_token"],
            "n_raw_or_harmonized_rows": len(rows),
            "n_official_eligible": len(eligible),
            "n_complete_95": len(complete),
            "n_complete_70": sum(bool(r["complete_70"]) for r in rows),
            "median_published_life": _safe_percentile(np.asarray([r["published_cycle_life"] for r in complete], float), 50),
            "min_published_life": _safe_percentile(np.asarray([r["published_cycle_life"] for r in complete], float), 0),
            "max_published_life": _safe_percentile(np.asarray([r["published_cycle_life"] for r in complete], float), 100),
        }
        for ch in NONCAPACITY_CHANNELS:
            summary[f"n_complete_with_{ch}_viable"] = sum(bool(r["complete_95"] and r[f"{ch}_viable"]) for r in rows)
        batch_summary.append(summary)

    complete_rows = [r for r in inventory if r["complete_95"]]
    viable_counts = {
        ch: sum(bool(r["complete_95"] and r[f"{ch}_viable"]) for r in inventory)
        for ch in NONCAPACITY_CHANNELS
    }
    selected_signal = max(viable_counts, key=viable_counts.get) if viable_counts else ""
    n_selected = viable_counts.get(selected_signal, 0)
    g1 = len(complete_rows) >= 80
    g2 = n_selected >= 80
    eligible_batches_selected = [
        b["batch_label"] for b in batch_summary
        if b.get(f"n_complete_with_{selected_signal}_viable", 0) >= 20
    ] if selected_signal else []
    g3 = len(eligible_batches_selected) >= 2
    report_core = {
        "n_harmonized_rows": len(inventory),
        "n_official_eligible": sum(bool(r["official_eligible"]) for r in inventory),
        "n_complete_95": len(complete_rows),
        "n_complete_70": sum(bool(r["complete_70"]) for r in inventory),
        "viable_counts_among_complete_95": viable_counts,
        "preferred_signal": "IR" if viable_counts.get("IR", 0) >= 80 else selected_signal,
        "selected_signal_by_coverage": selected_signal,
        "selected_signal_n": n_selected,
        "batches_with_at_least_20_selected_signal_units": eligible_batches_selected,
        "gates": {
            "G1_at_least_80_complete_units": g1,
            "G2_at_least_80_complete_units_with_noncapacity_signal": g2,
            "G3_at_least_two_batches_with_20_signal_eligible_units": g3,
        },
        "gate_interpretation": (
            "cohort audit provisional pass; proceed to manual review of exclusions and signal trajectories before fitting the primary estimator."
            if g1 and g2 and g3 else
            "cohort audit gate not fully satisfied; inspect batch and channel outputs before deciding on fallback data."
        ),
    }

    _csv_write(results_dir / "matr_cell_inventory.csv", inventory)
    _csv_write(results_dir / "matr_channel_missingness.csv", missingness)
    _csv_write(results_dir / "matr_eol_comparison.csv", eol_rows)
    _csv_write(results_dir / "matr_batch_summary.csv", batch_summary)
    return inventory, missingness, eol_rows, batch_summary, report_core


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matr", required=True, type=Path, help="Directory containing MATR .mat files")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="Package root")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--validation-dir", type=Path, default=None)
    parser.add_argument("--skip-hash", action="store_true", help="Skip SHA-256 of raw .mat files")
    args = parser.parse_args()

    root = args.root.resolve()
    matr = args.matr.expanduser().resolve()
    results_dir = (args.results_dir or (root / "results/matr_cohort")).resolve()
    validation_dir = (args.validation_dir or (root / "validation")).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)
    if not matr.is_dir():
        raise FileNotFoundError(f"MATR directory not found: {matr}")

    schema_lines = [
        "MATR HDF5 schema audit",
        f"created_utc={dt.datetime.now(dt.timezone.utc).isoformat()}",
        f"matr_root={matr}",
        "",
    ]
    raw = read_raw_cells(matr, schema_lines)
    harmonized = harmonize(raw)
    inventory, missingness, eol_rows, batch_summary, core = build_outputs(harmonized, results_dir)

    mat_files = sorted(matr.glob("*.mat"))
    hash_rows: list[dict[str, Any]] = []
    for path in mat_files:
        stat = path.stat()
        print(f"hashing {path.name} ({stat.st_size / (1024**3):.2f} GiB)..." if not args.skip_hash else f"recording {path.name} (hash skipped)...")
        hash_rows.append({
            "file": path.name,
            "size_bytes": stat.st_size,
            "modified_utc": dt.datetime.fromtimestamp(stat.st_mtime, tz=dt.timezone.utc).isoformat(),
            "sha256": "SKIPPED" if args.skip_hash else _sha256(path),
        })
    _csv_write(validation_dir / "matr_raw_file_hashes.csv", hash_rows)
    (validation_dir / "matr_hdf5_schema.txt").write_text("\n".join(schema_lines), encoding="utf-8")

    report = {
        "audit": "matr_data_audit",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "package_root": str(root),
        "matr_root": str(matr),
        "raw_files": hash_rows,
        "definitions": {
            "complete_95": "officially eligible and either recording/published-life >=0.95 or tolerant rated-capacity EOL observed",
            "signal_viable": "coverage >=0.90, at least 20 finite observations, nonconstant; IR additionally >=0.95 positive",
            "author_eol": "first QDischarge < 0.88 Ah",
            "rated80_tolerant_eol": f"first QDischarge < {EOL_ABS + EOL_TOL:.4f} Ah",
            "initial80_eol": "first QDischarge < 0.8 * median(first five valid discharge capacities)",
        },
        **core,
        "outputs": {
            "cell_inventory": str((results_dir / "matr_cell_inventory.csv").relative_to(root)),
            "channel_missingness": str((results_dir / "matr_channel_missingness.csv").relative_to(root)),
            "eol_comparison": str((results_dir / "matr_eol_comparison.csv").relative_to(root)),
            "batch_summary": str((results_dir / "matr_batch_summary.csv").relative_to(root)),
            "schema": str((validation_dir / "matr_hdf5_schema.txt").relative_to(root)),
            "raw_hashes": str((validation_dir / "matr_raw_file_hashes.csv").relative_to(root)),
        },
        "cautions": [
            "Gate decisions are provisional until cell-level exclusions and trajectories are manually reviewed.",
            "The 2017-06-30 continuation segments are merged into 2017-05-12 cells 1-5 and not counted as independent units.",
            "Author exclusions from LoadData.m are flagged; no raw file is modified.",
            "This audit does not fit a censoring model and does not create IPCW results.",
        ],
    }
    report_path = validation_dir / "matr_audit_report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")

    print("\nMATR AUDIT SUMMARY")
    print(f"  raw cells read: {len(raw)}")
    print(f"  harmonized rows: {core['n_harmonized_rows']}")
    print(f"  complete eligible units (95% rule): {core['n_complete_95']}")
    print(f"  viable non-capacity signals: {core['viable_counts_among_complete_95']}")
    print(f"  preferred signal: {core['preferred_signal']}")
    for key, value in core["gates"].items():
        print(f"  {key}: {'PASS' if value else 'FAIL'}")
    print(f"Wrote {report_path}")
    print("MATR AUDIT COMPLETED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
