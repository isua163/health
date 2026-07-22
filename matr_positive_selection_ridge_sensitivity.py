#!/usr/bin/env python3
"""Run the non-selective positive-selection ridge sensitivity for the RESS package.

The script does not select a preferred ridge. It reruns the same prespecified
positive-selection overlays and fold assignments at every requested ridge, then
summarises:

1. the near-aligned primary MATR TV-IPCW implementation check; and
2. the policy-driver omission audit (IR-only, Tmax-only, IR+Tmax).

All ridge candidates are reported. The AIPW outputs produced by the existing
estimator-signal script are ignored; ``--signal-mc 1`` is sufficient because
this add-on only consumes its TV-IPCW rows.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PRIMARY_BATCHES = ("MATR-05-12", "MATR-06-30", "MATR-04-12")


def ridge_tag(value: float) -> str:
    text = f"{value:g}".replace("-", "m").replace(".", "p")
    return f"ridge_{text}"


def run_command(cmd: list[str], cwd: Path, log_path: Path) -> int:
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(cwd / "code"),
        PYTHONHASHSEED="0",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND: " + subprocess.list2cmdline(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return int(proc.returncode)


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def find_signal_summary(out_dir: Path) -> Path:
    candidates = [
        out_dir / "estimator_signal_summary.csv",
        out_dir / "summary.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"No estimator-signal summary found under {out_dir}")


def numeric_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return frame


def primary_summary(ridge: float, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    est = numeric_frame(require_file(out_dir / "estimator_summary.csv"))
    est = est[np.isclose(est["beta"].astype(float), 1.0)].copy()
    rows: list[dict[str, object]] = []
    for batch in PRIMARY_BATCHES:
        sub = est[est["scope"] == batch].set_index("arm")
        needed = {"naive", "crossfit_tv_ipcw", "same_sample_tv_ipcw"}
        if not needed.issubset(sub.index):
            raise RuntimeError(f"Missing primary estimator rows for {batch}, ridge={ridge}")
        naive = sub.loc["naive"]
        cross = sub.loc["crossfit_tv_ipcw"]
        same = sub.loc["same_sample_tv_ipcw"]
        rows.append(
            {
                "ridge_slope": ridge,
                "batch": batch,
                "R": int(cross["R"]),
                "truth_net_rmst": float(cross["truth_net_rmst"]),
                "naive_gap_pct": float(naive["mean_signed_gap_pct"]),
                "crossfit_gap_pct": float(cross["mean_signed_gap_pct"]),
                "crossfit_mcse_pct": float(cross["mcse_signed_gap_pct"]),
                "same_sample_gap_pct": float(same["mean_signed_gap_pct"]),
                "crossfit_minus_naive_pp": float(cross["mean_signed_gap_pct"] - naive["mean_signed_gap_pct"]),
                "crossfit_minus_same_sample_pp": float(cross["mean_signed_gap_pct"] - same["mean_signed_gap_pct"]),
            }
        )

    support = numeric_frame(require_file(out_dir / "support_summary.csv"))
    support = support[
        np.isclose(support["beta"].astype(float), 1.0)
        & (support["arm"] == "crossfit_tv_ipcw")
        & np.isclose(support["checkpoint_fraction_H"].astype(float), 1.0)
    ].copy()
    support.insert(0, "ridge_slope", ridge)
    keep_support = [
        "ridge_slope", "scope", "R", "median_n_at_risk",
        "p10_ess_over_n_at_risk", "median_ess_over_n_at_risk",
        "p95_max_weight", "max_max_weight",
        "fraction_replicates_with_exp_clipping",
    ]
    support = support[keep_support].rename(columns={"scope": "batch"})

    fits = numeric_frame(require_file(out_dir / "fit_diagnostics.csv"))
    fits = fits[
        np.isclose(fits["beta"].astype(float), 1.0)
        & fits["fit_scope"].astype(str).str.startswith("crossfit_fold_")
    ].copy()
    fit_rows: list[dict[str, object]] = []
    for batch in PRIMARY_BATCHES:
        sub = fits[fits["batch"] == batch]
        if sub.empty:
            raise RuntimeError(f"Missing positive-selection fit rows for {batch}, ridge={ridge}")
        success = sub["success"].astype(str).str.lower().isin({"true", "1", "yes"})
        slope = pd.to_numeric(sub["slope"], errors="coerce")
        fit_rows.append(
            {
                "ridge_slope": ridge,
                "batch": batch,
                "n_crossfit_fits": int(len(sub)),
                "fit_failure_fraction": float(1.0 - success.mean()),
                "mean_fitted_slope": float(slope.mean()),
                "median_fitted_slope": float(slope.median()),
                "p10_fitted_slope": float(slope.quantile(0.10)),
                "p90_fitted_slope": float(slope.quantile(0.90)),
            }
        )
    return pd.DataFrame(rows), support, pd.DataFrame(fit_rows)


def signal_summary(ridge: float, out_dir: Path) -> pd.DataFrame:
    summary = numeric_frame(find_signal_summary(out_dir))
    rows: list[dict[str, object]] = []
    for batch in PRIMARY_BATCHES:
        sub = summary[summary["batch"] == batch].copy()
        key = {(str(r.signal_set), str(r.estimator)): r for r in sub.itertuples(index=False)}
        required = [
            ("policy_IR", "naive"),
            ("IR-only", "crossfit_tv_ipcw"),
            ("Tmax-only", "crossfit_tv_ipcw"),
            ("IR+Tmax", "crossfit_tv_ipcw"),
        ]
        missing = [item for item in required if item not in key]
        if missing:
            raise RuntimeError(f"Missing estimator-signal rows for {batch}, ridge={ridge}: {missing}")
        naive = key[("policy_IR", "naive")]
        ir = key[("IR-only", "crossfit_tv_ipcw")]
        tmax = key[("Tmax-only", "crossfit_tv_ipcw")]
        both = key[("IR+Tmax", "crossfit_tv_ipcw")]
        rows.append(
            {
                "ridge_slope": ridge,
                "batch": batch,
                "R": int(ir.n),
                "naive_gap_pct": float(naive.mean),
                "ir_only_gap_pct": float(ir.mean),
                "ir_only_mcse_pct": float(ir.mcse),
                "tmax_only_gap_pct": float(tmax.mean),
                "tmax_only_mcse_pct": float(tmax.mcse),
                "ir_plus_tmax_gap_pct": float(both.mean),
                "ir_plus_tmax_mcse_pct": float(both.mcse),
                "omitted_driver_extra_pp": float(tmax.mean - ir.mean),
                "adding_tmax_to_driver_pp": float(both.mean - ir.mean),
            }
        )
    return pd.DataFrame(rows)


def range_audit(frame: pd.DataFrame, group: str, columns: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for value, sub in frame.groupby(group, sort=False):
        for column in columns:
            x = pd.to_numeric(sub[column], errors="coerce").dropna()
            rows.append(
                {
                    group: value,
                    "metric": column,
                    "minimum": float(x.min()),
                    "maximum": float(x.max()),
                    "range": float(x.max() - x.min()),
                    "sign_consistent": bool((x >= 0).all() or (x <= 0).all()),
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Unpacked package root")
    parser.add_argument("--matr", required=True, type=Path, help="Directory containing original MATR files")
    parser.add_argument("--ridges", nargs="+", type=float, default=[4.0, 16.0, 64.0])
    parser.add_argument("--R", type=int, default=200, help="Overlay replicates per ridge")
    parser.add_argument("--signal-mc", type=int, default=1, help="AIPW recursion count; ignored in this TV-IPCW audit")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--skip-signal-audit", action="store_true", help="Run only the near-aligned primary ridge sweep")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    matr = args.matr.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Package root not found: {root}")
    if not matr.is_dir():
        raise SystemExit(f"MATR root not found: {matr}")
    if args.R < 1:
        raise SystemExit("--R must be positive")
    if not args.ridges or any((not np.isfinite(x)) or x < 0 for x in args.ridges):
        raise SystemExit("All ridge values must be finite and non-negative")

    primary_script = require_file(root / "code" / "matr_primary_analysis.py")
    signal_script = require_file(root / "code" / "matr_estimator_signal_analysis.py")
    out = (args.out_dir.expanduser().resolve() if args.out_dir else root / "results" / "matr_positive_ridge_sensitivity")
    out.mkdir(parents=True, exist_ok=True)

    primary_frames: list[pd.DataFrame] = []
    support_frames: list[pd.DataFrame] = []
    fit_frames: list[pd.DataFrame] = []
    signal_frames: list[pd.DataFrame] = []
    run_records: list[dict[str, object]] = []

    for ridge in args.ridges:
        tag = ridge_tag(ridge)
        ridge_dir = out / tag
        primary_out = ridge_dir / "primary"
        signal_out = ridge_dir / "signal_audit"
        primary_out.mkdir(parents=True, exist_ok=True)
        signal_out.mkdir(parents=True, exist_ok=True)

        primary_cmd = [
            sys.executable, str(primary_script),
            "--root", str(root), "--matr", str(matr),
            "--out-dir", str(primary_out),
            "--R", str(args.R), "--ridge-slope", str(ridge),
            "--analysis-label", f"nonselective_positive_ridge_{ridge:g}",
            "--report-name", f"local_positive_ridge_{tag}_validation.json",
        ]
        primary_code = run_command(primary_cmd, root, ridge_dir / "primary.log")
        try:
            p, s, f = primary_summary(ridge, primary_out)
        except Exception as exc:
            raise RuntimeError(
                f"Primary ridge run failed or produced incomplete output for ridge={ridge:g}; "
                f"return_code={primary_code}. See {ridge_dir / 'primary.log'}"
            ) from exc
        primary_frames.append(p)
        support_frames.append(s)
        fit_frames.append(f)

        signal_code: int | None = None
        if not args.skip_signal_audit:
            signal_cmd = [
                sys.executable, str(signal_script),
                "--root", str(root), "--matr", str(matr),
                "--out-dir", str(signal_out),
                "--R", str(args.R), "--mc", str(args.signal_mc),
                "--ridge-slope", str(ridge),
            ]
            signal_code = run_command(signal_cmd, root, ridge_dir / "signal_audit.log")
            try:
                signal_frames.append(signal_summary(ridge, signal_out))
            except Exception as exc:
                raise RuntimeError(
                    f"Signal-audit ridge run failed or produced incomplete output for ridge={ridge:g}; "
                    f"return_code={signal_code}. See {ridge_dir / 'signal_audit.log'}"
                ) from exc

        run_records.append(
            {
                "ridge_slope": ridge,
                "primary_return_code": primary_code,
                "signal_return_code": signal_code,
                "primary_output": str(primary_out),
                "signal_output": None if args.skip_signal_audit else str(signal_out),
            }
        )

    primary_all = pd.concat(primary_frames, ignore_index=True)
    support_all = pd.concat(support_frames, ignore_index=True)
    fit_all = pd.concat(fit_frames, ignore_index=True)
    primary_all.to_csv(out / "primary_positive_selection_ridge_summary.csv", index=False)
    support_all.to_csv(out / "primary_positive_selection_support_summary.csv", index=False)
    fit_all.to_csv(out / "primary_positive_selection_fit_summary.csv", index=False)

    range_rows = range_audit(
        primary_all,
        "batch",
        ["crossfit_gap_pct", "crossfit_minus_naive_pp", "crossfit_minus_same_sample_pp"],
    )

    signal_all: pd.DataFrame | None = None
    if signal_frames:
        signal_all = pd.concat(signal_frames, ignore_index=True)
        signal_all.to_csv(out / "policy_driver_ridge_summary.csv", index=False)
        range_rows.extend(
            range_audit(
                signal_all,
                "batch",
                ["ir_only_gap_pct", "tmax_only_gap_pct", "omitted_driver_extra_pp", "adding_tmax_to_driver_pp"],
            )
        )

    ranges = pd.DataFrame(range_rows)
    ranges.to_csv(out / "across_ridge_range_audit.csv", index=False)

    report = {
        "analysis": "non-selective positive-selection ridge sensitivity",
        "interpretation": (
            "All requested ridge candidates are reported using common seeds and folds. "
            "This is a robustness audit, not a ridge-selection exercise."
        ),
        "root": str(root),
        "matr": str(matr),
        "ridges": [float(x) for x in args.ridges],
        "R": int(args.R),
        "signal_mc": int(args.signal_mc),
        "signal_audit_included": not args.skip_signal_audit,
        "runs": run_records,
        "maximum_across_ridge_range": float(ranges["range"].max()) if not ranges.empty else None,
        "all_reported_metric_signs_consistent": bool(ranges["sign_consistent"].all()) if not ranges.empty else None,
        "outputs": {
            "primary": str(out / "primary_positive_selection_ridge_summary.csv"),
            "support": str(out / "primary_positive_selection_support_summary.csv"),
            "fits": str(out / "primary_positive_selection_fit_summary.csv"),
            "signal": None if signal_all is None else str(out / "policy_driver_ridge_summary.csv"),
            "ranges": str(out / "across_ridge_range_audit.csv"),
        },
    }
    (out / "ridge_sensitivity_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("POSITIVE-SELECTION RIDGE SENSITIVITY COMPLETED")
    print(f"out_dir={out}")
    print("No preferred ridge was selected; inspect across_ridge_range_audit.csv.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
