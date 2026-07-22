#!/usr/bin/env python3
"""Validate the estimator-signal expanded estimator benchmark and apply the frozen gate."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

BATCHES = {"MATR-05-12", "MATR-06-30", "MATR-04-12"}
SIGNALS = {"IR-only", "Tmax-only", "IR+Tmax"}
ESTIMATORS = {"crossfit_tv_ipcw", "crossfit_longitudinal_aipw",
              "outcome_gformula_diagnostic"}


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--expected-R", type=int, default=200)
    args = ap.parse_args()
    out = args.out_dir.resolve()

    report = json.loads((out / "estimator_signal_report.json").read_text(encoding="utf-8"))
    reps = read_csv(out / "estimator_signal_replicates.csv")
    summary = read_csv(out / "estimator_signal_summary.csv")
    fits = read_csv(out / "estimator_signal_fit_diagnostics.csv")
    hazards = read_csv(out / "estimator_signal_hazard_summary.csv")

    rep_ids = sorted({int(float(r["replicate"])) for r in reps})
    combos = {(r["batch"], r["signal_set"], r["estimator"]) for r in summary}
    expected_combos = {(b, s, e) for b in BATCHES for s in SIGNALS for e in ESTIMATORS}
    computational_checks = {
        "report_pass": report.get("status") == "PASS",
        "replicate_count_complete": rep_ids == list(range(args.expected_R)),
        "all_signal_estimator_combinations": expected_combos.issubset(combos),
        "all_estimates_finite": all(np.isfinite(float(r["estimate"])) for r in reps),
        "no_post_censor_records": all(int(float(r["post_censor_records_used"])) == 0 for r in fits),
        "finite_fit_gradients": all(np.isfinite(float(r["censor_grad"])) and
                                    np.isfinite(float(r["event_grad"])) for r in fits),
        "hazard_summaries_complete": len(hazards) == args.expected_R * len(BATCHES) * len(SIGNALS),
    }

    aipw = [r for r in summary if r["estimator"] == "crossfit_longitudinal_aipw"]
    tv = [r for r in summary if r["estimator"] == "crossfit_tv_ipcw"]
    max_abs_aipw = max(abs(float(r["mean"])) for r in aipw)
    max_abs_tv = max(abs(float(r["mean"])) for r in tv)
    total_clip = sum(int(float(r["clip_count"])) for r in hazards)
    total_times = sum(int(float(r["n_hazard_times"])) for r in hazards)
    clip_fraction = total_clip / max(total_times, 1)

    # Frozen scientific gate.  A computationally complete but unstable AIPW is
    # retained as a benchmark failure; it is not promoted into B=4000 fleet inference.
    aipw_stable_for_fleet_bootstrap = bool(max_abs_aipw <= 5.0 and clip_fraction <= 0.05)
    status = "PASS" if all(computational_checks.values()) else "REVIEW_REQUIRED"
    validation = {
        "analysis": "estimator-signal_expanded_benchmark_validation",
        "status": status,
        "computational_checks": computational_checks,
        "scientific_gate": {
            "max_abs_aipw_mean_bias_pct": max_abs_aipw,
            "max_abs_tv_ipcw_mean_bias_pct": max_abs_tv,
            "aipw_hazard_clip_fraction": clip_fraction,
            "aipw_stable_for_fleet_bootstrap": aipw_stable_for_fleet_bootstrap,
            "aipw_role_if_gate_fails": "retain as negative diagnostic comparator; do not tune and do not include in the B=4000 joint redesign analysis",
        },
        "next_action": (
            "Proceed to the signal extension and joint redesign analysis with the TV-IPCW signal comparison and empirical B=4000 redesign ranges; "
            "report AIPW as an endpoint-benchmark instability result."
            if not aipw_stable_for_fleet_bootstrap else
            "AIPW passed the frozen stability gate and may be considered for a future joint redesign analysis."
        ),
    }
    (out / "estimator_signal_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8")

    print("estimator-signal EXPANDED BENCHMARK VALIDATION COMPLETED")
    print(f"status={status}")
    for key, value in computational_checks.items():
        print(f"{key}={value}")
    print(f"max_abs_aipw_mean_bias_pct={max_abs_aipw:.6f}")
    print(f"max_abs_tv_ipcw_mean_bias_pct={max_abs_tv:.6f}")
    print(f"aipw_hazard_clip_fraction={clip_fraction:.6f}")
    print(f"aipw_stable_for_fleet_bootstrap={aipw_stable_for_fleet_bootstrap}")
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
