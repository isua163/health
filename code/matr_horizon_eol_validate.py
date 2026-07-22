#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--expected-R", type=int, default=200)
    a = ap.parse_args()
    out = a.out_dir.resolve()

    required = [
        "matr_horizon_eol_horizon_EOL_report.json",
        "matr_horizon_eol_horizon_definition_audit.csv",
        "matr_horizon_eol_horizon_replicates.csv",
        "matr_horizon_eol_horizon_summary.csv",
        "matr_horizon_eol_horizon_exact_estimand_gaps.csv",
        "matr_horizon_eol_EOL_endpoint_reconstruction.csv",
        "matr_horizon_eol_EOL_policy_estimability.csv",
        "matr_horizon_eol_EOL_replicates.csv",
        "matr_horizon_eol_EOL_summary.csv",
        "matr_horizon_eol_EOL_exact_estimand_gaps.csv",
    ]
    checks = {f"has_{x}": (out / x).exists() for x in required}
    if not all(checks.values()):
        print("matr_horizon_eol HORIZON/EOL VALIDATION COMPLETED")
        print("status=FAIL")
        for k, v in checks.items():
            print(f"{k}={v}")
        return 2

    report = json.loads((out / "matr_horizon_eol_horizon_EOL_report.json").read_text(encoding="utf-8"))
    h = read(out / "matr_horizon_eol_horizon_replicates.csv")
    e = read(out / "matr_horizon_eol_EOL_replicates.csv")
    ep = read(out / "matr_horizon_eol_EOL_endpoint_reconstruction.csv")
    ha = read(out / "matr_horizon_eol_horizon_definition_audit.csv")
    estm = read(out / "matr_horizon_eol_EOL_policy_estimability.csv")
    gaps = read(out / "matr_horizon_eol_EOL_exact_estimand_gaps.csv")

    batches = {"MATR-05-12", "MATR-06-30", "MATR-04-12"}
    multipliers = {0.6, 0.8, 1.0}
    fractions = {0.78, 0.8, 0.82}
    modes = {"adaptive_batch_median", "fixed_original_batch_median"}
    estimators = {"naive", "oracle_product_limit", "oracle_HT_RMST", "crossfit_TV_IPCW"}

    estimable_pairs = {
        (round(float(r["EOL_fraction"]), 2), r["batch"])
        for r in estm if r["analysis_status"] == "estimable"
    }
    nonestimable_pairs = {
        (round(float(r["EOL_fraction"]), 2), r["batch"])
        for r in estm if r["analysis_status"] == "not_estimable_policy_incompatible"
    }
    expected_eol_per_rep = len(estimable_pairs) * len(modes) * len(estimators)

    observed_eol_pairs = {
        (round(float(r["EOL_fraction"]), 2), r["batch"]) for r in e
    }
    row_counts = {}
    for r in e:
        key = (
            int(float(r["replicate"])),
            round(float(r["EOL_fraction"]), 2),
            r["batch"],
            r["horizon_mode"],
            r["estimator"],
        )
        row_counts[key] = row_counts.get(key, 0) + 1

    gap_nonestimable = {
        (round(float(r["EOL_fraction"]), 2), r["batch"])
        for r in gaps if r["analysis_status"] == "not_estimable_policy_incompatible"
    }

    checks.update({
        "report_pass": report.get("status") == "PASS",
        "horizon_row_count": len(h) == a.expected_R * 3 * 3 * 4,
        "EOL_row_count_dynamic": len(e) == a.expected_R * expected_eol_per_rep,
        "endpoint_row_count": len(ep) == 124 * 3,
        "estimability_row_count": len(estm) == 3 * 3,
        "exact_gap_row_count": len(gaps) == 3 * 3 * 2,
        "horizon_batches": {r["batch"] for r in h} == batches,
        "horizon_multipliers": {round(float(r["horizon_multiplier"]), 2) for r in h} == multipliers,
        "EOL_fractions_endpoint": {round(float(r["EOL_fraction"]), 2) for r in ep} == fractions,
        "horizon_modes": {r["horizon_mode"] for r in e} == modes,
        "estimators_horizon": {r["estimator"] for r in h} == estimators,
        "estimators_EOL": {r["estimator"] for r in e} == estimators,
        "finite_horizon": all(np.isfinite(float(r["estimate"])) for r in h),
        "finite_EOL": all(np.isfinite(float(r["estimate"])) for r in e),
        "EOL80_exact_match": all(
            abs(float(r["difference_from_frozen"])) < 1e-12
            for r in ep if abs(float(r["EOL_fraction"]) - 0.8) < 1e-9
        ),
        "H_audit_three_batches": len(ha) == 3 and {r["batch"] for r in ha} == batches,
        "all_80pct_batches_estimable": all((0.8, b) in estimable_pairs for b in batches),
        "May82_structural_boundary_documented": (0.82, "MATR-05-12") in nonestimable_pairs,
        "unattainable_pairs_absent_from_overlay_rows": observed_eol_pairs.isdisjoint(nonestimable_pairs),
        "all_estimable_pairs_present": observed_eol_pairs == estimable_pairs,
        "nonestimable_pairs_present_in_gap_table": nonestimable_pairs == gap_nonestimable,
        "one_row_per_expected_key": all(v == 1 for v in row_counts.values()),
        "no_unreported_calibration_failure": all(
            r["analysis_status"] in {"estimable", "not_estimable_policy_incompatible"}
            for r in estm
        ),
    })

    status = "PASS" if all(checks.values()) else "FAIL"
    payload = {
        "analysis": "matr_horizon_eol_validation",
        "status": status,
        "checks": checks,
        "expected_R": a.expected_R,
        "estimable_pairs": [
            {"EOL_fraction": f, "batch": b} for f, b in sorted(estimable_pairs)
        ],
        "nonestimable_pairs": [
            {"EOL_fraction": f, "batch": b} for f, b in sorted(nonestimable_pairs)
        ],
        "interpretation": (
            "A structurally unattainable EOL-by-batch policy design is documented and omitted from overlay estimation; "
            "the frozen target is not replaced by a different policy."
        ),
    }
    (out / "matr_horizon_eol_horizon_EOL_validation.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    print("matr_horizon_eol HORIZON/EOL VALIDATION COMPLETED")
    print(f"status={status}")
    print(f"estimable_pairs={len(estimable_pairs)}")
    print(f"nonestimable_pairs={sorted(nonestimable_pairs)}")
    for k, v in checks.items():
        print(f"{k}={v}")
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
