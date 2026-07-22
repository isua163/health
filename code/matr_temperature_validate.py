#!/usr/bin/env python3
"""Validate temperature extension four-batch Tmax benchmark outputs."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np

BATCHES = {"MATR-05-12", "MATR-06-30", "MATR-04-12", "MATR-CLO"}
EST = {"naive", "oracle_product_limit", "oracle_ht_rmst", "crossfit_tv_ipcw"}


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--expected-R", type=int, default=200)
    a=ap.parse_args()
    p=a.out_dir.resolve()
    report=json.loads((p/"temperature_extension_report.json").read_text(encoding="utf-8"))
    rep=read_csv(p/"temperature_extension_replicates.csv")
    fit=read_csv(p/"temperature_extension_fit_diagnostics.csv")
    summ=read_csv(p/"temperature_extension_summary.csv")
    folds=read_csv(p/"temperature_extension_fold_assignment.csv")

    combos={(r["batch"],r["estimator"]) for r in summ}
    expected={(b,e) for b in BATCHES for e in EST}
    reps={int(float(r["replicate"])) for r in rep}
    fit_grad=np.array([float(r["grad_norm"]) for r in fit],float)
    realized={b:np.mean([float(r["realized_censor"]) for r in rep if r["batch"]==b and r["estimator"]=="naive"]) for b in BATCHES}
    checks={
        "report_pass": report.get("status")=="PASS",
        "replicate_count_complete": reps==set(range(a.expected_R)),
        "all_combinations": combos==expected,
        "expected_row_count": len(rep)==a.expected_R*len(BATCHES)*len(EST),
        "all_estimates_finite": all(np.isfinite(float(r["estimate"])) for r in rep),
        "all_fit_gradients_finite": bool(np.all(np.isfinite(fit_grad))),
        "all_fits_successful": all(str(r["success"]).lower() in {"true","1"} for r in fit),
        "no_post_censor_records": all(int(float(r["post_censor_records_used"]))==0 for r in fit),
        "fold_rows_169": len(folds)==169,
        "realized_censor_reasonable": all(0.25 <= x <= 0.55 for x in realized.values()),
    }
    status="PASS" if all(checks.values()) else "REVIEW_REQUIRED"
    out={"analysis":"temperature extension_Tmax_four_batch_validation","status":status,
         "checks":checks,"mean_realized_censor":realized}
    (p/"temperature_extension_validation.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    print("temperature extension TMAX FOUR-BATCH VALIDATION COMPLETED")
    print(f"status={status}")
    for k,v in checks.items(): print(f"{k}={v}")
    for b,v in sorted(realized.items()): print(f"{b}_mean_realized_censor={v:.4f}")
    return 0 if status=="PASS" else 2

if __name__=="__main__": raise SystemExit(main())
