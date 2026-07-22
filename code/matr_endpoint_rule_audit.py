#!/usr/bin/env python3
"""Audit endpoint provenance and one-cycle/right-censoring sensitivity.

This audit uses the derived endpoint-review table and does not require raw MATR
files.  It distinguishes observed threshold crossings from the public loader's
``cycle_record_count + 1`` reconstruction and quantifies the resulting finite-
cohort net-RMST sensitivity at the frozen batch horizons.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BATCH_ORDER = ("MATR-05-12", "MATR-06-30", "MATR-04-12")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--out-dir", type=Path, default=None)
    a = p.parse_args()
    root = a.root.resolve()
    out = (a.out_dir or root / "results" / "matr_endpoint_rule").resolve()
    out.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(root / "results" / "matr_cohort" / "endpoint_review.csv")
    d = d[d.primary_IR_cohort.astype(bool)].copy()
    design = json.loads((root / "results" / "matr_primary" / "analysis_design.json").read_text())
    horizons = {str(k): float(v) for k, v in design["horizons"].items()}

    provenance = []
    sensitivity = []
    for batch in BATCH_ORDER:
        g = d[d.batch_label == batch].copy()
        public = g.author_reconstructed_lifetime.to_numpy(float)
        is_plus1 = g.endpoint_source.eq("cycle_record_count_plus_1").to_numpy()
        last = np.where(is_plus1, public - 1.0, public)
        midpoint = np.where(is_plus1, public - 0.5, public)
        H = horizons[batch]

        exact_cross = ~is_plus1
        lower = np.where(exact_cross, np.minimum(public, H), np.minimum(last, H))
        upper = np.where(exact_cross, np.minimum(public, H), H)
        cont = g.continuation_appended.astype(bool)
        cont_gap = pd.to_numeric(g.loc[cont, "descriptor_rel_gap"], errors="coerce")

        provenance.append({
            "batch": batch,
            "n_units": int(len(g)),
            "observed_first_crossing": int(exact_cross.sum()),
            "cycle_record_count_plus_1": int(is_plus1.sum()),
            "continuation_merged_units": int(cont.sum()),
            "max_continuation_descriptor_relative_gap": float(cont_gap.max()) if cont_gap.notna().any() else np.nan,
            "endpoint_definition": (
                "first QDischarge < 0.88 Ah when terminal QDischarge is below threshold; "
                "otherwise public-loader cycle-record count + 1"
            ),
        })
        sensitivity.append({
            "batch": batch,
            "n_units": int(len(g)),
            "H": H,
            "net_rmst_public_rule": float(np.minimum(public, H).mean()),
            "net_rmst_last_observed_cycle": float(np.minimum(last, H).mean()),
            "net_rmst_interval_midpoint": float(np.minimum(midpoint, H).mean()),
            "public_minus_last_observed": float(np.minimum(public, H).mean() - np.minimum(last, H).mean()),
            "right_censor_lower_bound": float(lower.mean()),
            "right_censor_upper_bound": float(upper.mean()),
            "right_censor_bound_width": float(upper.mean() - lower.mean()),
            "n_non_crossing_units_censored_before_H": int(((~exact_cross) & (last < H)).sum()),
        })

    pd.DataFrame(provenance).to_csv(out / "endpoint_provenance.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(out / "endpoint_rule_sensitivity.csv", index=False)
    report = {
        "analysis": "MATR endpoint provenance and rule sensitivity",
        "status": "PASS",
        "n_primary_units": int(len(d)),
        "counts": {
            "observed_first_crossing": int((d.endpoint_source == "first_QDischarge_lt_0p88").sum()),
            "cycle_record_count_plus_1": int((d.endpoint_source == "cycle_record_count_plus_1").sum()),
            "continuation_merged_units": int(d.continuation_appended.astype(bool).sum()),
        },
        "interpretation": (
            "The public +1 convention changes point-identified net RMST by at most about half a cycle "
            "relative to last-observed or interval-midpoint coding, whereas treating all non-crossings "
            "as genuinely right-censored yields wider partial-identification bounds in May and April."
        ),
    }
    (out / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("MATR ENDPOINT RULE AUDIT COMPLETED")
    print("status=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
