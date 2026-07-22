#!/usr/bin/env python3
"""Consolidate validated sensitivity outputs into manuscript-ready evidence tables.

The script does not select tuning parameters, suppress adverse findings or
replace structurally unattainable policy designs with a different target.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def read(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def find(rows, **kw):
    out = [
        r for r in rows
        if all(
            str(r[k]) == str(v) if not isinstance(v, float)
            else abs(float(r[k]) - v) < 1e-9
            for k, v in kw.items()
        )
    ]
    if len(out) != 1:
        raise RuntimeError(f"expected one row for {kw}, got {len(out)}")
    return out[0]


def fmt(x, d=3):
    return f"{float(x):.{d}f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    a = ap.parse_args()
    root = a.root.resolve()
    out = (a.out_dir or root / "results" / "sensitivity").resolve()
    out.mkdir(parents=True, exist_ok=True)

    a_dir = root / "results" / "matr_policy"
    b_dir = root / "results" / "matr_aipw"
    c_dir = root / "results" / "matr_horizon_eol"
    validations = [
        a_dir / "validation.json",
        b_dir / "validation.json",
        c_dir / "validation.json",
    ]
    validation_payloads = [
        json.loads(p.read_text(encoding="utf-8")) if p.exists()
        else {"status": "MISSING"}
        for p in validations
    ]

    eng = read(a_dir / "engineering_comparison.csv")
    b_env = read(b_dir / "envelopes.csv")
    b_sum = read(b_dir / "summary.csv")
    h_sum = read(c_dir / "horizon_summary.csv")
    e_sum = read(c_dir / "eol_summary.csv")
    e_est = read(c_dir / "eol_policy_estimability.csv")
    h_audit = read(c_dir / "horizon_definition_audit.csv")

    batches = ["MATR-05-12", "MATR-06-30", "MATR-04-12"]
    intensity = []
    for target in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        for b in batches:
            intensity.append(find(
                eng, rule="continuous_cloglog",
                target_replacement=target, batch=b
            ))

    rule40 = []
    for rule in ["continuous_cloglog", "soft_threshold", "periodic_25", "periodic_50"]:
        for b in batches:
            rule40.append(find(
                eng, rule=rule, target_replacement=0.4, batch=b
            ))

    aipw = []
    for b in batches:
        for signal in ["IR-only", "Tmax-only", "IR+Tmax"]:
            env = find(b_env, batch=b, signal_set=signal)
            ref = find(
                b_sum, batch=b, signal_set=signal, event_ridge=2.0,
                transition_ridge=0.1, estimator="crossfit_longitudinal_AIPW"
            )
            aipw.append({
                **env,
                "frozen_reference_mcse": ref["mcse"],
                "frozen_reference_p2p5": ref["p2p5"],
                "frozen_reference_p97p5": ref["p97p5"],
            })

    horizon = []
    for b in batches:
        for mult in [0.6, 0.8, 1.0]:
            naive = find(h_sum, batch=b, horizon_multiplier=mult, estimator="naive")
            fitted = find(h_sum, batch=b, horizon_multiplier=mult, estimator="crossfit_TV_IPCW")
            horizon.append({
                "batch": b,
                "horizon_multiplier": mult,
                "naive_bias_pct_net": naive["mean"],
                "crossfit_bias_pct_net": fitted["mean"],
                "correction_pp": float(naive["mean"]) - float(fitted["mean"]),
                "naive_mcse": naive["mcse"],
                "crossfit_mcse": fitted["mcse"],
            })

    estimability = {
        (round(float(r["EOL_fraction"]), 2), r["batch"]): r
        for r in e_est
    }
    eol = []
    for b in batches:
        for frac in [0.78, 0.8, 0.82]:
            for mode in ["adaptive_batch_median", "fixed_original_batch_median"]:
                status = estimability[(frac, b)]["analysis_status"]
                if status == "estimable":
                    naive = find(
                        e_sum, EOL_fraction=frac, batch=b,
                        horizon_mode=mode, estimator="naive"
                    )
                    fitted = find(
                        e_sum, EOL_fraction=frac, batch=b,
                        horizon_mode=mode, estimator="crossfit_TV_IPCW"
                    )
                    eol.append({
                        "batch": b,
                        "EOL_fraction": frac,
                        "horizon_mode": mode,
                        "analysis_status": "estimable",
                        "naive_bias_pct_net": naive["mean"],
                        "crossfit_bias_pct_net": fitted["mean"],
                        "correction_pp": float(naive["mean"]) - float(fitted["mean"]),
                        "naive_mcse": naive["mcse"],
                        "crossfit_mcse": fitted["mcse"],
                    })
                else:
                    eol.append({
                        "batch": b,
                        "EOL_fraction": frac,
                        "horizon_mode": mode,
                        "analysis_status": status,
                        "naive_bias_pct_net": "",
                        "crossfit_bias_pct_net": "",
                        "correction_pp": "",
                        "naive_mcse": "",
                        "crossfit_mcse": "",
                    })

    sys.path.insert(0, str(root / "code" / "src"))
    import sensitivity_common as c

    c.write_csv(out / "sensitivity_policy_intensity_headline.csv", intensity)
    c.write_csv(out / "sensitivity_policy_rule_40pct_headline.csv", rule40)
    c.write_csv(out / "sensitivity_aipw_regularization_headline.csv", aipw)
    c.write_csv(out / "sensitivity_horizon_headline.csv", horizon)
    c.write_csv(out / "sensitivity_EOL_headline.csv", eol)

    macros = ["% Auto-generated by matr_sensitivity_summary.py"]
    for b, tag in [
        ("MATR-05-12", "May"),
        ("MATR-06-30", "June"),
        ("MATR-04-12", "April"),
    ]:
        p10 = find(intensity, batch=b, target_replacement=0.1)
        p40 = find(intensity, batch=b, target_replacement=0.4)
        p60 = find(intensity, batch=b, target_replacement=0.6)
        macros += [
            f"\\newcommand{{\\PolicyAudit{tag}GapTen}}{{{fmt(p10['estimand_gap_cycles'], 2)}}}",
            f"\\newcommand{{\\PolicyAudit{tag}GapForty}}{{{fmt(p40['estimand_gap_cycles'], 2)}}}",
            f"\\newcommand{{\\PolicyAudit{tag}GapSixty}}{{{fmt(p60['estimand_gap_cycles'], 2)}}}",
            f"\\newcommand{{\\PolicyAudit{tag}CorrectionTen}}{{{fmt(p10['TV_IPCW_correction_cycles'], 2)}}}",
            f"\\newcommand{{\\PolicyAudit{tag}CorrectionForty}}{{{fmt(p40['TV_IPCW_correction_cycles'], 2)}}}",
            f"\\newcommand{{\\PolicyAudit{tag}CorrectionSixty}}{{{fmt(p60['TV_IPCW_correction_cycles'], 2)}}}",
        ]
        env = find(aipw, batch=b, signal_set="IR-only")
        macros += [
            f"\\newcommand{{\\PolicyAudit{tag}AIPWMin}}{{{fmt(env['min_mean_bias_pct_net'], 2)}}}",
            f"\\newcommand{{\\PolicyAudit{tag}AIPWMax}}{{{fmt(env['max_mean_bias_pct_net'], 2)}}}",
        ]
    (out / "sensitivity_generated_results.tex").write_text(
        "\n".join(macros) + "\n", encoding="utf-8"
    )

    nonestimable = [r for r in e_est if r["analysis_status"] != "estimable"]
    checks = {
        "policy_validation_pass": validation_payloads[0].get("status") == "PASS",
        "aipw_validation_pass": validation_payloads[1].get("status") == "PASS",
        "horizon_eol_validation_pass": validation_payloads[2].get("status") == "PASS",
        "intensity_rows": len(intensity) == 18,
        "rule40_rows": len(rule40) == 12,
        "aipw_rows": len(aipw) == 9,
        "horizon_rows": len(horizon) == 9,
        "EOL_rows_with_status": len(eol) == 18,
        "structural_nonestimability_retained": len(nonestimable) >= 1,
        "May82_not_forced_to_different_policy": any(
            abs(float(r["EOL_fraction"]) - 0.82) < 1e-9
            and r["batch"] == "MATR-05-12"
            and r["analysis_status"] == "not_estimable_policy_incompatible"
            for r in e_est
        ),
    }
    report = {
        "analysis": "sensitivity_evidence_finalization",
        "status": "PASS" if all(checks.values()) else "REVIEW_REQUIRED",
        "checks": checks,
        "interpretation_constraints": [
            "replacement rates are imposed diagnostic intensities, not empirical maintenance prevalence",
            "estimand-gap dominance must be stated conditionally by intensity and rule",
            "AIPW grid is a fairness audit and no penalty is selected",
            "EOL sensitivity preserves the frozen 124 unit IDs",
            "structurally unattainable EOL-by-batch policy designs are reported as not estimable and are not replaced by a different target",
            "no cross-chemistry or cross-asset transport claim",
        ],
        "validation_reports": [str(p) for p in validations],
        "horizon_definition_audit": h_audit,
        "EOL_nonestimable_designs": nonestimable,
    }
    c.json_dump(out / "sensitivity_final_evidence_report.json", report)

    print("SENSITIVITY EVIDENCE SUMMARY COMPLETED")
    print(f"status={report['status']}")
    for r in intensity:
        if float(r["target_replacement"]) in {0.1, 0.4, 0.6}:
            print(
                f"{r['batch']:12s} target={float(r['target_replacement']):.1f} "
                f"gap={float(r['estimand_gap_cycles']):.2f} "
                f"correction={float(r['TV_IPCW_correction_cycles']):.2f} "
                f"ratio={float(r['gap_to_abs_correction_ratio']):.2f}"
            )
    print("EOL_nonestimable_designs=" + str([
        (r["EOL_fraction"], r["batch"], r["analysis_status"])
        for r in nonestimable
    ]))
    print(f"out_dir={out}")
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
