#!/usr/bin/env python3
"""Install audited manuscript macros and promote recomputed revision results when available."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

BATCHES = {
    "MATR-05-12": "May",
    "MATR-06-30": "June",
    "MATR-04-12": "April",
}


def _renew(name: str, value: object) -> str:
    return rf"\renewcommand{{\{name}}}{{{value}}}"


def _fmt(x: object, digits: int = 3) -> str:
    return f"{float(x):.{digits}f}"


def _sci_tex(x: object, digits: int = 2) -> str:
    value = float(x)
    if value == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10 ** exponent)
    return rf"{mantissa:.{digits}f}\times10^{{{exponent}}}"


def _simulation_overrides(root: Path) -> list[str]:
    path = root / "results" / "simulation" / "estimator_summary.csv"
    if not path.exists():
        return []
    d = pd.read_csv(path)
    mapping = {
        ("quadratic_misspec", "naive_km"): "SimQuadNaive",
        ("quadratic_misspec", "ipcw_tv_fitted"): "SimQuadFitted",
        ("quadratic_misspec", "ipcw_tv_oracle"): "SimQuadOracle",
        ("coarse_updates", "naive_km"): "SimCoarseNaive",
        ("coarse_updates", "ipcw_tv_fitted"): "SimCoarseFitted",
        ("coarse_updates", "ipcw_tv_oracle"): "SimCoarseOracle",
        ("strong_selection", "ipcw_tv_fitted"): "SimStrongFitted",
        ("heavy_censoring", "ipcw_tv_fitted"): "SimHeavyFitted",
        ("small_n", "ipcw_tv_fitted"): "SimSmallFitted",
    }
    lines: list[str] = []
    for (scenario, arm), macro in mapping.items():
        row = d[(d["scenario"] == scenario) & (d["arm"] == arm)]
        if len(row) != 1:
            return []
        lines.append(_renew(macro, _fmt(row.iloc[0]["mean_bias_pct"], 2)))
    return lines


def _bootstrap_overrides(root: Path) -> tuple[list[str], bool]:
    path = root / "results" / "matr_bootstrap" / "redesign_summary.csv"
    if not path.exists():
        return [], False
    d = pd.read_csv(path)
    required = {
        "naive_signed_gap_pct",
        "crossfit_tv_ipcw_signed_gap_pct",
        "oracle_tv_ipcw_signed_gap_pct",
        "crossfit_minus_naive_signed_error_pp",
        "crossfit_minus_naive_absolute_error_pp",
    }
    if not required.issubset(set(d.get("statistic", []))):
        return [], False
    lines: list[str] = [
        _renew("BootstrapTruthReady", 1),
        _renew("MatrBootB", int(d["prefix_B"].max())),
        _renew("MatrBootR", int(d["R_inner"].max())),
    ]
    stat_map = {
        "naive_signed_gap_pct": "Naive",
        "crossfit_tv_ipcw_signed_gap_pct": "Cross",
        "crossfit_minus_naive_signed_error_pp": "Diff",
        "crossfit_minus_naive_absolute_error_pp": "AbsDiff",
    }
    for batch, prefix in BATCHES.items():
        for stat, stem in stat_map.items():
            row = d[(d["batch_label"] == batch) & (d["statistic"] == stat)]
            if len(row) != 1:
                raise RuntimeError(f"Expected one bootstrap row for {batch}/{stat}; found {len(row)}")
            r = row.iloc[0]
            lines.extend([
                _renew(f"Matr{prefix}{stem}BootMean", _fmt(r["outer_mean"])),
                _renew(f"Matr{prefix}{stem}Lo", _fmt(r["descriptive_percentile_lo"])),
                _renew(f"Matr{prefix}{stem}Hi", _fmt(r["descriptive_percentile_hi"])),
            ])
        oracle = d[(d["batch_label"] == batch) & (d["statistic"] == "oracle_tv_ipcw_signed_gap_pct")]
        if len(oracle) != 1:
            raise RuntimeError(f"Expected one oracle bootstrap row for {batch}; found {len(oracle)}")
        r = oracle.iloc[0]
        lines.extend([
            _renew(f"Matr{prefix}OraclePLLo", _fmt(r["descriptive_percentile_lo"])),
            _renew(f"Matr{prefix}OraclePLHi", _fmt(r["descriptive_percentile_hi"])),
        ])
    return lines, True


def _primary_overrides(root: Path) -> tuple[list[str], bool, dict[str, dict[str, float]]]:
    est_path = root / "results" / "matr_primary" / "estimator_summary.csv"
    gap_path = root / "results" / "matr_primary" / "estimand_gap.csv"
    support_path = root / "results" / "matr_primary" / "support_summary.csv"
    if not all(p.exists() for p in (est_path, gap_path, support_path)):
        return [], False, {}
    est = pd.read_csv(est_path)
    gap = pd.read_csv(gap_path)
    support = pd.read_csv(support_path)
    est = est[est["beta"].astype(float) == 1.0]
    gap = gap[gap["beta"].astype(float) == 1.0]
    support = support[(support["beta"].astype(float) == 1.0) &
                      (support["checkpoint_fraction_H"].astype(float) == 1.0)]
    arms = {"naive", "crossfit_tv_ipcw", "same_sample_tv_ipcw", "oracle_tv_ipcw", "oracle_ht_rmst"}
    if not arms.issubset(set(est["arm"])):
        return [], False, {}
    lines: list[str] = [_renew("PrimaryRecomputeReady", 1)]
    records: dict[str, dict[str, float]] = {}
    contrasts = []
    for batch, prefix in BATCHES.items():
        rows = est[est["scope"] == batch].set_index("arm")
        if not arms.issubset(set(rows.index)):
            raise RuntimeError(f"Missing primary estimator rows for {batch}")
        g = gap[gap["scope"] == batch]
        if len(g) != 1:
            raise RuntimeError(f"Expected one estimand row for {batch}; found {len(g)}")
        g = g.iloc[0]
        s = support[(support["scope"] == batch) & (support["arm"] == "crossfit_tv_ipcw")]
        if len(s) != 1:
            raise RuntimeError(f"Expected one support row for {batch}; found {len(s)}")
        s = s.iloc[0]
        naive = float(rows.loc["naive", "mean_signed_gap_pct"])
        cross = float(rows.loc["crossfit_tv_ipcw", "mean_signed_gap_pct"])
        same = float(rows.loc["same_sample_tv_ipcw", "mean_signed_gap_pct"])
        oracle_pl = float(rows.loc["oracle_tv_ipcw", "mean_signed_gap_pct"])
        oracle_ht = float(rows.loc["oracle_ht_rmst", "mean_signed_gap_pct"])
        diff = cross - naive
        net = float(g["truth_net_rmst"])
        crude = float(g["exact_crude_rmst"])
        estimand_gap = float(g["estimand_gap"])
        correction_cycles = -diff * net / 100.0
        naive_cycles = naive * net / 100.0
        cross_cycles = cross * net / 100.0
        ratio = estimand_gap / correction_cycles
        contrasts.append(diff)
        records[batch] = {
            "naive_bias": naive,
            "crossfit_bias": cross,
            "same_sample_bias": same,
            "crossfit_minus_same": cross - same,
            "contrast": diff,
            "oracle_pl_bias": oracle_pl,
            "oracle_ht_bias": oracle_ht,
            "crossfit_minus_oracle": cross - oracle_pl,
            "n": float(rows.loc["naive", "n_units"]),
            "H": float(rows.loc["naive", "H"]),
            "net_rmst": net,
            "crude": crude,
            "estimand_gap": estimand_gap,
            "estimand_gap_pct": float(g["estimand_gap_pct_of_net"]),
            "p10_ess_risk_H": float(s["p10_ess_over_n_at_risk"]),
            "p95_max_weight_H": float(s["p95_max_weight"]),
        }
        lines.extend([
            _renew(f"Matr{prefix}N", int(rows.loc["naive", "n_units"])),
            _renew(f"Matr{prefix}H", _fmt(rows.loc["naive", "H"], 1)),
            _renew(f"Matr{prefix}Naive", _fmt(naive)),
            _renew(f"Matr{prefix}Cross", _fmt(cross)),
            _renew(f"Matr{prefix}Same", _fmt(same)),
            _renew(f"Matr{prefix}CrossMinusSame", _fmt(cross - same)),
            _renew(f"Matr{prefix}Diff", _fmt(diff)),
            _renew(f"Matr{prefix}OraclePL", _fmt(oracle_pl)),
            _renew(f"Matr{prefix}OracleHT", _fmt(oracle_ht)),
            _renew(f"Matr{prefix}FitOracleGap", _fmt(cross - oracle_pl)),
            _renew(f"Matr{prefix}Net", _fmt(net, 2)),
            _renew(f"Matr{prefix}Crude", _fmt(crude, 2)),
            _renew(f"Matr{prefix}EstimandGap", _fmt(estimand_gap, 2)),
            _renew(f"Matr{prefix}EstimandGapPct", _fmt(g["estimand_gap_pct_of_net"])),
            _renew(f"Matr{prefix}ESS", _fmt(s["p10_ess_over_n_at_risk"])),
            _renew(f"Matr{prefix}PninetyFiveW", _fmt(s["p95_max_weight"], 2)),
            _renew(f"Matr{prefix}NaiveCycles", _fmt(naive_cycles, 2)),
            _renew(f"Matr{prefix}CrossCycles", _fmt(cross_cycles, 2)),
            _renew(f"Matr{prefix}CorrectionCycles", _fmt(correction_cycles, 2)),
            _renew(f"Matr{prefix}GapRatio", _fmt(ratio, 2)),
        ])
    lines.append(_renew("MatrEqualDiff", _fmt(np.mean(contrasts))))
    return lines, True, records


def _any_exit_overrides(root: Path) -> tuple[list[str], bool]:
    path = root / "results" / "matr_primary" / "estimand_gap.csv"
    if not path.exists():
        return [], False
    d = pd.read_csv(path)
    required = {"beta", "scope", "exact_any_exit_rmst", "any_exit_minus_net"}
    if not required.issubset(d.columns):
        return [], False
    d = d[d["beta"].astype(float) == 1.0]
    lines: list[str] = [_renew("AnyExitReady", 1)]
    for batch, prefix in BATCHES.items():
        row = d[d["scope"] == batch]
        if len(row) != 1:
            raise RuntimeError(f"Expected one any-exit row for {batch}; found {len(row)}")
        r = row.iloc[0]
        lines.extend([
            _renew(f"Matr{prefix}AnyExit", _fmt(r["exact_any_exit_rmst"], 2)),
            _renew(f"Matr{prefix}AnyExitMinusNet", _fmt(r["any_exit_minus_net"], 2)),
            _renew(f"Matr{prefix}AnyExitShortfall", _fmt(-float(r["any_exit_minus_net"]), 2)),
        ])
    return lines, True


def _ridge_overrides(root: Path) -> tuple[list[str], bool]:
    audit_path = root / "validation" / "ridge_selection_audit.json"
    cand_path = root / "results" / "matr_ridge_audit" / "candidate_summary.csv"
    if not (audit_path.exists() and cand_path.exists()):
        return [], False
    audit = json.loads(audit_path.read_text(encoding="utf-8-sig"))
    if audit.get("status") != "PASS":
        return [], False
    selected = float(audit["selected_ridge_slope"])
    d = pd.read_csv(cand_path)
    d = d[d["ridge_slope"].astype(float) == selected]
    if len(d) != 3:
        return [], False
    return [
        _renew("RidgeAuditReady", 1),
        _renew("MatrRidgeSelected", _fmt(selected, 0)),
        _renew("MatrRidgePtenESSMin", _fmt(d["p10_ess_over_risk"].min())),
        _renew("MatrRidgePtenESSMax", _fmt(d["p10_ess_over_risk"].max())),
        _renew("MatrRidgePninetyNineWMin", _fmt(d["p99_max_weight"].min(), 2)),
        _renew("MatrRidgePninetyNineWMax", _fmt(d["p99_max_weight"].max(), 2)),
    ], True



def _positive_ridge_overrides(root: Path) -> tuple[list[str], bool]:
    directory = root / "results" / "matr_positive_ridge_sensitivity"
    primary_path = directory / "primary_positive_selection_ridge_summary.csv"
    support_path = directory / "primary_positive_selection_support_summary.csv"
    fit_path = directory / "primary_positive_selection_fit_summary.csv"
    signal_path = directory / "policy_driver_ridge_summary.csv"
    validation_path = directory / "ridge_4_validation.json"
    paths = [primary_path, support_path, fit_path, signal_path, validation_path]
    if not all(path.exists() for path in paths):
        return [], False

    primary = pd.read_csv(primary_path)
    support = pd.read_csv(support_path)
    fits = pd.read_csv(fit_path)
    signal = pd.read_csv(signal_path)
    validation = json.loads(validation_path.read_text(encoding="utf-8-sig"))

    ridges = [4.0, 16.0, 64.0]
    ridge_names = {4.0: "Four", 16.0: "Sixteen", 64.0: "SixtyFour"}
    if set(np.round(primary["ridge_slope"].astype(float), 8)) != set(ridges):
        return [], False
    if set(primary["batch"]) != set(BATCHES):
        return [], False

    lines: list[str] = [
        _renew("PositiveRidgeReady", 1),
        _renew("PositiveRidgeR", int(primary["R"].min())),
    ]
    global_ess_min = float(support["p10_ess_over_n_at_risk"].min())
    global_w_max = float(support["max_max_weight"].max())
    lines.extend([
        _renew("MatrRidgeESSGlobalMin", _fmt(global_ess_min)),
        _renew("MatrRidgeWGlobalMax", _fmt(global_w_max, 2)),
    ])

    all_add_tmax = pd.to_numeric(signal["adding_tmax_to_driver_pp"], errors="coerce")
    lines.extend([
        _renew("MatrAllAddTmaxMin", _fmt(all_add_tmax.min())),
        _renew("MatrAllAddTmaxMax", _fmt(all_add_tmax.max())),
    ])

    for batch, prefix in BATCHES.items():
        p = primary[primary["batch"] == batch].copy()
        s = support[support["batch"] == batch].copy()
        f = fits[fits["batch"] == batch].copy()
        q = signal[signal["batch"] == batch].copy()
        if not all(len(frame) == 3 for frame in (p, s, f, q)):
            raise RuntimeError(f"Expected three positive-ridge rows for {batch}")
        truth = float(p["truth_net_rmst"].iloc[0])
        move_cycles: list[float] = []
        for ridge in ridges:
            stem = ridge_names[ridge]
            pr = p[np.isclose(p["ridge_slope"].astype(float), ridge)].iloc[0]
            fr = f[np.isclose(f["ridge_slope"].astype(float), ridge)].iloc[0]
            pp = float(pr["crossfit_minus_naive_pp"])
            cycles = -pp * truth / 100.0
            move_cycles.append(cycles)
            lines.extend([
                _renew(f"Matr{prefix}Move{stem}", _fmt(pp)),
                _renew(f"Matr{prefix}MoveCycles{stem}", _fmt(cycles, 2)),
                _renew(f"Matr{prefix}Slope{stem}", _fmt(fr["mean_fitted_slope"])),
            ])
        lines.extend([
            _renew(f"Matr{prefix}MoveCyclesMin", _fmt(min(move_cycles), 2)),
            _renew(f"Matr{prefix}MoveCyclesMax", _fmt(max(move_cycles), 2)),
            _renew(f"Matr{prefix}RidgeESSMin", _fmt(s["p10_ess_over_n_at_risk"].min())),
            _renew(f"Matr{prefix}RidgeWMax", _fmt(s["max_max_weight"].max(), 2)),
            _renew(f"Matr{prefix}CrossSameMin", _fmt(p["crossfit_minus_same_sample_pp"].min())),
            _renew(f"Matr{prefix}CrossSameMax", _fmt(p["crossfit_minus_same_sample_pp"].max())),
            _renew(f"Matr{prefix}OmittedMin", _fmt(q["omitted_driver_extra_pp"].min())),
            _renew(f"Matr{prefix}OmittedMax", _fmt(q["omitted_driver_extra_pp"].max())),
            _renew(f"Matr{prefix}TmaxOnlyMin", _fmt(q["tmax_only_gap_pct"].min())),
            _renew(f"Matr{prefix}TmaxOnlyMax", _fmt(q["tmax_only_gap_pct"].max())),
            _renew(f"Matr{prefix}AddTmaxMin", _fmt(q["adding_tmax_to_driver_pp"].min())),
            _renew(f"Matr{prefix}AddTmaxMax", _fmt(q["adding_tmax_to_driver_pp"].max())),
        ])

    event = validation.get("beta1_crossfit_event_support_by_batch", {})
    for batch, prefix in BATCHES.items():
        row = event.get(batch)
        if not isinstance(row, dict):
            return [], False
        lines.extend([
            _renew(f"MatrRidgeFour{prefix}EventPninetyFive", _fmt(row["p95_max_weighted_hazard_increment"])),
            _renew(f"MatrRidgeFour{prefix}EventMax", _fmt(row["max_max_weighted_hazard_increment"])),
        ])
    lines.append(_renew(
        "MatrRidgeFourTotalFits",
        int(validation.get("solver_diagnostics", {}).get("total_fits", 0)),
    ))
    return lines, True


def _bootstrap_support_overrides(root: Path) -> tuple[list[str], bool]:
    path = root / "results" / "matr_bootstrap" / "support_summary.csv"
    if not path.exists():
        return [], False
    d = pd.read_csv(path)
    d = d[d["arm"] == "crossfit_tv_ipcw"].set_index("batch_label")
    if not set(BATCHES).issubset(set(d.index)):
        return [], False
    rows = d.loc[list(BATCHES)]
    lines = [
        _renew("BootstrapSupportReady", 1),
        _renew("MatrBootCrossPninetyNineWMin", _fmt(rows["p99_max_weight"].min(), 2)),
        _renew("MatrBootCrossPninetyNineWMax", _fmt(rows["p99_max_weight"].max(), 2)),
        _renew("MatrBootCrossPoneESSMin", _fmt(rows["p01_min_ess"].min())),
        _renew("MatrBootCrossPoneESSMax", _fmt(rows["p01_min_ess"].max())),
        _renew("MatrBootCrossMaxWMay", _fmt(d.loc["MATR-05-12", "max_weight"], 2)),
        _renew("MatrBootCrossMaxWJune", _sci_tex(d.loc["MATR-06-30", "max_weight"])),
        _renew("MatrBootCrossMaxWApril", _sci_tex(d.loc["MATR-04-12", "max_weight"])),
    ]
    return lines, True


def _update_figure_reference(root: Path, records: dict[str, dict[str, float]], primary_ready: bool) -> None:
    if not primary_ready:
        return
    candidates = [
        root / "results" / "reference_values.json",
        root / "manuscript" / "results" / "reference_values.json",
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        return
    d = json.loads(path.read_text(encoding="utf-8-sig"))
    for batch, values in records.items():
        target = d["matr"]["batches"][batch]
        target.update(values)
    d["matr"]["equal_batch_contrast"] = float(np.mean([v["contrast"] for v in records.values()]))
    path.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--output", type=Path, default=None)
    a = p.parse_args()
    root = a.root.resolve()
    template = root / "results" / "generated_results_template.tex"
    flat_output = root / "generated_results.tex"
    nested_output = root / "manuscript" / "generated_results.tex"
    if a.output is not None:
        out = a.output.resolve()
    elif (root / "manuscript.tex").exists() or flat_output.exists():
        out = flat_output.resolve()
    else:
        out = nested_output.resolve()
    src = template if template.exists() else out
    if not src.exists():
        raise FileNotFoundError(
            f"Missing macro template. Expected {template} or existing {out}."
        )
    text = src.read_text(encoding="utf-8-sig")
    marker = "% Dynamic revision overrides written by build_manuscript_results.py"
    if marker in text:
        text = text.split(marker, 1)[0].rstrip()
    additions: list[str] = ["", "% Dynamic revision overrides written by build_manuscript_results.py"]
    additions.extend(_simulation_overrides(root))
    primary, primary_ready, records = _primary_overrides(root)
    boot, boot_ready = _bootstrap_overrides(root)
    any_exit, any_exit_ready = _any_exit_overrides(root)
    ridge, ridge_ready = _ridge_overrides(root)
    positive_ridge, positive_ridge_ready = _positive_ridge_overrides(root)
    boot_support, boot_support_ready = _bootstrap_support_overrides(root)
    additions.extend(primary)
    additions.extend(boot)
    additions.extend(any_exit)
    additions.extend(ridge)
    additions.extend(positive_ridge)
    additions.extend(boot_support)
    additions.append(f"% full primary recomputation ready: {primary_ready}")
    additions.append(f"% resample-specific bootstrap truth ready: {boot_ready}")
    additions.append(f"% any-exit estimand ready: {any_exit_ready}")
    additions.append(f"% ridge audit ready: {ridge_ready}")
    additions.append(f"% positive-selection ridge sensitivity ready: {positive_ridge_ready}")
    additions.append(f"% bootstrap support diagnostics ready: {boot_support_ready}")
    out.write_text(text.rstrip() + "\n" + "\n".join(additions) + "\n", encoding="utf-8")

    _update_figure_reference(root, records, primary_ready)

    boot_src = root / "results" / "matr_bootstrap" / "redesign_summary.csv"
    if (root / "manuscript.tex").exists():
        fig_dst = root / "results" / "bootstrap_redesign_summary.csv"
    else:
        fig_dst = root / "manuscript" / "results" / "bootstrap_redesign_summary.csv"
    if boot_src.exists() and boot_ready:
        fig_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(boot_src, fig_dst)

    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
