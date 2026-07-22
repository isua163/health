#!/usr/bin/env python3
"""Validate the clean submission reproducibility package."""
from __future__ import annotations

import argparse
import compileall
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

TITLE = r"\title{Health-triggered replacement: estimands and dependent censoring}"
REQUIRED_DIRS = [
    "code/src", "code/tests", "results/simulation", "results/matr_cohort",
    "results/matr_primary", "results/matr_positive_ridge_sensitivity", "results/matr_bootstrap", "results/matr_estimator_signal",
    "results/matr_temperature", "results/matr_policy", "results/matr_aipw",
    "results/matr_horizon_eol", "results/matr_endpoint_rule", "results/xjtu", "manuscript/figures",
]
REQUIRED_SCRIPTS = [
    "simulation_benchmark.py", "matr_data.py", "matr_endpoint_reconstruction.py",
    "matr_primary_analysis.py", "matr_bootstrap_design.py", "matr_bootstrap_run.py",
    "matr_bootstrap_summarize.py", "matr_bootstrap_validate.py",
    "matr_endpoint_rule_audit.py", "matr_ridge_selection_reaudit.py",
    "matr_positive_selection_ridge_sensitivity.py",
    "matr_estimator_signal_analysis.py", "matr_temperature_extension.py",
    "matr_policy_sensitivity.py", "matr_aipw_regularization.py",
    "matr_horizon_eol_sensitivity.py", "xjtu_primary_analysis.py",
    "xjtu_support_diagnostics.py", "build_manuscript_results.py",
    "validate_figures.py", "run_tests.py",
]
REQUIRED_RESULTS = [
    "results/simulation/estimator_summary.csv",
    "results/matr_cohort/cohort_decision.csv",
    "results/matr_cohort/endpoint_review.csv",
    "results/matr_primary/estimator_summary.csv",
    "results/matr_primary/paired_correction_summary.csv",
    "results/matr_primary/estimand_gap.csv",
    "results/matr_positive_ridge_sensitivity/primary_positive_selection_ridge_summary.csv",
    "results/matr_positive_ridge_sensitivity/primary_positive_selection_support_summary.csv",
    "results/matr_positive_ridge_sensitivity/primary_positive_selection_fit_summary.csv",
    "results/matr_positive_ridge_sensitivity/policy_driver_ridge_summary.csv",
    "results/matr_positive_ridge_sensitivity/across_ridge_range_audit.csv",
    "results/matr_endpoint_rule/endpoint_provenance.csv",
    "results/matr_endpoint_rule/endpoint_rule_sensitivity.csv",
    "results/matr_bootstrap/redesign_summary.csv",
    "results/matr_bootstrap/support_summary.csv",
    "results/matr_estimator_signal/summary.csv",
    "results/matr_temperature/four_batch_summary.csv",
    "results/matr_policy/engineering_comparison.csv",
    "results/matr_aipw/envelopes.csv",
    "results/matr_horizon_eol/horizon_summary.csv",
    "results/matr_horizon_eol/eol_summary.csv",
    "results/xjtu/crossfit_summary.csv",
    "results/xjtu/horizon_support.csv",
    "results/xjtu/weight_diagnostics_summary.csv",
]
HISTORY_PATTERN = re.compile(r"\b" + "M" + r"(?:1[0-9]|[1-9])[a-z]?\b")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    report_path = (args.report or root / "validation" / "release_validation.json").resolve()
    failures: list[str] = []
    checks: dict[str, object] = {}

    checks["required_directories"] = all((root / item).is_dir() for item in REQUIRED_DIRS)
    if not checks["required_directories"]:
        failures.append("one or more required directories are missing")

    missing_scripts = [name for name in REQUIRED_SCRIPTS if not (root / "code" / name).is_file()]
    checks["required_scripts"] = not missing_scripts
    if missing_scripts:
        failures.append(f"missing required scripts: {missing_scripts}")

    missing_results = [name for name in REQUIRED_RESULTS if not (root / name).is_file()]
    checks["required_result_tables"] = not missing_results
    if missing_results:
        failures.append(f"missing required result tables: {missing_results}")

    py_files = sorted((root / "code").rglob("*.py")) + sorted((root / "manuscript" / "figures").glob("generate_Figure_*.py"))
    history_hits: list[str] = []
    for path in py_files:
        text = path.read_text(encoding="utf-8")
        for match in HISTORY_PATTERN.finditer(text):
            # APR18650M1A is a battery model, not an internal workflow label.
            if "APR18650M1A" in text[max(0, match.start()-20):match.end()+20]:
                continue
            line = text.count("\n", 0, match.start()) + 1
            history_hits.append(f"{path.relative_to(root)}:{line}:{match.group(0)}")
    checks["no_internal_workflow_labels_in_scripts"] = not history_hits
    if history_hits:
        failures.append(f"internal workflow labels remain in scripts: {history_hits[:10]}")

    manuscript = root / "manuscript" / "manuscript.tex"
    tex = manuscript.read_text(encoding="utf-8") if manuscript.is_file() else ""
    checks["title_exact"] = TITLE in tex
    if not checks["title_exact"]:
        failures.append("manuscript title does not match the final title")

    citations = re.findall(r"\\cite[a-zA-Z]*\{([^}]+)\}", tex)
    max_citations = max((len([x for x in group.split(",") if x.strip()]) for group in citations), default=0)
    checks["maximum_references_per_citation"] = max_citations
    if max_citations > 4:
        failures.append(f"a citation location contains {max_citations} references")

    cited_keys = {key.strip() for group in citations for key in group.split(",") if key.strip()}
    bib_keys = set(re.findall(r"\\bibitem\{([^}]+)\}", tex))
    undefined = sorted(cited_keys - bib_keys)
    uncited = sorted(bib_keys - cited_keys)
    checks["undefined_citation_keys"] = undefined
    checks["uncited_bibliography_keys"] = uncited
    if undefined:
        failures.append(f"undefined citation keys: {undefined}")
    if uncited:
        failures.append(f"uncited bibliography entries: {uncited}")

    macro_template = root / "results" / "generated_results_template.tex"
    macro_installed = root / "manuscript" / "generated_results.tex"
    expected_macro = root / "validation" / "_generated_results_expected.tex"
    try:
        subprocess.run(
            [sys.executable, str(root / "code" / "build_manuscript_results.py"),
             "--root", str(root), "--output", str(expected_macro)],
            cwd=root, check=True, stdout=subprocess.DEVNULL,
        )
        checks["generated_results_match"] = (
            macro_template.is_file() and macro_installed.is_file()
            and expected_macro.read_bytes() == macro_installed.read_bytes()
        )
    except Exception as exc:
        checks["generated_results_match"] = False
        checks["generated_results_error"] = str(exc)
    finally:
        expected_macro.unlink(missing_ok=True)
    if not checks["generated_results_match"]:
        failures.append("manuscript generated_results.tex is stale relative to audited/dynamic results")

    figure_names = [f"Figure_{i}.pdf" for i in range(1, 5)]
    checks["figure_pdfs_present"] = all((root / "manuscript" / "figures" / name).is_file() for name in figure_names)
    if not checks["figure_pdfs_present"]:
        failures.append("one or more manuscript figure PDFs are missing")

    checks["python_compile"] = compileall.compile_dir(root / "code", quiet=1, force=True)
    if not checks["python_compile"]:
        failures.append("Python compilation failed")

    pdf = root / "manuscript" / "manuscript.pdf"
    checks["manuscript_pdf_present"] = pdf.is_file() and pdf.stat().st_size > 50_000
    checks["manuscript_pdf_sha256"] = sha256(pdf) if checks["manuscript_pdf_present"] else None
    if not checks["manuscript_pdf_present"]:
        failures.append("compiled manuscript PDF is missing or implausibly small")

    status = "PASS" if not failures else "FAIL"
    report = {
        "package": root.name,
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "checks": checks,
        "failures": failures,
        "python_files_checked": len(py_files),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"RELEASE VALIDATION {status}")
    print(f"report={report_path}")
    if failures:
        for failure in failures:
            print(f" - {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
