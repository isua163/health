#!/usr/bin/env python3
"""Validate the revised flat QREI submission package.

By default this validates the source/PDF/code package without requiring the raw-
data calculations. Use ``--require-generated-results`` after the Windows
revision workflow to require all reviewer-requested output contracts.
"""
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

TITLE = r"\title{A structured audit of censoring corrections under health-triggered replacement}"
REQUIRED_DIRS = ["code/src", "code/tests", "figures", "revision"]
REQUIRED_SCRIPTS = [
    "simulation_benchmark.py", "matr_data.py", "matr_endpoint_reconstruction.py",
    "matr_primary_analysis.py", "matr_positive_selection_ridge_sensitivity.py",
    "revision_metrics.py", "matr_nested_preprocessing_audit.py",
    "build_revision_macros.py", "build_manuscript_results.py",
    "validate_figures.py", "run_tests.py",
]
REQUIRED_STATIC_FILES = [
    "manuscript.tex", "manuscript.pdf", "generated_results.tex",
    "revision_results.tex", "environment.yml", "run_revision_windows.bat",
    "REVISION_README.md", "response_to_reviewers.md",
]
REQUIRED_GENERATED = [
    "results/matr_primary/replicates.csv",
    "results/matr_primary/estimator_summary.csv",
    "results/revision_metrics/revision_accuracy_overlays.csv",
    "results/revision_metrics/revision_accuracy_summary.csv",
    "results/matr_nested_preprocessing/replicates.csv",
    "results/matr_nested_preprocessing/estimator_summary.csv",
    "results/matr_nested_preprocessing/paired_summary.csv",
    "results/matr_nested_preprocessing/fold_preprocessing_and_fit.csv",
    "results/matr_nested_preprocessing/heldout_calibration.csv",
    "results/matr_nested_preprocessing/weight_diagnostics.csv",
    "results/matr_nested_preprocessing/event_time_diagnostics.csv",
    "results/matr_nested_preprocessing/unit_dominance.csv",
    "results/matr_nested_preprocessing/design.json",
    "results/matr_positive_ridge_sensitivity/primary_positive_selection_ridge_summary.csv",
    "validation/matr_cohort_validation.json",
]


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
    parser.add_argument("--require-generated-results", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    report_path = (args.report or root / "validation" / "release_validation.json").resolve()
    failures: list[str] = []
    checks: dict[str, object] = {}

    missing_dirs = [item for item in REQUIRED_DIRS if not (root / item).is_dir()]
    checks["required_directories"] = not missing_dirs
    if missing_dirs:
        failures.append(f"missing required directories: {missing_dirs}")

    missing_scripts = [name for name in REQUIRED_SCRIPTS if not (root / "code" / name).is_file()]
    checks["required_scripts"] = not missing_scripts
    if missing_scripts:
        failures.append(f"missing required scripts: {missing_scripts}")

    missing_static = [name for name in REQUIRED_STATIC_FILES if not (root / name).is_file()]
    checks["required_static_files"] = not missing_static
    if missing_static:
        failures.append(f"missing required static files: {missing_static}")

    missing_generated = [name for name in REQUIRED_GENERATED if not (root / name).is_file()]
    checks["generated_result_contracts_present"] = not missing_generated
    checks["generated_result_contracts_missing"] = missing_generated
    if args.require_generated_results and missing_generated:
        failures.append(f"missing generated result files: {missing_generated}")

    manuscript = root / "manuscript.tex"
    tex = manuscript.read_text(encoding="utf-8") if manuscript.is_file() else ""
    checks["title_exact"] = TITLE in tex
    if not checks["title_exact"]:
        failures.append("manuscript title does not match the revised title")

    cited_keys = {
        key.strip()
        for group in re.findall(r"\\cite[a-zA-Z]*\{([^}]+)\}", tex)
        for key in group.split(",") if key.strip()
    }
    bib_keys = set(re.findall(r"\\bibitem\{([^}]+)\}", tex))
    undefined = sorted(cited_keys - bib_keys)
    uncited = sorted(bib_keys - cited_keys)
    checks["undefined_citation_keys"] = undefined
    checks["uncited_bibliography_keys"] = uncited
    if undefined:
        failures.append(f"undefined citation keys: {undefined}")
    if uncited:
        failures.append(f"uncited bibliography entries: {uncited}")

    expected_revision = root / "validation" / "_revision_results_expected.tex"
    expected_revision.parent.mkdir(parents=True, exist_ok=True)
    returned_manifest = root / "results" / "returned_results_manifest.json"
    full_macro_inputs = [
        root / "results" / "revision_metrics" / "revision_accuracy_summary.csv",
        root / "results" / "matr_nested_preprocessing" / "estimator_summary.csv",
        root / "results" / "matr_nested_preprocessing" / "paired_summary.csv",
        root / "results" / "matr_nested_preprocessing" / "weight_diagnostics.csv",
        root / "results" / "matr_nested_preprocessing" / "unit_dominance.csv",
        root / "results" / "matr_nested_preprocessing" / "heldout_calibration.csv",
        root / "results" / "matr_nested_preprocessing" / "event_time_diagnostics.csv",
        root / "results" / "matr_nested_preprocessing" / "fold_preprocessing_and_fit.csv",
    ]
    if all(path.exists() for path in full_macro_inputs):
        try:
            subprocess.run(
                [sys.executable, str(root / "code" / "build_revision_macros.py"),
                 "--root", str(root), "--output", str(expected_revision)],
                cwd=root, check=True, stdout=subprocess.DEVNULL,
            )
            checks["revision_results_match"] = (
                (root / "revision_results.tex").read_bytes() == expected_revision.read_bytes()
            )
            checks["revision_results_validation_mode"] = "rebuilt_from_complete_local_outputs"
        except Exception as exc:  # pragma: no cover - diagnostic path
            checks["revision_results_match"] = False
            checks["revision_results_error"] = str(exc)
        finally:
            expected_revision.unlink(missing_ok=True)
    elif returned_manifest.exists():
        try:
            returned = json.loads(returned_manifest.read_text(encoding="utf-8"))
            mismatches = []
            for rel, expected_hash in returned.get("files", {}).items():
                target = root / rel
                if not target.is_file() or sha256(target) != expected_hash:
                    mismatches.append(rel)
            ready_text = (root / "revision_results.tex").read_text(encoding="utf-8")
            ready_flags = (
                r"\renewcommand{\RevisionMetricsReady}{1}" in ready_text
                and r"\renewcommand{\NestedAuditReady}{1}" in ready_text
            )
            checks["returned_result_hash_mismatches"] = mismatches
            checks["revision_results_match"] = not mismatches and ready_flags
            checks["revision_results_validation_mode"] = "verified_integrated_return_manifest"
        except Exception as exc:  # pragma: no cover - diagnostic path
            checks["revision_results_match"] = False
            checks["revision_results_error"] = str(exc)
    else:
        checks["revision_results_match"] = False
        checks["revision_results_validation_mode"] = "insufficient_inputs"
    if not checks["revision_results_match"]:
        failures.append("revision_results.tex is stale or not verifiable from returned result tables")

    figure_check = subprocess.run(
        [sys.executable, str(root / "code" / "validate_figures.py"), "--root", str(root)],
        cwd=root, capture_output=True, text=True, check=False,
    )
    checks["figure_validation"] = figure_check.returncode == 0
    if figure_check.returncode != 0:
        failures.append("figure validation failed")
        checks["figure_validation_output"] = figure_check.stdout + figure_check.stderr

    checks["python_compile"] = compileall.compile_dir(root / "code", quiet=1, force=True)
    if not checks["python_compile"]:
        failures.append("Python compilation failed")

    pdf = root / "manuscript.pdf"
    checks["manuscript_pdf_present"] = pdf.is_file() and pdf.stat().st_size > 50_000
    checks["manuscript_pdf_sha256"] = sha256(pdf) if checks["manuscript_pdf_present"] else None
    if not checks["manuscript_pdf_present"]:
        failures.append("compiled manuscript PDF is missing or implausibly small")

    status = "PASS" if not failures else "FAIL"
    report = {
        "package": root.name,
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "require_generated_results": bool(args.require_generated_results),
        "checks": checks,
        "failures": failures,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"RELEASE VALIDATION {status}")
    print(f"report={report_path}")
    for failure in failures:
        print(f" - {failure}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
