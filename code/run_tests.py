#!/usr/bin/env python3
"""Run the deterministic scientific test suite on Windows, Linux, or macOS.

Artifact-contract tests are skipped only when their generated inputs are absent.
Use ``--require-artifacts`` after running the full MATR workflow to require every
contract test.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REAL_PIPELINE_NODES = [
    "test_hi_functions",
    "test_csv_reader_and_directory",
    "test_overlay_reproduces_optimism",
    "test_mat_reader",
    "test_tv_ipcw_recovers",
    "test_femto_reader",
    "test_cv_sweep_reproduces",
    "test_condition_inference_and_unit_equal_health_threshold",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-artifacts",
        action="store_true",
        help="Fail rather than skip tests whose generated validation/results files are absent.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    test_files = sorted((root / "code" / "tests").glob("test_*.py"))
    if not test_files:
        print("SCIENTIFIC TEST SUITE FAILED: no tests found", file=sys.stderr)
        return 2

    required_by_test = {
        "test_cohort_status_contract.py": [
            root / "validation" / "matr_cohort_validation.json",
        ],
        "test_positive_ridge_contract.py": [
            root / "results" / "matr_positive_ridge_sensitivity" /
            "primary_positive_selection_ridge_summary.csv",
            root / "results" / "matr_positive_ridge_sensitivity" /
            "primary_positive_selection_support_summary.csv",
        ],
    }

    targets: list[str] = []
    skipped: list[str] = []
    for path in test_files:
        missing = [p for p in required_by_test.get(path.name, []) if not p.exists()]
        if missing:
            message = f"{path.name}: missing " + ", ".join(str(p.relative_to(root)) for p in missing)
            if args.require_artifacts:
                print(f"SCIENTIFIC TEST SUITE FAILED: {message}", file=sys.stderr)
                return 2
            skipped.append(message)
            continue
        rel = str(path.relative_to(root))
        if path.name == "test_real_pipeline.py":
            targets.extend(f"{rel}::{node}" for node in REAL_PIPELINE_NODES)
        else:
            targets.append(rel)

    env = os.environ.copy()
    env.update({
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    })

    for message in skipped:
        print(f"SKIP_ARTIFACT_CONTRACT {message}", flush=True)
    for target in targets:
        print(f"TEST_TARGET {target}", flush=True)
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *targets],
        cwd=root,
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        print("SCIENTIFIC TEST SUITE FAILED", file=sys.stderr)
        return completed.returncode
    print(f"SCIENTIFIC TEST SUITE PASSED ({len(targets)} targets; {len(skipped)} artifact contracts skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
