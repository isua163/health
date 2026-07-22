#!/usr/bin/env python3
"""Run the submission package's deterministic scientific test suite."""
from __future__ import annotations

import shlex
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
    root = Path(__file__).resolve().parents[1]
    test_files = sorted((root / "code" / "tests").glob("test_*.py"))
    if not test_files:
        print("SCIENTIFIC TEST SUITE FAILED: no tests found", file=sys.stderr)
        return 2

    env_prefix = (
        "env PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1"
    )
    targets: list[str] = []
    for path in test_files:
        rel = str(path.relative_to(root))
        if path.name == "test_real_pipeline.py":
            targets.extend(f"{rel}::{node}" for node in REAL_PIPELINE_NODES)
        else:
            targets.append(rel)

    for target in targets:
        print(f"TEST_TARGET {target}", flush=True)
        command = f"{env_prefix} pytest -q {shlex.quote(target)}"
        completed = subprocess.run(["bash", "-lc", command], cwd=root)
        if completed.returncode != 0:
            print(f"SCIENTIFIC TEST SUITE FAILED: {target}", file=sys.stderr)
            return completed.returncode
    print(f"SCIENTIFIC TEST SUITE PASSED ({len(targets)} targets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
