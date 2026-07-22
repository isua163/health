#!/usr/bin/env python3
"""Validate manuscript-facing MATR bootstrap summaries."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--results-dir', type=Path, default=None)
    a = p.parse_args()
    root = a.root.resolve()
    r = (a.results_dir or root / 'results' / 'matr_bootstrap').resolve()
    x = pd.read_csv(r / 'redesign_summary.csv')
    s = pd.read_csv(r / 'support_summary.csv')
    expected = {
        'naive_signed_gap_pct', 'oracle_tv_ipcw_signed_gap_pct',
        'crossfit_tv_ipcw_signed_gap_pct',
        'crossfit_minus_naive_signed_error_pp',
        'crossfit_minus_naive_absolute_error_pp',
    }
    truth_path = r / 'resample_truth_audit.csv'
    truth = pd.read_csv(truth_path) if truth_path.is_file() else pd.DataFrame()
    checks = {
        'fifteen_summary_rows': len(x) == 15,
        'three_batches': x.batch_label.nunique() == 3,
        'all_statistics_present': expected.issubset(set(x.statistic)),
        'support_rows_present': len(s) >= 9,
        'all_finite_descriptive_percentiles': bool(
            x[['descriptive_percentile_lo', 'descriptive_percentile_hi']].notna().all().all()
        ),
        'resample_truth_audit_present': truth_path.is_file(),
        'resample_truth_unique_within_outer_batch': bool(
            not truth.empty and not truth.duplicated(['outer_b', 'batch_label']).any()
        ),
        'resample_truth_positive': bool(
            not truth.empty and (truth.truth_net_rmst.astype(float) > 0).all()
        ),
    }
    status = 'PASS' if all(checks.values()) else 'FAIL'
    out = {'analysis': 'MATR descriptive redesign validation', 'status': status, 'checks': checks}
    (r / 'validation.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print('MATR BOOTSTRAP VALIDATION COMPLETED')
    print(f'status={status}')
    return 0 if status == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
