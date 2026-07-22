#!/usr/bin/env python3
"""Summarize five-unit XJTU static-IPCW fit failures from condition-sensitivity replicates."""
from pathlib import Path
import argparse
import pandas as pd


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,default=Path('results/xjtu/condition_sensitivity_replicates.csv'))
    ap.add_argument('--output',type=Path,default=Path('results/xjtu/static_fit_failure_diagnostics.csv'))
    a=ap.parse_args()
    d=pd.read_csv(a.input)
    q=d[(d.arm=='ipcw_static')&(d.analysis_scope=='within_condition')].copy()
    rows=[]
    for (group,weighting),g in q.groupby(['group','threshold_weighting'],sort=True):
        ok=g.fit_success.astype(int)==1
        rows.append(dict(
            group=group,
            threshold_weighting=weighting,
            attempted=len(g),
            successful=int(ok.sum()),
            failed=int((~ok).sum()),
            failure_fraction=float((~ok).mean()),
            mean_censor_beta0_success=float(g.loc[ok,'realized_censor_beta0'].mean()),
            mean_censor_beta0_failure=float(g.loc[~ok,'realized_censor_beta0'].mean()),
            mean_censor_beta1_success=float(g.loc[ok,'realized_censor_beta1'].mean()),
            mean_censor_beta1_failure=float(g.loc[~ok,'realized_censor_beta1'].mean()),
            failure_message_contains_max_iterations=float(g.loc[~ok,'fit_message'].fillna('').str.contains('maximum iterations').mean()),
        ))
    out=pd.DataFrame(rows)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(a.output,index=False)
    print(out.to_string(index=False))
    print('wrote',a.output)

if __name__=='__main__': main()
