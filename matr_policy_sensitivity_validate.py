#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
import numpy as np

def read(path):
    with path.open('r',encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out-dir',required=True,type=Path); ap.add_argument('--expected-R',type=int,default=200); a=ap.parse_args(); out=a.out_dir.resolve()
    required=['report.json','replicates.csv','summary.csv','estimand_gaps.csv','engineering_comparison.csv','fit_diagnostics.csv','support.csv']
    checks={f'has_{x}':(out/x).exists() for x in required}
    if not all(checks.values()):
        print('MATR POLICY SENSITIVITY VALIDATION COMPLETED'); print('status=FAIL'); [print(f'{k}={v}') for k,v in checks.items()]; return 2
    report=json.loads((out/'report.json').read_text(encoding='utf-8'))
    reps=read(out/'replicates.csv'); summ=read(out/'summary.csv'); gaps=read(out/'estimand_gaps.csv'); eng=read(out/'engineering_comparison.csv')
    rules={'continuous_cloglog','soft_threshold','periodic_25','periodic_50'}; targets={0.1,0.2,0.3,0.4,0.5,0.6}; batches={'MATR-05-12','MATR-06-30','MATR-04-12'}; est={'naive','oracle_product_limit','oracle_HT_RMST','crossfit_TV_IPCW'}
    keys={(r['rule'],round(float(r['target_replacement']),2),r['batch'],r['estimator'],int(float(r['replicate']))) for r in reps}
    checks.update({
      'report_pass':report.get('status')=='PASS',
      'row_count':len(reps)==a.expected_R*len(rules)*len(targets)*len(batches)*len(est),
      'complete_keys':len(keys)==len(reps),
      'rules_complete':{r['rule'] for r in reps}==rules,
      'targets_complete':{round(float(r['target_replacement']),2) for r in reps}==targets,
      'batches_complete':{r['batch'] for r in reps}==batches,
      'estimators_complete':{r['estimator'] for r in reps}==est,
      'finite_estimates':all(np.isfinite(float(r['estimate'])) for r in reps),
      'realized_rates_valid':all(0<=float(r['realized_replacement'])<=1 for r in reps),
      'summary_complete':len(summ)==len(rules)*len(targets)*len(batches)*len(est),
      'gap_complete':len(gaps)==len(rules)*len(targets)*len(batches),
      'engineering_complete':len(eng)==len(rules)*len(targets)*len(batches),
      'primary_40_present':all(any(r['rule']=='continuous_cloglog' and abs(float(r['target_replacement'])-.4)<1e-9 and r['batch']==b for r in eng) for b in batches),
    })
    status='PASS' if all(checks.values()) else 'FAIL'
    payload={'analysis':'matr_policy_sensitivity_validation','status':status,'checks':checks,'expected_R':a.expected_R}
    (out/'validation.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print('MATR POLICY SENSITIVITY VALIDATION COMPLETED'); print(f'status={status}'); [print(f'{k}={v}') for k,v in checks.items()]
    return 0 if status=='PASS' else 2
if __name__=='__main__': raise SystemExit(main())
