#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np

def read(p):
    with p.open('r',encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out-dir',required=True,type=Path); ap.add_argument('--expected-R',type=int,default=50); a=ap.parse_args(); out=a.out_dir.resolve()
    req=['report.json','replicates.csv','summary.csv','fit_diagnostics.csv','hazard_diagnostics.csv','envelopes.csv']
    checks={f'has_{x}':(out/x).exists() for x in req}
    if not all(checks.values()):
        print('MATR AIPW REGULARIZATION VALIDATION COMPLETED');print('status=FAIL');[print(f'{k}={v}') for k,v in checks.items()];return 2
    reps=read(out/'replicates.csv'); summ=read(out/'summary.csv'); fits=read(out/'fit_diagnostics.csv'); env=read(out/'envelopes.csv'); report=json.loads((out/'report.json').read_text(encoding='utf-8'))
    batches={'MATR-05-12','MATR-06-30','MATR-04-12'}; sig={'IR-only','Tmax-only','IR+Tmax'}; eg={.5,2.,8.}; tg={.03,.1,.3,1.}; est={'crossfit_TV_IPCW','crossfit_longitudinal_AIPW','outcome_gformula_diagnostic'}
    expected=a.expected_R*len(batches)*len(sig)*len(eg)*len(tg)*len(est)
    keys={(int(float(r['replicate'])),r['batch'],r['signal_set'],float(r['event_ridge']),float(r['transition_ridge']),r['estimator']) for r in reps}
    checks.update({'report_pass':report.get('status')=='PASS','row_count':len(reps)==expected,'unique_complete_keys':len(keys)==len(reps),'batches_complete':{r['batch'] for r in reps}==batches,'signals_complete':{r['signal_set'] for r in reps}==sig,'event_grid_complete':{float(r['event_ridge']) for r in reps}==eg,'transition_grid_complete':{float(r['transition_ridge']) for r in reps}==tg,'estimators_complete':{r['estimator'] for r in reps}==est,'finite_estimates':all(np.isfinite(float(r['estimate'])) for r in reps),'no_post_censor':all(int(float(r['post_censor_records_used']))==0 for r in fits),'summary_complete':len(summ)==len(batches)*len(sig)*len(eg)*len(tg)*len(est),'envelopes_complete':len(env)==len(batches)*len(sig),'no_selection':all(str(r['selection_prohibited']).lower() in {'true','1'} for r in env)})
    status='PASS' if all(checks.values()) else 'FAIL'; payload={'analysis':'matr_aipw_regularization_validation','status':status,'checks':checks,'expected_R':a.expected_R}; (out/'validation.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print('MATR AIPW REGULARIZATION VALIDATION COMPLETED');print(f'status={status}');[print(f'{k}={v}') for k,v in checks.items()];return 0 if status=='PASS' else 2
if __name__=='__main__':raise SystemExit(main())
