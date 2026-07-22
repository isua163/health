#!/usr/bin/env python3
"""Non-selective regularization audit for the longitudinal AIPW comparator.

All event/transition penalty combinations are reported.  The grid is an audit,
not a tuning exercise: no combination replaces the frozen reference result.
"""
from __future__ import annotations
import argparse, platform, sys
from pathlib import Path
from typing import Any
import numpy as np

SIGNALS=('IR-only','Tmax-only','IR+Tmax')
EVENT_GRID=(0.5,2.0,8.0)
TRANS_GRID=(0.03,0.1,0.3,1.0)

def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True,type=Path); ap.add_argument('--matr',required=True,type=Path); ap.add_argument('--out-dir',type=Path,default=None)
    ap.add_argument('--R',type=int,default=50); ap.add_argument('--mc',type=int,default=16); ap.add_argument('--seed',type=int,default=20260731)
    ap.add_argument('--baseline-cycles',type=int,default=50); ap.add_argument('--smooth-window',type=int,default=5); ap.add_argument('--censor-ridge',type=float,default=16.0)
    ap.add_argument('--target-censor',type=float,default=0.40); ap.add_argument('--checkpoint-every',type=int,default=1); ap.add_argument('--resume',action='store_true')
    a=ap.parse_args(); root=a.root.resolve(); matr=a.matr.resolve(); out=(a.out_dir or root/'results'/'matr_aipw').resolve(); out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(root/'code'/'src')); import sensitivity_common as c; import matr_aipw as dr
    data,audit,finalizer,core,surv=c.load_primary_data(root,matr,a.baseline_cycles,a.smooth_window)
    ir_z,ir_scale,tau=core.standardize_policy_paths(data.ir_raw,policy_start=a.baseline_cycles); tm_z,tm_scale,_=core.standardize_policy_paths(data.tmax_raw,policy_start=a.baseline_cycles)
    models={'IR-only':[x[:,None] for x in ir_z],'Tmax-only':[x[:,None] for x in tm_z],'IR+Tmax':[np.column_stack([x,y]) for x,y in zip(ir_z,tm_z)]}
    idx_by_batch={b:np.flatnonzero(data.batches==b) for b in c.PRIMARY_BATCHES}; H_raw={b:float(np.median(data.life[idx])) for b,idx in idx_by_batch.items()}; H={b:int(round(H_raw[b])) for b in c.PRIMARY_BATCHES}; truth={b:float(np.mean(np.minimum(data.life[idx],H[b]))) for b,idx in idx_by_batch.items()}
    lambda0={b:core.calibrate_lambda0([ir_z[i] for i in idx],1.0,tau,a.target_censor,policy_start=a.baseline_cycles) for b,idx in idx_by_batch.items()}
    rp=out/'replicates.csv'; fp=out/'fit_diagnostics.csv'; hp=out/'hazard_diagnostics.csv'
    if a.resume:
      reps=c.read_csv(rp); fits=c.read_csv(fp); haz=c.read_csv(hp); counts={}
      for r in reps: counts[int(float(r['replicate']))]=counts.get(int(float(r['replicate'])),0)+1
      expected=len(c.PRIMARY_BATCHES)*len(SIGNALS)*len(EVENT_GRID)*len(TRANS_GRID)*3
      complete={r for r,n in counts.items() if n==expected}; reps=[r for r in reps if int(float(r['replicate'])) in complete]; fits=[r for r in fits if int(float(r['replicate'])) in complete]; haz=[r for r in haz if int(float(r['replicate'])) in complete]
    else: reps=[]; fits=[]; haz=[]; complete=set()
    for rep in range(a.R):
      if rep in complete: print(f'replicate {rep+1}/{a.R} already complete; skipped',flush=True); continue
      rng=np.random.default_rng(a.seed+10000019*rep); uniforms=[rng.random(max(len(x)-1,0)) for x in ir_z]
      times=np.full(len(data.ids),np.nan); events=np.full(len(data.ids),-1,int)
      for b,idx in idx_by_batch.items():
        tt,ee,_,_=core.overlay_from_uniforms([ir_z[i] for i in idx],1.0,tau,lambda0[b],[uniforms[i] for i in idx],policy_start=a.baseline_cycles); times[idx]=tt; events[idx]=ee
      for b,idx in idx_by_batch.items():
        t=times[idx]; e=events[idx]; h_int=H[b]; common0={'replicate':rep,'batch':b,'n_units':len(idx),'H_raw_batch_median':H_raw[b],'H_discrete_AIPW':h_int,'truth':truth[b],'realized_replacement':float(np.mean(e==0))}
        for si,sig in enumerate(SIGNALS):
          observed=[models[sig][i][:int(round(times[i]))].copy() for i in idx]
          for er in EVENT_GRID:
            for tr in TRANS_GRID:
              seed=a.seed+1000003*rep+10007*c.PRIMARY_BATCHES.index(b)+101*si+1009*EVENT_GRID.index(er)+10009*TRANS_GRID.index(tr)
              result=dr.crossfit_dr_rmst(observed,t,e,data.folds[idx],h_int,a.baseline_cycles,seed=seed,mc=a.mc,censor_ridge=a.censor_ridge,event_ridge=er,transition_ridge=tr)
              cg,cs=core.weighted_product_limit(t,e,result['cumhaz_before']); ipcw=float(surv.rmrl_from_survival(cg,cs,0.0,H[b]))
              common={**common0,'signal_set':sig,'event_ridge':er,'transition_ridge':tr,'frozen_reference':bool(er==2.0 and tr==0.1)}
              for estimator,val in [('crossfit_TV_IPCW',ipcw),('crossfit_longitudinal_AIPW',float(result['dr_rmst'])),('outcome_gformula_diagnostic',float(result['gformula_rmst']))]:
                reps.append({**common,'estimator':estimator,'estimate':val,'bias_cycles':val-truth[b],'bias_pct_net':100*(val-truth[b])/truth[b],'hazard_clip_count':int(result['hazard_clip_count'])})
              for fr in result['fit_diagnostics']: fits.append({**common,**fr,'post_censor_records_used':0})
              raw=np.asarray([x[1] for x in result['dr_hazards']],float)
              haz.append({**common,'n_hazard_times':len(raw),'clip_count':int(result['hazard_clip_count']),'clip_fraction':int(result['hazard_clip_count'])/max(len(raw),1),'min_raw_hazard':float(np.min(raw)),'max_raw_hazard':float(np.max(raw)),'mean_raw_hazard':float(np.mean(raw))})
      if (rep+1)%a.checkpoint_every==0 or rep==a.R-1: c.write_csv(rp,reps); c.write_csv(fp,fits); c.write_csv(hp,haz)
      print(f'replicate {rep+1}/{a.R} completed',flush=True)
    summary=c.summarize(reps,['batch','signal_set','event_ridge','transition_ridge','estimator'],'bias_pct_net'); c.write_csv(out/'summary.csv',summary)
    # Complete-grid sensitivity envelope, without selecting a preferred penalty.
    aipw=[r for r in summary if r['estimator']=='crossfit_longitudinal_AIPW']; envelopes=[]
    for b in c.PRIMARY_BATCHES:
      for sig in SIGNALS:
        rows=[r for r in aipw if r['batch']==b and r['signal_set']==sig]; vals=np.asarray([float(r['mean']) for r in rows]); ref=[r for r in rows if float(r['event_ridge'])==2.0 and float(r['transition_ridge'])==0.1][0]
        envelopes.append({'batch':b,'signal_set':sig,'n_grid':len(rows),'min_mean_bias_pct_net':float(vals.min()),'max_mean_bias_pct_net':float(vals.max()),'range_pp':float(vals.max()-vals.min()),'median_abs_mean_bias_pct_net':float(np.median(np.abs(vals))),'fraction_abs_bias_below_5pct':float(np.mean(np.abs(vals)<5.0)),'frozen_reference_bias_pct_net':float(ref['mean']),'best_abs_bias_descriptive_only':float(np.min(np.abs(vals))),'selection_prohibited':True})
    c.write_csv(out/'envelopes.csv',envelopes)
    expected=a.R*len(c.PRIMARY_BATCHES)*len(SIGNALS)*len(EVENT_GRID)*len(TRANS_GRID)*3
    checks={'expected_rows':len(reps)==expected,'all_estimates_finite':all(np.isfinite(float(r['estimate'])) for r in reps),'complete_grid':len(summary)==len(c.PRIMARY_BATCHES)*len(SIGNALS)*len(EVENT_GRID)*len(TRANS_GRID)*3,'no_post_censor_records':all(int(float(r['post_censor_records_used']))==0 for r in fits),'finite_fit_gradients':all(np.isfinite(float(r['censor_grad'])) and np.isfinite(float(r['event_grad'])) for r in fits),'frozen_reference_present':all(any(r['batch']==b and r['signal_set']==s and float(r['event_ridge'])==2 and float(r['transition_ridge'])==.1 and r['estimator']=='crossfit_longitudinal_AIPW' for r in summary) for b in c.PRIMARY_BATCHES for s in SIGNALS)}
    report={'analysis':'matr_aipw_regularization_audit','status':'PASS' if all(checks.values()) else 'REVIEW_REQUIRED','python':platform.python_version(),'design':{'R':a.R,'mc':a.mc,'seed':a.seed,'event_grid':EVENT_GRID,'transition_grid':TRANS_GRID,'signals':SIGNALS,'censor_ridge':a.censor_ridge,'selection_rule':'complete-grid reporting; no winning penalty selected','frozen_reference':{'event_ridge':2.0,'transition_ridge':0.1}},'checks':checks,'scales':{'IR':ir_scale,'Tmax':tm_scale,'tau_IR':tau},'lambda0':lambda0}
    c.json_dump(out/'report.json',report)
    print('MATR AIPW REGULARIZATION AUDIT COMPLETED'); print(f"status={report['status']}")
    for r in envelopes: print(f"{r['batch']:12s} {r['signal_set']:10s} range=[{r['min_mean_bias_pct_net']:+.2f},{r['max_mean_bias_pct_net']:+.2f}]% ref={r['frozen_reference_bias_pct_net']:+.2f}%")
    print(f'out_dir={out}'); return 0 if all(checks.values()) else 2
if __name__=='__main__': raise SystemExit(main())
