#!/usr/bin/env python3
"""Replacement-intensity and maintenance-rule sensitivity scan.

The scan keeps the frozen 124-cell IR cohort and declared endpoint benchmarks.  Target
replacement fractions are imposed diagnostic design points, not estimates of
operational prevalence.  Four positive-probability rules are compared:
continuous cloglog (the primary rule), a steeper soft threshold, and the same
soft threshold evaluated at 25- or 50-cycle inspection occasions.
"""
from __future__ import annotations
import argparse, json, math, platform, sys
from pathlib import Path
from typing import Any
import numpy as np


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',required=True,type=Path); ap.add_argument('--matr',required=True,type=Path)
    ap.add_argument('--out-dir',type=Path,default=None); ap.add_argument('--R',type=int,default=200)
    ap.add_argument('--seed',type=int,default=20260730); ap.add_argument('--baseline-cycles',type=int,default=50)
    ap.add_argument('--smooth-window',type=int,default=5); ap.add_argument('--ridge-slope',type=float,default=16.0)
    ap.add_argument('--checkpoint-every',type=int,default=5); ap.add_argument('--resume',action='store_true')
    ap.add_argument('--weight-caps',type=float,nargs='*',default=[],help='Optional absolute fitted-weight caps for a prespecified positivity sensitivity grid')
    a=ap.parse_args(); root=a.root.resolve(); matr=a.matr.resolve(); out=(a.out_dir or root/'results'/'matr_policy').resolve(); out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(root/'code'/'src'))
    import sensitivity_common as c
    data,audit,finalizer,core,surv=c.load_primary_data(root,matr,a.baseline_cycles,a.smooth_window)
    ir_z,scale,_=core.standardize_policy_paths(data.ir_raw,policy_start=a.baseline_cycles)
    idx_by_batch={b:np.flatnonzero(data.batches==b) for b in c.PRIMARY_BATCHES}
    H={b:float(np.median(data.life[idx])) for b,idx in idx_by_batch.items()}
    truth={b:float(np.mean(np.minimum(data.life[idx],H[b]))) for b,idx in idx_by_batch.items()}
    rules={
      'continuous_cloglog':{'beta':1.0,'inspection_interval':1},
      'soft_threshold':{'beta':4.0,'inspection_interval':1},
      'periodic_25':{'beta':4.0,'inspection_interval':25},
      'periodic_50':{'beta':4.0,'inspection_interval':50},
    }
    targets=[0.10,0.20,0.30,0.40,0.50,0.60]
    tau={r:c.scheduled_tau(core,ir_z,0.70,a.baseline_cycles,s['inspection_interval']) for r,s in rules.items()}
    lambdas:dict[tuple[str,float,str],float]={}
    for r,s in rules.items():
      for target in targets:
        for b,idx in idx_by_batch.items():
          lambdas[(r,target,b)]=c.calibrate_lambda0_scheduled([ir_z[i] for i in idx],s['beta'],tau[r],target,a.baseline_cycles,s['inspection_interval'])

    # Exact policy-conditioned estimand gaps do not depend on overlay draws.
    gap_rows=[]
    for r,s in rules.items():
      for target in targets:
        for b,idx in idx_by_batch.items():
          batch_paths=[ir_z[i] for i in idx]
          crude,g=c.exact_crude_scheduled(batch_paths,s['beta'],tau[r],lambdas[(r,target,b)],H[b],a.baseline_cycles,s['inspection_interval'])
          any_exit=c.exact_any_exit_scheduled(batch_paths,s['beta'],tau[r],lambdas[(r,target,b)],H[b],a.baseline_cycles,s['inspection_interval'])
          gap=crude-truth[b]
          gap_rows.append({'rule':r,'target_replacement':target,'batch':b,'n_units':len(idx),'H':H[b],'net_rmst':truth[b],'exact_crude_functional':crude,'exact_any_exit_rmst':any_exit,'estimand_gap_cycles':gap,'estimand_gap_pct_net':100*gap/truth[b],'any_exit_minus_net':any_exit-truth[b],'any_exit_minus_net_pct':100*(any_exit-truth[b])/truth[b],'crude_minus_any_exit':crude-any_exit,'terminal_G_mean':float(np.mean(g)),'beta':s['beta'],'inspection_interval':s['inspection_interval'],'tau':tau[r],'lambda0':lambdas[(r,target,b)]})
    c.write_csv(out/'estimand_gaps.csv',gap_rows)

    rep_path=out/'replicates.csv'; fit_path=out/'fit_diagnostics.csv'; sup_path=out/'support.csv'
    cap_rep_path=out/'weight_truncation_replicates.csv'; cap_sup_path=out/'weight_truncation_support.csv'
    if any((not np.isfinite(x)) or x<=0 for x in a.weight_caps): raise ValueError('weight caps must be finite and positive')
    if a.resume:
      reps=c.read_csv(rep_path); fits=c.read_csv(fit_path); support=c.read_csv(sup_path)
      cap_reps=c.read_csv(cap_rep_path); cap_support=c.read_csv(cap_sup_path)
      counts={}
      for row in reps: counts[int(float(row['replicate']))]=counts.get(int(float(row['replicate'])),0)+1
      expected=len(rules)*len(targets)*len(c.PRIMARY_BATCHES)*4
      complete={r for r,n in counts.items() if n==expected}
      reps=[r for r in reps if int(float(r['replicate'])) in complete]; fits=[r for r in fits if int(float(r['replicate'])) in complete]; support=[r for r in support if int(float(r['replicate'])) in complete]
      cap_reps=[r for r in cap_reps if int(float(r['replicate'])) in complete]; cap_support=[r for r in cap_support if int(float(r['replicate'])) in complete]
    else: reps=[]; fits=[]; support=[]; cap_reps=[]; cap_support=[]; complete=set()

    for rep in range(a.R):
      if rep in complete:
        print(f'replicate {rep+1}/{a.R} already complete; skipped',flush=True); continue
      rng=np.random.default_rng(a.seed+10000019*rep)
      uniforms=[rng.random(max(len(x)-1,0)) for x in ir_z]
      for r,s in rules.items():
        interval=s['inspection_interval']; beta=s['beta']
        for target in targets:
          for b,idx in idx_by_batch.items():
            tt,ee,obs,oracle=c.overlay_scheduled([ir_z[i] for i in idx],beta,tau[r],lambdas[(r,target,b)],[uniforms[i] for i in idx],a.baseline_cycles,interval)
            fold=data.folds[idx]
            fitted,ff=c.fit_crossfit_scheduled(core,obs,ee,fold,a.baseline_cycles,interval,a.ridge_slope)
            nt,ns,_=surv.km(tt,ee); naive=float(surv.rmrl_from_survival(nt,ns,0.0,H[b]))
            ot,os=core.weighted_product_limit(tt,ee,oracle); oracle_pl=float(surv.rmrl_from_survival(ot,os,0.0,H[b]))
            ft,fs=core.weighted_product_limit(tt,ee,fitted); cross=float(surv.rmrl_from_survival(ft,fs,0.0,H[b]))
            oracle_ht=float(core.ht_ipcw_rmst(tt,ee,oracle,H[b]))
            common={'replicate':rep,'rule':r,'target_replacement':target,'batch':b,'n_units':len(idx),'H':H[b],'truth':truth[b],'realized_replacement':float(np.mean(ee==0)),'beta':beta,'inspection_interval':interval}
            for estimator,val in [('naive',naive),('oracle_product_limit',oracle_pl),('oracle_HT_RMST',oracle_ht),('crossfit_TV_IPCW',cross)]:
              reps.append({**common,'estimator':estimator,'estimate':val,'bias_cycles':val-truth[b],'bias_pct_net':100*(val-truth[b])/truth[b]})
            for fold_id,fit in zip(sorted(np.unique(fold)),ff):
              fits.append({**common,'fold':int(fold_id),'intercept':fit.intercept,'slope':fit.slope,'success':fit.success,'n_iter':fit.n_iter,'objective':fit.objective,'grad_norm':fit.grad_norm,'message':fit.message,'method':fit.method})
            wd=core.weight_diagnostics(tt,fitted,[H[b]])[0]
            ed=core.weighted_event_diagnostics(tt,ee,fitted,horizon=H[b])
            support.append({**common,'ess_over_n_at_H':wd['ess_over_n_at_risk'],'n_at_risk_H':wd['n_at_risk'],'p95_weight_H':wd['weight_p95'],'max_weight_H':wd['max_weight'],'n_exp_clipped_H':wd['n_exp_clipped'],'max_event_hazard_increment':max([float(x['weighted_hazard_increment']) for x in ed],default=float('nan'))})
            for cap in a.weight_caps:
              ct,cs=core.weighted_product_limit(tt,ee,fitted,weight_cap=float(cap)); capped=float(surv.rmrl_from_survival(ct,cs,0.0,H[b]))
              cwd=core.weight_diagnostics(tt,fitted,[H[b]],weight_cap=float(cap))[0]
              cap_reps.append({**common,'weight_cap':float(cap),'estimate':capped,'bias_cycles':capped-truth[b],'bias_pct_net':100*(capped-truth[b])/truth[b]})
              cap_support.append({**common,'weight_cap':float(cap),'ess_over_n_at_H':cwd['ess_over_n_at_risk'],'n_at_risk_H':cwd['n_at_risk'],'p95_weight_H':cwd['weight_p95'],'max_weight_H':cwd['max_weight'],'fraction_weight_capped_H':cwd['fraction_weight_capped']})
      if (rep+1)%a.checkpoint_every==0 or rep==a.R-1:
        c.write_csv(rep_path,reps); c.write_csv(fit_path,fits); c.write_csv(sup_path,support)
        if a.weight_caps: c.write_csv(cap_rep_path,cap_reps); c.write_csv(cap_sup_path,cap_support)
      print(f'replicate {rep+1}/{a.R} completed',flush=True)

    summary=c.summarize(reps,['rule','target_replacement','batch','estimator'],'bias_pct_net')
    c.write_csv(out/'summary.csv',summary)
    if a.weight_caps:
      cap_summary=c.summarize(cap_reps,['rule','target_replacement','batch','weight_cap'],'bias_pct_net')
      c.write_csv(out/'weight_truncation_summary.csv',cap_summary)
    # Engineering comparison table, based on overlay means and exact estimand gaps.
    lookup={(r['rule'],float(r['target_replacement']),r['batch'],r['estimator']):r for r in summary}
    gap_lookup={(r['rule'],float(r['target_replacement']),r['batch']):r for r in gap_rows}
    eng=[]
    for r in rules:
      for target in targets:
        for b in c.PRIMARY_BATCHES:
          n=lookup[(r,target,b,'naive')]; x=lookup[(r,target,b,'crossfit_TV_IPCW')]; g=gap_lookup[(r,target,b)]
          correction_pct=float(n['mean'])-float(x['mean']); correction_cycles=truth[b]*correction_pct/100
          ratio=float(g['estimand_gap_cycles'])/abs(correction_cycles) if abs(correction_cycles)>1e-12 else float('inf')
          eng.append({'rule':r,'target_replacement':target,'batch':b,'naive_bias_pct_net':n['mean'],'crossfit_bias_pct_net':x['mean'],'TV_IPCW_correction_pp':correction_pct,'TV_IPCW_correction_cycles':correction_cycles,'estimand_gap_cycles':g['estimand_gap_cycles'],'estimand_gap_pct_net':g['estimand_gap_pct_net'],'gap_to_abs_correction_ratio':ratio})
    c.write_csv(out/'engineering_comparison.csv',eng)
    expected=a.R*len(rules)*len(targets)*len(c.PRIMARY_BATCHES)*4
    checks={'expected_replicate_rows':len(reps)==expected,'all_finite_estimates':all(np.isfinite(float(r['estimate'])) for r in reps),'all_fits_success':all(str(r['success']).lower() in {'true','1'} for r in fits),'target_calibration_exact':all(abs(c.expected_replacement_fraction([ir_z[i] for i in idx_by_batch[b]],rules[r]['beta'],tau[r],lambdas[(r,t,b)],a.baseline_cycles,rules[r]['inspection_interval'])-t)<1e-8 for r in rules for t in targets for b in c.PRIMARY_BATCHES),'primary_horizon_is_batch_median':all(abs(H[b]-float(np.median(data.life[idx_by_batch[b]])))<1e-12 for b in c.PRIMARY_BATCHES)}
    report={'analysis':'matr_policy_policy_intensity_rule_scan','status':'PASS' if all(checks.values()) else 'REVIEW_REQUIRED','python':platform.python_version(),'design':{'R':a.R,'seed':a.seed,'targets':targets,'rules':rules,'ridge_slope':a.ridge_slope,'policy_scale':scale,'tau_by_rule':tau,'weight_caps':a.weight_caps,'interpretation':'imposed diagnostic replacement intensities; not operational prevalence estimates. Optional cap grid is a positivity sensitivity, not the primary estimator.'},'checks':checks,'horizons':H,'lambda0':{f'{r}|{t:.2f}|{b}':v for (r,t,b),v in lambdas.items()}}
    c.json_dump(out/'report.json',report)
    print('matr_policy POLICY INTENSITY AND RULE SCAN COMPLETED'); print(f"status={report['status']}")
    for row in eng:
      if row['rule']=='continuous_cloglog' and row['target_replacement'] in {0.1,0.4,0.6}:
        print(f"{row['batch']:12s} target={row['target_replacement']:.1f} naive={float(row['naive_bias_pct_net']):+.3f}% crossfit={float(row['crossfit_bias_pct_net']):+.3f}% gap={float(row['estimand_gap_cycles']):.2f} cycles correction={float(row['TV_IPCW_correction_cycles']):.2f} cycles")
    print(f'out_dir={out}')
    return 0 if all(checks.values()) else 2
if __name__=='__main__': raise SystemExit(main())
