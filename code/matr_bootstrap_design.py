#!/usr/bin/env python3
"""Create deterministic batch-stratified fleet-bootstrap and overlay manifests."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

BATCHES=("MATR-05-12","MATR-06-30","MATR-04-12")
BATCH_CODE={"MATR-05-12":512,"MATR-06-30":630,"MATR-04-12":412}

def grouped_folds(counts: np.ndarray, rng: np.random.Generator, n_folds: int=5):
    active=np.flatnonzero(counts)
    order=active[np.lexsort((rng.random(len(active)),-counts[active]))]
    load=np.zeros(n_folds,int); groups=np.zeros(n_folds,int); assignment={}
    for idx in order:
        candidates=np.flatnonzero(load==load.min())
        if len(candidates)>1:
            candidates=candidates[groups[candidates]==groups[candidates].min()]
        f=int(candidates[0]); assignment[int(idx)]=f
        load[f]+=int(counts[idx]); groups[f]+=1
    return assignment,load,groups

def main()->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,type=Path)
    p.add_argument('--B',type=int,default=4000)
    p.add_argument('--R-inner',type=int,default=32)
    p.add_argument('--bootstrap-seed',type=int,default=20261201)
    p.add_argument('--fold-seed',type=int,default=20261211)
    p.add_argument('--overlay-seed',type=int,default=20261221)
    p.add_argument('--out-dir',type=Path,default=None)
    a=p.parse_args(); root=a.root.resolve(); out=(a.out_dir or root/'results'/'matr_bootstrap').resolve(); out.mkdir(parents=True,exist_ok=True)
    if a.B<1 or a.R_inner<1: raise ValueError('B and R-inner must be positive')
    inventory=pd.read_csv(root/'results'/'matr_cohort'/'fold_assignment.csv')
    design=json.loads((root/'results'/'matr_primary'/'analysis_design.json').read_text())
    expected=design['batch_counts']
    got=inventory.groupby('batch_label')['unit_id'].nunique().astype(int).to_dict()
    if got!=expected: raise RuntimeError(f'cohort counts differ from design: {got}')
    outer=[]; seeds=[]
    for b in range(a.B):
        for batch in BATCHES:
            g=inventory[inventory.batch_label==batch].reset_index(drop=True); n=len(g)
            draw=np.random.default_rng(np.random.SeedSequence([a.bootstrap_seed,b,BATCH_CODE[batch]])).integers(0,n,size=n)
            counts=np.bincount(draw,minlength=n)
            frng=np.random.default_rng(np.random.SeedSequence([a.fold_seed,b,BATCH_CODE[batch]]))
            assignment,load,groups=grouped_folds(counts,frng,5)
            for i,c in enumerate(counts):
                if c:
                    outer.append({'outer_b':b,'batch_label':batch,'source_unit_id':str(g.loc[i,'unit_id']),
                                  'source_lifetime':float(g.loc[i,'lifetime']),'multiplicity':int(c),
                                  'crossfit_fold':int(assignment[i]),'fold_total_positions':int(load[assignment[i]]),
                                  'fold_unique_source_units':int(groups[assignment[i]])})
            for beta in (0.0,1.0):
                for r in range(1,a.R_inner+1):
                    ss=np.random.SeedSequence([a.overlay_seed,b,BATCH_CODE[batch],int(beta),r])
                    seed=int(ss.generate_state(1,dtype=np.uint64)[0])
                    seeds.append({'outer_b':b,'batch_label':batch,'beta':beta,'inner_r':r,'overlay_seed_uint64':seed})
    pd.DataFrame(outer).to_csv(out/'outer_manifest.csv',index=False)
    pd.DataFrame(seeds).to_csv(out/'overlay_seed_manifest.csv',index=False)
    report={'analysis':'batch-stratified unit bootstrap design','status':'PASS','B':a.B,'R_inner':a.R_inner,
            'seeds':{'bootstrap':a.bootstrap_seed,'fold':a.fold_seed,'overlay':a.overlay_seed},
            'outer_rows':len(outer),'seed_rows':len(seeds)}
    (out/'design_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('MATR BOOTSTRAP DESIGN COMPLETED'); print('status=PASS'); print(f'B={a.B}; R_inner={a.R_inner}')
    return 0
if __name__=='__main__': raise SystemExit(main())
