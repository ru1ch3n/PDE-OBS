"""Versioned CPU aggregation of a complete score grid; no outcome-based gates."""
import argparse,copy,hashlib,itertools,json,math
from pathlib import Path
import numpy as np
from audit_core import PDES,METHODS,VIEWS,validate_grid

LABELS=['R50','H','V','CL','BL','LI','BD','R65','R80']
FIXED=['burgers/cno','darcy/cno','darcy/fno','darcy/pino','darcy/ufno_2d',
       'heat/cno','helmholtz/cno','helmholtz/deeponet','helmholtz/fno','helmholtz/pino',
       'helmholtz/transolver','navier_stokes/cno','reaction_diffusion/cno']

def stats(values,n):
    x=np.asarray(values,dtype=float)
    if x.shape!=(n,) or n<2 or not np.isfinite(x).all():raise ValueError('complete finite summary population required')
    return dict(mean=float(x.mean()),std=float(x.std(ddof=1)),n=n,ddof=1)

def format_pair(mean,std,places=4,scale=1.0):
    if places not in (2,4) or not all(math.isfinite(x) for x in (mean,std,scale)) or std<0 or scale<=0:raise ValueError('display inputs')
    values=[0.0 if round(x/scale,places)==0 else x/scale for x in (mean,std)]
    return f'{values[0]:.{places}f} \\pm {values[1]:.{places}f}'

def analyze(records):
    validate_grid(records)
    pairs=[];transfers=[];density=[];horizons=[];matrices=[];epochs=[];training=[];cellstd=[]
    for pde,method in itertools.product(PDES,METHODS):
        group=[records[f'{pde}/{method}/{view}'] for view in VIEWS]
        for row in group:
            if set(row['blocks'])!=set(VIEWS):raise ValueError('nine-view grid')
        E=np.array([[r['blocks'][w]['joint']['mean'] for w in VIEWS] for r in group],dtype=float)
        if E.shape!=(9,9) or not np.isfinite(E).all() or np.any(E<=0):raise ValueError('invalid error or undefined ratio denominator')
        S=np.array([[r['blocks'][w]['joint'].get('std',np.nan) for w in VIEWS] for r in group],dtype=float)
        ep=[int(r['actual_epochs']) for r in group];task='recovery' if pde in PDES[:3] else 'rollout'
        diagonal=np.diag(E);off=~np.eye(9,dtype=bool);D=stats(diagonal,9);C=stats(E[off],72)
        ratio=E/diagonal[None,:];delta=E-diagonal[None,:]
        name=pde+'/'+method
        pair=dict(pair=name,pde=pde,method=method,task=task,D=D,C=C,C_over_D=C['mean']/D['mean'],
          fixed_original500=name in FIXED,all500=all(e==500 for e in ep),min_epochs=min(ep),max_epochs=max(ep),
          median_off_error=float(np.median(E[off])),p90_off_error=float(np.quantile(E[off],.9)),max_off_error=float(E[off].max()),
          positive_destination=int((delta[off]>0).sum()))
        pairs.append(pair);matrices.append(E);epochs.append(ep);cellstd.append(S)
        for i,row in enumerate(group):
            training.append({k:row[k] for k in ('identity','pde','method','train_view','actual_epochs','training_cohort')})
            for j,w in enumerate(VIEWS):
                if i!=j:
                    count=8192 if i<4 and j<4 else 8284 if 4<=i<7 and 4<=j<7 else 0
                    transfers.append(dict(pair=name,pde=pde,method=method,task=task,train_view=VIEWS[i],test_view=w,
                       error=float(E[i,j]),destination_reference=float(E[j,j]),difference=float(delta[i,j]),ratio=float(ratio[i,j]),
                       log10_ratio=float(np.log10(ratio[i,j])),count_group=count))
            if i in (0,7,8):
                for j in (7,8):
                    r=float(E[i,j]/E[i,0])
                    density.append(dict(pair=name,pde=pde,method=method,train_view=LABELS[i],test_view=LABELS[j],
                       error=float(E[i,j]),R50_error=float(E[i,0]),ratio=r,log10_ratio=math.log10(r)))
        if task=='rollout':
            for h in (1,2,3):
                H=np.array([[next(v['mean'] for v in r['blocks'][w]['horizons'] if v['horizon']==h) for w in VIEWS] for r in group])
                horizons.append(dict(pair=name,pde=pde,method=method,horizon=h,D=stats(np.diag(H),9),C=stats(H[off],72)))
    def selected_summary(pair_names):
        p=[x for x in pairs if x['pair'] in pair_names];t=[x for x in transfers if x['pair'] in pair_names];d=[x for x in density if x['pair'] in pair_names]
        eq=[];dens=[]
        for count,task in itertools.product((8192,8284),('recovery','rollout')):
            items=[x for x in t if x['count_group']==count and x['task']==task]
            eq.append(dict(count_group=count,task=task,n=len(items),positive=sum(x['difference']>0 for x in items),
              median_ratio=float(np.median([x['ratio'] for x in items])),min_ratio=min(x['ratio'] for x in items),max_ratio=max(x['ratio'] for x in items)))
        for v,w in itertools.product(('R50','R65','R80'),('R65','R80')):
            items=[x for x in d if x['train_view']==v and x['test_view']==w]
            dens.append(dict(train_view=v,test_view=w,n=len(items),lower=sum(x['ratio']<1 for x in items),
              median_ratio=float(np.median([x['ratio'] for x in items])),min_ratio=min(x['ratio'] for x in items),max_ratio=max(x['ratio'] for x in items)))
        return dict(pairs=len(p),pairs_C_gt_D=sum(x['C']['mean']>x['D']['mean'] for x in p),median_C_over_D=float(np.median([x['C_over_D'] for x in p])),
          positive_destination=sum(x['difference']>0 for x in t),destination_n=len(t),equal_count=eq,density=dens)
    full=selected_summary([x['pair'] for x in pairs]);subset=selected_summary(FIXED)
    E=np.array(matrices);R=E/np.diagonal(E,axis1=1,axis2=2)[:,None,:]
    main=[]
    for pde in PDES:
        part=E[[x['pde']==pde for x in pairs]]
        diag=np.diagonal(part,axis1=1,axis2=2).ravel();off=part[:,~np.eye(9,dtype=bool)].ravel()
        main.append(dict(pde=pde,matched_cells=stats(diag,63),cross_cells=stats(off,504),
          median_C_over_D=float(np.median([x['C_over_D'] for x in pairs if x['pde']==pde]))))
    cohorts={}
    for row in training:cohorts[row['training_cohort']]=cohorts.get(row['training_cohort'],0)+1
    return dict(schema='pdeobs-new-prediction-statistics/20260925-v1',settings=441,blocks=3969,
       strict_rescoring=True,stats=full,fixed_subset=subset,fixed_pairs=FIXED,pairs=pairs,transfers=transfers,density=density,
       horizons=horizons,training=training,cohorts=cohorts,main=main,
       arrays={'E':E.tolist(),'cell_std':np.array(cellstd).tolist(),'epochs':epochs,'R':R.tolist(),
         'median_R':np.median(R,axis=0).tolist(),'median_log_R':np.median(np.log10(R),axis=0).tolist()},
       statistics_note='Single-cell SD:200 test records. D/C SD:9/72 cell means. Main PDE rows:63/504 cell means. ddof=1; dependent contrasts, not seed uncertainty.')

def from_archive(path):
    original=json.loads(Path(path).read_text(encoding='utf-8'))['records'];records={}
    for key,row in original.items():
        new={k:row[k] for k in ('identity','pde','method','train_view','actual_epochs','training_cohort')};new['blocks']={}
        for view,b in row['blocks'].items():
            h=[]
            for i in (1,2,3):
                if str(i) in b['per_horizon']:h.append({'horizon':i,'mean':b['per_horizon'][str(i)]})
            new['blocks'][view]={'joint':{'mean':b['relative_l2']},'horizons':h}
        records[key]=new
    return records

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive-regression',type=Path);p.add_argument('--index',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    records=from_archive(a.archive_regression) if a.archive_regression else json.loads(a.index.read_text())['records']
    result=analyze(records)
    if a.archive_regression:
        result['strict_rescoring']=False;result['arrays'].pop('cell_std');result['note']='Historical regression only; no standard deviations invented.'
    with a.out.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({'stats':result['stats'],'fixed_subset':result['fixed_subset']}))
