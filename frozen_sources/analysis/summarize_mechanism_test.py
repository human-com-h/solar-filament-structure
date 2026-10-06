import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import csv,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2];REV=ROOT/'project';OUT=REV/'mechanism_final_analysis'
def read(p):return json.loads(p.read_text())
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,data):
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def boot(d,groups,seed):
    rng=np.random.default_rng(seed);u=sorted(set(groups));n=len(u);g=np.array(groups)
    sums=np.array([d[g==x].sum(0) for x in u]);counts=np.array([(g==x).sum() for x in u]);draws=[]
    for _ in range(100):
        w=rng.multinomial(n,np.full(n,1/n),size=1000);draws.append((w@sums)/(w@counts)[:,None])
    return np.quantile(np.concatenate(draws),[.025,.975],axis=0)
lock=read(OUT/'locked_test_protocol.json');lockhash=hashlib.sha256((OUT/'locked_test_protocol.json').read_bytes()).hexdigest();methods=list(dict.fromkeys(r['method'] for r in lock['runs']));seeds=sorted({r['seed'] for r in lock['runs']})
manifest=rows(REV/'mechanism_training/temporal_manifest.csv');meta={r['physical_observation_id']:r for r in manifest if r['split']=='internal_test'}
def summarize(kind):
    if kind=='region':
        src=REV/'mechanism_locked_test/results';ids=lock['test_ids'];metrics=['dice','iou','precision','recall','cldice','msc','msgr','foreground_fraction','false_positive_fraction','predicted_to_reference_area'];radii=[1,3,5];primary=3
        assert read(src/'COMPLETE.json')['limited_pilot'] is False
    else:
        src=REV/'mechanism_branch_test';ids=read(src/'signature.json')['ids'];metrics=['component_recall','pixel_recall'];radii=[0,1,2];primary=0
    assert read(src/'COMPLETE.json')['protocol_sha256']==lockhash
    allrows=[];arr={};points=[]
    for run in lock['runs']:
        for oid in ids:
            data=read(src/run['name']/f'{oid}.json');rr=data['rows'] if kind=='region' else data
            assert len(rr)==6
            for r in rr:
                assert r['physical_observation_id']==oid
                allrows.append({**r,'method':run['method'],'seed':run['seed']})
        for mode in ['fixed_0.5','validation_selected']:
            for radius in radii:
                rr=[r for r in allrows if r['method']==run['method'] and r['seed']==run['seed'] and r['mode']==mode and r['radius']==radius]
                assert [r['physical_observation_id'] for r in rr]==ids
                a=np.array([[r[k] for k in metrics] for r in rr]);assert np.isfinite(a).all()
                arr[run['method'],run['seed'],mode,radius]=a
                points.append({'method':run['method'],'seed':run['seed'],'mode':mode,'radius':radius,**dict(zip(metrics,map(float,a.mean(0))))})
    multi=[]
    for method in methods:
        for mode in ['fixed_0.5','validation_selected']:
            for radius in radii:
                a=np.array([arr[method,s,mode,radius].mean(0) for s in seeds]);entry={'method':method,'mode':mode,'radius':radius,'n_seeds':3,'n_observations':len(ids)}
                for j,k in enumerate(metrics):entry[k+'_mean']=float(a[:,j].mean());entry[k+'_sd']=float(a[:,j].std(ddof=1))
                multi.append(entry)
    intervals=[];sensitivity=[]
    groups=[meta[oid]['temporal_group'] for oid in ids];years=[meta[oid]['year'] for oid in ids]
    for seed in seeds:
        for comparator in [m for m in methods if m!='SABR']:
            d=arr['SABR',seed,'validation_selected',primary]-arr[comparator,seed,'validation_selected',primary]
            for unit,labels in [('observation',ids),('temporal_group',groups)]:
                ci=boot(d,labels,seed+71)
                for j,k in enumerate(metrics):intervals.append({'seed':seed,'comparison':'SABR-'+comparator,'metric':k,'resampling_unit':unit,'n_units':len(set(labels)),'mean_delta':float(d[:,j].mean()),'ci95_low':float(ci[0,j]),'ci95_high':float(ci[1,j]),'resamples':100000})
            for scope,labels in [('year',years),('leave_one_temporal_group_out',groups)]:
                for label in sorted(set(labels)):
                    keep=np.array(labels)==label
                    if scope.startswith('leave'):keep=~keep
                    for j,k in enumerate(metrics):sensitivity.append({'seed':seed,'comparison':'SABR-'+comparator,'scope':scope,'label':label,'n_observations':int(keep.sum()),'metric':k,'mean_delta':float(d[keep,j].mean())})
    write(OUT/f'{kind}_per_observation.csv',allrows);write(OUT/f'{kind}_per_seed.csv',points);write(OUT/f'{kind}_multiseed.csv',multi);write(OUT/f'{kind}_paired_intervals.csv',intervals);write(OUT/f'{kind}_sensitivity.csv',sensitivity)
    print(kind,'done',len(ids),'observations',len(set(groups)),'groups',flush=True)
    for r in multi:
        if r['mode']=='validation_selected' and r['radius']==primary:print(json.dumps(r),flush=True)
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('kind',choices=['region','branch']);args=ap.parse_args();summarize(args.kind)
