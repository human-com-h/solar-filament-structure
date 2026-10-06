"""Portable standard-library reconstruction of all eight-method descriptive tables."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from filament_structure.io import Path
from collections import defaultdict
import argparse,csv,json,statistics

def read(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,rows):
    with Path(p).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def mean(values):return statistics.mean(values) if values else None
def compare(actual,expected,key_fields):
    a={tuple(str(x[k]) for k in key_fields):x for x in actual}
    b={tuple(x[k] for k in key_fields):x for x in expected}
    assert len(a)==len(actual) and len(b)==len(expected) and a.keys()==b.keys()
    max_error=0.
    for key in a:
        assert a[key].keys()==b[key].keys()
        for field,v in a[key].items():
            target=b[key][field]
            if v is None:assert target=='',(key,field,target)
            elif isinstance(v,(float,int)):
                error=abs(float(target)-v);max_error=max(max_error,error)
                assert error<1e-12,(key,field,error)
            else:assert str(v)==target,(key,field,v,target)
    return {'rows':len(actual),'maximum_numeric_difference':max_error,'numeric_tolerance':1e-12}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--output',type=Path)
    args=p.parse_args();root=args.workspace/'project';h=root/'region_clDice_analysis_20261004'
    out=args.output or args.workspace/'summary_replay';out.mkdir(parents=True,exist_ok=True)
    m=['dice','iou','precision','recall','cldice','msc','msgr']
    ss=read(root/'strong_baseline_final_analysis_20261001/region_per_observation.csv')+read(h/'test_analysis/segmentation_per_observation.csv')
    grouped=defaultdict(list)
    for x in ss:grouped[x['method'],x['seed'],x['mode'],x['radius']].append(x)
    seg=[]
    for (method,seed,mode,radius),rows in sorted(grouped.items()):
        assert len(rows)==108
        valid=[x for x in rows if x.get('status','OK')=='OK']
        seg.append(dict(method=method,seed=seed,mode=mode,radius=radius,status='OK' if valid else 'INFEASIBLE_NA',**{k:mean([float(x[k]) for x in valid]) if valid else 'NA' for k in m}))
    ap=read(root/'morphology_application_results_20261002/per_observation.csv')+read(h/'test_analysis/application/per_observation.csv')
    grouped=defaultdict(list)
    for x in ap:grouped[x['method'],x['seed'],x['mode']].append(x)
    app=[]
    for (method,seed,mode),rows in sorted(grouped.items()):
        assert len(rows)==108
        valid=[x for x in rows if x.get('status','OK')=='OK']
        successful=[x for x in valid if x['length_status']=='OK']
        app.append(dict(method=method,seed=seed,mode=mode,status='OK' if valid else 'INFEASIBLE_NA',success_rate=mean([float(x['success_rate']) for x in valid]),successful_references=sum(int(x['successful_references']) for x in valid) if valid else None,successful_copies=sum(int(x['successful_copies']) for x in valid) if valid else None,successful_observations=len(successful) if valid else None,**{k:mean([float(x[k]) for x in successful]) for k in ['signed_relative_error','absolute_relative_error','absolute_error_px']}))
    combined=[]
    grouped=defaultdict(list)
    for x in seg:
        for metric in m:grouped['segmentation',x['method'],x['mode'],x['radius'],metric].append(x[metric])
    for x in app:
        for metric in ['success_rate','signed_relative_error','absolute_relative_error','absolute_error_px']:grouped['application',x['method'],x['mode'],'',metric].append(x[metric])
    for (endpoint,method,mode,radius,metric),values in sorted(grouped.items()):
        assert len(values)==3
        v=[x for x in values if x not in [None,'NA']]
        combined.append(dict(endpoint=endpoint,method=method,mode=mode,radius=radius,metric=metric,numeric_seeds=len(v),mean=mean(v),sample_sd=statistics.stdev(v) if len(v)>1 else None))
    qa=[]
    for name,data,keys in [('all_eight_methods_segmentation_per_seed',seg,['method','seed','mode','radius']),('all_eight_methods_application_per_seed',app,['method','seed','mode']),('all_eight_methods_mean_sd',combined,['endpoint','method','mode','radius','metric'])]:
        expected=read(h/'report_tables'/(name+'.csv'))
        qa.append(dict(table=name,**compare(data,expected,keys)))
        write(out/(name+'.csv'),data)
    result={'status':'PASS_PORTABLE_STANDARD_LIBRARY_DESCRIPTIVE_REPLAY','tables':qa,'gpu_required':False,'raw_images_required':False,'bootstrap_or_mask_scoring_repeated':False,'scope':'Reconstructs descriptive summaries from the supplied accepted observation scores. Numerical equality is checked at 1e-12, not byte equality; bootstrap draws and scoring retain their separate frozen implementations.'}
    (out/'REPLAY_QA.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
