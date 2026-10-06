"""Accept the frozen GPU return, then score axes and paired statistics locally."""
from pathlib import Path
import argparse,csv,hashlib,json,sys,time
from datetime import datetime,timezone
import numpy as np
from PIL import Image
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];P=HERE/'evaluation_package'
APP=ROOT/'project/morphology_application_execution_20261001'
sys.path.insert(0,str(APP))
from frozen_bootstrap import boot
import run_application as application

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def dump(p,x):application.dump(p,x)
def rows(p):return list(csv.DictReader(Path(p).open(encoding='utf-8-sig')))
def table(p,records):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if not records:return
    fields=list(dict.fromkeys(k for r in records for k in r))
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(records)

def accept(root,out):
    lock=read(P/'LOCKED_EVALUATION.json');sig=read(root/'signature.json');done=read(root/'COMPLETE.json')
    assert sig['lock_sha256']==sha(P/'LOCKED_EVALUATION.json') and sig['code_lock_sha256']==sha(P/'EVAL_CODE_LOCK.json')
    assert sig['environment_sha256']==digest(sig['environment'])
    assert {k:v for k,v in sig['environment'].items() if k!='platform'}=={k:v for k,v in lock['target_environment'].items() if k!='platform'}
    assert sig['application_lock_sha256']==lock['application_lock_sha256']
    assert done['complete'] and done['status']=='THREE_NEW_MODELS_TEST_AND_NATIVE_MASKS_COMPLETE' and done['signature']==sig
    assert not done.get('engineering_fixture',False),'Engineering fixtures are not accepted scientific results'
    assert done['runs']==3 and done['observations_per_run']==108 and done['validation_exit_codes']==done['test_exit_codes']==[0,0]
    assert not done['training_performed'] and not done['threshold_search_on_test']
    for name in ('LOCKED_EVALUATION.json','LOCKED_VALIDATION.json','EVAL_CODE_LOCK.json','GT_PROVENANCE.json','TRAINING_ACCEPTANCE.json','VALIDATION_CLDICE_REFERENCE.json','SKELETON_POLICY.json'):
        assert sha(root/'protocol'/name)==sha(P/name)
    hashes=read(root/'OUTPUT_SHA256.json')
    for rel,h in hashes.items():assert sha(root/rel)==h,('Returned file drift',rel)
    parity=read(root/'VALIDATION_PARITY.json');assert parity['complete'] and parity['signature']==sig and len(parity['checks'])==6
    assert {(x['run'],x['rank']) for x in parity['checks']}=={(r['name'],i) for r in lock['runs'] for i in (0,1)}
    for x in parity['checks']:
        run=next(r for r in lock['runs'] if r['name']==x['run'])
        assert x['expected']==run['validation_smoke_scores'] and x['observation_id']==run['validation_smoke_id'] and x['tolerance']==1e-7
        cl=read(P/'VALIDATION_CLDICE_REFERENCE.json')[x['run']]['cldice_at_0_5']
        assert x['expected_cldice_at_0_5']==cl
        error=max(max(abs(a-b) for a,b in zip(x['actual'],x['expected'])),abs(x['cldice_at_0_5']-cl))
        assert error==x['max_error'] and error<=1e-7
    for rank in (0,1):
        w=read(root/f'worker_{rank}_test_COMPLETE.json');assert w['signature']==sig and w['complete'] and w['rank']==rank and w['observations_per_run']==54
    manifests=read(root/'MASK_MANIFESTS.json');assert len(manifests)==6
    matrix={};actual_pngs=set();reconstructed=[]
    for run in lock['runs']:
        folder=root/run['name']/'receipts';assert {p.stem for p in folder.glob('*.json')}==set(lock['test_ids'])
        for oid in lock['test_ids']:
            receipt=read(folder/(oid+'.json'));payload={k:v for k,v in receipt.items() if k!='payload_sha256'}
            assert digest(payload)==receipt['payload_sha256'] and receipt['signature']==sig
            assert receipt['run']==run['name'] and receipt['observation_id']==oid and receipt['seed']==run['seed']
            assert receipt['input_image_sha256']==lock['images'][oid]['sha256'] and receipt['weights_sha256']==run['weights_sha256'] and receipt['tensor_sha256']==run['tensor_sha256']
            expected_modes={'fixed_0.5':.5,**({'validation_selected':run['threshold']} if run['threshold'] is not None else {})}
            assert {m['mode']:m['threshold'] for m in receipt['modes']}==expected_modes
            for m in receipt['modes']:
                path=f"{run['name']}/{m['mode']}/{oid}.png";assert m['path']==path
                assert sha(root/path)==m['mask']['sha256'] and (root/path).stat().st_size==m['mask']['bytes']
                with Image.open(root/path) as im:
                    assert im.size==(2048,2048) and im.mode=='L';a=np.asarray(im)
                assert set(np.unique(a))<={0,255} and int(np.count_nonzero(a))==m['mask']['foreground_pixels']
                assert m['mask']['width']==m['mask']['height']==2048;actual_pngs.add(path)
                physical=m['scores']['physical'];assert {r['radius'] for r in physical}=={1,3,5}
                assert len(m['scores']['annotation_copies'])==len(lock['ground_truth'][oid]['copies'])
                for r in physical:
                    assert r['physical_observation_id']==oid and all(np.isfinite(r[k]) and 0<=r[k]<=1 for k in lock['metrics'])
                    reconstructed.append({'run':run['name'],'method':'UNet_region_clDice','seed':run['seed'],'mode':m['mode'],
                        'threshold':m['threshold'],'observation':oid,**r})
                matrix[(run['name'],m['mode'],oid)]={'observation':oid,'mode':m['mode'],'threshold':m['threshold'],'path':path,**m['mask']}
        for mode in ('fixed_0.5','validation_selected'):
            m=next(m for m in manifests if m['run']==run['name'] and m['mode']==mode)
            t=.5 if mode=='fixed_0.5' else run['threshold'];assert m['threshold']==t and m['seed']==run['seed'] and m['method']=='UNet_region_clDice'
            assert m['checkpoint_sha256']==run['source_archive_sha256'] and m['export_checkpoint_sha256']==run['weights_sha256'] and m['tensor_sha256']==run['tensor_sha256']
            assert m['planned_observations']==108 and m['status']==('OK' if t is not None else 'INFEASIBLE_NA')
            assert m['masks']==([matrix[run['name'],mode,oid] for oid in lock['test_ids']] if t is not None else [])
            assert m['observations']==len(m['masks'])
    assert {p.relative_to(root).as_posix() for p in root.rglob('*.png')}==actual_pngs
    assert len(actual_pngs)==done['png_files']==sum(108 for m in manifests if m['status']=='OK')
    exported=rows(root/'segmentation_per_observation.csv')
    assert len(exported)==len(reconstructed)
    for a,b in zip(exported,reconstructed):
        assert set(a)==set(b) and all(float(a[k])==v if isinstance(v,(int,float)) else a[k]==v for k,v in b.items())
    table(out/'segmentation_per_observation.csv',reconstructed)
    result={'status':'ALL_NEW_TEST_RECEIPTS_MASKS_AND_METRICS_ACCEPTED','input':str(root),'signature':sig,
            'returned_sha_manifest_sha256':sha(root/'OUTPUT_SHA256.json'),'png_files':len(actual_pngs),'receipts':324,
            'numeric_segmentation_rows':len(reconstructed),'complete':True}
    dump(out/'TEST_ACCEPTANCE.json',result)
    return lock,sig,manifests,reconstructed

def score_application(root,out,lock,export_sig,manifests):
    scientific_sig,app_lock=application.scoring_signature()
    assert scientific_sig['application_lock_sha256']==lock['application_lock_sha256']
    scientific_sig['export_signature']=export_sig
    refs=read(application.APP/'REFERENCE_AXES.json');meta={}
    for r in rows(application.APP/'reference_manifest.csv'):
        meta[r['physical_observation_id']]={'year':r['year'],'temporal_group':r['temporal_group']}
    assert sorted(meta)==lock['test_ids'] and len({r['temporal_group'] for r in meta.values()})==13
    local=out/'application';local.mkdir(exist_ok=True)
    if (local/'signature.json').exists():assert read(local/'signature.json')==scientific_sig
    else:dump(local/'signature.json',scientific_sig)
    perref=[];percopy=[];perobs=[];perseed=[]
    length_fields=application.LENGTH_FIELDS
    for m in manifests:
        assert m['status']=='OK','This return should have all three feasible selected thresholds'
        base={k:m[k] for k in ('run','method','seed','mode','threshold')};current=[]
        for i,rec in enumerate(m['masks'],1):
            oid=rec['observation'];r=application.score_record(root,local,m,rec,refs,scientific_sig)['result']
            assert np.isfinite(r['per_observation']['success_rate']) and len(r['per_reference'])==sum(len(c['references']) for c in refs[oid])
            common={**base,'observation':oid,**meta[oid]}
            perref.extend({**common,**v} for v in r['per_reference']);percopy.extend({**common,**v} for v in r['per_copy'])
            obs={**common,**r['per_observation']};perobs.append(obs);current.append(obs)
            if i%10==0 or i==108:print('Application:',m['run'],m['mode'],i,'/108',flush=True)
        successful=[r for r in current if r['length_status']=='OK']
        perseed.append({**base,'success_rate':float(np.mean([r['success_rate'] for r in current])),
            'planned_observations':108,'planned_annotation_copies':170,'planned_references':1368,
            'successful_observations':len(successful),'successful_references':sum(r['successful_references'] for r in current),
            'candidate_count':float(np.mean([r['candidate_count'] for r in current])),
            **{k:float(np.mean([r[k] for r in successful])) if successful else None for k in length_fields}})
    assert len(perobs)==648 and len(percopy)==1020 and len(perref)==8208
    for name,data in [('per_reference',perref),('per_copy',percopy),('per_observation',perobs),('per_seed',perseed)]:table(local/(name+'.csv'),data)
    dump(local/'COMPLETE.json',{'complete':True,'signature':scientific_sig,'observations_per_mode':108,'modes':6,
          'per_reference_rows':8208,'references_are_annotation_consistency_targets':True,'lengths_conditional_on_success':True})
    return meta,perobs,perref,perseed

def paired_statistics(out,lock,meta,newseg,newapp):
    oldseg=rows(ROOT/'project/strong_baseline_final_analysis_20261001/region_per_observation.csv')
    oldapp=rows(ROOT/'project/morphology_application_results_20261002/per_observation.csv')
    ids=lock['test_ids'];comparisons=[];intervals=[];sensitivity=[];point_summary=[]
    for endpoint,new,old,metrics,radii,offset in [
        ('segmentation',newseg,oldseg,lock['metrics'],lock['radii'],7),
        ('application',newapp,oldapp,['success_rate'],[None],71)]:
        index={(r['method'],int(r['seed']),r['mode'],int(r['radius']) if endpoint=='segmentation' else None,
                r.get('observation',r.get('physical_observation_id'))):r for r in old}
        newidx={(int(r['seed']),r['mode'],int(r['radius']) if endpoint=='segmentation' else None,r['observation']):r for r in new}
        for seed in [r['seed'] for r in lock['runs']]:
            for mode in lock['modes']:
                for radius in radii:
                    x=np.array([[float(newidx[seed,mode,radius,o][m]) for m in metrics] for o in ids])
                    for j,m in enumerate(metrics):point_summary.append({'endpoint':endpoint,'method':'UNet_region_clDice','seed':seed,'mode':mode,
                         'radius':radius,'metric':m,'mean':float(x[:,j].mean()),'observations':108})
                    for control in lock['comparison_methods']:
                        records=[index[control,seed,mode,radius,o] for o in ids]
                        feasible=all(r.get('status','OK') in ('OK','SUCCESS','FAIL') for r in records)
                        d=x-np.array([[float(r[m]) for m in metrics] for r in records]) if feasible else None
                        base={'endpoint':endpoint,'comparison':'UNet_region_clDice-minus-'+control,'seed':seed,'mode':mode,'radius':radius,
                              'status':'OK' if feasible else 'INFEASIBLE_NA','planned_observations':108,'paired_observations':108 if feasible else 0}
                        for j,m in enumerate(metrics):comparisons.append({**base,'metric':m,'mean_difference':float(d[:,j].mean()) if feasible else None})
                        for unit in ('observation','temporal_group'):
                            groups=ids if unit=='observation' else [meta[o]['temporal_group'] for o in ids]
                            ci=boot(d,groups,seed+offset) if feasible else None
                            for j,m in enumerate(metrics):intervals.append({**base,'metric':m,'unit':unit,'planned_units':len(set(groups)),
                                'mean_difference':float(d[:,j].mean()) if feasible else None,'ci_low':float(ci[0,j]) if feasible else None,
                                'ci_high':float(ci[1,j]) if feasible else None,'draws':100000 if feasible else 0,'rng_seed':seed+offset,
                                'interpretation':'Exploratory unadjusted descriptive interval'})
                        for kind,key in [('per_year','year'),('leave_one_group_out','temporal_group')]:
                            for v in sorted({r[key] for r in meta.values()}):
                                keep=np.array([meta[o][key]==v if kind=='per_year' else meta[o][key]!=v for o in ids])
                                for j,m in enumerate(metrics):sensitivity.append({**base,'metric':m,'analysis':kind,'year_or_omitted_group':v,
                                     'planned_subset_observations':int(keep.sum()),'mean_difference':float(d[keep,j].mean()) if feasible else None})
    multiseed=[]
    keys={(r['endpoint'],r['comparison'],r['mode'],r['radius'],r['metric']) for r in comparisons}
    for key in sorted(keys,key=str):
        selected=[r for r in comparisons if tuple(r[k] for k in ['endpoint','comparison','mode','radius','metric'])==key]
        assert len(selected)==3;values=[r['mean_difference'] for r in selected];ok=all(v is not None for v in values)
        multiseed.append(dict(zip(['endpoint','comparison','mode','radius','metric'],key))|{'numeric_seeds':sum(v is not None for v in values),
             'mean_difference':float(np.mean(values)) if ok else None,'sample_sd':float(np.std(values,ddof=1)) if ok else None,
             'status':'OK' if ok else 'INFEASIBLE_NA'})
    for name,data in [('new_per_seed',point_summary),('paired_effects',comparisons),('paired_intervals',intervals),
                      ('sensitivity',sensitivity),('paired_three_seed_summary',multiseed)]:table(out/(name+'.csv'),data)

def common_success_lengths(out,lock,newref):
    old=rows(ROOT/'project/morphology_application_results_20261002/per_reference.csv')
    index={(r['method'],int(r['seed']),r['mode'],r['observation'],r['sample'],r['annotation_id']):r for r in old}
    effects=[];details=[]
    for seed in [r['seed'] for r in lock['runs']]:
        for mode in lock['modes']:
            for control in lock['comparison_methods']:
                subset=[r for r in newref if r['seed']==seed and r['mode']==mode];paired=[]
                infeasible=any(index[control,seed,mode,r['observation'],r['sample'],r['annotation_id']].get('mode_status')=='INFEASIBLE_NA' for r in subset)
                for r in subset:
                    c=index[control,seed,mode,r['observation'],r['sample'],r['annotation_id']]
                    if r['status']=='SUCCESS' and c['status']=='SUCCESS':
                        paired.append({'seed':seed,'mode':mode,'control':control,'observation':r['observation'],'sample':r['sample'],
                            'annotation_id':r['annotation_id'],'absolute_relative_error_difference':float(r['absolute_relative_error'])-float(c['absolute_relative_error'])})
                grouped={}
                for r in paired:grouped.setdefault((r['observation'],r['sample']),[]).append(r['absolute_relative_error_difference'])
                observations={}
                for (o,s),v in grouped.items():observations.setdefault(o,[]).append(float(np.mean(v)))
                values=np.array([np.mean(v) for o,v in sorted(observations.items())],float)
                effects.append({'seed':seed,'mode':mode,'control':control,'common_success_references':len(paired),
                    'common_success_copies':len(grouped),'common_success_observations':len(observations),
                    'mean_absolute_relative_error_difference':float(values.mean()) if len(values) else None,
                    'status':'INFEASIBLE_NA' if infeasible else 'OK' if len(values) else 'FAIL_NA',
                    'na_reason':'Original selected control has no feasible validation threshold' if infeasible else '',
                    'scope':'Conditional common-success annotation length error; negative favors new control'})
                details.extend(paired)
    table(out/'common_success_length_per_seed.csv',effects);table(out/'common_success_length_pairs.csv',details)

def main():
    p=argparse.ArgumentParser();p.add_argument('--masks',required=True);p.add_argument('--output',default=str(HERE/'test_analysis'));a=p.parse_args()
    root=Path(a.masks);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);start=time.time()
    for rel,h in read(HERE/'LOCAL_ANALYSIS_CODE_LOCK.json').items():assert sha(HERE/rel)==h
    lock,sig,manifests,newseg=accept(root,out)
    meta,newapp,newref,seeds=score_application(root,out,lock,sig,manifests)
    paired_statistics(out,lock,meta,newseg,newapp);common_success_lengths(out,lock,newref)
    dump(out/'COMPLETE.json',{'complete':True,'status':'NEW_CONTROL_TEST_APPLICATION_AND_PAIRED_STATISTICS_COMPLETE',
         'created_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.time()-start,'lock_sha256':sha(P/'LOCKED_EVALUATION.json'),
         'local_analysis_code_lock_sha256':sha(HERE/'LOCAL_ANALYSIS_CODE_LOCK.json'),'unit':'108 observations; 13 temporal groups sensitivity',
         'seeds_are_technical_replications':True,'no_confirmatory_significance_claimed':True,'old_results_modified':False})
    dump(out/'OUTPUT_SHA256.json',{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='OUTPUT_SHA256.json'})
    print('New control analysis complete:',out,flush=True)
if __name__=='__main__':main()
