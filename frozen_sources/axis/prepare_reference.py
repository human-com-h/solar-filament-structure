"""Freeze reference-only metadata; no model predictions are read."""
from pathlib import Path
import json,csv,hashlib,sys
from collections import defaultdict
import numpy as np
from axis_evaluator import polyline,length,self_test

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,obj):(HERE/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def freeze():
    ann=ROOT/'data/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'
    manifest=ROOT/'project/mechanism_training/temporal_manifest.csv'
    qc=ROOT/'project/mechanism_training/manual_spine_qc_exclusions.csv'
    oldp=ROOT/'project/mechanism_final_analysis/locked_test_protocol.json'
    newp=ROOT/'project/strong_baseline_final_eval/package_dual_gpu_v2/LOCKED_EVALUATION.json'
    old=json.loads(oldp.read_text());new=json.loads(newp.read_text());test=old['test_ids']
    assert sha(ann)==old['annotation_sha256']==new['annotation_sha256']
    assert sha(manifest)==old['manifest_sha256']==new['manifest_sha256']
    coco=json.loads(ann.read_text());byimage=defaultdict(list)
    for a in coco['annotations']:byimage[str(a['image_id'])].append(a)
    bysample={str(im['id']):im for im in coco['images']}
    with manifest.open(encoding='utf-8-sig') as f:records=[r for r in csv.DictReader(f) if r['split']=='internal_test']
    with qc.open(encoding='utf-8-sig') as f:excluded={r['annotation_id'] for r in csv.DictReader(f)}
    refs={oid:[] for oid in test};audit=[]
    for rec in records:
        sample=rec['annotation_sample_id'];im=bysample[sample];valid=[]
        for a in sorted(byimage[sample],key=lambda x:str(x['id'])):
            reason='';points=[];L=None
            try:
                if str(a['id']) in excluded:raise ValueError('frozen_spine_qc_exclusion')
                p=polyline(a.get('spine',[]));L=length(p)
                if (p<0).any() or (p[:,0]>im['width']-1).any() or (p[:,1]>im['height']-1).any():raise ValueError('spine_outside_native_image')
                if L<=0:raise ValueError('zero_length')
                points=p.tolist();valid.append({'annotation_id':str(a['id']),'points':points,'length_px':L})
            except ValueError as e:reason=str(e)
            audit.append({**{k:rec[k] for k in ['physical_observation_id','annotation_sample_id','year','temporal_group']},
                          'annotation_id':str(a['id']),'eligible':not bool(reason),'exclusion_reason':reason,'length_px':L})
        refs[rec['physical_observation_id']].append({'sample':sample,'references':valid})
    assert len(refs)==108 and len(records)==170
    for oid in refs:refs[oid].sort(key=lambda x:x['sample'])
    write('REFERENCE_AXES.json',refs)
    with (HERE/'reference_manifest.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(audit[0]));w.writeheader();w.writerows(audit)
    runtime=self_test();write('ENGINEERING_SELF_TEST.json',runtime)
    runs=[]
    for r in old['runs']+new['runs']:
        status=r.get('selected_status','FEASIBLE')
        runs.append({'name':r['name'],'method':r['method'],'seed':r['seed'],'checkpoint_sha256':r['sha256'],
                     'selected_status':'OK' if status=='FEASIBLE' else 'INFEASIBLE_NA',
                     'selected_threshold':r.get('threshold')})
    lock={'version':'1.0-reference-and-extractor-freeze','date':'2026-10-01',
          'status':'PROTOCOL_LOCKED_NOT_SCORED','primary':'main-axis extraction success','secondary':'projected length error conditional on success',
          'scope':'annotation consistency; retrospective exploratory; not physical ground truth',
          'radius_native_px':3,'reference_coverage_min':.9,'axis_precision_min':.9,'polyline_sampling_step_px':.5,
          'reference_annotation_sha256':sha(ann),'temporal_manifest_sha256':sha(manifest),'qc_sha256':sha(qc),
          'reference_axes_sha256':sha(HERE/'REFERENCE_AXES.json'),'reference_manifest_sha256':sha(HERE/'reference_manifest.csv'),
          'protocol_sha256':sha(HERE/'PROTOCOL_CN.md'),'axis_evaluator_sha256':sha(HERE/'axis_evaluator.py'),
          'segmentation_locks':[sha(oldp),sha(newp)],'engineering_runtime':runtime['versions'],
          'observations':sorted(test),'runs':runs,'planned_observations':108,'planned_annotation_copies':170,'temporal_groups':13,
          'reference_instances':len(audit),'eligible_instances':sum(r['eligible'] for r in audit),
          'excluded_instances':sum(not r['eligible'] for r in audit),'inference_or_training_performed':False,
          'bootstrap_rule':'frozen boot; 100000; seed+71; paired observation and frozen group; no multiplicity correction',
          'length_missing_rule':'FAIL_NA; conditional errors and yields only; no zero imputation'}
    write('LOCKED_APPLICATION.json',lock)
    write('REFERENCE_AUDIT.json',{'status':'PASS_REFERENCE_ONLY','eligible_instances':lock['eligible_instances'],
       'excluded_instances':lock['excluded_instances'],'all_observations_have_eligible_spine':all(any(c['references'] for c in v) for v in refs.values()),
       'copies_without_eligible_spine':[c['sample'] for v in refs.values() for c in v if not c['references']],
       'engineering_tests':runtime['status'],'real_predictions_scored':0})
    print(json.dumps({'reference_instances':len(audit),'eligible':lock['eligible_instances'],'excluded':lock['excluded_instances'],
                      'lock_sha256':sha(HERE/'LOCKED_APPLICATION.json'),'status':'NOT_SCORED'}))
if __name__=='__main__':freeze()
