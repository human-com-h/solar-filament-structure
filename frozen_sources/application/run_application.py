"""Resumable locked application scoring; no model inference or remote calls."""
from pathlib import Path
import argparse,csv,json,os,platform,sys,threading,time
import numpy as np,psutil,scipy,skimage
from PIL import Image
from mask_acceptance import read,sha,digest,accept
STAGE=Path(__file__).resolve().parent;ROOT=STAGE.parents[1];APP=ROOT/'project/manuscript_revision_20261001/morphology_application'
sys.path.insert(0,str(APP));import axis_evaluator as ax
LENGTH_FIELDS=['signed_relative_error','absolute_relative_error','absolute_error_px']
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_name(p.name+'.tmp')
 with temp.open('w',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(temp,p)
def scoring_signature():
 lock=read(APP/'LOCKED_APPLICATION.json')
 for name,key in [('axis_evaluator.py','axis_evaluator_sha256'),('REFERENCE_AXES.json','reference_axes_sha256'),('reference_manifest.csv','reference_manifest_sha256'),('PROTOCOL_CN.md','protocol_sha256')]:assert sha(APP/name)==lock[key]
 current={'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'skimage':skimage.__version__};assert current==lock['engineering_runtime']
 assert sha(STAGE/'frozen_bootstrap.py')==sha(ROOT/'project/strong_baseline_final_analysis_20261001/frozen_bootstrap.py')
 for rel,h in read(STAGE/'SCORING_CODE_LOCK.json').items():assert sha(STAGE/rel)==h
 return {'application_lock_sha256':sha(APP/'LOCKED_APPLICATION.json'),'axis_evaluator_sha256':sha(APP/'axis_evaluator.py'),
  'reference_axes_sha256':sha(APP/'REFERENCE_AXES.json'),'reference_manifest_sha256':sha(APP/'reference_manifest.csv'),
  'scoring_code_lock_sha256':sha(STAGE/'SCORING_CODE_LOCK.json'),'bootstrap_sha256':sha(STAGE/'frozen_bootstrap.py'),'runtime':current},lock
def score_one(mask,refs):
 start=time.perf_counter();peak=psutil.Process().memory_info().rss;done=threading.Event()
 def monitor():
  nonlocal peak
  while not done.wait(.05):peak=max(peak,psutil.Process().memory_info().rss)
 thread=threading.Thread(target=monitor,daemon=True);thread.start()
 try:
  axes,engineering=ax.extract_axes(mask);extract=time.perf_counter()-start;filaments=[];copies=[]
  for copy in refs:
   rows,s=ax.score_axes(axes,copy['references']);successful=[r for r in rows if r['status']=='SUCCESS']
   for r in rows:filaments.append({'sample':copy['sample'],'length_status':'OK' if r['status']=='SUCCESS' else 'FAIL_NA',**r})
   copies.append({'sample':copy['sample'],**s,'matching_yield':s['successful_references']/s['candidate_count'] if s['candidate_count'] else None,
    'length_status':'OK' if successful else 'FAIL_NA',**{k:float(np.mean([r[k] for r in successful])) if successful else None for k in LENGTH_FIELDS}})
  vals=[c['success_rate'] for c in copies if c['success_rate'] is not None];lc=[c for c in copies if c['length_status']=='OK']
  summary={'success_rate':float(np.mean(vals)) if vals else None,'planned_copies':len(copies),'evaluable_copies':len(vals),
   'eligible_references':sum(c['eligible_references'] for c in copies),'successful_references':sum(c['successful_references'] for c in copies),
   'successful_copies':len(lc),'length_status':'OK' if lc else 'FAIL_NA',**{k:float(np.mean([c[k] for c in lc])) if lc else None for k in LENGTH_FIELDS},**engineering}
  return {'per_reference':filaments,'per_copy':copies,'per_observation':summary,'extraction_seconds':extract,'total_seconds':time.perf_counter()-start,'peak_rss_bytes':peak}
 finally:done.set();thread.join()
def load_cached(p):
 record=read(p);h=record.pop('payload_sha256');assert digest(record)==h,'Score receipt corruption';return record
def score_record(root,out,m,rec,refs,sig):
 oid=rec['observation'];dest=out/'receipts'/m['run']/m['mode']/(oid+'.json')
 binding={'run':m['run'],'method':m['method'],'seed':m['seed'],'mode':m['mode'],'observation':oid,'threshold':m['threshold'],
  'checkpoint_sha256':m['checkpoint_sha256'],'mask_sha256':rec['sha256'],'signature':sig}
 if dest.exists():
  x=load_cached(dest);assert x['binding']==binding;return x
 # Equal binary mask at the same observation gives exactly equal extraction and matching; no reference-based filtering.
 cache=out/'mask_score_cache'/(digest({'oid':oid,'mask':rec['sha256'],'sig':sig})+'.json')
 if cache.exists():
  cached=load_cached(cache);assert cached['mask_sha256']==rec['sha256'] and cached['observation']==oid and cached['signature']==sig;result=cached['result'];reused=True
 else:
  with Image.open(root/rec['path']) as im:mask=np.asarray(im)>0
  result=score_one(mask,refs[oid]);cached={'mask_sha256':rec['sha256'],'observation':oid,'signature':sig,'result':result}
  dump(cache,{**cached,'payload_sha256':digest(cached)});reused=False
 x={'binding':binding,'result':result,'identical_mask_cache_reused':reused};dump(dest,{**x,'payload_sha256':digest(x)});return x
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--masks',required=True);ap.add_argument('--smoke',required=True);ap.add_argument('--output',required=True);ap.add_argument('--continue-full',action='store_true');a=ap.parse_args()
 root=Path(a.masks);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);sig,lock=scoring_signature();manifests,acceptance=accept(root,Path(a.smoke))
 sig['export_signature']=read(root/'signature.json')
 if (out/'signature.json').exists():assert read(out/'signature.json')==sig,'Scoring resume signature mismatch'
 else:dump(out/'signature.json',sig)
 dump(out/'MASK_ACCEPTANCE.json',acceptance);dump(out/'INPUT_SHA256.json',{'mask_export':acceptance['export']['input_sha256'],'target_smoke':acceptance['smoke']['input_sha256']})
 refs=read(APP/'REFERENCE_AXES.json');assert sorted(refs)==lock['observations'] and sum(len(v) for v in refs.values())==170
 assert sum(len(c['references']) for v in refs.values() for c in v)==1368
 numeric=[m for m in manifests if m['status']=='OK'];pilot_ids=[lock['observations'][i] for i in [0,12,107]]
 dump(out/'PLANNED_MODE_MATRIX.json',[{k:m[k] for k in ['run','method','seed','mode','status','na_reason','threshold','planned_observations']} for m in manifests])
 start=time.perf_counter();current=None
 try:
  pilot=[]
  for m in numeric:
   for oid in pilot_ids:
    rec=next(x for x in m['masks'] if x['observation']==oid);current={'run':m['run'],'mode':m['mode'],'observation':oid}
    x=score_record(root,out,m,rec,refs,sig);pilot.append({**current,**{k:x['result'][k] for k in ['extraction_seconds','total_seconds','peak_rss_bytes']},'cache_reused':x['identical_mask_cache_reused']})
   print('Native pilot',m['run'],m['mode'],flush=True)
  gate={'status':'PASS_EXACT_NATIVE_PILOT_FULL_SCORING_PENDING','fixed_pilot_ids':pilot_ids,'planned_mode_observations':117,'records':pilot,
   'max_seconds':max(x['total_seconds'] for x in pilot),'max_peak_rss_bytes':max(x['peak_rss_bytes'] for x in pilot),
   'free_disk_bytes':psutil.disk_usage(str(out)).free,'runtime':sig['runtime'],'signature':sig,
   'warning':'Pilot uses three predetermined observations; full-cohort complexity may differ. No successes/NA are screened by cost.'}
  assert gate['free_disk_bytes']>1024**3,'TECHNICAL_ERROR insufficient scoring disk reserve'
  dump(out/'ACTUAL_NATIVE_RESOURCE_GATE.json',gate)
  if not a.continue_full:
   dump(out/'STATUS.json',{'complete':False,'status':'ACTUAL_NATIVE_PILOT_PASSED_FULL_SCORING_PENDING'});return
  for m in numeric:
   for i,rec in enumerate(sorted(m['masks'],key=lambda x:x['observation'])):
    current={'run':m['run'],'mode':m['mode'],'observation':rec['observation']};score_record(root,out,m,rec,refs,sig)
    if (i+1)%10==0:print('Scored',m['run'],m['mode'],i+1,'/108',flush=True)
  from application_statistics import analyze
  analyze(out,manifests,refs,lock,sig)
  dump(out/'COMPLETE.json',{'status':'ALL_LOCKED_APPLICATION_SCORES_AND_STATISTICS_COMPLETE','complete':True,'runs':21,'planned_modes':42,
   'numeric_modes':39,'selected_na':3,'observations_per_numeric_mode':108,'numeric_per_reference_rows':39*1368,'planned_per_reference_rows_including_na':42*1368,
   'bootstrap_draws':100000,'unit':'108 observation and 13 original temporal groups','elapsed_seconds':time.perf_counter()-start,'signature':sig})
  dump(out/'OUTPUT_SHA256.json',{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='OUTPUT_SHA256.json' and not p.name.endswith('.tmp')})
 except BaseException as exc:
  dump(out/'STATUS.json',{'status':'TECHNICAL_ERROR','complete':False,'case':current,'error':str(exc),'signature':sig});raise
if __name__=='__main__':main()
