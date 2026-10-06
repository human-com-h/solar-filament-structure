"""Frozen exploratory calibration; local CPU; writes only its own new directory."""
from pathlib import Path
from datetime import datetime,timezone
from collections import defaultdict,Counter
from itertools import combinations
import ast,csv,hashlib,json,platform,sys,time
import numpy as np,scipy,skimage,cv2
from PIL import Image
from scipy.spatial import cKDTree
ROOT=Path('E:/SABR'); HERE=Path(__file__).resolve().parent
APP=ROOT/'project/manuscript_revision_20261002/morphology_application'
OLD=ROOT/'project/morphology_application_results_20261002'
MASKS=ROOT/'output/Kaggle_SABR_native_masks_output/mask_export'
sys.dont_write_bytecode=True
sys.path.insert(0,str(APP)); import axis_evaluator as ax
LF=['signed_relative_error','absolute_relative_error','absolute_error_px']
CATS=['SUCCESS','no_candidate','qualifying_edge_unassigned','reference_direction_only','candidate_direction_only','separate_direction_candidates','neither_direction_reaches_90pct']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def csvread(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def csvwrite(name,rows):
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with (HERE/(name+'.csv')).open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
def mean(v):
 v=[float(x) for x in v if x is not None and x!=''];return float(np.mean(v)) if v else None
def std(v):
 v=[float(x) for x in v if x is not None and x!=''];return float(np.std(v,ddof=1)) if len(v)>1 else None
def key(r):return r['observation'],r['sample'],r['annotation_id']
def summarize_lengths(rows):return {k:mean([r[k] for r in rows]) for k in LF}
def axes_json(axes):return [{**a,'points':np.asarray(a['points']).tolist()} for a in axes]
def diagnostic(axes,refs):
 rows,s=ax.score_axes(axes,refs)
 qs=[ax.sample(a['points']) for a in axes]; qt=[cKDTree(q) for q in qs]
 for r,ref in zip(rows,sorted(refs,key=lambda x:x['annotation_id'])):
  p=ax.sample(ref['points']);pt=cKDTree(p)
  geom=[(float((t.query(p)[0]<=3).mean()),float((pt.query(q)[0]<=3).mean())) for q,t in zip(qs,qt)]
  maxr=max((v[0] for v in geom),default=None);maxp=max((v[1] for v in geom),default=None)
  qualifying=any(R>=.9 and P>=.9 for R,P in geom)
  if r['status']=='SUCCESS':cat='SUCCESS'
  elif not axes:cat='no_candidate'
  elif qualifying:cat='qualifying_edge_unassigned'
  elif maxr>=.9 and maxp>=.9:cat='separate_direction_candidates'
  elif maxr>=.9:cat='reference_direction_only'
  elif maxp>=.9:cat='candidate_direction_only'
  else:cat='neither_direction_reaches_90pct'
  best=max(range(len(geom)),key=lambda j:(min(geom[j]),sum(geom[j]),-axes[j]['candidate_id'])) if geom else None
  r.update(geometric_category=cat,max_reference_coverage=maxr,max_candidate_precision=maxp,
   best_balanced_candidate_id=axes[best]['candidate_id'] if best is not None else None,
   best_reference_coverage=geom[best][0] if best is not None else None,
   best_candidate_precision=geom[best][1] if best is not None else None,
   qualifying_candidate_count=sum(R>=.9 and P>=.9 for R,P in geom))
 assert sum(r['status']=='SUCCESS' for r in rows)==s['successful_references']
 ids=[r['candidate_id'] for r in rows if r['status']=='SUCCESS'];assert len(ids)==len(set(ids))
 return rows,s

def main():
 start=time.perf_counter();freeze=read(HERE/'ANALYSIS_FREEZE.json')
 for rel,h in freeze['input_sha256'].items():assert sha(ROOT/rel)==h,rel
 code_lock=HERE/'ANALYSIS_CODE_LOCK.json';myhash=sha(__file__)
 if code_lock.exists():assert read(code_lock)['sha256']==myhash
 else:dump(code_lock,{'sha256':myhash,'frozen_before_new_scoring_at_utc':datetime.now(timezone.utc).isoformat()})
 lock=read(APP/'LOCKED_APPLICATION.json');refs=read(APP/'REFERENCE_AXES.json')
 runtime={'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'skimage':skimage.__version__}
 assert runtime==lock['engineering_runtime'];assert sha(APP/'axis_evaluator.py')==lock['axis_evaluator_sha256']
 assert sha(ROOT/'data/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json')==lock['reference_annotation_sha256']
 oldhash=read(OLD/'OUTPUT_SHA256.json')
 for name in ['per_copy.csv','per_observation.csv','per_reference.csv','per_seed.csv']:assert sha(OLD/name)==oldhash[name]
 # Extract only the frozen rasterize function to avoid importing any model/GPU module.
 src=(ROOT/'project/mechanism_training/frozen_evaluator.py').read_text(encoding='utf-8-sig')
 node=next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name=='rasterize')
 ns={'np':np,'cv2':cv2};exec(compile(ast.Module(body=[node],type_ignores=[]),'<unchanged_frozen_rasterize>','exec'),ns)
 rasterize=ns['rasterize']; coco=read(ROOT/'data/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json')
 ims={str(x['id']):x for x in coco['images']};anns=defaultdict(list)
 for a in coco['annotations']:anns[str(a['image_id'])].append(a)
 metadata={r['physical_observation_id']:{'year':int(r['year']),'temporal_group':int(r['temporal_group'])} for r in csvread(ROOT/'project/mechanism_training/temporal_manifest.csv') if r['split']=='internal_test'}
 refrows=[];copyrows=[];obsrows=[];reference_input_hashes=[]
 for i,(oid,copies) in enumerate(sorted(refs.items())):
  cr=[]
  for copy in copies:
   sample=copy['sample'];mask=rasterize(ims[sample],anns[sample]);maskhash=hashlib.sha256(mask.tobytes()).hexdigest()
   dest=HERE/'reference_mask_receipts'/(sample+'.json');dest.parent.mkdir(exist_ok=True)
   bind={'observation':oid,'sample':sample,'native_union_mask_sha256':maskhash,'analysis_freeze_sha256':sha(HERE/'ANALYSIS_FREEZE.json'),'analysis_code_sha256':myhash}
   if dest.exists():
    rec=read(dest);assert rec['binding']==bind;rows=rec['rows'];s=rec['summary'];engineering=rec['engineering']
   else:
    axes,engineering=ax.extract_axes(mask);rows,s=diagnostic(axes,copy['references'])
    dump(dest,{'binding':bind,'rows':rows,'summary':s,'engineering':engineering,'axes':axes_json(axes)})
   reference_input_hashes.append({'observation':oid,'sample':sample,'native_union_mask_sha256':maskhash})
   for r in rows:refrows.append({'observation':oid,'sample':sample,**metadata[oid],**r})
   good=[r for r in rows if r['status']=='SUCCESS']
   row={'observation':oid,'sample':sample,**metadata[oid],**s,'matching_yield':s['successful_references']/s['candidate_count'] if s['candidate_count'] else None,
    'length_status':'OK' if good else 'FAIL_NA',**summarize_lengths(good),**engineering}
   copyrows.append(row);cr.append(row)
  lc=[r for r in cr if r['length_status']=='OK']
  obsrows.append({'observation':oid,**metadata[oid],'planned_copies':len(cr),'success_rate':mean([r['success_rate'] for r in cr]),
   'eligible_references':sum(r['eligible_references'] for r in cr),'successful_references':sum(r['successful_references'] for r in cr),
   'candidate_count_copy_mean':mean([r['candidate_count'] for r in cr]),'matching_yield':mean([r['matching_yield'] for r in cr]),
   'successful_copies':len(lc),'length_status':'OK' if lc else 'FAIL_NA',**{k:mean([r[k] for r in lc]) for k in LF}})
  if (i+1)%10==0:print('Reference masks',i+1,'/108 observations',flush=True)
 assert len(copyrows)==170 and len(refrows)==1368 and len(obsrows)==108
 csvwrite('reference_mask_per_reference',refrows);csvwrite('reference_mask_per_copy',copyrows);csvwrite('reference_mask_per_observation',obsrows);csvwrite('reference_native_mask_hashes',reference_input_hashes)
 rs={'observations':108,'copies':170,'eligible_references':1368,'successful_references':sum(r['status']=='SUCCESS' for r in refrows),
  'success_rate':mean([r['success_rate'] for r in obsrows]),'matching_yield':mean([r['matching_yield'] for r in obsrows]),
  'successful_copies':sum(r['length_status']=='OK' for r in copyrows),'successful_observations':sum(r['length_status']=='OK' for r in obsrows),
  **{k:mean([r[k] for r in obsrows if r['length_status']=='OK']) for k in LF}}
 dump(HERE/'REFERENCE_MASK_SUMMARY.json',rs)
 for group in ['year','temporal_group']:
  out=[]
  for g in sorted(set(r[group] for r in obsrows)):
   rr=[r for r in obsrows if r[group]==g];out.append({group:g,'observations':len(rr),'success_rate':mean([r['success_rate'] for r in rr]),'eligible_references':sum(r['eligible_references'] for r in rr),'successful_references':sum(r['successful_references'] for r in rr)})
  csvwrite('reference_mask_per_'+group,out)
 # Every unordered copy pair, both directions for count symmetry QA.
 pairrows=[];pairref=[];pair_symmetry=True
 for oid,copies in sorted(refs.items()):
  for left,right in combinations(copies,2):
   aa=[{'candidate_id':j,'points':r['points'],'length':ax.length(r['points'])} for j,r in enumerate(sorted(right['references'],key=lambda x:x['annotation_id']))]
   bb=[{'candidate_id':j,'points':r['points'],'length':ax.length(r['points'])} for j,r in enumerate(sorted(left['references'],key=lambda x:x['annotation_id']))]
   rows,s=ax.score_axes(aa,left['references']);_,rev=ax.score_axes(bb,right['references']);assert s['successful_references']==rev['successful_references']
   pair_symmetry &= s['successful_references']==rev['successful_references']
   nL=len(left['references']);nR=len(right['references']);M=s['successful_references'];good=[r for r in rows if r['status']=='SUCCESS']
   pairrows.append({'observation':oid,**metadata[oid],'left_sample':left['sample'],'right_sample':right['sample'],'left_references':nL,'right_references':nR,'matched_pairs':M,
    'left_matched_fraction':M/nL if nL else None,'right_matched_fraction':M/nR if nR else None,'symmetric_matched_fraction':2*M/(nL+nR) if nL+nR else None,
    'unmatched_left':nL-M,'unmatched_right':nR-M,'length_status':'OK' if good else 'FAIL_NA',**summarize_lengths(good)})
   rightrefs=sorted(right['references'],key=lambda x:x['annotation_id'])
   for r in rows:pairref.append({'observation':oid,'left_sample':left['sample'],'right_sample':right['sample'],'right_annotation_id':rightrefs[r['candidate_id']]['annotation_id'] if r['status']=='SUCCESS' else None,**r})
 assert len(pairrows)==84
 po=[]
 for oid in sorted(set(r['observation'] for r in pairrows)):
  rr=[r for r in pairrows if r['observation']==oid];po.append({'observation':oid,**metadata[oid],'copy_pairs':len(rr),'symmetric_matched_fraction':mean([r['symmetric_matched_fraction'] for r in rr]),'left_matched_fraction':mean([r['left_matched_fraction'] for r in rr]),'right_matched_fraction':mean([r['right_matched_fraction'] for r in rr]),'matched_pair_instances':sum(r['matched_pairs'] for r in rr),'length_status':'OK' if any(r['length_status']=='OK' for r in rr) else 'FAIL_NA',**{k:mean([r[k] for r in rr if r['length_status']=='OK']) for k in LF}})
 assert len(po)==40
 csvwrite('annotation_copy_per_pair',pairrows);csvwrite('annotation_copy_per_reference',pairref);csvwrite('annotation_copy_per_observation',po)
 ps={'observations':40,'copy_pairs':84,'matched_pair_instances':sum(r['matched_pairs'] for r in pairrows),'left_instance_count':sum(r['left_references'] for r in pairrows),'right_instance_count':sum(r['right_references'] for r in pairrows),'symmetric_matched_fraction':mean([r['symmetric_matched_fraction'] for r in po]),'direction_count_symmetry':pair_symmetry,'independent_annotator_identity_available':False,'expert_confirmed_physical_correspondence':False,**{k:mean([r[k] for r in po if r['length_status']=='OK']) for k in LF}}
 dump(HERE/'ANNOTATION_COPY_SUMMARY.json',ps);print('All 84 copy pairs scored',flush=True)
 # Reuse frozen full-cohort summaries for all planned candidate modes.
 pc=csvread(OLD/'per_copy.csv');ob=csvread(OLD/'per_observation.csv');pr=csvread(OLD/'per_reference.csv');seedrows=csvread(OLD/'per_seed.csv');yields=[];yieldobs=[]
 for sr in seedrows:
  rr=[r for r in pc if r['run']==sr['run'] and r['mode']==sr['mode']];oo=[r for r in ob if r['run']==sr['run'] and r['mode']==sr['mode']]
  b={k:sr[k] for k in ['run','method','seed','mode','status','na_reason','threshold']}
  if sr['status']!='OK':yields.append({**b,'planned_observations':108,'evaluable_yield_observations':0,'matching_yield':None,'candidate_count_observation_mean':None,'unmatched_candidates_copy_mean':None,'pooled_copy_matching_yield':None});continue
  yy=[]
  for o in oo:
   cr=[r for r in rr if r['observation']==o['observation']];ys=[r['matching_yield'] for r in cr if r['matching_yield']!='']
   q={'run':sr['run'],'method':sr['method'],'seed':sr['seed'],'mode':sr['mode'],'observation':o['observation'],'planned_copies':len(cr),'evaluable_yield_copies':len(ys),'matching_yield':mean(ys),'candidate_count':int(o['candidate_count']),'unmatched_candidates_copy_mean':mean([r['unmatched_candidates'] for r in cr])}
   yy.append(q);yieldobs.append(q)
  yields.append({**b,'planned_observations':108,'evaluable_yield_observations':sum(r['matching_yield'] is not None for r in yy),'matching_yield':mean([r['matching_yield'] for r in yy]),'candidate_count_observation_mean':mean([r['candidate_count'] for r in yy]),'unmatched_candidates_copy_mean':mean([r['unmatched_candidates_copy_mean'] for r in yy]),'successful_references_copy_sum':sum(int(r['successful_references']) for r in rr),'candidate_count_copy_sum':sum(int(r['candidate_count']) for r in rr),'pooled_copy_matching_yield':sum(int(r['successful_references']) for r in rr)/sum(int(r['candidate_count']) for r in rr) if sum(int(r['candidate_count']) for r in rr) else None})
 assert len(yields)==42
 csvwrite('candidate_yield_per_seed',yields);csvwrite('candidate_yield_per_observation',yieldobs)
 ys=[]
 for method,mode in sorted(set((r['method'],r['mode']) for r in yields)):
  rr=[r for r in yields if r['method']==method and r['mode']==mode];out={'method':method,'mode':mode,'planned_seeds':3,'numeric_seeds':sum(r['status']=='OK' for r in rr)}
  for k in ['matching_yield','candidate_count_observation_mean','unmatched_candidates_copy_mean','pooled_copy_matching_yield']:out[k+'_mean']=mean([r[k] for r in rr]);out[k+'_sd']=std([r[k] for r in rr])
  ys.append(out)
 csvwrite('candidate_yield_three_seed_summary',ys)
 # Paired conditional length errors on the identical common success set.
 index=defaultdict(dict)
 for r in pr:index[r['run'],r['mode']][key(r)]=r
 common=[];commonref=[];commonobs=[]
 methods=sorted(set(r['method'] for r in seedrows)-{'SABR'})
 for seed in [20260831,20260901,20260902]:
  for mode in ['validation_selected','fixed_0.5']:
   sab=index[f'SABR_seed{seed}',mode]
   for meth in methods:
    comp=index[f'{meth}_seed{seed}',mode];assert set(sab)==set(comp)
    z=[]
    for k,a in sab.items():
     b=comp[k]
     if a['status']=='SUCCESS' and b['status']=='SUCCESS':
      row={'seed':seed,'mode':mode,'comparator':meth,'observation':k[0],'sample':k[1],'annotation_id':k[2]}
      for metric in LF:row['sabr_'+metric]=float(a[metric]);row['comparator_'+metric]=float(b[metric]);row['difference_'+metric]=float(a[metric])-float(b[metric])
      z.append(row);commonref.append(row)
    cc=defaultdict(list)
    for r in z:cc[r['observation'],r['sample']].append(r)
    oo=defaultdict(list)
    for (oid,sample),rr in cc.items():oo[oid].append({metric:mean([r[metric] for r in rr]) for metric in rr[0] if metric.startswith(('sabr_','comparator_','difference_'))})
    oout=[]
    for oid,cr in oo.items():
     row={'seed':seed,'mode':mode,'comparator':meth,'observation':oid,'common_success_copies':len(cr),**{metric:mean([r[metric] for r in cr]) for metric in cr[0]}}
     oout.append(row);commonobs.append(row)
    sr=next(r for r in seedrows if r['method']==meth and int(r['seed'])==seed and r['mode']==mode)
    out={'seed':seed,'mode':mode,'comparator':meth,'status':'OK' if z else ('INFEASIBLE_NA' if sr['status']!='OK' else 'NO_COMMON_SUCCESS_NA'),'planned_references':1368,'common_success_references':len(z),'common_success_copies':len(cc),'common_success_observations':len(oo)}
    for metric in LF:
     for pref in ['sabr_','comparator_','difference_']:out[pref+metric]=mean([r[pref+metric] for r in oout])
    common.append(out)
 assert len(common)==36
 csvwrite('common_success_length_per_seed',common);csvwrite('common_success_length_per_reference',commonref);csvwrite('common_success_length_per_observation',commonobs)
 cs=[]
 for meth,mode in sorted(set((r['comparator'],r['mode']) for r in common)):
  rr=[r for r in common if r['comparator']==meth and r['mode']==mode];out={'comparator':meth,'mode':mode,'planned_seeds':3,'numeric_seeds':sum(r['status']=='OK' for r in rr),'common_success_references_per_seed':','.join(str(r['common_success_references']) for r in rr),'common_success_observations_per_seed':','.join(str(r['common_success_observations']) for r in rr)}
  for metric in LF:
   for pref in ['sabr_','comparator_','difference_']:out[pref+metric+'_mean']=mean([r[pref+metric] for r in rr]);out[pref+metric+'_sd']=std([r[pref+metric] for r in rr])
  cs.append(out)
 csvwrite('common_success_length_three_seed_summary',cs);print('Candidate yield and all 36 conditional paired lengths complete',flush=True)
 # Focal selected-run geometry, exactly replayed against frozen scores.
 predrows=[];accepted_masks=[];manifest_hashes=[];cache={};parity_count=0
 manifests=read(MASKS/'MASK_MANIFESTS.json')
 focal=[m for m in manifests if m['mode']=='validation_selected' and m['run'].split('_seed')[0] in ['SABR','SRL','BCE_Dice']]
 assert len(focal)==9
 for fm in focal:
  mp=MASKS/fm['path'];assert sha(mp)==fm['sha256'];m=read(mp);manifest_hashes.append({'path':mp.relative_to(ROOT).as_posix(),'sha256':sha(mp)})
  assert sorted(r['observation'] for r in m['masks'])==sorted(refs)
  old=index[m['run'],m['mode']]
  for i,rec in enumerate(sorted(m['masks'],key=lambda r:r['observation'])):
   oid=rec['observation'];p=MASKS/rec['path'];assert sha(p)==rec['sha256'];accepted_masks.append({'run':m['run'],'observation':oid,'path':p.relative_to(ROOT).as_posix(),'sha256':rec['sha256']})
   dest=HERE/'prediction_diagnostic_receipts'/m['run']/(oid+'.json');dest.parent.mkdir(parents=True,exist_ok=True)
   bind={'run':m['run'],'observation':oid,'mask_sha256':rec['sha256'],'analysis_freeze_sha256':sha(HERE/'ANALYSIS_FREEZE.json'),'analysis_code_sha256':myhash}
   if dest.exists():
    x=read(dest);assert x['binding']==bind;rr=x['rows']
   else:
    ck=(oid,rec['sha256'])
    if ck not in cache:
     with Image.open(p) as im:arr=np.asarray(im)
     assert arr.shape==(2048,2048) and set(np.unique(arr))<={0,255}
     cache[ck]=ax.extract_axes(arr>0)
    axes,engineering=cache[ck];rr=[]
    for copy in refs[oid]:
     rows,s=diagnostic(axes,copy['references'])
     rr.extend({'observation':oid,'sample':copy['sample'],**metadata[oid],**r} for r in rows)
    dump(dest,{'binding':bind,'rows':rr,'engineering':engineering,'axes':axes_json(axes)})
   for r in rr:
    orig=old[key(r)];assert r['status']==orig['status']
    assert r['candidate_id']==(int(orig['candidate_id']) if orig['candidate_id'] else None)
    for k in LF:
     assert (r[k] is None and orig[k]=='') or float(orig[k])==r[k],(m['run'],oid,r['annotation_id'],k)
    predrows.append({'run':m['run'],'method':m['method'],'seed':m['seed'],'mode':m['mode'],**r});parity_count+=1
   if (i+1)%20==0:print('Focal geometry',m['run'],i+1,'/108',flush=True)
  print('Focal geometry complete',m['run'],flush=True)
 assert parity_count==9*1368
 csvwrite('prediction_failure_per_reference',predrows);csvwrite('prediction_input_mask_hashes',accepted_masks);csvwrite('prediction_input_manifest_hashes',manifest_hashes)
 catrows=[]
 groups=[('REFERENCE_MASK',None,'REFERENCE_MASK',refrows)]+[(m['method'],m['seed'],m['run'],[r for r in predrows if r['run']==m['run']]) for m in [read(MASKS/f['path']) for f in focal]]
 for method,seed,run,rr in groups:
  counts=Counter(r['geometric_category'] for r in rr)
  for cat in CATS:catrows.append({'run':run,'method':method,'seed':seed,'geometric_category':cat,'reference_instances':len(rr),'count':counts[cat],'reference_instance_fraction':counts[cat]/len(rr)})
 csvwrite('failure_category_counts',catrows)
 for rel,h in freeze['input_sha256'].items():assert sha(ROOT/rel)==h,rel
 qa={'status':'PASS_ALL_NEW_DIAGNOSTICS_COMPLETE','source_input_hashes_unchanged':True,'runtime_exact_lock_match':runtime,'opencv_version':cv2.__version__,
  'unchanged_frozen_rasterize_ast_extracted':True,'same_r3_both90_matching':True,'no_new_threshold_search':True,'old_prediction_reference_rows_exactly_replayed':parity_count,
  'reference_mask_copies':len(copyrows),'reference_mask_reference_instances':len(refrows),'copy_pairs':len(pairrows),'multicopy_observations':len(po),
  'candidate_mode_rows_including_na':len(yields),'candidate_na_rows':sum(r['status']!='OK' for r in yields),'paired_length_comparisons':len(common),
  'one_to_one_assignment_checks':True,'copy_pair_direction_cardinality_equal':pair_symmetry,'prediction_masks_sha256_verified':len(accepted_masks),'no_training_inference_or_remote_tasks':True,
  'elapsed_seconds':time.perf_counter()-start,'completed_at_utc':datetime.now(timezone.utc).isoformat()}
 dump(HERE/'ENGINEERING_QA.json',qa)
 dump(HERE/'SUMMARY.json',{'reference_mask':rs,'annotation_copy':ps,'selected_candidate_yields':[r for r in ys if r['mode']=='validation_selected'],'selected_common_success_lengths':[r for r in cs if r['mode']=='validation_selected'],'geometric_categories':catrows,'engineering_qa':qa})
 dump(HERE/'OUTPUT_SHA256.json',{p.relative_to(HERE).as_posix():sha(p) for p in sorted(HERE.rglob('*')) if p.is_file() and p.name!='OUTPUT_SHA256.json' and '__pycache__' not in str(p)})
 print(json.dumps({'reference_mask':rs,'annotation_copy':ps,'qa':qa},ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as exc:
  dump(HERE/'TECHNICAL_ERROR.json',{'status':'TECHNICAL_ERROR_NOT_SCIENTIFIC_NA','error':str(exc),'time_utc':datetime.now(timezone.utc).isoformat()});raise
