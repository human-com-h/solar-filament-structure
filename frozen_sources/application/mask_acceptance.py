"""Independent CPU-only acceptance. COMPLETE alone is never sufficient."""
from pathlib import Path
import hashlib,json
import numpy as np
from PIL import Image
STAGE=Path(__file__).resolve().parent;P=STAGE/'package'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def png(p,record):
 assert sha(p)==record['sha256'] and Path(p).stat().st_size==record['bytes'],'PNG hash/size mismatch'
 with Image.open(p) as im:
  assert im.size==(2048,2048) and im.mode=='L';a=np.asarray(im);assert set(np.unique(a))<={0,255}
 assert record['width']==record['height']==2048 and record['foreground_pixels']==int(np.count_nonzero(a))
def validate(root,phase,lock):
 root=Path(root);sig=read(root/'signature.json');done=read(root/'COMPLETE.json')
 assert sig=={'export_lock_sha256':sha(P/'LOCKED_EXPORT.json'),'code_lock_sha256':sha(P/'EXPORT_CODE_LOCK.json'),
  'application_lock_sha256':lock['application_lock_sha256'],'environment':lock['target_environment'],
  'environment_sha256':lock['target_environment_sha256'],'phase':phase,'local_engineering':False,'schedule':lock['schedule']}
 assert digest(sig['environment'])==sig['environment_sha256']
 assert done['status']==('TARGET_MASK_EXPORT_SMOKE_PASSED' if phase=='smoke' else 'FROZEN_NATIVE_MASK_EXPORT_COMPLETE')
 assert done['complete'] and done['exit_codes']==[0,0] and done['runs']==21
 assert done['test_segmentation_metrics_recomputed'] is False and done['training_performed'] is False
 for k in ['export_lock_sha256','code_lock_sha256','application_lock_sha256','environment_sha256']:assert done[k]==sig[k]
 workers=[read(root/f'worker_gpu{i}_COMPLETE.json') for i in [0,1]]
 ids=[lock['smoke_id']] if phase=='smoke' else lock['test_ids'];n=len(ids)
 for i,w in enumerate(workers):
  assert w['complete'] and w['rank']==i and w['physical_gpu']==str(i) and w['signature']==sig
  assert w['assigned_runs']==lock['schedule'][str(i)] and w['observations_per_run']==n
  assert (root/f'worker_gpu{i}.log').is_file()
 if phase=='smoke':
  checks=[x for w in workers for x in w['checks']];assert len(checks)==21 and {x['run'] for x in checks}=={r['name'] for r in lock['runs']}
  assert done['checks']==checks
  for x in checks:
   r=next(r for r in lock['runs'] if r['name']==x['run']);assert x['expected']==r['smoke_expected'] and x['observation']==lock['smoke_id']
   assert x['fields']==r['smoke_fields'] and len(x['actual'])==len(x['expected'])==7
   assert x['tolerance']==1e-7 and np.isfinite(x['actual']).all()
   error=max(abs(a-b) for a,b in zip(x['actual'],x['expected']));assert error==x['max_error'] and error<=1e-7
 index=read(root/'MASK_MANIFESTS.json');assert len(index)==42 and len({(x['run'],x['mode']) for x in index})==42
 matrix=read(root/'MODE_MATRIX.json');assert len(matrix)==42
 manifests=[];pngs=set();receipts=set();input_hashes={}
 for r in lock['runs']:
  for oid in ids:
   rel=f"{r['name']}/receipts/{oid}.json";rec=read(root/rel);receipts.add(rel)
   assert rec['run']==r['name'] and rec['observation']==oid and rec['signature']==sig
   assert rec['checkpoint_sha256']==r['source_checkpoint_sha256'] and rec['export_checkpoint_sha256']==r['export_checkpoint_sha256']
   assert rec['tensor_sha256']==r['tensor_sha256'] and rec['input_image_sha256']==lock['images'][oid]['sha256']
   input_hashes[rel]=sha(root/rel)
  for mode in ['fixed_0.5','validation_selected']:
   ent=next(x for x in index if x['run']==r['name'] and x['mode']==mode);path=root/ent['path'];assert sha(path)==ent['sha256']
   m=read(path);t=.5 if mode=='fixed_0.5' else r['selected_threshold'];status='OK' if t is not None else 'INFEASIBLE_NA'
   assert m['run']==r['name'] and m['method']==r['method'] and m['seed']==r['seed'] and m['mode']==mode
   assert m['status']==ent['status']==status and m['threshold']==t and m['planned_observations']==n
   assert m['checkpoint_sha256']==r['source_checkpoint_sha256'] and m['export_checkpoint_sha256']==r['export_checkpoint_sha256']
   assert m['tensor_sha256']==r['tensor_sha256'] and m['segmentation_provenance']==r['segmentation_provenance']
   assert m['export_signature']==sig and m['environment']==sig['environment'] and m['environment_sha256']==sig['environment_sha256']
   row=next(x for x in matrix if x['run']==r['name'] and x['mode']==mode)
   assert row=={k:m[k] for k in ['run','method','seed','mode','status','na_reason','threshold','planned_observations','observations']}
   if status=='INFEASIBLE_NA':assert m['masks']==[] and m['observations']==0 and m['na_reason']
   else:
    assert len(m['masks'])==m['observations']==n and sorted(x['observation'] for x in m['masks'])==sorted(ids)
    for x in m['masks']:
     oid=x['observation'];rel=f"{r['name']}/{mode}/{oid}.png";assert x['path']==rel and x['mode']==mode and x['threshold']==t
     receipt=read(root/f"{r['name']}/receipts/{oid}.json")
     assert [y for y in receipt['masks'] if y['mode']==mode]==[{k:v for k,v in x.items() if k!='observation'}]
     png(root/rel,x);pngs.add(rel);input_hashes[rel]=x['sha256']
   input_hashes[ent['path']]=ent['sha256'];manifests.append(m)
 for r in lock['runs']:
  for oid in ids:
   rec=read(root/f"{r['name']}/receipts/{oid}.json");expected=['fixed_0.5']+(['validation_selected'] if r['selected_status']=='OK' else [])
   assert sorted(x['mode'] for x in rec['masks'])==sorted(expected)
 assert {p.relative_to(root).as_posix() for p in root.rglob('*.png')}==pngs
 assert {p.relative_to(root).as_posix() for p in root.glob('*_seed*/receipts/*.json')}==receipts
 assert len(pngs)==39*n and done['png_files']==39*n and done['observations_per_run']==n
 assert done['planned_mode_combinations']==42 and done['numeric_mode_combinations']==39 and done['selected_na']==3
 for p in root.glob('*.json'):input_hashes[p.name]=sha(p)
 for p in root.glob('*.log'):input_hashes[p.name]=sha(p)
 return manifests,{'phase':phase,'status':'PASS_ALL_FILES_VALIDATED','runs':21,'planned_modes':42,'numeric_modes':39,'selected_na':3,
  'observations_per_numeric_mode':n,'png_files':len(pngs),'receipts':len(receipts),'worker_exit_codes':done['exit_codes'],
  'target_environment_sha256':sig['environment_sha256'],'input_sha256':input_hashes}
def accept(mask_root,smoke_root):
 lock=read(P/'LOCKED_EXPORT.json')
 for rel,h in read(P/'EXPORT_CODE_LOCK.json').items():assert sha(P/rel)==h,('Package drift',rel)
 assert sha(P/'LOCKED_APPLICATION.json')==lock['application_lock_sha256']
 _,smoke=validate(smoke_root,'smoke',lock);manifests,result=validate(mask_root,'export',lock)
 return manifests,{'status':'PASS_TARGET_SMOKE_AND_ALL_NATIVE_MASKS','smoke':smoke,'export':result}
