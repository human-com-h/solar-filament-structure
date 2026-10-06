"""Bounded real train/validation smoke, never the nine scientific runs."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import argparse,json,subprocess,sys,traceback,gc
import numpy as np
import torch
from methods import *
from runtime import *
from data import prepare,Dataset,old,audit_cache
from annotation_helpers import deterministic_choice

def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b
def fingerprint(a):
    if torch.is_tensor(a):return {'dtype':str(a.dtype),'shape':list(a.shape),'sha256':hashlib.sha256(a.cpu().numpy().tobytes()).hexdigest()}
    if isinstance(a,np.ndarray):return {'dtype':str(a.dtype),'shape':list(a.shape),'sha256':hashlib.sha256(a.tobytes()).hexdigest()}
    if isinstance(a,dict):return {str(k):fingerprint(v) for k,v in a.items()}
    if isinstance(a,(list,tuple)):return [fingerprint(x) for x in a]
    return a
def scientific_state(c):
    result={k:fingerprint(c[k]) for k in ['model','optimizer','best_model','best','best_epoch','stale','epoch','rng','initial_model_digest','cfg']}
    result['history']=[{k:r[k] for k in ['epoch','train_loss','val_physical_macro_dice']} for r in c['history']]
    return result
from local_smoke import invariant_tests

def run_method(a,method):
    root=Path(a.output);results={};runid=f'{method}_seed20260831'
    for sub,stop,action in [('continuous',0,'train'),('resumed',1,'train'),('resumed',0,'train'),('resumed',0,'export')]:
        cmd=[sys.executable,str(HERE/'engine.py'),action,'--method',method,'--seed','20260831','--smoke','--json-path',a.json_path,'--image-dir',a.image_dir,'--cache',a.cache,'--output',str(root/sub),'--stop-after-epoch',str(stop)]
        # Bounded subprocess; all scientific dimensions remain frozen.
        with (root/f'{method}_{sub}_{action}_{stop}.log').open('w',encoding='utf-8') as log:
            p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=a.timeout)
        if p.returncode:
            failure=root/sub/runid/'FAILURE.json'
            return {'status':'FAILED','phase':f'{sub}_{action}_{stop}','error':json.loads(failure.read_text()) if failure.exists() else f'exit {p.returncode}'}
        if sub=='continuous':
            left=load_archive(root/sub/runid/'recovery.zip');reference=scientific_state(left)
            dump(root/f'{method}_continuous_state_fingerprint.json',reference);del left;gc.collect()
            # Disposable engineering checkpoint only; keep fingerprints and logs. Avoid
            # accumulating multiple 6.9GiB MORDEN snapshots in the 20GB workspace.
            (root/sub/runid/'recovery.zip').unlink()
    right=load_archive(root/'resumed'/runid/'recovery.zip')
    assert scientific_state(right)==reference,'Full scientific resume state differs'
    report={'status':'PASS','resume_full_state_exact':True,'resume_history_scientific_fields_exact':True,'history_timing_not_compared':True,'initialization_matches_archived':right['initialization_matches_archived'],'history':right['history'],'archive':json.loads((root/'resumed'/runid/'archive_resources.json').read_text()),'validation_export':True,'cfg':right['cfg']}
    del right;gc.collect();return report
def main(a):
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True);code=verify_code()
    report={'status':'NOT_READY','code_sha256':code,'environment':environment(),'environment_sha256':digest(environment()),'formal_training_started':False,'methods':{}}
    report['invariants']=invariant_tests(root);dump(root/'SMOKE_REPORT.json',report)
    report['cache']=prepare(a.json_path,a.image_dir,a.cache,True)
    parity={}
    for seed in DESIGN['seeds']:
        sequences=[]
        for cls in ('legacy','new'):
            ds=old.TrainingData(a.cache,seed,limit=4) if cls=='legacy' else Dataset(a.cache,seed,True)
            seq=[]
            for epoch in (1,2,150):
                ds.epoch=epoch
                loader=torch.utils.data.DataLoader(ds,batch_size=2,shuffle=True,generator=torch.Generator().manual_seed(seed+epoch*100003),num_workers=0)
                for b in loader:seq.extend(zip(b['oid'],b['aid']) if cls=='legacy' else zip(b[2],b[3]))
            sequences.append(seq)
        assert sequences[0]==sequences[1];parity[str(seed)]=digest(sequences[0])
    report['legacy_dataloader_sampling_parity']=parity
    for method in METHODS:
        try:report['methods'][method]=run_method(a,method)
        except Exception as e:report['methods'][method]={'status':'FAILED','error':repr(e),'traceback':traceback.format_exc()}
        dump(root/'SMOKE_REPORT.json',report);print(method,report['methods'][method]['status'],flush=True)
    passed=[m for m,v in report['methods'].items() if v['status']=='PASS']
    initialized=all(v['matches'] for v in report['invariants']['initialization'].values())
    target='T4' in str(report['environment']['gpu'])
    report['status']='LOCAL_SMOKE_PASSED' if a.local and len(passed)==len(METHODS) else ('TARGET_SMOKE_PASSED' if not a.local and len(passed)==len(METHODS) else 'SMOKE_INCOMPLETE')
    report['passed_methods']=passed
    dump(root/'SMOKE_REPORT.json',report)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('json-path','image-dir','cache','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--local',action='store_true');p.add_argument('--skip-morden',action='store_true');p.add_argument('--timeout',type=int,default=3600)
    main(p.parse_args())
