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
def invariant_tests(root):
    result={};loss=criterion('UNet_softDice_clDice')
    assert loss.soft_skeletonize.num_iter==10 and not loss.exclude_background
    cases=[]
    for pred,target in [(0,0),(0,1),(1,0),(1,1),(.3,0),(.3,1)]:
        p=torch.full((2,1,32,32),float(pred),requires_grad=True);y=torch.full_like(p,float(target));v=loss(y,p);v.backward()
        assert torch.isfinite(v) and torch.isfinite(p.grad).all();cases.append(float(v.detach()))
    # Count actual erosion calls: one opening + (erode + opening) per iteration.
    sk=loss.soft_skeletonize;original=sk.soft_erode;calls=[]
    def counted(x):calls.append(1);return original(x)
    sk.soft_erode=counted;sk(torch.rand(2,1,32,32));sk.soft_erode=original
    assert len(calls)==21
    result['cldice']={'empty_and_single_channel_cases':cases,'actual_iterations':10,'erode_calls':len(calls)}
    for method in METHODS:
        z=torch.tensor([[[[-2.,0.,2.]]]],requires_grad=True);y=torch.tensor([[[[0.,1.,1.]]]])
        output=torch.sigmoid(z) if method=='MORDEN_Focal' else z
        p=probability(output,method)
        assert torch.equal(p,torch.sigmoid(z))
        if method=='MORDEN_Focal':assert p is output
        v=loss_value(output,y,method,criterion(method));v.backward();assert torch.isfinite(z.grad).all()
    result['output_domains']='PASS'
    # Use actual checkpoints elsewhere for resume. Here inject interruption into a tiny snapshot.
    state={'model':{'w':torch.ones(3)},'cfg':{'test':'atomic_archive'}}
    run=Path(root)/'archive_fault_test';save_archive(state,run,root);before=sha(run/'recovery.zip')
    try:save_archive({**state,'model':{'w':torch.zeros(3)}},run,root,True)
    except RuntimeError as e:assert 'injected' in str(e)
    else:raise AssertionError('Fault injection did not fire')
    assert before==sha(run/'recovery.zip') and equal(load_archive(run/'recovery.zip'),state)
    with run_lock(run/'.run.lock'):
        child=subprocess.run([sys.executable,'-c',"from runtime import run_lock; import sys;\nwith run_lock(sys.argv[1]): pass",str(run/'.run.lock')],cwd=HERE,capture_output=True,text=True)
        assert child.returncode!=0
    result['atomic_previous_archive_preserved']=True;result['same_run_concurrent_writer_rejected']=True
    init={};sampling={}
    expected=json.loads((HERE/'INITIAL_DIGESTS.json').read_text())
    for seed in DESIGN['seeds']:
        seed_all(seed);model=factory('UNet_softDice_clDice');actual=model_digest(model);del model
        init[str(seed)]={'actual':actual,'expected':expected[str(seed)],'matches':actual==expected[str(seed)]}
        methods_sequences=[]
        for method in METHODS:
            seed_all(seed)
            # Model construction consumes different RNG; sampler must remain independent.
            torch.rand(37 if method=='FlatUNet_BCE' else 93)
            seq=[]
            for epoch in (1,2,150):
                groups=Dataset(root,seed,False).groups
                generator=torch.Generator().manual_seed(seed+epoch*100003)
                # DataLoader consumes a base-seed draw before RandomSampler.
                torch.empty((),dtype=torch.int64).random_(generator=generator)
                order=torch.randperm(len(groups),generator=generator).tolist()
                seq.append([(groups[i][0],deterministic_choice(groups[i][1],seed,epoch,groups[i][0])) for i in order])
            methods_sequences.append(digest(seq))
        assert len(set(methods_sequences))==1;sampling[str(seed)]=methods_sequences[0]
    result['initialization']=init;result['sampling_all_seeds']=sampling
    with torch.device('meta'):
        result['parameters']={m:sum(p.numel() for p in factory(m).parameters()) for m in METHODS}
    return result

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
        insufficient_local_capacity=a.local and (torch.cuda.get_device_properties(0).total_memory<10*1024**3 or psutil.virtual_memory().available<12*1024**3)
        if method=='MORDEN_Focal' and (insufficient_local_capacity or a.skip_morden):
            # Known 4.61e8 parameters: floor excludes activations/workspaces and CPU best-state.
            floor=CONFIG[method]['parameters']*16
            report['methods'][method]={'status':'NOT_RUN_LOCAL_CAPACITY' if a.local else 'NOT_RUN_RESOURCE_PROBE_FAILED','gpu':torch.cuda.get_device_name(0),'parameter_gradient_adam_floor_bytes':floor,'available_cpu_ram_bytes':psutil.virtual_memory().available,'reason':'Full MORDEN engineering smoke pending; no altered batch/model substitute.'}
        else:
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
