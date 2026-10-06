"""CPU-only bounded engineering checks; no real scientific predictions."""
from pathlib import Path
import argparse,csv,gc,json,subprocess,sys,tempfile
import numpy as np
import torch
from methods import *
from runtime import *
from data import Dataset,old,audit_cache
from annotation_helpers import deterministic_choice
from vendor.cldice.cldice import soft_cldice

def same(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b

def invariant_tests(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    results={};loss=criterion(METHODS[0])
    z=torch.linspace(-3,3,2*32*32).reshape(2,1,32,32).requires_grad_()
    y=torch.zeros_like(z);y[0,0,5:20,13]=1;y[1,0,3:23,5:18]=1
    actual=loss_value(z,y,METHODS[0],loss)
    expected=bce_dice_loss(z,y)+.05*soft_cldice(iter_=10,smooth=1.,exclude_background=False)(y,torch.sigmoid(z))
    ga=torch.autograd.grad(actual,z,retain_graph=True)[0];gb=torch.autograd.grad(expected,z)[0]
    assert torch.equal(actual,expected) and torch.equal(ga,gb)
    assert torch.isfinite(ga).all()
    sk=loss.structure.soft_skeletonize;original=sk.soft_erode;calls=[]
    def counted(x):calls.append(1);return original(x)
    sk.soft_erode=counted;sk(torch.rand(2,1,32,32));sk.soft_erode=original
    assert len(calls)==21
    results['historical_region_plus_upstream_structure_value_gradient_exact']=True
    results['actual_skeleton_iterations']=10
    vals=[]
    for logit,target in [(-8,0),(-8,1),(8,0),(8,1),(0,0),(0,1)]:
        z=torch.full((2,1,32,32),float(logit),requires_grad=True);y=torch.full_like(z,float(target))
        v=loss(z,y);v.backward();assert torch.isfinite(v) and torch.isfinite(z.grad).all();vals.append(float(v))
    results['empty_and_full_foreground_finite_losses']=vals
    init={};expected_init=json.loads((HERE/'INITIAL_DIGESTS.json').read_text())
    for seed in DESIGN['seeds']:
        seed_all(seed);model=factory(METHODS[0]);value=model_digest(model);del model
        init[str(seed)]={'actual':value,'archived_expected':expected_init[str(seed)],'matches':value==expected_init[str(seed)]}
    results['initialization']=init
    assert sha(HERE/'legacy/unet.py')==old.protocol()['source_provenance']['unet.py']['sha256']
    assert sha(HERE/'legacy/temporal_manifest.csv')==DESIGN['manifest_sha256']
    results['unet_and_manifest_preserved']=True
    records=old.rows();counts={s:len({r['physical_observation_id'] for r in records if r['split']==s}) for s in DESIGN['split_counts']}
    assert counts==DESIGN['split_counts'];results['physical_split_counts']=counts
    sampling={}
    for seed in DESIGN['seeds']:
        legacy=old.TrainingData(root,seed);new=Dataset(root,seed,False)
        assert legacy.groups==new.groups
        seqs=[]
        for ds in (legacy,new):
            seq=[]
            for epoch in (1,2,150):
                gen=torch.Generator().manual_seed(seed+epoch*100003)
                torch.empty((),dtype=torch.int64).random_(generator=gen)
                for i in torch.randperm(len(ds),generator=gen).tolist():
                    oid,aids=ds.groups[i];seq.append((epoch,oid,deterministic_choice(aids,seed,epoch,oid)))
            seqs.append(seq)
        assert seqs[0]==seqs[1];sampling[str(seed)]=digest(seqs[0])
    results['legacy_sampling_all_seeds_epochs_1_2_150']=sampling
    # Synthetic cache with real manifest keys tests isolation and actual loader
    # ordering without reading any physical image or creating a science score.
    from PIL import Image
    cache=root/'synthetic_cache';(cache/'images').mkdir(parents=True);(cache/'targets').mkdir()
    allowed=[r for r in records if r['split'] in ('train','validation')]
    keep=set()
    for split in ('train','validation'):
        keep.update(sorted({r['physical_observation_id'] for r in allowed if r['split']==split})[:4])
    allowed=[r for r in allowed if r['physical_observation_id'] in keep]
    for oid in keep:Image.fromarray(np.zeros((64,64),np.uint8)).save(cache/'images'/f'{oid}.png')
    for r in allowed:np.savez(cache/'targets'/f"{r['annotation_sample_id']}.npz",mask=np.zeros((64,64),bool),srl=np.zeros((64,64),bool),labels=np.zeros((64,64),np.uint16))
    sig={'manifest_sha256':DESIGN['manifest_sha256'],'annotation_sha256':DESIGN['annotation_sha256'],
         'pipeline_sha256':sha(HERE/'legacy/data_pipeline.py'),'source_provenance':old.protocol()['source_provenance'],'smoke':True,
         'annotation_ids':sorted(r['annotation_sample_id'] for r in allowed),'image_sha256':{r['file_name']:'SYNTHETIC_ENGINEERING_INPUT' for r in allowed}}
    dump(cache/'signature.json',sig);dump(cache/'COMPLETE.json',sig);audit_cache(cache,True)
    sentinel=cache/'images/UNEXPECTED_TEST_SENTINEL.png';sentinel.write_bytes(b'forbidden')
    try:audit_cache(cache,True)
    except AssertionError:results['unexpected_cache_image_rejected']=True
    else:raise AssertionError('Unexpected image entered cache')
    sentinel.unlink()
    parity={}
    for seed in DESIGN['seeds']:
        sequences=[]
        for ds in (old.TrainingData(cache,seed,limit=4),Dataset(cache,seed,True)):
            seq=[]
            for epoch in (1,2,150):
                ds.epoch=epoch
                loader=torch.utils.data.DataLoader(ds,batch_size=2,shuffle=True,generator=torch.Generator().manual_seed(seed+epoch*100003),num_workers=0)
                for b in loader:seq.extend(zip(b['oid'],b['aid']) if isinstance(b,dict) else zip(b[2],b[3]))
            sequences.append(seq)
        assert sequences[0]==sequences[1];parity[str(seed)]=digest(sequences[0])
    results['actual_legacy_dataloader_parity_synthetic_cache']=parity
    from engine import require_release
    from types import SimpleNamespace
    no_gate=root/'no_gate.json';dump(no_gate,{'status':'NOT_READY_FOR_EXECUTION'})
    cfg={'environment':{'torch':'2.10.0+cu128','cuda':'12.8','numpy':'2.0.2','gpu':'Tesla T4'}}
    try:require_release(SimpleNamespace(smoke=False,gate=str(no_gate)),cfg)
    except AssertionError:results['formal_training_without_target_gate_rejected']=True
    else:raise AssertionError('Ungated formal training accepted')
    bad={**cfg,'environment':{**cfg['environment'],'torch':'2.2.2'}}
    try:require_release(SimpleNamespace(smoke=False,gate=str(no_gate)),bad)
    except AssertionError:results['wrong_environment_formal_training_rejected']=True
    else:raise AssertionError('Wrong environment accepted')
    # Failure injection must preserve the previously committed bytes and state.
    state={'model':{'w':torch.ones(3)},'cfg':{'test':'transaction'},'epoch':1}
    run=root/'archive_fault';save_archive(state,run,root);before=sha(run/'recovery.zip')
    try:save_archive({**state,'model':{'w':torch.zeros(3)}},run,root,True)
    except RuntimeError as e:assert 'injected' in str(e)
    else:raise AssertionError('Fault did not fire')
    assert sha(run/'recovery.zip')==before and same(load_archive(run/'recovery.zip'),state)
    results['transaction_fault_keeps_complete_archive']=True
    with run_lock(run/'.run.lock'):
        cmd=[sys.executable,'-c','from runtime import run_lock; import sys\nwith run_lock(sys.argv[1]): pass',str(run/'.run.lock')]
        child=subprocess.run(cmd,cwd=HERE,capture_output=True,text=True)
        assert child.returncode!=0
    results['concurrent_same_run_rejected']=True
    try:disk_check(root,20_000_000_000)
    except RuntimeError:results['disk_guard_rejects_over_budget']=True
    else:raise AssertionError('Overbudget accepted')
    return results

def cpu_epoch(model,opt,loss,epoch):
    gen=torch.Generator().manual_seed(20260831+epoch*100003)
    x=torch.rand(2,1,64,64,generator=gen)
    y=torch.zeros_like(x);y[:,:,10:45,17:22]=1;y[:,:,29:34,10:43]=1
    model.train();opt.zero_grad(set_to_none=True);out=model(x);v=loss(out,y)
    v.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    opt.step()
    return float(v.detach())

def cpu_resume(root):
    seed=20260831;seed_all(seed);m=factory(METHODS[0]);o=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=1e-5);loss=criterion(METHODS[0])
    initial=model_digest(m);h=[cpu_epoch(m,o,loss,1),cpu_epoch(m,o,loss,2)]
    reference={'model':{k:v.clone() for k,v in m.state_dict().items()},'optimizer':o.state_dict(),'rng':rng(),'history':h}
    del m,o;gc.collect();seed_all(seed);m=factory(METHODS[0]);o=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=1e-5)
    h=[cpu_epoch(m,o,loss,1)]
    state={'model':m.state_dict(),'optimizer':o.state_dict(),'best_model':{k:v.clone() for k,v in m.state_dict().items()},'best':0.,'best_epoch':1,'stale':0,'history':h,'epoch':1,'rng':rng(),'initial_model_digest':initial,'cfg':{'purpose':'CPU synthetic 64x64 smoke only'}}
    save_archive(state,root/'cpu_resume',root);del m,o,state;gc.collect()
    restored=load_archive(root/'cpu_resume/recovery.zip')
    m=factory(METHODS[0]);o=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=1e-5)
    m.load_state_dict(restored['model']);o.load_state_dict(restored['optimizer']);restore_rng(restored['rng'])
    h=restored['history']+[cpu_epoch(m,o,loss,2)]
    assert same(m.state_dict(),reference['model']) and same(o.state_dict(),reference['optimizer'])
    assert same(rng(),reference['rng']) and h==reference['history']
    return {'model_optimizer_rng_loss_exact':True,'shape':[2,1,64,64],'epochs':2,'real_data_used':False,'initial_digest':initial}

def main(output):
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    report={'status':'NOT_PASSED','code_sha256':verify_code(),'environment':environment(),'formal_training_started':False,'gpu_training_started':False}
    with tempfile.TemporaryDirectory(prefix='cpu_smoke_',dir=root) as t:
        temp=Path(t);report['invariants']=invariant_tests(temp)
        report['cpu_resume']=cpu_resume(temp)
        # Code mutation must fail before models or data are touched.
        import shutil
        altered=temp/'altered';shutil.copytree(HERE,altered,ignore=shutil.ignore_patterns('__pycache__','evidence'))
        with (altered/'methods.py').open('a') as f:f.write('\n# intentional drift\n')
        result=subprocess.run([sys.executable,'-c','from runtime import verify_code; verify_code()'],cwd=altered,capture_output=True,text=True)
        assert result.returncode!=0 and 'drift' in result.stderr
        report['code_drift_rejected']=True
    report['status']='LOCAL_CPU_SMOKE_PASSED'
    report['limits']='CPU synthetic 64x64 only; target T4 full-resolution training/resume/export resource gate still mandatory.'
    dump(root/'LOCAL_CPU_SMOKE.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);main(p.parse_args().output)
