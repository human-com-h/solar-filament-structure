"""Kaggle Run All discovery, diagnostic gates, and explicit per-session run plans."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json,subprocess,sys,time,zipfile,shutil
from pathlib import Path
from methods import HERE,DESIGN,CONFIG,METHODS
from runtime import *
from data import prepare,old
PLANS={str(i+1):[("FlatUNet_BCE",s),("UNet_softDice_clDice",s)] for i,s in enumerate(DESIGN['seeds'])}

def discover(input_root):
    input_root=Path(input_root)
    annotations=[p for p in input_root.rglob('*.json') if 'annotation' in p.name.lower() and sha(p)==DESIGN['annotation_sha256']]
    assert len(annotations)==1,f'Need exactly one frozen annotation JSON, found {len(annotations)}'
    names=sorted({r['file_name'] for r in old.rows() if r['split'] in ('train','validation')})
    candidates=[p.parent for p in input_root.rglob(names[0])]
    image_dirs=[p for p in candidates if all((p/n).is_file() for n in names)]
    assert len(image_dirs)==1,f'Need one image directory containing all train+validation filenames, found {len(image_dirs)}'
    return str(annotations[0]),str(image_dirs[0])
def call(script,args,check=True,env=None,log=None):
    cmd=[sys.executable,str(HERE/script),*map(str,args)]
    return subprocess.run(cmd,check=check,env=env,stdout=log,stderr=subprocess.STDOUT if log else None)
def diagnostic(a):
    work=Path(a.output);work.mkdir(parents=True,exist_ok=True)
    annotation,images=discover(a.input);cache=Path('/tmp/sabr_strong_smoke_cache')
    prepare(annotation,images,cache,True)
    call('target_preflight.py',['--output',work/'preflight'])
    args=['--json-path',annotation,'--image-dir',images,'--cache',cache,'--output',work/'target_smoke','--timeout',3600]
    call('smoke.py',args)
    local=HERE/'evidence/LOCAL_SMOKE_REPORT.json'
    call('release.py',['--local',local,'--preflight',work/'preflight/TARGET_ENV_PREFLIGHT.json','--resource',HERE/'MORDEN_CHECKPOINT_EXCLUSION.json','--target',work/'target_smoke/SMOKE_REPORT.json','--output',work/'RELEASE_GATE.json'])
    print('Diagnostic phase only; NO formal run was launched.',flush=True)
def restore(inputs,output,runs):
    root=Path(output)
    for method,seed in runs:
        runid=f'{method}_seed{seed}';dest=root/runid;dest.mkdir(parents=True,exist_ok=True)
        if (dest/'recovery.zip').exists():continue
        choices=[]
        for p in Path(inputs).rglob('recovery.zip'):
            with zipfile.ZipFile(p) as z:
                cfg=json.loads(z.read('identity.json'))
                if cfg.get('method')==method and cfg.get('seed')==seed and cfg.get('smoke') is False:
                    choices.append((json.loads(z.read('epoch.json'))['epoch'],p,cfg,'zip'))
        for p in Path(inputs).rglob('identity.json'):
            parent=p.parent
            if not (parent/'state.pt').is_file() or not (parent/'epoch.json').is_file():continue
            cfg=json.loads(p.read_text())
            if cfg.get('method')==method and cfg.get('seed')==seed and cfg.get('smoke') is False:
                choices.append((json.loads((parent/'epoch.json').read_text())['epoch'],parent,cfg,'expanded'))
        if not choices:continue
        epoch,p,cfg,kind=max(choices,key=lambda x:x[0])
        ties=[(q,k) for e,q,c,k in choices if e==epoch]
        assert len(ties)==1 or (all(k=='zip' for q,k in ties) and len({sha(q) for q,k in ties})==1),'Ambiguous different archives for the same epoch'
        assert cfg['code_sha256']==verify_code() and cfg['environment_sha256']==digest(environment()),'Recovery code/environment drift'
        with run_lock(dest/'.run.lock'):
            tmp=dest/'recovery.zip.partial'
            if kind=='zip':
                with zipfile.ZipFile(p) as z:assert z.testzip() is None
                disk_check('/kaggle/working',p.stat().st_size);shutil.copyfile(p,tmp)
                assert sha(tmp)==sha(p)
            else:
                # Kaggle Dataset upload can expand recovery.zip. Repack without
                # deserializing or changing any tensor/state bytes.
                disk_check('/kaggle/working',(p/'state.pt').stat().st_size+1024**2)
                with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
                    for name in ('state.pt','identity.json','epoch.json'):z.write(p/name,name)
                with zipfile.ZipFile(tmp) as z:assert z.testzip() is None
            os.replace(tmp,dest/'recovery.zip');dump(dest/'config.json',cfg)
        print('Restored',runid,'epoch',epoch,flush=True)
def session_worker(a):
    # This process sees exactly one GPU through CUDA_VISIBLE_DEVICES.
    common=['--method',a.method,'--seed',a.seed,'--json-path',a.annotation,'--image-dir',a.images,'--cache',a.cache,'--output',a.output,'--deadline',a.deadline,'--gate',a.gate]
    call('engine.py',['train',*common])
    status=json.loads((Path(a.output)/f'{a.method}_seed{a.seed}/STATUS.json').read_text())
    if status['complete']:call('engine.py',['export',*common])
def formal(a):
    # No automatic upgrades, no default nine-run loop. One explicit wave per session.
    started=time.time();deadline=started+9*3600
    gate=json.loads(Path(a.gate).read_text());assert gate['status']=='READY_FOR_EXECUTION','Formal execution locked: evidence gates incomplete'
    assert gate['code_sha256']==verify_code() and gate['environment_sha256']==digest(environment())
    for key in ('LOCAL_SMOKE_PASSED','TARGET_ENV_PREFLIGHT_PASSED','MORDEN_EXCLUSION_VERIFIED','TARGET_SMOKE_PASSED','code_consistency','target_environment_consistency'):assert gate['checks'].get(key) is True
    runs=PLANS[a.plan];assert len(runs)<=torch.cuda.device_count()
    assert sum(m=='MORDEN_Focal' for m,s in runs)<=1
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    for p in output.glob('*_seed*/recovery.zip'):
        assert p.parent.name in {f'{m}_seed{s}' for m,s in runs},'Do not accumulate other runs in this workspace'
    # Serialize/extract no duplicate best/last files. Reserve old+new snapshot per run.
    budget=sum(CONFIG[m]['parameters']*32+64*1024**2 for m,s in runs)
    existing=sum(p.stat().st_size for p in output.glob('*_seed*/recovery.zip'))
    disk_check('/kaggle/working',max(0,budget-existing))
    assert psutil.virtual_memory().available>sum(10*1024**3 if m=='MORDEN_Focal' else 2*1024**3 for m,s in runs),'Insufficient CPU RAM headroom for this wave'
    annotation,images=discover(a.input);cache=Path('/tmp/sabr_strong_train_cache')
    prepare(annotation,images,cache,False);restore(a.input,output,runs)
    procs=[];logs=[]
    try:
        for gpu,(method,seed) in enumerate(runs):
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
            log=(output/f'{method}_seed{seed}.log').open('a',encoding='utf-8');logs.append(log)
            cmd=[sys.executable,str(Path(__file__)),'worker','--method',method,'--seed',str(seed),'--annotation',annotation,'--images',images,'--cache',str(cache),'--output',str(output),'--deadline',str(deadline),'--gate',a.gate]
            procs.append(subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT))
        codes=[p.wait() for p in procs]
        dump(output/'SESSION_STATUS.json',{'plan':a.plan,'runs':runs,'exit_codes':codes,'budget_seconds':9*3600,'elapsed_seconds':time.time()-started,'independent_single_gpu_processes':True})
        if any(codes):raise RuntimeError('One or more processes failed; previous complete recovery.zip remains available')
    finally:
        for log in logs:log.close()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['diagnostic','formal','worker']);p.add_argument('--input',default='/kaggle/input');p.add_argument('--output',default='/kaggle/working/sabr_strong');p.add_argument('--plan',choices=list(PLANS));p.add_argument('--gate',default='')
    for name in ('method','seed','annotation','images','cache','deadline'):p.add_argument('--'+name)
    a=p.parse_args()
    if a.action=='diagnostic':diagnostic(a)
    elif a.action=='formal':formal(a)
    else:session_worker(a)
