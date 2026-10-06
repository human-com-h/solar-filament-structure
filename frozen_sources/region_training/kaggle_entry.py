"""One manual Run All: embedded checks, one selected seed on two GPUs, auto-resume."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import argparse,gc,json,shutil,subprocess,sys,tempfile,time,zipfile
import torch,psutil
from methods import HERE,DESIGN,METHODS,CONFIG
from runtime import *
from data import prepare,old
METHOD=METHODS[0]
# Runs are selected explicitly per session; there is no seed queue.
SESSION_LIMIT_SECONDS=12*3600
FINAL_COLLECTION_RESERVE_SECONDS=15*60
EXECUTION_BUDGET_SECONDS=SESSION_LIMIT_SECONDS-FINAL_COLLECTION_RESERVE_SECONDS
MIN_NEW_RUN_SECONDS=1800

def discover(input_root):
    root=Path(input_root)
    candidates=[p for p in root.rglob('*.json') if 'annotation' in p.name.lower() and sha(p)==DESIGN['annotation_sha256']]
    assert len(candidates)==1,f'需唯一原annotation JSON，找到{len(candidates)}份；移除重复挂载。'
    names=sorted({r['file_name'] for r in old.rows() if r['split'] in ('train','validation')})
    dirs=[p.parent for p in root.rglob(names[0]) if all((p.parent/n).is_file() for n in names)]
    assert len(dirs)==1,f'需唯一包含全部train/validation文件的图像目录，找到{len(dirs)}份。'
    return str(candidates[0]),str(dirs[0])

def call(script,args,timeout=None):
    return run_process([sys.executable,str(HERE/script),*map(str,args)],timeout=timeout)

def recovery_candidates(inputs,method,seed):
    choices=[]
    for p in Path(inputs).rglob('recovery.zip'):
        with zipfile.ZipFile(p) as z:
            cfg=json.loads(z.read('identity.json'))
            if cfg.get('method')==method and cfg.get('seed')==seed and cfg.get('smoke') is False:
                choices.append((p,'archive',cfg,sha(p),('expanded',p.parent)))
    for p in Path(inputs).rglob('identity.json'):
        parent=p.parent
        if not all((parent/n).is_file() for n in ('state.pt','epoch.json')):continue
        cfg=json.loads(p.read_text())
        if cfg.get('method')==method and cfg.get('seed')==seed and cfg.get('smoke') is False:
            choices.append((parent,'expanded',cfg,sha(parent/'state.pt'),('expanded',parent)))
    # Kaggle normally expands outer output.zip. If kept as a file, unpack only
    # the matching nested recovery archives into a dedicated read-only staging.
    for p in Path(inputs).rglob('output.zip'):
        with zipfile.ZipFile(p) as z:
            entries=[n for n in z.namelist() if n.endswith(f'{method}_seed{seed}/recovery.zip')]
            for n in entries:
                temp=Path(tempfile.gettempdir())/'study_region_cldice_restore'/str(seed)/('archive_'+sha(p)+'.zip')
                temp.parent.mkdir(parents=True,exist_ok=True)
                if not temp.exists():
                    with z.open(n) as src,temp.open('wb') as dst:shutil.copyfileobj(src,dst)
                with zipfile.ZipFile(temp) as inner:
                    cfg=json.loads(inner.read('identity.json'))
                    if cfg.get('method')==method and cfg.get('seed')==seed and cfg.get('smoke') is False:
                        choices.append((temp,'archive',cfg,sha(temp),('outer_zip',(p,n.rsplit('/',1)[0]+'/'))))
    return choices

def copy_run_evidence(companions,dest):
    kind,source=companions
    names={'config.json','STATUS.json','history.json','VALIDATION_COMPLETE.json'}
    if kind=='expanded':
        for name in names:
            if (source/name).is_file():shutil.copy2(source/name,dest/name)
        if (source/'validation').is_dir():shutil.copytree(source/'validation',dest/'validation',dirs_exist_ok=True)
    else:
        archive,prefix=source
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                if not name.startswith(prefix):continue
                relative=name[len(prefix):]
                if relative not in names and not relative.startswith('validation/'):continue
                path=(dest/relative).resolve();assert path.is_relative_to(dest.resolve())
                if name.endswith('/'):continue
                path.parent.mkdir(parents=True,exist_ok=True)
                with z.open(name) as src,path.open('wb') as dst:shutil.copyfileobj(src,dst)

def audit_recovery(run,method,seed):
    archive=Path(run)/'recovery.zip';c=load_archive(archive);cfg=c['cfg']
    assert cfg['method']==method and cfg['seed']==seed and cfg['smoke'] is False
    assert cfg['code_sha256']==verify_code() and cfg['environment_sha256']==digest(environment())
    assert cfg['design_sha256']==sha(HERE/'FROZEN_DESIGN.json') and cfg['manifest_sha256']==DESIGN['manifest_sha256']
    assert cfg['protocol_sha256']==sha(HERE/'legacy/protocol.json') and cfg['method_config_sha256']==digest(CONFIG[method])
    with zipfile.ZipFile(archive) as z:assert json.loads(z.read('identity.json'))==cfg
    assert c['initial_model_digest']==json.loads((HERE/'INITIAL_DIGESTS.json').read_text())[str(seed)]
    assert all(k in c for k in ('optimizer','rng','best_model','best_epoch','stale','history','model','rng_by_rank'))
    assert len(c['rng_by_rank'])==2 and cfg['distributed']['world_size']==2
    assert c['epoch']==len(c['history']) and 1<=c['epoch']<=150
    receipt={'seed':seed,'archive_sha256':sha(archive),'epoch':c['epoch'],'best_epoch':c['best_epoch'],
             'training_complete':c['epoch']>=150 or c['stale']>=20,'full_state_present':True,
             'code_environment_initialization_checked':True,'validation_complete':False}
    sigfile=Path(run)/'VALIDATION_COMPLETE.json'
    if sigfile.exists():
        assert receipt['training_complete'],'Paused training cannot be skipped as completed validation'
        sig=json.loads(sigfile.read_text());assert sig['cfg']==cfg and sig['archive_sha256']==receipt['archive_sha256']
        assert (Path(run)/'validation/curves.csv').is_file()
        receipt['validation_complete']=True
    del c;gc.collect();dump(Path(run)/'RECOVERY_RECEIPT.json',receipt)
    return receipt

def restore(inputs,output,runs):
    receipts={}
    for method,seed in runs:
        dest=Path(output)/f'{method}_seed{seed}';dest.mkdir(parents=True,exist_ok=True)
        if (dest/'recovery.zip').exists():
            receipts[str(seed)]=audit_recovery(dest,method,seed);continue
        choices=recovery_candidates(inputs,method,seed)
        if not choices:
            receipts[str(seed)]={'seed':seed,'state':'FRESH_PENDING','training_complete':False,'validation_complete':False};continue
        assert len(choices)==1,'同seed恢复点有多份，拒绝自动选最新。仅挂载一份该seed权威output.zip。'
        p,kind,cfg,_,companions=choices[0]
        assert cfg['code_sha256']==verify_code() and cfg['environment_sha256']==digest(environment()),'恢复代码/环境漂移'
        tmp=dest/'recovery.zip.partial'
        with run_lock(dest/'.run.lock'):
            if kind=='archive':
                with zipfile.ZipFile(p) as z:assert z.testzip() is None
                disk_check(Path(output).parent,p.stat().st_size);shutil.copyfile(p,tmp)
                assert sha(tmp)==sha(p)
            else:
                disk_check(Path(output).parent,(p/'state.pt').stat().st_size+1024**2)
                with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
                    for n in ('state.pt','identity.json','epoch.json'):z.write(p/n,n)
                with zipfile.ZipFile(tmp) as z:assert z.testzip() is None
            os.replace(tmp,dest/'recovery.zip');dump(dest/'config.json',cfg)
        copy_run_evidence(companions,dest)
        receipts[str(seed)]=audit_recovery(dest,method,seed)
        print('恢复',dest.name,flush=True)
    return receipts

def gate_candidates(inputs):
    gates=[]
    for p in Path(inputs).rglob('RELEASE_GATE.json'):
        data=json.loads(p.read_text())
        if data.get('status')=='READY_FOR_EXECUTION' and data.get('code_sha256')==verify_code():gates.append((p,data))
    for p in Path(inputs).rglob('output.zip'):
        with zipfile.ZipFile(p) as z:
            for name in z.namelist():
                if name.endswith('/RELEASE_GATE.json'):
                    data=json.loads(z.read(name))
                    if data.get('status')=='READY_FOR_EXECUTION' and data.get('code_sha256')==verify_code():gates.append((p,data))
    return gates

def embedded_checks(a,annotation,images,deadline):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    local=json.loads((HERE/'evidence/LOCAL_CPU_SMOKE.json').read_text())
    assert local['status']=='LOCAL_CPU_SMOKE_PASSED' and local['code_sha256']==verify_code()
    ddp_cpu=json.loads((HERE/'evidence/DDP_CPU_SMOKE.json').read_text())
    assert ddp_cpu['status']=='DDP_CPU_SMOKE_PASSED' and ddp_cpu['code_sha256']==verify_code()
    cache=Path(tempfile.gettempdir())/'study_region_cldice_smoke'
    prepare(annotation,images,cache,True)
    remaining=lambda:max(1,deadline-time.time()-900)
    call('target_preflight.py',['--output',out/'preflight','--seed',a.seed],timeout=min(1800,remaining()))
    call('ddp_smoke.py',['--json-path',annotation,'--image-dir',images,'--cache',cache,'--output',out/'target_smoke','--seed',a.seed,'--timeout',3600],timeout=min(3600,remaining()))
    pre=json.loads((out/'preflight/TARGET_ENV_PREFLIGHT.json').read_text())
    target=json.loads((out/'target_smoke/SMOKE_REPORT.json').read_text())
    checks={'LOCAL_CPU_SMOKE_PASSED':local['status']=='LOCAL_CPU_SMOKE_PASSED',
            'DDP_CPU_PARITY_PASSED':ddp_cpu['status']=='DDP_CPU_SMOKE_PASSED',
            'TARGET_ENV_PREFLIGHT_PASSED':pre['status']=='TARGET_ENV_PREFLIGHT_PASSED',
            'TARGET_SMOKE_PASSED':target['status']=='TARGET_SMOKE_PASSED',
            'TARGET_DDP_SMOKE_PASSED':target.get('execution_backend')=='DDP2_SyncBatchNorm_global_loss' and target.get('resume_model_optimizer_two_rank_rng_best_history_exact') is True,
            'code_consistency':local['code_sha256']==pre['code_sha256']==target['code_sha256']==verify_code(),
            'target_environment_consistency':pre['environment_sha256']==target['environment_sha256']==digest(environment())}
    gate={'status':'READY_FOR_EXECUTION' if all(checks.values()) else 'NOT_READY_FOR_EXECUTION',
          'checks':checks,'code_sha256':verify_code(),'environment_sha256':digest(environment()),
          'passed_methods':target['passed_methods'],'formal_training_started':False,
          'target_report_sha256':sha(out/'target_smoke/SMOKE_REPORT.json'),'preflight_sha256':sha(out/'preflight/TARGET_ENV_PREFLIGHT.json')}
    dump(out/'RELEASE_GATE.json',gate);assert gate['status']=='READY_FOR_EXECUTION'
    print('本session必要检查通过，继续同一notebook的正式训练。',flush=True)
    return gate

def distributed_call(action,a,annotation,images,cache,out,deadline):
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=2',str(HERE/'ddp_engine.py'),action,
             '--method',METHOD,'--seed',str(a.seed),'--json-path',annotation,'--image-dir',images,
             '--cache',str(cache),'--output',str(out),'--deadline',str(deadline),'--gate',str(out/'RELEASE_GATE.json'),'--epoch-estimate','600']
    run_process(command,timeout=max(1,deadline-time.time()))

def run_session(a,started=None):
    # Notebook bootstrap, cache, checks, restoration, training and export all
    # share this original timestamp. Embedded checks never reset the clock.
    assert a.seed in DESIGN['seeds'], 'SEED must be one of the three frozen seeds'
    runs=[(METHOD,a.seed)]
    started=float(started if started is not None else getattr(a,'session_started',0) or time.time())
    deadline=started+EXECUTION_BUDGET_SECONDS;out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    session={'session_started_unix':started,'execution_deadline_unix':deadline,'execution_budget_seconds':EXECUTION_BUDGET_SECONDS,
             'platform_limit_seconds':SESSION_LIMIT_SECONDS,'final_collection_reserve_seconds':FINAL_COLLECTION_RESERVE_SECONDS,
             'gpu_memory_pooling':False,'runs':runs,'selected_seed':a.seed,'execution_backend':'DDP2_SyncBatchNorm_global_loss','scope':'train/validation only; no test evaluation'}
    dump(out/'SESSION_STATUS.json',session)
    assert torch.cuda.device_count()>=2,'请选择T4×2；同一个seed由两张卡共同训练。'
    assert all('T4' in torch.cuda.get_device_name(i) for i in range(2))
    assert psutil.virtual_memory().available>4*1024**3
    disk_check(out.parent,sum(CONFIG[m]['parameters']*32+64*1024**2 for m,s in runs))
    annotation,images=discover(a.input)
    gate=embedded_checks(a,annotation,images,deadline)
    cache=Path(tempfile.gettempdir())/'study_region_cldice_train'
    prepare(annotation,images,cache,False)
    receipts=restore(a.input,out,runs)
    engineering={'session_started_unix':started,'remaining_after_checks_and_cache_seconds':deadline-time.time(),
                 'code_sha256':verify_code(),'environment_sha256':digest(environment()),'embedded_checks':gate,
                 'authoritative_checkpoint_audits':receipts,'external_gate_file_required':False}
    dump(out/'SESSION_ENGINEERING.json',engineering)
    try:
        receipt=receipts[str(a.seed)]
        session['skipped_completed']=bool(receipt.get('validation_complete'))
        if not receipt.get('validation_complete') and deadline-time.time()>=MIN_NEW_RUN_SECONDS:
            training_complete=bool(receipt.get('training_complete'))
            if not receipt.get('training_complete'):
                distributed_call('train',a,annotation,images,cache,out,deadline)
                training_complete=json.loads((out/f'{METHOD}_seed{a.seed}/STATUS.json').read_text())['complete']
            if training_complete and deadline-time.time()>900:
                distributed_call('export',a,annotation,images,cache,out,deadline)
        session['complete']=(out/f'{METHOD}_seed{a.seed}/VALIDATION_COMPLETE.json').exists()
        session['seed_finished']=a.seed if session['complete'] else None
    finally:
        session['elapsed_seconds']=time.time()-started;dump(out/'SESSION_STATUS.json',session)
    return session

def collect(output):
    root=Path(output);dest=root.parent/'output.zip';tmp=dest.with_name('output.zip.partial')
    if not root.exists():return
    # Never collect scratch caches or partial epoch archives.
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_STORED,allowZip64=True) as z:
        for p in sorted(root.rglob('*')):
            if p.is_file() and 'synthetic_cache' not in p.parts and not p.name.endswith(('.partial','.tmp','.lock')):
                z.write(p,str(Path('study_region_cldice')/p.relative_to(root)))
        for name in ('FROZEN_DESIGN.json','CODE_LOCK.json','INITIAL_DIGESTS.json','DESIGN_CN.md'):
            z.write(HERE/name,'protocol/'+name)
    with zipfile.ZipFile(tmp) as z:assert z.testzip() is None
    os.replace(tmp,dest);print('回收文件',dest,'bytes',dest.stat().st_size,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run'],nargs='?',default='run')
    p.add_argument('--input',default='/kaggle/input');p.add_argument('--output',default='/kaggle/working/study_region_cldice');p.add_argument('--gate',default='');p.add_argument('--session-started',type=float,default=0)
    p.add_argument('--seed',type=int,choices=DESIGN['seeds'],required=True)
    a=p.parse_args()
    try:run_session(a)
    finally:collect(a.output)
