"""Stage the accepted training closure outside the release and invoke its engine.

The seven single-GPU methods retain their original batch2 training. The added
region control always launches two NCCL ranks with SyncBatchNorm and the frozen
autograd-aware global loss. Staged code locks identify this portable layout and
are never substituted for the archival identities in provenance/.
"""
from pathlib import Path
import shutil,subprocess,sys,json
from .io import *
from .frozen import verify_sources

def copy_exact(source,dest):
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():
        if sha(source)!=sha(dest):raise ValueError('Staged source differs: '+str(dest))
    else:shutil.copyfile(source,dest)
def stage(args):
    verify_sources();root=output(args.output_root)/'execution_sources'
    if args.method in ('BCE_Dice','SRL','SABR','ResidualPixel','SRL_FG'):
        dest=root/'mechanism';source=REPO/'frozen_sources/mechanism'
        for p in source.glob('*.py'):copy_exact(p,dest/p.name)
        legacy=dest
    else:
        group='region_training' if args.method=='UNet_region_clDice' else 'baselines'
        dest=root/group
        for p in (REPO/'frozen_sources'/group).iterdir():
            if p.is_file() and p.name!='CODE_LOCK.json' and not (group=='baselines' and p.name=='methods.py'):copy_exact(p,dest/p.name)
        legacy=dest/'legacy'
        for p in (REPO/'frozen_sources/mechanism').glob('*.py'):copy_exact(p,legacy/p.name)
        for p in (REPO/'frozen_sources/vendor').rglob('*'):
            if p.is_file():copy_exact(p,dest/'vendor'/p.relative_to(REPO/'frozen_sources/vendor'))
        if group=='baselines':
            # Removed method is absent from accepted CONFIG. Its eager imports
            # needlessly required a full, unused MORDEN stack. Only those imports
            # and its sys.path entry are removed in the execution copy.
            src=(REPO/'frozen_sources/baselines/methods.py').read_text()
            removals=["sys.path.insert(0,str(HERE/'vendor/morden'))\n","from segmentation.utils.models.filament_seg import FilamentSeg\n","from segmentation.utils.criteria.focal_loss import FocalLoss\n"]
            for before in removals:
                if src.count(before)!=1:raise ValueError('Unexpected baseline import source')
                src=src.replace(before,'')
            adapted=dest/'methods.py'
            if adapted.exists() and adapted.read_text()!=src:raise ValueError('Staged baseline adapter differs')
            adapted.write_text(src,encoding='utf-8')
        # Workers inherit bytecode suppression; no cache files affect the lock.
    for name in ('temporal_manifest.csv','manual_spine_qc_exclusions.csv'):
        p=original(args.evidence_root,'project/mechanism_training/'+name)
        copy_exact(p,legacy/name)
    copy_exact(REPO/'configs/paper24/protocol.json',legacy/'protocol.json')
    if legacy!=dest:
        lock={p.relative_to(dest).as_posix():sha(p) for p in dest.rglob('*') if p.is_file() and ('legacy' in p.parts or 'vendor' in p.parts or p.parent==dest) and p.suffix in ('.py','.csv','.json') and p.name not in ('CODE_LOCK.json','ENGINEERING_STATUS.json')}
        dump(dest/'CODE_LOCK.json',lock)
    dump(root/'ADAPTATION.json',dict(original_identities='provenance/FROZEN_SOURCE_IDENTITIES.json',staged_directory=str(dest),legacy_sources_unchanged=True,scientific_functions_unchanged=True,baseline_change='remove eager imports for the already-excluded MORDEN method',staged_lock_is_archival_lock=False))
    return dest
def prepare_cache(args):
    import os
    dest=stage(args);legacy=dest/'legacy' if (dest/'legacy').exists() else dest
    annotation=safe(args.data_root,args.annotation);images=safe(args.data_root,args.images)
    code="import sys;sys.dont_write_bytecode=True;from data_pipeline import prepare;prepare(sys.argv[1],sys.argv[2],sys.argv[3],limit=0)"
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(legacy))
    subprocess.run([sys.executable,'-c',code,str(annotation),str(images),str(Path(args.cache).resolve())],env=env,check=True)
    return dict(status='PASS',cache=str(Path(args.cache).resolve()),train_and_validation_only=True)

def preflight(args):
    """Generate a gate from actual frozen smoke reports, never asserted flags.

    This command includes GPU engineering steps and is deliberately separate
    from validate. It does not start a formal 150-epoch run.
    """
    import os,torch
    if not torch.cuda.is_available():raise RuntimeError('Target GPU preflight requires CUDA; no GPU checks were run.')
    if args.method in ('BCE_Dice','SRL','SABR','ResidualPixel','SRL_FG'):
        raise ValueError('The original five-method trainer has no release-gate command; use validate and its original runtime checks.')
    dest=stage(args);out=output(args.output_root)/'preflight'
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
    annotation=str(safe(args.data_root,args.annotation));images=str(safe(args.data_root,args.images))
    def call(name,params):subprocess.run([sys.executable,str(dest/name),*map(str,params)],cwd=dest,env=env,check=True)
    common=['--json-path',annotation,'--image-dir',images,'--cache',str(Path(args.cache).resolve())]
    if args.method=='UNet_region_clDice':
        if sys.platform!='linux' or torch.cuda.device_count()!=2:raise RuntimeError('Region target preflight requires Linux and two visible T4 GPUs.')
        call('local_smoke.py',['--output',dest/'evidence'])
        call('cpu_ddp_smoke.py',['--output',dest/'evidence'])
        # embedded_checks retains its original environment, gradient, recovery
        # and code-consistency assertions. It never invokes run_session.
        code="import sys,time;sys.dont_write_bytecode=True;from types import SimpleNamespace;from kaggle_entry import embedded_checks;embedded_checks(SimpleNamespace(output=sys.argv[1],seed=int(sys.argv[2])),sys.argv[3],sys.argv[4],time.time()+7200)"
        subprocess.run([sys.executable,'-c',code,str(out),str(args.seed),annotation,images],cwd=dest,env=env,check=True)
    else:
        if not args.resource_record:raise ValueError('Supply --resource-record with the checksum-defined MORDEN resource failure JSON; it records unmeasured performance.')
        prepare=['--json-path',annotation,'--image-dir',images,'--cache',str(Path(args.cache).resolve()),'--output',str(out/'local_smoke'),'--local']
        call('smoke.py',prepare)
        call('target_preflight.py',['--output',out/'target_environment'])
        call('smoke.py',[*common,'--output',out/'target_smoke'])
        call('release.py',['--local',out/'local_smoke/SMOKE_REPORT.json','--preflight',out/'target_environment/TARGET_ENV_PREFLIGHT.json','--target',out/'target_smoke/SMOKE_REPORT.json','--resource',required(args.resource_record,'MORDEN resource failure record'),'--output',out/'RELEASE_GATE.json'])
    report=read(out/'RELEASE_GATE.json')
    if report['status']!='READY_FOR_EXECUTION':raise RuntimeError('Actual preflight gate did not pass.')
    return dict(status=report['status'],gate=str(out/'RELEASE_GATE.json'),formal_training_started=False)
def train(args):
    import torch,os
    if not torch.cuda.is_available():raise RuntimeError('GPU unavailable. Training has not started. CPU checks use validate --with-models.')
    dest=stage(args);args_output=output(args.output_root)/'training'
    command=[sys.executable]
    common=['--json-path',str(safe(args.data_root,args.annotation)),'--cache',str(Path(args.cache).resolve()),'--output',str(args_output),'--method',args.method,'--seed',str(args.seed)]
    if args.method=='UNet_region_clDice':
        if torch.cuda.device_count()!=2:raise RuntimeError('region+clDice requires exactly two visible GPUs')
        if sys.platform!='linux':raise RuntimeError('The frozen region backend requires Linux/NCCL')
        command+=['-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(dest/'ddp_engine.py'),'train']
    elif (dest/'engine.py').exists():command+=[str(dest/'engine.py'),'train']
    else:command+=[str(dest/'trainer.py')]
    command+=common
    if (dest/'engine.py').exists():
        if not args.gate:raise ValueError('Missing actual target release gate; use the frozen preflight workflow in docs/TRAINING.md.')
        command+=['--image-dir',str(safe(args.data_root,args.images)),'--gate',str(required(args.gate,'target preflight release gate'))]
    else:command+=['--workers','2']
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
    subprocess.run(command,cwd=dest,env=env,check=True)
    return dict(status='EXECUTED',method=args.method,seed=args.seed,engine=str(command[1:]))
