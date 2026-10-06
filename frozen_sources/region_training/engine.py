import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import argparse,json,time,gc
import numpy as np
import torch
from PIL import Image
from methods import HERE,DESIGN,CONFIG,METHODS,factory,criterion,probability,loss_value
from data import Dataset,audit_cache,old
from runtime import *

@torch.no_grad()
def validate(model,cache,method,smoke):
    model.eval();values=[]
    for oid,aids in old.validation_groups(2 if smoke else 0):
        with Image.open(Path(cache)/'images'/f'{oid}.png') as im:x=np.asarray(im,dtype=np.float32)/255.
        p=probability(model(torch.from_numpy(x[None,None]).cuda()),method)
        assert torch.isfinite(p).all() and p.min()>=0 and p.max()<=1
        pred=(p[0,0]>=.5).cpu().numpy();scores=[]
        for aid in aids:
            with np.load(Path(cache)/'targets'/f'{aid}.npz') as d:y=d['mask']
            scores.append(float((2*(pred&y).sum()+1e-6)/(pred.sum()+y.sum()+1e-6)))
        values.append(float(np.mean(scores)))
    return float(np.mean(values))

def identity(args):
    code=verify_code();old.check_inputs(args.json_path)
    env=environment()
    return {'method':args.method,'seed':args.seed,'method_config_sha256':digest(CONFIG[args.method]),'design_sha256':sha(HERE/'FROZEN_DESIGN.json'),'protocol_sha256':sha(HERE/'legacy/protocol.json'),'manifest_sha256':sha(HERE/'legacy/temporal_manifest.csv'),'code_sha256':code,'environment':env,'environment_sha256':digest(env),'cache':audit_cache(args.cache,args.smoke),'smoke':args.smoke,'batch_size':2,'resolution':1024,'max_epochs':2 if args.smoke else 150,'patience':20,'workers':0,'distributed':{'backend':'nccl','world_size':2,'global_batch_size':2,'per_rank_batch_size':1,'normalization':'SyncBatchNorm','structure_reduction':'autograd-aware global sufficient-statistic SUM'}}

def require_release(args,cfg):
    if args.smoke:return
    require_runtime_spec(cfg['environment'])
    assert cfg['environment']['torch']=='2.10.0+cu128' and cfg['environment']['cuda']=='12.8'
    assert cfg['environment']['numpy']=='2.0.2' and 'T4' in cfg['environment']['gpu']
    gate=json.loads(Path(args.gate).read_text())
    assert gate['status']=='READY_FOR_EXECUTION','Formal training locked until all gates pass'
    assert gate['code_sha256']==cfg['code_sha256'] and gate['environment_sha256']==cfg['environment_sha256'],'Gate belongs to different code/environment'
    assert set(gate['passed_methods'])==set(METHODS)
    for name in ('LOCAL_CPU_SMOKE_PASSED','DDP_CPU_PARITY_PASSED','TARGET_ENV_PREFLIGHT_PASSED','TARGET_SMOKE_PASSED','TARGET_DDP_SMOKE_PASSED','code_consistency','target_environment_consistency'):
        assert gate['checks'].get(name) is True,name

def train(args):
    assert args.smoke, 'Formal training uses ddp_engine.py with both GPUs'
    assert torch.cuda.is_available();assert args.seed in DESIGN['seeds'];assert args.method in METHODS
    cfg=identity(args);require_release(args,cfg)
    run=Path(args.output)/f'{args.method}_seed{args.seed}';run.mkdir(parents=True,exist_ok=True)
    with run_lock(run/'.run.lock'):
        cp=run/'config.json'
        if cp.exists():assert json.loads(cp.read_text())==cfg,'Resume identity/config/environment drift'
        else:dump(cp,cfg)
        seed_all(args.seed);model=factory(args.method).cuda();initial=model_digest(model)
        expected=json.loads((HERE/'INITIAL_DIGESTS.json').read_text())
        init_match=initial==expected[str(args.seed)]
        # A local smoke may diagnose version drift; it cannot authorize formal training.
        if not args.smoke:assert init_match,'Frozen UNet initialization mismatch'
        opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5);loss_fn=criterion(args.method)
        best=-1.;stale=0;history=[];start=1;best_state=None;best_epoch=0
        archive=run/'recovery.zip'
        if archive.exists():
            c=load_archive(archive);assert c['cfg']==cfg and c['initial_model_digest']==initial
            model.load_state_dict(c['model']);opt.load_state_dict(c['optimizer'])
            for state in opt.state.values():
                for key,v in state.items():
                    if torch.is_tensor(v) and key!='step':state[key]=v.cuda()
            best=c['best'];stale=c['stale'];history=c['history'];start=c['epoch']+1;best_state=c['best_model'];best_epoch=c['best_epoch']
            restore_rng(c['rng']);del c;gc.collect()
        ds=Dataset(args.cache,args.seed,args.smoke);paused=False;archive_stats=None
        for epoch in range(start,cfg['max_epochs']+1):
            if stale>=20:break
            last=history[-1]['seconds'] if history else float(getattr(args,'epoch_estimate',3600))
            reserve=max(900,2*float((archive_stats or {}).get('archive_seconds',0)))
            if args.deadline and time.time()+last*1.25+reserve>=args.deadline:paused=True;break
            ds.epoch=epoch;generator=torch.Generator().manual_seed(args.seed+epoch*100003)
            loader=torch.utils.data.DataLoader(ds,batch_size=2,shuffle=True,generator=generator,num_workers=0,pin_memory=True)
            model.train();losses=[];steps=[];started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
            with Monitor(args.output) as mon:
                for x,y,oids,aids in loader:
                    # Incomplete epoch never commits. Previous recovery.zip remains valid.
                    if args.deadline and time.time()+reserve>=args.deadline:paused=True;break
                    x=x.cuda();y=y.cuda();torch.cuda.synchronize();t=time.perf_counter()
                    opt.zero_grad(set_to_none=True);out=model(x)
                    loss=loss_value(out,y,args.method,loss_fn)
                    if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
                    loss.backward()
                    if not all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters() if p.requires_grad):raise RuntimeError('Missing/non-finite gradient')
                    opt.step();torch.cuda.synchronize();steps.append(time.perf_counter()-t);losses.append(float(loss.detach()))
                    del x,y,out,loss
                if paused:break
                score=validate(model,args.cache,args.method,args.smoke)
            row={'epoch':epoch,'train_loss':float(np.mean(losses)),'val_physical_macro_dice':score,'seconds':time.perf_counter()-started,'step_seconds':steps,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved(),'peak_process_rss_bytes':mon.ram,'peak_system_used_bytes':mon.system_used}
            history.append(row)
            if score>best+1e-6:
                best=score;stale=0;best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:stale+=1
            state={'model':model.state_dict(),'optimizer':opt.state_dict(),'best_model':best_state,'best':best,'best_epoch':best_epoch,'stale':stale,'history':history,'epoch':epoch,'cfg':cfg,'rng':rng(),'initial_model_digest':initial,'initialization_matches_archived':init_match}
            archive_stats=save_archive(state,run,args.output);del state
            dump(run/'archive_resources.json',archive_stats);dump(run/'history.json',history)
            print(args.method,args.seed,'epoch',epoch,'archive_bytes',archive_stats['zip_bytes'],flush=True)
            if args.stop_after_epoch and epoch>=args.stop_after_epoch:paused=True;break
        complete=bool(history) and (stale>=20 or history[-1]['epoch']>=cfg['max_epochs'])
        dump(run/'STATUS.json',{'complete':complete,'paused':paused,'epochs':len(history),'initialization_matches_archived':init_match,'smoke':args.smoke})
    return complete

def export(args):
    # Frozen metric math, native-grid nearest remapping, original 39 thresholds.
    import frozen_evaluator as fe
    from threshold_curve import THRESHOLDS,FIELDS,curve,reference_check
    seed_all(args.seed) # Validation export also explicitly forbids TF32 and nondeterminism.
    cfg=identity(args);run=Path(args.output)/f'{args.method}_seed{args.seed}'
    with run_lock(run/'.run.lock'):
        ck=load_archive(run/'recovery.zip');assert ck['cfg']==cfg
        assert ck['epoch']>=cfg['max_epochs'] or ck['stale']>=20
        model=factory(args.method).cuda();model.load_state_dict(ck['best_model']);del ck;gc.collect();model.eval()
        ids=[x[0] for x in old.validation_groups(2 if args.smoke else 0)]
        coco=json.loads(Path(args.json_path).read_text());excluded=old.exclusions()
        # Filter before GT construction; no test image IO or test metrics.
        allowed={im['id'] for im in coco['images'] if fe.physical_id(im['file_name']) in set(ids)}
        coco['images']=[im for im in coco['images'] if im['id'] in allowed]
        coco['annotations']=[a for a in coco['annotations'] if a['image_id'] in allowed]
        for a in coco['annotations']:
            if str(a['id']) in excluded:a['spine']=[]
        gt=fe.build_gt(coco,set(ids));paths={fe.physical_id(im['file_name']):Path(args.image_dir)/im['file_name'] for im in coco['images']}
        ds=fe.Images(ids,paths);out=run/'validation';out.mkdir(exist_ok=True)
        sig={'cfg':cfg,'archive_sha256':sha(run/'recovery.zip'),'ids':ids,'thresholds':[float(t) for t in THRESHOLDS]}
        if (out/'signature.json').exists():assert json.loads((out/'signature.json').read_text())==sig
        else:dump(out/'signature.json',sig)
        for i in range(len(ds)):
            if args.deadline and time.time()+900>=args.deadline:return False
            x,oid,size=ds[i];dest=out/f'{oid}.json'
            if dest.exists():continue
            with torch.inference_mode():p=probability(model(x[None].cuda()),args.method).cpu().numpy()[0,0]
            assert np.isfinite(p).all() and p.min()>=0 and p.max()<=1
            native=np.asarray(Image.fromarray(p).resize(size,Image.Resampling.NEAREST));scores=curve(native,gt[oid])
            parity=reference_check(oid,native,gt[oid],scores) if i==0 else None
            dump(dest,{'observation_id':oid,'scores':scores.tolist(),'reference_check':parity})
        means=np.mean([json.loads((out/f'{oid}.json').read_text())['scores'] for oid in ids],axis=0)
        rows=[{'threshold':float(t),'n_observations':len(ids),**dict(zip(FIELDS,map(float,means[i])))} for i,t in enumerate(THRESHOLDS)]
        fe.write_csv(out/'curves.csv',rows,list(rows[0]));dump(run/'VALIDATION_COMPLETE.json',sig)
    return True

def parser():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['train','export']);p.add_argument('--method',choices=METHODS,required=True);p.add_argument('--seed',type=int,required=True)
    for name in ['json-path','image-dir','cache','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--stop-after-epoch',type=int,default=0);p.add_argument('--deadline',type=float,default=0);p.add_argument('--epoch-estimate',type=float,default=3600);p.add_argument('--gate',default='')
    return p
if __name__=='__main__':
    a=parser().parse_args()
    try:(train if a.action=='train' else export)(a)
    except BaseException as e:
        run=Path(a.output)/f'{a.method}_seed{a.seed}'
        dump(run/'FAILURE.json',{'error':repr(e),'type':type(e).__name__,'resource_status':'RESOURCE_GATE_FAILED' if isinstance(e,torch.cuda.OutOfMemoryError) or 'RESOURCE_GATE_FAILED' in str(e) else 'EXECUTION_FAILED','gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
        raise
