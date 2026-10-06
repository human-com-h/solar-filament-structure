import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import argparse,csv,hashlib,json,random,time
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
from data_pipeline import HERE,protocol,sha,dump,TrainingData,validation_groups,prepare,check_inputs
from objectives import objective,METHODS
from unet import UNet

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True);torch.set_num_threads(2)
def save(obj,path):
    path=Path(path);tmp=path.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(path)
def state_rng():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}
def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch']);torch.cuda.set_rng_state_all(state['cuda'])
def model_digest(model):
    h=hashlib.sha256()
    for key,t in sorted(model.state_dict().items()):h.update(key.encode());h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()
@torch.no_grad()
def validate(model,cache,device,limit=0):
    model.eval();values=[]
    for oid,aids in validation_groups(limit):
        image=np.asarray(Image.open(Path(cache)/'images'/f'{oid}.png'),dtype=np.float32)/255.
        x=torch.from_numpy(image[None,None]).to(device);p=(torch.sigmoid(model(x))[0,0]>=.5).cpu().numpy();pc=p.sum();scores=[]
        for aid in aids:
            with np.load(Path(cache)/'targets'/f'{aid}.npz') as data:y=data['mask']
            scores.append(float((2*(p&y).sum()+1e-6)/(pc+y.sum()+1e-6)))
        values.append(float(np.mean(scores)))
    return float(np.mean(values))

def train(args):
    base=check_inputs(args.json_path);assert args.method in METHODS and args.seed in base['seeds']
    assert torch.cuda.is_available(),'GPU required, including the full-resolution smoke check.'
    cache=Path(args.cache);out=Path(args.output)/f'{args.method}_seed{args.seed}';out.mkdir(parents=True,exist_ok=True)
    cache_sig=json.loads((cache/'COMPLETE.json').read_text());assert cache_sig['manifest_sha256']==base['manifest_sha256'] and cache_sig['smoke']==args.smoke
    cfg={'method':args.method,'seed':args.seed,'protocol_sha256':sha(HERE/'protocol.json'),'code_sha256':{p.name:sha(p) for p in HERE.glob('*.py')},
         'cache_signature_sha256':sha(cache/'signature.json'),'smoke':args.smoke,'batch_size':2,'resolution':1024,'base_channels':32,
         'max_epochs':2 if args.smoke else base['max_epochs'],'patience':base['patience'],'workers':args.workers,
         'torch':torch.__version__,'numpy':np.__version__,'gpu':torch.cuda.get_device_name(0),'deterministic':True,'tf32':False,'amp':False}
    cfgfile=out/'config.json'
    if cfgfile.exists():assert json.loads(cfgfile.read_text())==cfg,'Resume configuration/environment differs.'
    else:dump(cfgfile,cfg)
    if (out/'TRAINING_COMPLETE.json').exists():return True
    seed_all(args.seed);device=torch.device('cuda:0');model=UNet(base=32).to(device)
    assert sum(p.numel() for p in model.parameters())==7762465
    opt=torch.optim.AdamW(model.parameters(),lr=base['lr'],weight_decay=base['weight_decay'])
    initial=model_digest(model);best=-1.;stale=0;history=[];start=1;best_state=None;best_epoch=0
    if (out/'last.pt').exists():
        c=torch.load(out/'last.pt',map_location='cpu',weights_only=False);assert c['cfg']==cfg
        model.load_state_dict(c['model']);opt.load_state_dict(c['optimizer'])
        for state in opt.state.values():
            for key,v in state.items():
                if torch.is_tensor(v) and key!='step':state[key]=v.to(device)
        best=c['best'];stale=c['stale'];history=c['history'];start=c['epoch']+1;best_state=c['best_model'];best_epoch=c['best_epoch']
        assert c['initial_model_digest']==initial
        restore_rng(c['rng']);save({'model':best_state,'epoch':best_epoch,'val_dice':best,'cfg':cfg},out/'best.pt')
    ds=TrainingData(cache,args.seed,limit=4 if args.smoke else 0)
    stopped=False
    if stale<base['patience']:
        for epoch in range(start,cfg['max_epochs']+1):
            last_seconds=history[-1]['seconds'] if history else 0
            if args.deadline and time.time()+max(last_seconds,120)+90>=args.deadline:stopped=True;break
            ds.epoch=epoch
            generator=torch.Generator().manual_seed(args.seed+epoch*100003)
            loader=DataLoader(ds,batch_size=2,shuffle=True,generator=generator,num_workers=args.workers,pin_memory=True,persistent_workers=False)
            model.train();totals=[];started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
            for batch in loader:
                batch={k:(v.to(device,non_blocking=True) if torch.is_tensor(v) else v) for k,v in batch.items()}
                opt.zero_grad(set_to_none=True);logits=model(batch['x']);loss,region,structure=objective(logits,batch,args.method)
                if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss.')
                loss.backward();opt.step();totals.append([float(loss.detach()),float(region.detach()),float(structure.detach())])
            score=validate(model,cache,device,limit=2 if args.smoke else 0)
            means=np.mean(totals,axis=0)
            row={'epoch':epoch,'train_loss':float(means[0]),'region_loss':float(means[1]),'structure_loss':float(means[2]),'val_physical_macro_dice':score,
                 'seconds':time.perf_counter()-started,'peak_cuda_allocated_MB':torch.cuda.max_memory_allocated()/1048576,'batches':len(totals)}
            history.append(row)
            if score>best+base['min_delta']:
                best=score;stale=0;best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:stale+=1
            # last.pt is the authoritative self-contained epoch snapshot. Reconstruct best.pt from it on resume.
            checkpoint={'model':model.state_dict(),'optimizer':opt.state_dict(),'best_model':best_state,'best':best,'best_epoch':best_epoch,'stale':stale,'history':history,
                        'epoch':epoch,'cfg':cfg,'rng':state_rng(),'initial_model_digest':initial}
            save(checkpoint,out/'last.pt');save({'model':best_state,'epoch':best_epoch,'val_dice':best,'cfg':cfg},out/'best.pt')
            with (out/'history.csv').open('w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerows(history)
            print(args.method,args.seed,json.dumps(row),flush=True)
            if stale>=base['patience']:break
            if args.stop_after_epoch and epoch>=args.stop_after_epoch:stopped=True;break
    complete=bool(history) and (stale>=base['patience'] or history[-1]['epoch']>=cfg['max_epochs'])
    status={'method':args.method,'seed':args.seed,'complete':complete,'smoke':args.smoke,'epochs':len(history),'best_epoch':best_epoch,'best_validation_dice':best,
            'initial_model_digest':initial,'stopped_for_budget_or_epoch_cap':stopped,'scope':'training and validation only'}
    dump(out/('TRAINING_COMPLETE.json' if complete else 'PAUSED.json'),status)
    if complete and (out/'PAUSED.json').exists():(out/'PAUSED.json').unlink()
    return complete

def parser():
    ap=argparse.ArgumentParser();ap.add_argument('--json-path',required=True);ap.add_argument('--cache',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--method',choices=METHODS,required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--smoke',action='store_true');ap.add_argument('--stop-after-epoch',type=int,default=0);ap.add_argument('--deadline',type=float,default=0)
    return ap
if __name__=='__main__':train(parser().parse_args())
