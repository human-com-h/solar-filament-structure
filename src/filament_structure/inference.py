"""Locked state-dict inference and independent validation-threshold reconstruction."""
from pathlib import Path
from collections import defaultdict
import hashlib
from .io import *
from .data import inputs
from .models import factory
from .frozen import evaluator,threshold_functions

def weight_path(root,m):
    candidates=[safe(root,m['weights_relative_path']),safe(root,'weights/'+m['run']+'.pt')]
    raw=m['archive_member'].split('!')[-1]
    candidates.append(safe(root,raw))
    for p in candidates:
        if p.is_file():return verified(p,m['export_sha256'],'inference state-dict')
    raise FileNotFoundError('Missing state-dict for '+m['run']+'. Put the verified export at '+str(candidates[0])+'. No public download address is recorded.')
def load(m,root,device):
    import torch
    p=weight_path(root,m);saved=torch.load(p,map_location='cpu',weights_only=True)
    state=saved.get('model',saved.get('model_state_dict',saved.get('state_dict',saved)))
    state={k.removeprefix('module.'):v for k,v in state.items()}
    net=factory(m['method']);net.load_state_dict(state,strict=True)
    h=hashlib.sha256()
    for k,v in sorted(net.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    if h.hexdigest()!=m['tensor_sha256']:raise ValueError('Model tensor identity differs')
    return net.to(device).eval(),p
def probability(net,path,device):
    import numpy as np,torch
    from PIL import Image
    with Image.open(path) as im:
        size=im.size;x=np.asarray(im.convert('L').resize((1024,1024),Image.Resampling.BILINEAR),dtype=np.float32).copy()/255.
    with torch.inference_mode():p=torch.sigmoid(net(torch.from_numpy(x[None,None]).to(device))).cpu().numpy()[0,0]
    if not np.isfinite(p).all():raise ValueError('Nonfinite probability')
    return p,size
def configure(device):
    import torch
    if str(device).startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; use --device cpu for CPU inference.')
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
def infer(args):
    import numpy as np,torch
    from PIL import Image
    configure(args.device);m=model(args.run);_,rr,_,_,_=inputs(args,True)
    out=output(args.output_root);ids=sorted({r['physical_observation_id'] for r in rr if r['split']==args.split})
    thresholds={'fixed_0.5':.5,'validation_selected':m['selected_threshold']}
    t=thresholds[args.mode]
    base=dict(run=m['run'],method=m['method'],seed=m['seed'],mode=args.mode,threshold=t,split=args.split,planned_observations=len(ids),export_sha256=m['export_sha256'],tensor_sha256=m['tensor_sha256'],observations=ids)
    if t is None:
        dump(out/'MASK_MANIFEST.json',{**base,'status':'INFEASIBLE_NA','na_reason':'No nonempty validation candidate satisfies the same-seed BCE–Dice precision floor','masks':[]})
        return dict(status='INFEASIBLE_NA',run=m['run'])
    net,wp=load(m,args.weights_root,args.device)
    paths={r['physical_observation_id']:safe(safe(args.data_root,args.images),r['file_name']) for r in rr if r['split']==args.split}
    masks=[]
    for oid in ids:
        prob,size=probability(net,paths[oid],args.device)
        # Threshold on the training grid, then nearest-neighbour binary remapping.
        arr=Image.fromarray((prob>=t).astype(np.uint8)*255).resize(size,Image.Resampling.NEAREST)
        dest=out/'masks'/(oid+'.png');dest.parent.mkdir(exist_ok=True)
        if dest.exists():raise FileExistsError('Use new inference output: '+str(dest))
        arr.save(dest)
        masks.append(dict(observation=oid,path=dest.relative_to(out).as_posix(),sha256=sha(dest),input_image_sha256=sha(paths[oid]),width=size[0],height=size[1],foreground_pixels=int(np.count_nonzero(np.asarray(arr)))))
    dump(out/'MASK_MANIFEST.json',{**base,'status':'OK','masks':masks,'torch':torch.__version__,'device':args.device,'threshold_selected_on_test':False})
    return dict(status='PASS',run=m['run'],masks=len(masks),manifest=str(out/'MASK_MANIFEST.json'))
def thresholds(args):
    import numpy as np
    from PIL import Image
    configure(args.device);m=model(args.run);_,rr,coco,_,_=inputs(args,True)
    ids=sorted({r['physical_observation_id'] for r in rr if r['split']=='validation'})
    excluded={r['annotation_id'] for r in qc(args.evidence_root)}
    for a in coco['annotations']:
        if str(a['id']) in excluded:a['spine']=[]
    fe=evaluator();gt=fe['build_gt'](coco,set(ids));tt,curve=threshold_functions()
    net,_=load(m,args.weights_root,args.device);records=[]
    paths={r['physical_observation_id']:safe(safe(args.data_root,args.images),r['file_name']) for r in rr if r['split']=='validation'}
    for oid in ids:
        p,size=probability(net,paths[oid],args.device)
        native=np.asarray(Image.fromarray(p).resize(size,Image.Resampling.NEAREST))
        records.append(curve(native,gt[oid]))
    means=np.mean(records,axis=0)
    floors=read(REPO/'frozen_sources/baselines/FROZEN_DESIGN.json')['operating_point']['precision_floors']
    floor=floors[str(m['seed'])]
    eligible=[i for i in range(len(tt)) if means[i,2]>=floor and means[i,6]>0]
    best=max(eligible,key=lambda i:(means[i,4],-means[i,5],means[i,0],float(tt[i]))) if eligible else None
    selected=None if best is None else float(tt[best]);out=output(args.output_root)
    fields=['dice','iou','precision','recall','msc','msgr','foreground_fraction']
    table(out/'validation_curves.csv',[dict(run=m['run'],threshold=float(t),n_observations=len(ids),**dict(zip(fields,map(float,means[i])))) for i,t in enumerate(tt)])
    result=dict(run=m['run'],selected_threshold=selected,locked_threshold=m['selected_threshold'],status='INFEASIBLE_NA' if best is None else 'OK',precision_floor=floor,eligible_candidates=len(eligible),matches_frozen_threshold=selected==m['selected_threshold'],test_used=False)
    dump(out/'THRESHOLD_CHECK.json',result)
    if not result['matches_frozen_threshold']:raise ValueError('Reconstructed validation selection differs; locked paper configuration has not been changed.')
    return result
