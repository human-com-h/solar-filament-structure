"""Validation-only, exact threshold sweep using archived U-Net checkpoints.

No model training, test-set threshold selection or physical-barb claims.
"""
import argparse
import csv
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import maximum_filter
import frozen_evaluator as fe

HERE = Path(__file__).resolve().parent
THRESHOLDS = np.unique(np.r_[0.01, np.arange(0.05,0.951,0.025),0.5,0.99].round(6)).astype(np.float32)
FIELDS = ['dice','iou','precision','recall','msc','msgr','foreground_fraction']

def counts(values):
    values = np.sort(values)
    return len(values)-np.searchsorted(values,THRESHOLDS,side='left')

def curve(prob, copies):
    """max over an r=3 disk commutes exactly with foreground thresholding."""
    assert prob.ndim == 2 and np.isfinite(prob).all()
    yy,xx = np.mgrid[-3:4,-3:4]
    nearby = maximum_filter(prob,footprint=xx*xx+yy*yy<=9,mode='constant',cval=-np.inf)
    flat = prob.ravel()
    pc = counts(flat)
    results=[]
    for gt in copies:
        if gt.shape != prob.shape or not gt.spines:
            raise ValueError('Unexpected shape or annotation copy without a valid spine; requires explicit policy.')
        tp=counts(flat[gt.mask_idx]); gc=len(gt.mask_idx)
        spine_scores=[]
        for _,xs,ys in gt.spines:
            c=nearby[ys,xs,None] >= THRESHOLDS[None,:]
            spine_scores.append(np.stack([c.mean(0),[fe.longest_gap(c[:,i])/len(c) for i in range(len(THRESHOLDS))]],axis=1))
        spine_mean=np.mean(spine_scores,axis=0)
        results.append(np.column_stack([(2*tp+fe.EPS)/(pc+gc+fe.EPS),
                       (tp+fe.EPS)/(pc+gc-tp+fe.EPS),(tp+fe.EPS)/(pc+fe.EPS),
                       (tp+fe.EPS)/(gc+fe.EPS),spine_mean,pc/len(flat)]))
    return np.mean(results,axis=0)

def reference_check(oid, prob, gt, scores):
    pred=prob>=0.5
    if not pred.any():
        # The frozen EDT path does not define the empty-foreground case reliably.
        # Our coverage is exactly zero and longest gap exactly one in that case.
        i=int(np.argmin(abs(THRESHOLDS-0.5)))
        assert scores[i,4] == 0 and scores[i,5] == 1
        return {'empty_prediction':True,'max_error':0.0}
    _,_,ref=fe.evaluate_one(oid,pred,gt)
    i=int(np.argmin(abs(THRESHOLDS-0.5)))
    error=max(abs(scores[i,j]-ref[3][m]) for j,m in enumerate(FIELDS[:6]))
    if error>1e-10: raise RuntimeError(f'Frozen evaluator parity failed: {error}')
    return {'empty_prediction':False,'max_error':float(error),'cldice_at_0_5':ref[3]['cldice']}

def run_worker(args):
    cfg=json.loads((HERE/'runs.json').read_text())
    coco=json.loads(Path(args.json_path).read_text(encoding='utf-8'))
    if fe.sha256(Path(args.json_path))!=cfg['annotation_sha256']:
        raise RuntimeError('Annotation file differs from frozen release.')
    if fe.sha256(HERE/'split_manifest.csv')!=cfg['manifest_sha256']:
        raise RuntimeError('Manifest hash mismatch.')
    with (HERE/'split_manifest.csv').open(encoding='utf-8-sig') as f:
        ids=sorted({r['physical_observation_id'] for r in csv.DictReader(f) if r['split']=='validation'})
    assert len(ids)==106
    if args.limit: ids=ids[:args.limit]
    gt=fe.build_gt(coco,set(ids))
    paths={fe.physical_id(im['file_name']):Path(args.image_dir)/im['file_name'] for im in coco['images'] if fe.physical_id(im['file_name']) in ids}
    if not all(p.is_file() for p in paths.values()): raise RuntimeError('Missing images.')
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable.')
    torch.set_num_threads(2)
    ds=fe.Images(ids,paths)
    for index,run in enumerate(cfg['runs']):
        if index%args.shards!=args.shard or (args.only and args.only!=run['name']): continue
        checkpoint=HERE/run['checkpoint']
        if fe.sha256(checkpoint)!=run['sha256']: raise RuntimeError('Checkpoint hash mismatch.')
        out=Path(args.output)/run['name']; out.mkdir(parents=True,exist_ok=True)
        signature={'run':run,'ids':ids,'thresholds':[float(t) for t in THRESHOLDS],
                   'annotation_sha256':cfg['annotation_sha256'],'manifest_sha256':cfg['manifest_sha256'],
                   'runner_sha256':fe.sha256(Path(__file__)),'evaluator_sha256':fe.sha256(HERE/'frozen_evaluator.py'),
                   'torch':torch.__version__,'numpy':np.__version__,'device':str(device)}
        meta=out/'signature.json'
        if meta.exists() and json.loads(meta.read_text())!=signature:
            raise RuntimeError('Resume signature differs; use a new output directory.')
        fe.dump_json(meta,signature)
        model=fe.load_model(checkpoint,device)
        for i in range(len(ds)):
            image,oid,size=ds[i]; result_file=out/f'{oid}.json'
            if result_file.exists(): continue
            with torch.inference_mode():
                prob=torch.sigmoid(model(image[None].to(device))).cpu().numpy()[0,0]
            native=np.asarray(Image.fromarray(prob).resize(size,Image.Resampling.NEAREST))
            scores=curve(native,gt[oid])
            check=reference_check(oid,native,gt[oid],scores) if i==0 else None
            result={'observation_id':oid,'scores':scores.tolist(),'reference_check':check}
            tmp=result_file.with_suffix('.tmp'); fe.dump_json(tmp,result); tmp.replace(result_file)
            if (i+1)%10==0 or i==0 or i+1==len(ds): print(run['name'],i+1,len(ds),flush=True)
        del model
        if device.type=='cuda': torch.cuda.empty_cache()

def summarize(args):
    cfg=json.loads((HERE/'runs.json').read_text()); summaries=[]; by_run={}
    for run in cfg['runs']:
        if args.only and args.only!=run['name']: continue
        folder=Path(args.output)/run['name']; meta=folder/'signature.json'
        if not meta.exists(): raise RuntimeError(f'Missing run: {run["name"]}')
        ids=json.loads(meta.read_text())['ids']
        records=[json.loads((folder/f'{oid}.json').read_text()) for oid in ids]
        means=np.mean([r['scores'] for r in records],axis=0); by_run[run['name']]=means
        for i,t in enumerate(THRESHOLDS):
            summaries.append({'method':run['method'],'seed':run['seed'],'n':len(ids),'threshold':float(t),**dict(zip(FIELDS,map(float,means[i])))})
    fe.write_csv(Path(args.output)/'validation_curves.csv',summaries,list(summaries[0]))
    selected=[]
    for run in cfg['runs']:
        if run['name'] not in by_run: continue
        baseline=by_run.get(f'BCE_Dice_seed{run["seed"]}')
        if baseline is None: continue
        target=float(baseline[np.argmin(abs(THRESHOLDS-0.5)),2]); means=by_run[run['name']]
        candidates=[i for i in range(len(THRESHOLDS)) if means[i,2]>=target and means[i,6]>0]
        best=max(candidates,key=lambda i:(means[i,4],-means[i,5],means[i,0],float(THRESHOLDS[i]))) if candidates else None
        selected.append({'name':run['name'],'validation_precision_floor':target,'selected_threshold':None if best is None else float(THRESHOLDS[best]),'status':'unattainable' if best is None else 'selected',
                         'validation_metrics':None if best is None else dict(zip(FIELDS,map(float,means[best])))})
    fe.dump_json(Path(args.output)/'selected_validation_thresholds.json',selected)
    fe.dump_json(Path(args.output)/'COMPLETE.json',{'stage':'validation-only','limited_pilot':bool(args.limit or args.only),'runs':len(by_run),'torch':torch.__version__,'python':platform.python_version(),'claim':'No independent test, no training, no superiority claim. Curves omit clDice except first-image parity diagnostics.'})

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--image-dir',required=True); ap.add_argument('--json-path',required=True)
    ap.add_argument('--output',required=True); ap.add_argument('--devices',default='cuda:0,cuda:1')
    ap.add_argument('--device',default='cuda:0'); ap.add_argument('--worker',action='store_true')
    ap.add_argument('--shards',type=int,default=1); ap.add_argument('--shard',type=int,default=0)
    ap.add_argument('--limit',type=int,default=0); ap.add_argument('--only')
    args=ap.parse_args()
    if args.worker: run_worker(args); return
    devices=args.devices.split(','); children=[]
    for i,dev in enumerate(devices):
        cmd=[sys.executable,str(Path(__file__).resolve()),'--worker','--image-dir',args.image_dir,'--json-path',args.json_path,'--output',args.output,'--device',dev,'--shards',str(len(devices)),'--shard',str(i),'--limit',str(args.limit)]
        if args.only: cmd+=['--only',args.only]
        children.append(subprocess.Popen(cmd))
    codes=[p.wait() for p in children]
    if any(codes): raise RuntimeError(f'Worker failure: {codes}; no completion archive produced.')
    summarize(args)
    import shutil
    archive=shutil.make_archive(str(Path(args.output).resolve().parent/'output'),'zip',root_dir=Path(args.output).resolve())
    print('Return this file:',archive,flush=True)

if __name__=='__main__': main()
