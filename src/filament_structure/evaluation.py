"""Frozen segmentation, training-grid residual recall and mask-only axis scoring."""
from collections import defaultdict
from pathlib import Path
from .io import *
from .data import inputs
from .frozen import evaluator,score_one

def masks(args):
    import numpy as np
    from PIL import Image
    source=required(args.mask_manifest,'mask manifest');mm=read(source);m=model(mm['run'])
    mode=mm['mode'];t=.5 if mode=='fixed_0.5' else m['selected_threshold']
    if mode not in ('fixed_0.5','validation_selected') or mm['threshold']!=t:raise ValueError('Mask operating point differs from locked model index')
    if mm['export_sha256']!=m['export_sha256'] or mm['tensor_sha256']!=m['tensor_sha256']:raise ValueError('Mask model identity differs')
    ids=sorted({r['physical_observation_id'] for r in manifest(args.evidence_root) if r['split']==mm['split']})
    if mm['observations']!=ids or mm['planned_observations']!=len(ids):raise ValueError('Mask cohort differs')
    if mm['status']=='INFEASIBLE_NA':
        if t is not None or mm['masks']:raise ValueError('Invalid selected infeasibility state')
        return mm,{}
    if mm['status']!='OK' or sorted(r['observation'] for r in mm['masks'])!=ids:raise ValueError('Missing or duplicate masks')
    index={}
    for r in mm['masks']:
        p=verified(safe(source.parent,r['path']),r['sha256'],'native mask')
        with Image.open(p) as im:
            a=np.asarray(im)
            if im.mode!='L' or a.shape!=(r['height'],r['width']) or not set(np.unique(a))<={0,255}:raise ValueError('Expected native binary L mask')
        index[r['observation']]=p
    return mm,index
def segmentation(args):
    import numpy as np
    from PIL import Image
    mm,paths=masks(args);out=output(args.output_root)
    fields=['dice','iou','precision','recall','cldice','msc','msgr'];base={k:mm[k] for k in ('run','method','seed','mode','threshold')}
    if not paths:
        rr=[dict(**base,status='INFEASIBLE_NA',physical_observation_id=o,radius=r,**{k:None for k in fields}) for o in mm['observations'] for r in (1,3,5)]
        table(out/'segmentation_per_observation.csv',rr);return dict(status='INFEASIBLE_NA',rows=len(rr))
    _,_,coco,_,_=inputs(args,False);fe=evaluator();gt=fe['build_gt'](coco,set(paths))
    rr=[];copies=[];spines=[]
    for oid,p in sorted(paths.items()):
        with Image.open(p) as im:pred=np.asarray(im)>0
        if any(c.shape!=pred.shape for c in gt[oid]):raise ValueError('Native mask shape differs from annotation')
        aa,ff,oo=fe['evaluate_one'](oid,pred,gt[oid])
        copies.extend(dict(**base,**r) for r in aa)
        for radius in (1,3,5):
            rr.append(dict(**base,status='OK',radius=radius,**oo[radius]))
            spines.extend(dict(**base,radius=radius,**r) for r in ff[radius])
    table(out/'segmentation_per_observation.csv',rr);table(out/'segmentation_per_copy.csv',copies);table(out/'spine_scores.csv',spines)
    table(out/'segmentation_summary.csv',[dict(**base,radius=r,observations=len(paths),**{k:float(np.mean([x[k] for x in rr if x['radius']==r])) for k in fields}) for r in (1,3,5)])
    dump(out/'SCORING_SCOPE.json',dict(kind='native mask rescoring',copies=sum(map(len,gt.values())),observations=len(paths),frozen_function='evaluate_one',empty_foreground_convention='unchanged frozen EDT convention',new_training=False))
    return dict(status='PASS',observation_rows=len(rr))
def axes(args):
    import numpy as np
    from PIL import Image
    mm,paths=masks(args)
    if mm['split']!='internal_test':raise ValueError('Locked reference axes cover internal_test only')
    refpath=original(args.evidence_root,'project/manuscript_revision_20261001/morphology_application/REFERENCE_AXES.json')
    lock=read(original(args.evidence_root,'project/manuscript_revision_20261001/morphology_application/LOCKED_APPLICATION.json'))
    verified(refpath,lock['reference_axes_sha256'],'reference axes');refs=read(refpath)
    if sorted(refs)!=mm['observations'] or sum(len(c['references']) for v in refs.values() for c in v)!=1368:raise ValueError('Full reference denominator changed')
    base={k:mm[k] for k in ('run','method','seed','mode','threshold')};out=output(args.output_root);rr=[];cc=[];oo=[]
    lengths=['signed_relative_error','absolute_relative_error','absolute_error_px']
    for oid in mm['observations']:
        if paths:
            with Image.open(paths[oid]) as im:result=score_one(np.asarray(im)>0,refs[oid])
            rr.extend(dict(**base,observation=oid,mode_status='OK',**r) for r in result['per_reference'])
            cc.extend(dict(**base,observation=oid,mode_status='OK',**r) for r in result['per_copy'])
            oo.append(dict(**base,observation=oid,status='OK',**result['per_observation']))
        else:
            for c in refs[oid]:
                for ref in c['references']:rr.append(dict(**base,observation=oid,sample=c['sample'],annotation_id=ref['annotation_id'],mode_status='INFEASIBLE_NA',status='INFEASIBLE_NA',length_status='INFEASIBLE_NA',**{k:None for k in lengths}))
                cc.append(dict(**base,observation=oid,sample=c['sample'],mode_status='INFEASIBLE_NA',success_rate=None,length_status='INFEASIBLE_NA',**{k:None for k in lengths}))
            oo.append(dict(**base,observation=oid,status='INFEASIBLE_NA',success_rate=None,length_status='INFEASIBLE_NA',**{k:None for k in lengths}))
    table(out/'per_reference.csv',rr);table(out/'per_copy.csv',cc);table(out/'per_observation.csv',oo)
    successful=[r for r in oo if r['length_status']=='OK']
    summary=dict(**base,status=mm['status'],planned_observations=108,planned_copies=170,planned_references=1368,success_rate=float(np.mean([r['success_rate'] for r in oo])) if paths else None,successful_observations=len(successful) if paths else None,successful_copies=sum(r['successful_copies'] for r in oo) if paths else None,successful_references=sum(r['successful_references'] for r in oo) if paths else None,length_status='INFEASIBLE_NA' if not paths else 'OK' if successful else 'FAIL_NA',**{k:float(np.mean([r[k] for r in successful])) if successful else None for k in lengths})
    table(out/'per_seed.csv',[summary]);dump(out/'SCORING_SCOPE.json',dict(kind='mask-only exact extraction then bidirectional one-to-one matching',mask_manifest_sha256=sha(args.mask_manifest),reference_axes_sha256=sha(refpath),bootstrap_repeated=False))
    return dict(status=mm['status'],references=len(rr),copies=len(cc),observations=len(oo))
def residual_recall(probability,labels,threshold,radius):
    """Identical disk-neighbour maximum and label counts to the original diagnostic."""
    import numpy as np
    ys,xs=np.nonzero(labels);ls=labels[ys,xs].astype(np.int64)
    if not len(ls):return None
    counts=np.bincount(ls);valid=np.flatnonzero(counts);valid=valid[valid>0]
    score=np.full(len(xs),-np.inf,dtype=np.float32)
    for dy in range(-radius,radius+1):
        for dx in range(-radius,radius+1):
            if dx*dx+dy*dy>radius*radius:continue
            yy=ys+dy;xx=xs+dx;inside=(yy>=0)&(yy<1024)&(xx>=0)&(xx<1024)
            score[inside]=np.maximum(score[inside],probability[yy[inside],xx[inside]])
    hits=np.bincount(ls[score>=threshold],minlength=len(counts))
    return dict(component_recall=float((hits[valid]/counts[valid]).mean()),pixel_recall=float(hits[valid].sum()/counts[valid].sum()))
def components(args):
    import numpy as np
    from .inference import configure,load,probability
    configure(args.device);m=model(args.run)
    if m['method'] not in ('BCE_Dice','SRL','SABR','ResidualPixel','SRL_FG'):raise ValueError('Component endpoint was not assessed for this method; no score is substituted.')
    _,rr,_,_,_=inputs(args,True);rr=[r for r in rr if r['split']=='internal_test'];groups=defaultdict(list)
    for r in rr:
        dest=required(safe(args.targets_root,'targets/'+r['annotation_sample_id']+'.npz'),'training-grid labels')
        with np.load(dest) as f:labels=f['labels']
        if labels.shape!=(1024,1024):raise ValueError('Component target resolution differs')
        if labels.any():groups[r['physical_observation_id']].append((r,labels))
    if len(groups)!=101 or sum(map(len,groups.values()))!=157 or len({r['temporal_group'] for v in groups.values() for r,_ in v})!=11:raise ValueError('Component cohort differs from 101/157/11')
    net,_=load(m,args.weights_root,args.device);records=[]
    for oid,cc in sorted(groups.items()):
        p,_=probability(net,safe(safe(args.data_root,args.images),cc[0][0]['file_name']),args.device)
        for radius in (0,1,2):
            for mode,t in [('fixed_0.5',.5),('validation_selected',m['selected_threshold'])]:
                v=[residual_recall(p,l,t,radius) for _,l in cc]
                records.append(dict(run=m['run'],method=m['method'],seed=m['seed'],mode=mode,radius=radius,physical_observation_id=oid,n_copies=len(cc),**{k:float(np.mean([x[k] for x in v])) for k in ('component_recall','pixel_recall')}))
    table(output(args.output_root)/'branch_per_observation.csv',records);return dict(status='PASS',observations=101,valid_copies=157,groups=11)
