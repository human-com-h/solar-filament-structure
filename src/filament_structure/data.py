"""Exact input validation and unchanged annotation-derived training targets."""
from collections import defaultdict,Counter
from pathlib import Path
import json
from .io import *
from .frozen import target_functions

def inputs(args,images=True):
    annotation=verified(safe(args.data_root,args.annotation),protocol()['annotation_sha256'],'MAGFiLO competition annotation JSON')
    rr=manifest(args.evidence_root);qc(args.evidence_root)
    coco=read(annotation);ims={str(x['id']):x for x in coco['images']};anns=defaultdict(list)
    for a in coco['annotations']:anns[str(a['image_id'])].append(a)
    if len(rr)!=1154 or len({r['physical_observation_id'] for r in rr})!=707:raise ValueError('Unexpected fixed cohort')
    counts={s:(len({r['physical_observation_id'] for r in rr if r['split']==s}),sum(r['split']==s for r in rr)) for s in ('train','validation','internal_test')}
    if counts!={'train':(492,812),'validation':(107,172),'internal_test':(108,170)}:raise ValueError('Fixed partition counts changed')
    for r in rr:
        im=ims[r['annotation_sample_id']]
        if im['file_name']!=r['file_name'] or Path(im['file_name']).stem!=r['physical_observation_id']:raise ValueError('Copy/observation/file mapping differs')
    if images:
        idx=evidence(args.evidence_root,'04_access/DATA_INPUT_INDEX.csv')
        image_rows=[r for r in rows(idx) if r.get('observation')]
        if len(image_rows)!=707:raise ValueError('Expected 707 image identities')
        for r in image_rows:
            p=verified(safe(safe(args.data_root,args.images),r['file_name']),r['sha256'],'observation image')
            if p.stat().st_size!=int(r['bytes']):raise ValueError('Image size differs')
    return annotation,rr,coco,ims,anns
def check(args):
    _,rr,_,_,_=inputs(args,not args.annotation_only)
    result=dict(status='PASS',copies=len(rr),observations=707,images_verified=not args.annotation_only,split_counts={s:sum(r['split']==s for r in rr) for s in ('train','validation','internal_test')},qc_exclusions=len(qc(args.evidence_root)))
    dump(output(args.output_root)/'DATA_CHECK.json',result);return result
def targets(args):
    import numpy as np
    from skimage.morphology import skeletonize,dilation,diamond
    _,rr,_,ims,anns=inputs(args,False);f=target_functions(args.evidence_root)
    selected_rows=[r for r in rr if r['split'] in args.splits]
    if args.sample_id:
        selected_rows=[r for r in selected_rows if r['annotation_sample_id']==args.sample_id]
        if not selected_rows:raise ValueError('Sample ID absent from selected frozen partition')
    out=output(args.output_root);summary=[]
    for r in selected_rows:
        sid=r['annotation_sample_id'];mask=f['render_union_mask'](sid,ims,anns,1024).astype(bool)
        srl=dilation(skeletonize(mask),diamond(2))&mask
        labels=f['label_copy'](ims[sid],anns[sid]);weights,offset=f['balanced_weights'](labels)
        dest=out/'targets'/(sid+'.npz');dest.parent.mkdir(exist_ok=True)
        if dest.exists():raise FileExistsError('Use new target output: '+str(dest))
        np.savez_compressed(dest,mask=mask,srl=srl,labels=labels,weights=weights,offset=offset)
        summary.append(dict(sample_id=sid,observation=r['physical_observation_id'],split=r['split'],components=int(labels.max()),residual_pixels=int((labels>0).sum()),srl_valid=bool(srl.any()),sha256=sha(dest)))
    table(out/'TARGET_MANIFEST.csv',summary);return dict(status='PASS',copies=len(summary),resolution=1024)
