"""Export validation curves only; fixed historical QC exclusions retain region masks."""
from pathlib import Path
import csv,json,time
import numpy as np
import torch
from PIL import Image
from data_pipeline import HERE,sha,dump,check_inputs,exclusions,validation_groups
import frozen_evaluator as fe
from threshold_curve import THRESHOLDS,FIELDS,curve,reference_check

def export(args):
    cfg=check_inputs(args.json_path);run=Path(args.output)/f'{args.method}_seed{args.seed}'
    complete=json.loads((run/'TRAINING_COMPLETE.json').read_text());assert complete['complete'] and complete['smoke']==args.smoke
    checkpoint=run/'best.pt';digest=sha(checkpoint);out=run/f'validation_{digest[:12]}';out.mkdir(exist_ok=True)
    coco=json.loads(Path(args.json_path).read_text());excluded=exclusions()
    for ann in coco['annotations']:
        if str(ann['id']) in excluded:ann['spine']=[]
    ids=[oid for oid,aids in validation_groups(limit=2 if args.smoke else 0)]
    signature={'checkpoint_sha256':digest,'protocol_sha256':sha(HERE/'protocol.json'),'exporter_sha256':sha(Path(__file__)),
               'evaluator_sha256':sha(HERE/'frozen_evaluator.py'),'curve_sha256':sha(HERE/'threshold_curve.py'),
               'spine_qc_excluded_annotation_ids':sorted(excluded),'ids':ids,'thresholds':[float(t) for t in THRESHOLDS],'torch':torch.__version__,'smoke':args.smoke}
    if (out/'signature.json').exists():assert json.loads((out/'signature.json').read_text())==signature
    else:dump(out/'signature.json',signature)
    gt=fe.build_gt(coco,set(ids));paths={fe.physical_id(im['file_name']):Path(args.image_dir)/im['file_name'] for im in coco['images'] if fe.physical_id(im['file_name']) in set(ids)}
    device=torch.device('cuda:0');model=fe.load_model(checkpoint,device);ds=fe.Images(ids,paths)
    for i in range(len(ds)):
        if args.deadline and time.time()+60>=args.deadline:return False
        image,oid,size=ds[i];dest=out/f'{oid}.json'
        if dest.exists():continue
        with torch.inference_mode():p=torch.sigmoid(model(image[None].to(device))).cpu().numpy()[0,0]
        native=np.asarray(Image.fromarray(p).resize(size,Image.Resampling.NEAREST));scores=curve(native,gt[oid])
        parity=reference_check(oid,native,gt[oid],scores) if i==0 else None
        dump(dest,{'observation_id':oid,'scores':scores.tolist(),'reference_check':parity})
        if (i+1)%25==0:print(args.method,args.seed,'validation curve',i+1,len(ds),flush=True)
    means=np.mean([json.loads((out/f'{oid}.json').read_text())['scores'] for oid in ids],axis=0)
    rows=[{'threshold':float(t),'n_observations':len(ids),**dict(zip(FIELDS,map(float,means[i])))} for i,t in enumerate(THRESHOLDS)]
    fe.write_csv(out/'curves.csv',rows,list(rows[0]));dump(run/'VALIDATION_COMPLETE.json',{'signature':signature,'directory':out.name})
    return True
