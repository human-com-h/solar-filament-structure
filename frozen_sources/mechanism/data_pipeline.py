from pathlib import Path
from collections import defaultdict
import csv,hashlib,json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from skimage.morphology import skeletonize,dilation,diamond
from annotation_helpers import render_union_mask,deterministic_choice
from branch_reference import branch_components_from_instance

HERE=Path(__file__).resolve().parent
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def dump(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,indent=2),encoding='utf-8');tmp.replace(path)
def rows():
    with (HERE/'temporal_manifest.csv').open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def exclusions():
    with (HERE/'manual_spine_qc_exclusions.csv').open(encoding='utf-8-sig') as f:return {r['annotation_id'] for r in csv.DictReader(f)}
def protocol():return json.loads((HERE/'protocol.json').read_text())
def check_inputs(annotation):
    cfg=protocol();assert sha(annotation)==cfg['annotation_sha256'];assert sha(HERE/'temporal_manifest.csv')==cfg['manifest_sha256']
    for name,info in cfg['source_provenance'].items():assert sha(HERE/name)==info['sha256'],name
    return cfg
def label_copy(im,anns):
    labels=np.zeros((1024,1024),np.uint16);lid=0;excluded=exclusions()
    for a in anns:
        if str(a['id']) in excluded:continue
        for comp in branch_components_from_instance(a,int(im['width']),int(im['height']),1024):
            if len(comp)<5:continue
            yy,xx=comp[:,0],comp[:,1];free=labels[yy,xx]==0;yy,xx=yy[free],xx[free]
            if not len(yy):continue
            lid+=1;labels[yy,xx]=lid
    return labels
def prepare(annotation,image_dir,cache,limit=0):
    cfg=check_inputs(annotation);cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    allowed=[r for r in rows() if r['split'] in ['train','validation']]
    if limit:
        keep=set()
        for split in ['train','validation']:
            keep.update(sorted({r['physical_observation_id'] for r in allowed if r['split']==split})[:limit])
        allowed=[r for r in allowed if r['physical_observation_id'] in keep]
    allowedids=sorted(r['annotation_sample_id'] for r in allowed)
    image_hashes={name:sha(Path(image_dir)/name) for name in sorted({r['file_name'] for r in allowed})}
    sig={'annotation_sha256':cfg['annotation_sha256'],'manifest_sha256':cfg['manifest_sha256'],'pipeline_sha256':sha(Path(__file__)),'source_provenance':cfg['source_provenance'],'annotation_ids':allowedids,'smoke':bool(limit),'image_sha256':image_hashes}
    sign=cache/'signature.json'
    if sign.exists():assert json.loads(sign.read_text())==sig,'Cache input signature differs.'
    else:dump(sign,sig)
    coco=json.loads(Path(annotation).read_text());images={str(im['id']):im for im in coco['images']};annotations=defaultdict(list)
    for ann in coco['annotations']:annotations[str(ann['image_id'])].append(ann)
    (cache/'images').mkdir(exist_ok=True);(cache/'targets').mkdir(exist_ok=True)
    for i,r in enumerate(allowed):
        sid=r['annotation_sample_id'];oid=r['physical_observation_id'];dest=cache/'targets'/f'{sid}.npz'
        ipath=cache/'images'/f'{oid}.png'
        if not ipath.exists():
            with Image.open(Path(image_dir)/r['file_name']) as im:x=im.convert('L').resize((1024,1024),Image.Resampling.BILINEAR)
            tmp=ipath.with_suffix('.tmp');x.save(tmp,format='PNG');tmp.replace(ipath)
        if not dest.exists():
            mask=render_union_mask(sid,images,annotations,1024).astype(bool)
            srl=dilation(skeletonize(mask),diamond(2))&mask
            labels=label_copy(images[sid],annotations[sid]);tmp=dest.with_suffix('.tmp')
            with tmp.open('wb') as f:np.savez_compressed(f,mask=mask,srl=srl,labels=labels)
            tmp.replace(dest)
        if (i+1)%50==0 or i+1==len(allowed):print(f'cache {i+1}/{len(allowed)}',flush=True)
    dump(cache/'COMPLETE.json',sig)
    return sig

def balanced_weights(labels,eps=1e-6):
    counts=np.bincount(labels.ravel());valid=np.flatnonzero(counts);valid=valid[valid>0]
    lut=np.zeros(len(counts),dtype=np.float32)
    if not len(valid):return lut[labels],np.float32(0)
    lut[valid]=1.0/(len(valid)*(counts[valid]+eps))
    offset=np.float32(np.mean(eps/(counts[valid]+eps)))
    return lut[labels],offset

class TrainingData(Dataset):
    def __init__(self,cache,seed,limit=0):
        self.cache=Path(cache);self.seed=seed;self.epoch=0;groups=defaultdict(list)
        for r in rows():
            if r['split']=='train':groups[r['physical_observation_id']].append(r['annotation_sample_id'])
        self.groups=sorted(groups.items())
        if limit:self.groups=self.groups[:limit]
    def __len__(self):return len(self.groups)
    def __getitem__(self,i):
        oid,aids=self.groups[i];aid=deterministic_choice(aids,self.seed,self.epoch,oid)
        x=np.asarray(Image.open(self.cache/'images'/f'{oid}.png'),dtype=np.float32)/255.
        with np.load(self.cache/'targets'/f'{aid}.npz') as data:
            y=data['mask'].astype(np.float32);srl=data['srl'].astype(np.float32);labels=data['labels']
        weights,offset=balanced_weights(labels)
        return {'x':torch.from_numpy(x[None]),'y':torch.from_numpy(y[None]),'srl':torch.from_numpy(srl[None]),
                'residual':torch.from_numpy((labels>0).astype(np.float32)[None]),'weights':torch.from_numpy(weights[None]),'offset':torch.tensor(offset),'oid':oid,'aid':aid}

def validation_groups(limit=0):
    groups=defaultdict(list)
    for r in rows():
        if r['split']=='validation':groups[r['physical_observation_id']].append(r['annotation_sample_id'])
    items=sorted(groups.items())
    return items[:limit] if limit else items
