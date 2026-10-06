from pathlib import Path
import json
from collections import Counter,defaultdict
import numpy as np
import torch
from PIL import Image
from methods import HERE,DESIGN
import data_pipeline as old
from annotation_helpers import deterministic_choice
from runtime import sha,digest,dump,run_lock
def audit_cache(cache,smoke):
    cache=Path(cache);sig=json.loads((cache/'signature.json').read_text());done=json.loads((cache/'COMPLETE.json').read_text())
    assert sig==done,'Incomplete cache'
    assert sig['manifest_sha256']==DESIGN['manifest_sha256'] and sig['annotation_sha256']==DESIGN['annotation_sha256']
    assert sig['pipeline_sha256']==sha(HERE/'legacy/data_pipeline.py') and sig['smoke']==smoke
    assert sig['source_provenance']==old.protocol()['source_provenance']
    records=old.rows()
    counts={s:len({r['physical_observation_id'] for r in records if r['split']==s}) for s in ['train','validation','internal_test']}
    assert counts==DESIGN['split_counts']
    allowed=[r for r in records if r['split'] in ('train','validation')]
    if smoke:
        keep=set()
        for s in ('train','validation'):keep.update(sorted({r['physical_observation_id'] for r in allowed if r['split']==s})[:4])
        allowed=[r for r in allowed if r['physical_observation_id'] in keep]
    ids={r['physical_observation_id'] for r in allowed};aids=sorted(r['annotation_sample_id'] for r in allowed)
    assert set(p.stem for p in (cache/'images').iterdir())==ids,'Unexpected/missing cache image (test forbidden)'
    assert sorted(p.stem for p in (cache/'targets').iterdir())==aids,'Unexpected/missing target'
    assert sig['annotation_ids']==aids
    assert set(sig['image_sha256'])=={r['file_name'] for r in allowed}
    assert not ids & {r['physical_observation_id'] for r in records if r['split']=='internal_test'}
    hashes={str(p.relative_to(cache)).replace('\\','/'):sha(p) for folder in ('images','targets') for p in sorted((cache/folder).iterdir())}
    return {'signature_sha256':sha(cache/'signature.json'),'content_sha256':digest(hashes),'test_cache_count':0,'images':len(ids),'annotation_copies':len(aids)}
def prepare(annotation,images,cache,smoke):
    with run_lock(Path(cache).parent/(Path(cache).name+'.lock')):
        old.prepare(annotation,images,cache,limit=4 if smoke else 0)
        result=audit_cache(cache,smoke)
    return result
class Dataset(torch.utils.data.Dataset):
    def __init__(self,cache,seed,smoke):
        self.cache=Path(cache);self.seed=seed;self.epoch=0;g=defaultdict(list)
        for r in old.rows():
            if r['split']=='train':g[r['physical_observation_id']].append(r['annotation_sample_id'])
        self.groups=sorted(g.items())[:4] if smoke else sorted(g.items())
    def __len__(self):return len(self.groups)
    def __getitem__(self,i):
        oid,aids=self.groups[i];aid=deterministic_choice(aids,self.seed,self.epoch,oid)
        with Image.open(self.cache/'images'/f'{oid}.png') as im:x=np.asarray(im,dtype=np.float32)/255.
        with np.load(self.cache/'targets'/f'{aid}.npz') as d:y=d['mask'].astype(np.float32)
        return torch.from_numpy(x[None]),torch.from_numpy(y[None]),oid,aid
