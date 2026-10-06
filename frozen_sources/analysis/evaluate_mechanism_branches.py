from pathlib import Path
from collections import defaultdict
import sys,json,time
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
ROOT=Path(__file__).resolve().parents[2];REV=ROOT/'project';PKG=REV/'mechanism_training';OUT=REV/'mechanism_branch_test'
sys.path.insert(0,str(PKG))
import frozen_evaluator as fe

def main():
    lockpath=REV/'mechanism_final_analysis/locked_test_protocol.json';lock=json.loads(lockpath.read_text());targetfile=OUT/'TARGETS.json';targetcfg=json.loads(targetfile.read_text());targets=defaultdict(list)
    for info in targetcfg['copies']:
        path=OUT/'labels'/f"{info['sample_id']}.npz";assert fe.sha256(path)==info['sha256']
        with np.load(path) as data:labels=data['labels']
        ys,xs=np.nonzero(labels);ls=labels[ys,xs].astype(np.int64)
        if not len(ls):continue
        counts=np.bincount(ls);valid=np.flatnonzero(counts);valid=valid[valid>0]
        targets[info['observation_id']].append((info['sample_id'],ys,xs,ls,counts,valid))
    ids=sorted(targets)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    signature={'protocol_sha256':fe.sha256(lockpath),'target_signature_sha256':fe.sha256(targetfile),'runner_sha256':fe.sha256(Path(__file__)),'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'ids':ids,'radii':[0,1,2],'tf32':False,'scope':'Training-derived residual component recall proxy, not independent physical branch annotation.'}
    sigfile=OUT/'signature.json'
    if sigfile.exists():assert json.loads(sigfile.read_text())==signature
    else:fe.dump_json(sigfile,signature)
    ann=ROOT/'data/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json';assert fe.sha256(ann)==lock['annotation_sha256'];coco=json.loads(ann.read_text())
    paths={fe.physical_id(im['file_name']):ROOT/'data/train/train_images'/im['file_name'] for im in coco['images'] if fe.physical_id(im['file_name']) in targets}
    ds=fe.Images(ids,paths);device=torch.device('cuda:0');start=time.perf_counter()
    for run in lock['runs']:
        path=ROOT/run['checkpoint'];assert fe.sha256(path)==run['sha256'];folder=OUT/run['name'];folder.mkdir(exist_ok=True);model=fe.load_model(path,device)
        for i in range(len(ds)):
            image,oid,_=ds[i];dest=folder/f'{oid}.json'
            if dest.exists():continue
            with torch.inference_mode():p=torch.sigmoid(model(image[None].to(device))).cpu().numpy()[0,0]
            sums=defaultdict(list)
            for sid,ys,xs,ls,counts,valid in targets[oid]:
                for radius in [0,1,2]:
                    score=np.full(len(xs),-np.inf,dtype=np.float32)
                    for dy in range(-radius,radius+1):
                        for dx in range(-radius,radius+1):
                            if dx*dx+dy*dy>radius*radius:continue
                            yy=ys+dy;xx=xs+dx;inside=(yy>=0)&(yy<1024)&(xx>=0)&(xx<1024)
                            score[inside]=np.maximum(score[inside],p[yy[inside],xx[inside]])
                    for mode,t in [('fixed_0.5',.5),('validation_selected',run['threshold'])]:
                        hits=np.bincount(ls[score>=t],minlength=len(counts));component=(hits[valid]/counts[valid]).mean();pixel=hits[valid].sum()/counts[valid].sum()
                        if i==0 and sid==targets[oid][0][0]:
                            binary=p>=t;direct=(distance_transform_edt(~binary)[ys,xs]<=radius) if binary.any() else np.zeros(len(xs),bool)
                            np.testing.assert_array_equal(score>=t,direct)
                        sums[mode,radius].append([float(component),float(pixel)])
            records=[{'method':run['method'],'seed':run['seed'],'mode':mode,'radius':radius,'physical_observation_id':oid,'n_copies':len(targets[oid]),'component_recall':float(np.mean(vals,axis=0)[0]),'pixel_recall':float(np.mean(vals,axis=0)[1])} for (mode,radius),vals in sums.items()]
            temp=dest.with_suffix('.tmp');fe.dump_json(temp,records);temp.replace(dest)
        print(run['name'],'branch',len(ds),'elapsed',round(time.perf_counter()-start,1),flush=True);del model;torch.cuda.empty_cache()
    fe.dump_json(OUT/'COMPLETE.json',{'runs':len(lock['runs']),'n_observations':len(ids),'protocol_sha256':fe.sha256(lockpath),'signature_sha256':fe.sha256(sigfile),'elapsed_seconds':time.perf_counter()-start})
if __name__=='__main__':main()
