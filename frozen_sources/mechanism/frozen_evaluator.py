#!/usr/bin/env python3
"""MAGFiLO Phase 9: frozen SRL vs SABR evaluation on the held-out internal test split."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy import stats
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from torch import nn
from torch.utils.data import DataLoader, Dataset

RESOLUTION, THRESHOLD = 1024, 0.5
RADII, PRIMARY_RADIUS = (1, 3, 5), 3
METRICS = ("dice", "iou", "precision", "recall", "cldice", "msc", "msgr")
JSON_SHA256 = "5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1"
MANIFEST_SHA256 = "8adbc51bad637a36daba477697fa15266456d64a070e76c4e8ea6e248bc608e6"
RUNS = [
    (20260831, "SRL",  "checkpoints/unet_srl_seed20260831.pt",        "73dd7b43d9d42f73a967fd8ea4bed682f880b077e47f54b1e4e53fdff439d2bc"),
    (20260831, "SABR", "checkpoints/unet_sabr_a025_seed20260831.pt", "ca5c2199623552db214dea15b04ccff1071fc75dff1faa717e91c2fd0c22c802"),
    (20260901, "SRL",  "checkpoints/unet_srl_seed20260901.pt",        "d8e013198f839aacd5077fa5227066596ea6408f1ecf69f4451ed8582025d818"),
    (20260901, "SABR", "checkpoints/unet_sabr_a025_seed20260901.pt", "8eca83f6b0da52f5a368a75f047608a340da75308a741db52548c87e0c36232a"),
    (20260902, "SRL",  "checkpoints/unet_srl_seed20260902.pt",        "347fc5d6ce70091e63a3409b10162db9a267bf8f19c4595ad1e933fa6bbd5fc9"),
    (20260902, "SABR", "checkpoints/unet_sabr_a025_seed20260902.pt", "16e9871201cf95c4526fe1265edcc404657d805d89b611d01f3b14b06787b866"),
]
EPS = 1e-6


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader(); w.writerows(rows)


class DoubleConv(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


class UNet(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.e1, self.e2 = DoubleConv(1, base), DoubleConv(base, base * 2)
        self.e3, self.e4 = DoubleConv(base * 2, base * 4), DoubleConv(base * 4, base * 8)
        self.b, self.pool = DoubleConv(base * 8, base * 16), nn.MaxPool2d(2)
        self.up4, self.d4 = nn.ConvTranspose2d(base * 16, base * 8, 2, 2), DoubleConv(base * 16, base * 8)
        self.up3, self.d3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2), DoubleConv(base * 8, base * 4)
        self.up2, self.d2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2), DoubleConv(base * 4, base * 2)
        self.up1, self.d1 = nn.ConvTranspose2d(base * 2, base, 2, 2), DoubleConv(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)
    def forward(self, x):
        x1=self.e1(x); x2=self.e2(self.pool(x1)); x3=self.e3(self.pool(x2)); x4=self.e4(self.pool(x3)); x=self.b(self.pool(x4))
        x=self.d4(torch.cat((self.up4(x),x4),1)); x=self.d3(torch.cat((self.up3(x),x3),1))
        x=self.d2(torch.cat((self.up2(x),x2),1)); return self.out(self.d1(torch.cat((self.up1(x),x1),1)))


def load_model(path: Path, device: torch.device) -> nn.Module:
    model = UNet()
    if sum(p.numel() for p in model.parameters()) != 7_762_465:
        raise RuntimeError("Unexpected U-Net architecture")
    try: ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError: ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt.get("state_dict")))
    if state is None: state = ckpt
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


class Images(Dataset):
    def __init__(self, ids, paths): self.ids, self.paths = ids, paths
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        oid = self.ids[i]
        with Image.open(self.paths[oid]) as im:
            im = im.convert("L"); size = im.size
            a = np.asarray(im.resize((RESOLUTION, RESOLUTION), Image.Resampling.BILINEAR), dtype=np.float32).copy()/255.0
        return torch.from_numpy(a)[None], oid, size


def collate(batch):
    return torch.stack([x[0] for x in batch]), [x[1] for x in batch], [x[2] for x in batch]


@dataclass
class CopyGT:
    sample_id: str
    shape: tuple[int, int]
    mask_idx: np.ndarray
    skel_idx: np.ndarray
    spines: list[tuple[str, np.ndarray, np.ndarray]]


def physical_id(name: str) -> str: return Path(name).stem


def rasterize(image, anns) -> np.ndarray:
    mask = np.zeros((int(image["height"]), int(image["width"])), np.uint8)
    for ann in anns:
        for polygon in ann.get("segmentation", []):
            if len(polygon) >= 6:
                pts = np.rint(np.asarray(polygon).reshape(-1,2)).astype(np.int32).reshape(-1,1,2)
                cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def sample_spine(values, width, height):
    p = np.asarray(values, dtype=np.float64).reshape(-1,2)
    if len(p) > 1:
        p = p[np.r_[True, np.linalg.norm(np.diff(p,axis=0),axis=1)>0]]
    if len(p) > 1:
        cumulative=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
        d=np.r_[np.arange(0.,cumulative[-1],.5),cumulative[-1]]
        p=np.column_stack((np.interp(d,cumulative,p[:,0]),np.interp(d,cumulative,p[:,1])))
    q=np.rint(p).astype(np.int32)
    return np.clip(q[:,0],0,width-1),np.clip(q[:,1],0,height-1)


def build_gt(coco, test_ids):
    anns=defaultdict(list)
    for a in coco["annotations"]: anns[str(a["image_id"])].append(a)
    result=defaultdict(list)
    for im in coco["images"]:
        oid=physical_id(im["file_name"])
        if oid not in test_ids: continue
        aa=anns[str(im["id"])]; mask=rasterize(im,aa); spines=[]
        for a in aa:
            if a.get("spine"):
                xs,ys=sample_spine(a["spine"],int(im["width"]),int(im["height"])); spines.append((str(a["id"]),xs,ys))
        result[oid].append(CopyGT(str(im["id"]),mask.shape,np.flatnonzero(mask.ravel()),np.flatnonzero(skeletonize(mask).ravel()),spines))
    for oid in result: result[oid].sort(key=lambda x:x.sample_id)
    if set(result)!=set(test_ids): raise RuntimeError("Frozen test IDs do not match annotation JSON")
    return result


def longest_gap(covered):
    best=cur=0
    for x in covered:
        cur=0 if x else cur+1; best=max(best,cur)
    return best


def evaluate_one(oid, pred, copies):
    flat=pred.ravel(); pc=int(flat.sum()); psk=np.flatnonzero(skeletonize(pred).ravel()); dist=distance_transform_edt(~pred)
    ann_rows=[]; filament={r:[] for r in RADII}; per_copy={r:defaultdict(list) for r in RADII}
    for gt in copies:
        tp=int(flat[gt.mask_idx].sum()); gc=len(gt.mask_idx)
        tprec=np.intersect1d(psk,gt.mask_idx,assume_unique=True).size/(len(psk)+EPS) if len(psk) else 0.
        tsens=int(flat[gt.skel_idx].sum())/(len(gt.skel_idx)+EPS) if len(gt.skel_idx) else (1. if not len(psk) else 0.)
        cl=2*tprec*tsens/(tprec+tsens) if tprec+tsens else 0.
        row={"physical_observation_id":oid,"annotation_sample_id":gt.sample_id,
             "dice":(2*tp+EPS)/(pc+gc+EPS),"iou":(tp+EPS)/(pc+gc-tp+EPS),
             "precision":(tp+EPS)/(pc+EPS),"recall":(tp+EPS)/(gc+EPS),"cldice":cl}
        ann_rows.append(row)
        for aid,xs,ys in gt.spines:
            for r in RADII:
                covered=dist[ys,xs] <= r; msc=float(covered.mean()); msgr=longest_gap(covered)/len(covered)
                filament[r].append({"physical_observation_id":oid,"annotation_sample_id":gt.sample_id,"annotation_id":aid,"msc":msc,"msgr":msgr})
                per_copy[r][gt.sample_id].append((msc,msgr))
    base={"physical_observation_id":oid}
    for m in METRICS[:5]: base[m]=float(np.mean([x[m] for x in ann_rows]))
    physical={}
    for r in RADII:
        copy_means=[np.mean(per_copy[r][g.sample_id],axis=0) for g in copies]
        physical[r]={**base,"msc":float(np.mean([x[0] for x in copy_means])),"msgr":float(np.mean([x[1] for x in copy_means]))}
    return ann_rows,filament,physical


def summary(rows, radius):
    out={"radius_native_px":radius,"threshold":THRESHOLD,"n_physical_observations":len(rows)}
    for m in METRICS:
        a=np.asarray([x[m] for x in rows]); out[m]={"mean":float(a.mean()),"std":float(a.std(ddof=1)),"median":float(np.median(a))}
    return out


def infer(model, ids, paths, pred_dir, device, batch_size, workers):
    pred_dir.mkdir(parents=True,exist_ok=True); missing=[x for x in ids if not (pred_dir/f"{x}.png").is_file()]
    loader=DataLoader(Images(missing,paths),batch_size=batch_size,shuffle=False,num_workers=workers,pin_memory=device.type=="cuda",collate_fn=collate)
    done=0
    with torch.inference_mode():
        for images,oids,sizes in loader:
            masks=(torch.sigmoid(model(images.to(device,non_blocking=True)))>=THRESHOLD).byte().cpu().numpy()[:,0]
            for mask,oid,size in zip(masks,oids,sizes):
                Image.fromarray(mask*255).resize(size,Image.Resampling.NEAREST).save(pred_dir/f"{oid}.png")
                done+=1
            print(f"  inference {done}/{len(missing)}",flush=True)


def evaluate_run(run_dir, ids, gt):
    ann=[]; fils={r:[] for r in RADII}; phys={r:[] for r in RADII}
    for i,oid in enumerate(ids,1):
        pred=np.asarray(Image.open(run_dir/"predictions_native"/f"{oid}.png").convert("L"))>0
        a,f,p=evaluate_one(oid,pred,gt[oid]); ann+=a
        for r in RADII: fils[r]+=f[r]; phys[r].append(p[r])
        if i%10==0 or i==len(ids): print(f"  metrics {i}/{len(ids)}",flush=True)
    for r in RADII:
        d=run_dir/f"internal_test_r{r}"
        write_csv(d/"annotation_copy_metrics.csv",ann,["physical_observation_id","annotation_sample_id",*METRICS[:5]])
        write_csv(d/"filament_spine_metrics.csv",fils[r],["physical_observation_id","annotation_sample_id","annotation_id","msc","msgr"])
        write_csv(d/"physical_observation_metrics.csv",phys[r],["physical_observation_id",*METRICS])
        dump_json(d/"summary.json",summary(phys[r],r))


def bootstrap(diff, seed, n=100_000, chunk=2000):
    rng=np.random.default_rng(seed); means=[]
    for start in range(0,n,chunk): means.append(diff[rng.integers(0,len(diff),size=(min(chunk,n-start),len(diff)))].mean(1))
    return np.quantile(np.concatenate(means),[.025,.975])


def final_tables(out: Path):
    seeds=(20260831,20260901,20260902); all_rows=[]; physical={}
    for seed,model,_,_ in RUNS:
        name=f"{model}_seed{seed}"; d=out/name/f"internal_test_r{PRIMARY_RADIUS}"
        s=json.loads((d/"summary.json").read_text()); row={"seed":seed,"model":model,**{m:s[m]["mean"] for m in METRICS}}; all_rows.append(row)
        physical[(seed,model)]={x["physical_observation_id"]:x for x in csv.DictReader((d/"physical_observation_metrics.csv").open())}
    write_csv(out/"phase9_test_run_summary_r3.csv",all_rows,["seed","model",*METRICS])
    lookup={(x["seed"],x["model"]):x for x in all_rows}; comp=[]; boots=[]
    for si,seed in enumerate(seeds):
        row={"seed":seed}; ids=sorted(physical[(seed,"SRL")])
        for mi,m in enumerate(METRICS):
            sv=lookup[(seed,"SRL")][m]; av=lookup[(seed,"SABR")][m]; row.update({f"SRL_{m}":sv,f"SABR_{m}":av,f"delta_{m}":av-sv})
            diff=np.asarray([float(physical[(seed,"SABR")][x][m])-float(physical[(seed,"SRL")][x][m]) for x in ids]); lo,hi=bootstrap(diff,20260904+si*10+mi)
            boots.append({"seed":seed,"metric":m,"mean_delta_SABR_minus_SRL":float(diff.mean()),"ci95_low":float(lo),"ci95_high":float(hi),"bootstrap_replicates":100000})
        comp.append(row)
    fields=["seed"]+[x for m in METRICS for x in (f"SRL_{m}",f"SABR_{m}",f"delta_{m}")]
    write_csv(out/"phase9_per_seed_comparison_r3.csv",comp,fields)
    write_csv(out/"phase9_paired_bootstrap_r3.csv",boots,["seed","metric","mean_delta_SABR_minus_SRL","ci95_low","ci95_high","bootstrap_replicates"])
    multi=[]
    for m in METRICS:
        s=np.asarray([lookup[(x,"SRL")][m] for x in seeds]); a=np.asarray([lookup[(x,"SABR")][m] for x in seeds]); d=a-s
        multi.append({"metric":m,"SRL_mean":s.mean(),"SRL_std":s.std(ddof=1),"SABR_mean":a.mean(),"SABR_std":a.std(ddof=1),"mean_delta":d.mean(),"delta_std":d.std(ddof=1)})
    write_csv(out/"phase9_multiseed_summary_r3.csv",multi,["metric","SRL_mean","SRL_std","SABR_mean","SABR_std","mean_delta","delta_std"])
    lines=["# Phase 9 held-out internal-test results","","|Metric|SRL mean ± SD|SABR mean ± SD|SABR − SRL|","|---|---:|---:|---:|"]
    for x in multi: lines.append(f"|{x['metric']}|{x['SRL_mean']:.4f} ± {x['SRL_std']:.4f}|{x['SABR_mean']:.4f} ± {x['SABR_std']:.4f}|{x['mean_delta']:+.5f}|")
    lines += ["","Do not tune λ, α, threshold, or checkpoints using these results."]
    (out/"PHASE9_RESULTS.md").write_text("\n".join(lines),encoding="utf-8")


def discover():
    root=Path("/kaggle/input")
    js=list(root.rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    if len(js)!=1: raise RuntimeError(f"Expected one MAGFiLO annotation JSON, found {len(js)}")
    candidates=[js[0].parent/"train_images",js[0].parent/"images"]
    images=next((x for x in candidates if x.is_dir()),None)
    if images is None: raise RuntimeError("train_images directory not found")
    return images,js[0]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--image-dir"); ap.add_argument("--json-path"); ap.add_argument("--output-dir",default="/kaggle/working/magfilo_phase9_final_test"); ap.add_argument("--batch-size",type=int,default=2); ap.add_argument("--num-workers",type=int,default=2); ap.add_argument("--allow-cpu",action="store_true"); args=ap.parse_args()
    package=Path(__file__).resolve().parent; out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    image_dir,json_path=(Path(args.image_dir),Path(args.json_path)) if args.image_dir and args.json_path else discover()
    manifest_path=package/"frozen_split_manifest.csv"
    if sha256(manifest_path)!=MANIFEST_SHA256 or sha256(json_path)!=JSON_SHA256: raise RuntimeError("Frozen manifest or annotation JSON SHA-256 mismatch")
    rows=list(csv.DictReader(manifest_path.open())); counts=Counter(x["split"] for x in rows)
    if counts!={"train":495,"validation":106,"test":106}: raise RuntimeError(f"Split changed: {counts}")
    test_ids=sorted(x["physical_observation_id"] for x in rows if x["split"]=="test")
    coco=json.loads(json_path.read_text()); paths={physical_id(x["file_name"]):image_dir/x["file_name"] for x in coco["images"] if physical_id(x["file_name"]) in test_ids}
    if len(paths)!=106 or not all(x.is_file() for x in paths.values()): raise RuntimeError("106 frozen test images were not resolved")
    gt=build_gt(coco,set(test_ids)); copies=sum(map(len,gt.values())); spines=sum(len(c.spines) for cs in gt.values() for c in cs)
    if (copies,spines)!=(172,1230): raise RuntimeError(f"GT counts changed: copies={copies}, spines={spines}")
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type=="cpu" and not args.allow_cpu: raise RuntimeError("Enable a Kaggle GPU, or deliberately pass --allow-cpu")
    dump_json(out/"frozen_protocol.json",{"split_counts":dict(counts),"test_observations":106,"annotation_copies":copies,"filament_spines":spines,"resolution":RESOLUTION,"threshold":THRESHOLD,"radii":RADII,"lambda":0.05,"sabr_alpha":0.25,"started_utc":datetime.now(timezone.utc).isoformat()})
    dump_json(out/"environment.json",{"python":sys.version,"torch":torch.__version__,"platform":platform.platform(),"device":str(device),"gpus":[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})
    for seed,kind,rel,expected_hash in RUNS:
        ckpt=package/rel
        if sha256(ckpt)!=expected_hash: raise RuntimeError(f"Checkpoint hash mismatch: {ckpt.name}")
        name=f"{kind}_seed{seed}"; print(f"[{name}]",flush=True); model=load_model(ckpt,device)
        if device.type=="cuda" and torch.cuda.device_count()>1: model=nn.DataParallel(model)
        infer(model,test_ids,paths,out/name/"predictions_native",device,args.batch_size,args.num_workers)
        del model; torch.cuda.empty_cache() if device.type=="cuda" else None
        evaluate_run(out/name,test_ids,gt)
    final_tables(out)
    dump_json(out/"TEST_EVALUATION_COMPLETE.json",{"completed_utc":datetime.now(timezone.utc).isoformat(),"complete":True,"runs":6})
    archive=shutil.make_archive("/kaggle/working/phase9_final_test_outputs","zip",root_dir=out.parent,base_dir=out.name)
    print(f"Phase 9 complete: {archive}",flush=True)


if __name__=="__main__": main()
