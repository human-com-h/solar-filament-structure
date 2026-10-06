"""Real observation plate; retrospective declared selection; no generated pixels."""
from pathlib import Path
from collections import defaultdict
import ast,csv,hashlib,json,sys
import cv2,numpy as np
from PIL import Image
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]
APP=ROOT/'project/manuscript_revision_20261002/morphology_application'
CAL=HERE.parent/'application_calibration'; OLD=ROOT/'project/morphology_application_results_20261002'
sys.path.insert(0,str(APP)); import axis_evaluator as ax
sys.path.insert(0,str(HERE/'figure_tools')); from audit_panel_alignment import require_matplotlib_panel_alignment
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':7,'svg.fonttype':'none','pdf.fonttype':42})
OUT=HERE/'figures';OUT.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def csvread(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def key(r):return r['observation'],r['sample'],r['annotation_id']
def polygons(a):return [np.asarray(p,float).reshape(-1,2) for p in a['segmentation']]
def eligible(a,L):
    q=np.concatenate(polygons(a));return L>=50 and float(np.ptp(q,axis=0).max())<=384

def main():
    refrows=csvread(CAL/'reference_mask_per_reference.csv')
    oldrows=csvread(OLD/'per_reference.csv')
    pred={key(r):r for r in oldrows if r['run']=='SABR_seed20260831' and r['mode']=='validation_selected'}
    sr={key(r):r for r in oldrows if r['run']=='SRL_seed20260831' and r['mode']=='validation_selected'}
    coco=read(ROOT/'data/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json')
    annotations={str(a['id']):a for a in coco['annotations']};ims={str(i['id']):i for i in coco['images']};by=defaultdict(list)
    for a in coco['annotations']:by[str(a['image_id'])].append(a)
    refs=read(APP/'REFERENCE_AXES.json')
    raw=(ROOT/'project/mechanism_training/frozen_evaluator.py').read_text(encoding='utf-8-sig')
    node=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=='rasterize')
    ns={'np':np,'cv2':cv2};exec(compile(ast.Module(body=[node],type_ignores=[]),'<frozen_rasterize>','exec'),ns)
    selected=[];used=set();strata=[('Reference and SABR succeed','SUCCESS','SUCCESS'),('Reference succeeds, SABR fails','SUCCESS','FAIL'),('Reference and SABR fail','FAIL','FAIL')]
    populations=[]
    for title,rs,ps in strata:
        pool=[r for r in refrows if r['status']==rs and pred[key(r)]['status']==ps and eligible(annotations[r['annotation_id']],float(r['reference_length_px']))]
        pool.sort(key=lambda r:(float(r['reference_length_px']),key(r)))
        order=sorted(range(len(pool)),key=lambda j:(abs(j-(len(pool)-1)/2),j))
        chosen=next((pool[j] for j in order if pool[j]['observation'] not in used),pool[order[0]])
        used.add(chosen['observation']); selected.append((title,chosen));populations.append({'stratum':title,'eligible_references':len(pool),'selected':key(chosen),'length_sort_rank':pool.index(chosen)+1})
    fig,axs=plt.subplots(3,4,figsize=(7.2047244,6.4173228));fig.subplots_adjust(left=.035,right=.985,bottom=.08,top=.89,wspace=.08,hspace=.61)
    headers=['H-alpha + manual spine','Reference mask + axis','SRL mask + axes','SABR mask + axes']
    records=[]
    for i,(title,row) in enumerate(selected):
        oid,sample,aid=key(row);a=annotations[aid];sp=np.asarray(a['spine'],float).reshape(-1,2)
        points=np.concatenate(polygons(a)+[sp]);lo=np.floor(points.min(0)-32);hi=np.ceil(points.max(0)+32);side=max(128,int((hi-lo).max()));center=(hi+lo)/2
        x0=max(0,min(2048-side,int(np.floor(center[0]-side/2))));y0=max(0,min(2048-side,int(np.floor(center[1]-side/2))));x1=x0+side;y1=y0+side
        ip=ROOT/'data/train/train_images'/(oid+'.jpeg');img=np.asarray(Image.open(ip).convert('L'))
        assert img.shape==(2048,2048)
        m=ns['rasterize'](ims[sample],by[sample]);receipt=read(CAL/'reference_mask_receipts'/(sample+'.json'))
        assert hashlib.sha256(m.tobytes()).hexdigest()==receipt['binding']['native_union_mask_sha256']
        axes=[{'candidate_id':q['candidate_id'],'points':np.asarray(q['points'],float),'length':q['length']} for q in receipt['axes']]
        columns=[(None,[],None),(m,axes,row)];paths=[]
        for method,lookup in [('SRL',sr),('SABR',pred)]:
            path=ROOT/'output/Kaggle_SABR_native_masks_output/mask_export'/(method+'_seed20260831')/'validation_selected'/(oid+'.png')
            pm=np.asarray(Image.open(path))>0;pa,_=ax.extract_axes(pm)
            cr=next(c for c in refs[oid] if c['sample']==sample)['references'];rr,_=ax.score_axes(pa,cr);selected_r=next(r for r in rr if r['annotation_id']==aid)
            assert selected_r['status']==lookup[key(row)]['status']
            columns.append((pm,pa,selected_r));paths.append({'method':method,'mask_file':path.relative_to(ROOT).as_posix(),'sha256':sha(path)})
        for j,(mask,axes,rr) in enumerate(columns):
            p=axs[i,j];p.imshow(img[y0:y1,x0:x1],cmap='gray',vmin=0,vmax=255,interpolation='nearest',extent=[x0,x1,y1,y0])
            if mask is not None:
                overlay=np.zeros((side,side,4),float);overlay[mask[y0:y1,x0:x1]>0]=[1,.65,.16,.26]
                p.imshow(overlay,interpolation='nearest',extent=[x0,x1,y1,y0])
                for q in axes:
                    z=np.asarray(q['points']).copy()
                    z[~((z[:,0]>=x0)&(z[:,0]<=x1)&(z[:,1]>=y0)&(z[:,1]<=y1))]=np.nan
                    p.plot(z[:,0],z[:,1],color='#e5a021',lw=.8,clip_on=True)
            p.plot(sp[:,0],sp[:,1],color='#34d8ed',lw=1.05,ls='--',clip_on=True)
            p.set_xlim(x0,x1);p.set_ylim(y1,y0);p.set_aspect('equal');p.set_xticks([]);p.set_yticks([])
            for s in p.spines.values():s.set_visible(False)
            p.text(-.02,1.05,chr(97+i*4+j),transform=p.transAxes,fontweight='bold',fontsize=8,ha='left',va='bottom')
            p.plot([x0+.07*side,x0+.07*side+20],[y1-.08*side]*2,color='white',lw=1.8)
            p.text(.07,.13,'20 px',transform=p.transAxes,color='white',fontsize=6.5,va='bottom')
            if rr is not None:
                lab=rr['status']
                if lab=='SUCCESS':lab+='  R/P={:.2f}/{:.2f}'.format(float(rr['reference_coverage']),float(rr['axis_precision']))
                p.text(.5,-.06,lab,transform=p.transAxes,ha='center',va='top',fontsize=6.5)
        records.append({'stratum':title,'observation':oid,'sample':sample,'annotation_id':aid,'reference_length_px':float(row['reference_length_px']),
          'raw_image_file':ip.relative_to(ROOT).as_posix(),'raw_image_sha256':sha(ip),'crop_xyxy_native_px':[x0,y0,x1,y1],
          'reference_union_mask_sha256':receipt['binding']['native_union_mask_sha256'],'prediction_masks':paths,
          'reference_status':row['status'],'SRL_status':sr[key(row)]['status'],'SABR_status':pred[key(row)]['status']})
    fig.canvas.draw()
    for j,name in enumerate(headers):
        pos=axs[0,j].get_position();fig.text((pos.x0+pos.x1)/2,.96,name,ha='center',fontsize=7)
    for i,(title,row) in enumerate(selected):
        pos=axs[i,0].get_position();fig.text(.035,pos.y1+.04,title,fontsize=7.2,fontweight='bold')
    fig.text(.035,.028,'Cyan dashed: manual spine     Gold solid: mask-derived axis     Amber: foreground mask',fontsize=7)
    require_matplotlib_panel_alignment(fig,json_out=OUT/'Figure_6.alignment.json',overlay_svg=OUT/'Figure_6.alignment.svg',strict=True)
    fig.savefig(OUT/'Figure_6.pdf')
    fig.savefig(OUT/'Figure_6.svg')
    fig.savefig(OUT/'Figure_6.png',dpi=600)
    fig.savefig(OUT/'Figure_6.tiff',dpi=600)
    plt.close(fig)
    report={'selection_protocol_sha256':sha(HERE.parent/'INTEGRATION_PROTOCOL_CN.md'),'selection':'retrospective illustration only; fixed median-length rule within three declared strata',
     'populations':populations,'records':records,'annotation_source_sha256':sha(ROOT/'data/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'),
     'extractor_sha256':sha(APP/'axis_evaluator.py'),'brightness_contrast':'All source images linear grayscale 0..255; no per-crop contrast or gamma adjustment',
     'scale':'native pixels only; no inferred physical/solar deprojection calibration','masks':'whole native-image extraction precedes display cropping',
     'stitching':'Distinct panels clearly separated; no scientific-image stitching','uncertainty':'Not applicable: deterministic examples; no aggregate estimates shown'}
    (OUT/'Figure_6_source.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(records,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
