"""CPU engineering checks; synthetic fixtures never count as paper evidence."""
from pathlib import Path
import ast,json,platform,sys
from collections import defaultdict
from .io import *
from .frozen import axis,branch,selected,evaluator,score_one,threshold_functions,verify_sources

def run_checks(with_models=False):
    import numpy as np,cv2,scipy,skimage
    ax=axis();checks=[]
    checks.append(dict(name='frozen_axis_self_test',result=ax.self_test()))
    mask=np.zeros((32,32),bool);mask[10,4:25]=True
    copies=[dict(sample='copy1',references=[dict(annotation_id='a',points=[[4.,10.],[24.,10.]])]),dict(sample='copy2',references=[dict(annotation_id='b'+str(i),points=[[4.,20.+i],[24.,20.+i]]) for i in range(3)])]
    result=score_one(mask,copies)
    assert result['per_observation']['success_rate']==.5
    assert result['per_observation']['successful_references']==1
    assert result['per_observation']['successful_copies']==1
    assert result['per_observation']['absolute_relative_error']==0
    fail=score_one(np.zeros_like(mask),copies)
    assert fail['per_observation']['success_rate']==0
    assert fail['per_observation']['length_status']=='FAIL_NA' and fail['per_observation']['absolute_relative_error'] is None
    assert all(r['length_status']=='FAIL_NA' and r['predicted_length_px'] is None for r in fail['per_reference'])
    checks.append(dict(name='copy_then_observation_and_failed_lengths',macro=.5,pooled_reference_fraction=.25,status='PASS'))
    # The core function must preserve a preownership-eligible component even
    # when ownership leaves fewer than five pixels.
    first=np.array([[10,i] for i in range(5)])
    second=np.array([[10,i] for i in range(3,8)])
    fake=lambda ann,*_: [first if ann['id']=='first' else second]
    labels=selected(REPO/'frozen_sources/mechanism/data_pipeline.py',['label_copy'],dict(np=np,exclusions=lambda:set(),branch_components_from_instance=fake))['label_copy'](dict(width=1024,height=1024),[dict(id='first'),dict(id='second')])
    assert (labels==1).sum()==5 and (labels==2).sum()==3
    checks.append(dict(name='five_pixel_filter_before_ownership_only',second_retained_pixels=3,status='PASS'))
    br=branch();ann=dict(id='cross',segmentation=[[100,490,900,490,900,510,100,510],[490,200,510,200,510,800,490,800]],spine=[100,500,900,500])
    comps=br.branch_components_from_instance(ann,1024,1024,1024)
    assert len(comps)>=2 and sum(len(c) for c in comps)>100
    assert br.branch_components_from_instance(dict(segmentation=ann['segmentation'],spine=[]),1024,1024,1024)==[]
    assert br.branch_components_from_instance(dict(segmentation=[],spine=ann['spine']),1024,1024,1024)==[]
    checks.append(dict(name='endpoint_path_residuals_and_rejection',components=len(comps),status='PASS'))
    render=selected(REPO/'frozen_sources/mechanism/annotation_helpers.py',['render_union_mask'],dict(np=np,cv2=cv2))['render_union_mask']
    m=render('s',{'s':dict(width=1024,height=1024)},{'s':[ann]},1024)
    ns=selected(REPO/'frozen_sources/mechanism/data_pipeline.py',['balanced_weights'],dict(np=np))
    labels=np.zeros((8,8),np.uint16);labels[0,:5]=1;labels[1,:7]=2;weights,offset=ns['balanced_weights'](labels)
    p=np.arange(64,dtype=np.float32).reshape(8,8)/64
    weighted=1-float((p*weights).sum())-float(offset)
    explicit=1-np.mean([(p[labels==k].sum()+1e-6)/((labels==k).sum()+1e-6) for k in (1,2)])
    assert abs(weighted-explicit)<1e-7
    checks.append(dict(name='component_balancing_equivalence',difference=float(abs(weighted-explicit)),status='PASS'))
    fe=evaluator();coco=dict(images=[dict(id='s',file_name='obs.png',width=32,height=32)],annotations=[dict(id='a',image_id='s',segmentation=[[4,10,24,10,24,11,4,11]],spine=[4,10,24,10])])
    gt=fe['build_gt'](coco,{'obs'});pred=fe['rasterize'](coco['images'][0],coco['annotations']);aa,ff,oo=fe['evaluate_one']('obs',pred,gt['obs'])
    assert oo[3]['dice']==1 and oo[3]['msc']==1 and oo[3]['msgr']==0
    tt,curve=threshold_functions();assert len(tt)==39 and tt.dtype==np.float32
    values=curve(pred.astype(np.float32),gt['obs']);i=int(np.argmin(abs(tt-.5)))
    assert max(abs(values[i,j]-oo[3][k]) for j,k in enumerate(['dice','iou','precision','recall','msc','msgr']))<1e-10
    checks.append(dict(name='native_segmentation_threshold_curve_parity',grid_size=len(tt),status='PASS'))
    # A candidate that covers a short reference but extends far beyond it must
    # fail the candidate-to-reference direction, even at full reference cover.
    aa=[dict(candidate_id=1,points=np.array([[0.,10.],[100.,10.]]),length=100.)]
    _,ss=ax.score_axes(aa,[dict(annotation_id='short',points=[[10.,10.],[20.,10.]])]);assert ss['successful_references']==0
    ring=np.zeros((16,16),bool);ring[3,3:13]=True;ring[12,3:13]=True;ring[3:13,3]=True;ring[3:13,12]=True
    a1,_=ax.extract_axes(ring);a2,_=ax.extract_axes(ring)
    assert len(a1)==len(a2)==1 and np.array_equal(a1[0]['points'],a2[0]['points'])
    checks.append(dict(name='bidirectional_matching_and_loop_ties',status='PASS'))
    from .evaluation import residual_recall
    # Compile the original radius-max loop itself, independent of the adapter.
    original_tree=ast.parse((REPO/'frozen_sources/analysis/evaluate_mechanism_branches.py').read_text())
    original_loop=next(n for n in ast.walk(original_tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='dy')
    p=np.random.default_rng(19).random((1024,1024),dtype=np.float32)
    labels=np.zeros_like(p,dtype=np.uint16);labels[0,:7]=1;labels[1023,1018:]=2;labels[20:23,30:35]=3
    ys,xs=np.nonzero(labels);ls=labels[ys,xs].astype(np.int64);counts=np.bincount(ls);valid=np.flatnonzero(counts);valid=valid[valid>0]
    for radius in (0,1,2):
        ns=dict(np=np,ys=ys,xs=xs,p=p,radius=radius,score=np.full(len(xs),-np.inf,dtype=np.float32))
        exec(compile(ast.Module(body=[original_loop],type_ignores=[]),'<original_component_radius_loop>','exec'),ns)
        for threshold in (.5,.875):
            hits=np.bincount(ls[ns['score']>=threshold],minlength=len(counts))
            result=residual_recall(p,labels,threshold,radius)
            assert result==dict(component_recall=float((hits[valid]/counts[valid]).mean()),pixel_recall=float(hits[valid].sum()/counts[valid].sum()))
    checks.append(dict(name='component_recall_original_loop_parity',radii=[0,1,2],boundary_pixels_included=True,status='PASS'))
    model_results=[]
    if with_models:
        import torch
        from .models import METHODS,factory,loss,loss_module
        torch.set_num_threads(2);torch.manual_seed(7)
        for method in METHODS:
            net=factory(method);net.eval();x=torch.linspace(0,1,2*64*64).reshape(2,1,64,64)
            logits=net(x);assert logits.shape==x.shape and logits.dtype==torch.float32
            y=torch.zeros_like(logits);y[:,:,20:44,20:44]=1;srl=torch.zeros_like(y);srl[:,:,30:33,20:44]=1
            residual=torch.zeros_like(y);residual[:,:,25:28,25:28]=1
            batch=dict(y=y,srl=srl,residual=residual,weights=residual/9.,offset=torch.zeros(2))
            value=loss(logits,batch,method);assert torch.isfinite(value);value.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
            if method in ('UNet_softDice_clDice','UNet_region_clDice'):assert loss_module(method).soft_skeletonize.num_iter==10
            if method in METHODS[:5]:
                empty={**batch,'srl':torch.zeros_like(srl),'residual':torch.zeros_like(residual)}
                regional=loss(logits.detach(),batch,'BCE_Dice')
                assert torch.equal(loss(logits.detach(),empty,method),regional)
            if method in ('SABR','ResidualPixel','SRL_FG'):
                fallback={**batch,'residual':torch.zeros_like(residual)}
                assert torch.equal(loss(logits.detach(),fallback,method),loss(logits.detach(),fallback,'SRL'))
            model_results.append(dict(method=method,parameters=sum(p.numel() for p in net.parameters()),loss=float(value.detach()),finite_gradients=True))
        checks.append(dict(name='eight_model_forward_loss_gradient',status='PASS',methods=model_results,shape=[2,1,64,64],gpu_training_performed=False))
    return dict(status='PASS_CPU_ENGINEERING',synthetic_only=True,checks=checks,frozen_files_checked=verify_sources(),environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,skimage=skimage.__version__,opencv=cv2.__version__),model_checks_executed=with_models,ddp_gpu_training_verified=False,new_paper_observations_scored=False)
def validate(args):
    report=run_checks(args.with_models);dump(output(args.output_root)/'CPU_VALIDATION.json',report);return report
def verify_release(args):
    index=rows(REPO/'provenance/RELEASE_FILES_SHA256.csv')
    expected={r['path'] for r in index}|{'provenance/RELEASE_FILES_SHA256.csv'}
    actual={p.relative_to(REPO).as_posix() for p in source_files()}
    if actual!=expected:raise ValueError('Release file set differs from checksum manifest')
    for r in index:
        p=REPO/r['path']
        if sha(p)!=r['sha256'] or p.stat().st_size!=int(r['bytes']):raise ValueError('Release file differs: '+r['path'])
    return dict(status='PASS',files=len(actual),frozen_sources=verify_sources(),root_git_metadata_excluded=(REPO/'.git').exists())
