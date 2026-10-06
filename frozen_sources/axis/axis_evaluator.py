"""Mask-only candidate extraction and reference-only scoring. No inference/network."""
from pathlib import Path
import argparse, json, hashlib, sys
import numpy as np
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import label
import skimage
from skimage.morphology import skeletonize
from PIL import Image

def polyline(points):
    p=np.asarray(points,dtype=float).reshape(-1,2)
    if len(p)>1:p=p[np.r_[True,np.linalg.norm(np.diff(p,axis=0),axis=1)>0]]
    if len(p)<2:raise ValueError('Reference/candidate needs two distinct points')
    if not np.isfinite(p).all():raise ValueError('Nonfinite coordinates')
    return p

def length(points):return float(np.linalg.norm(np.diff(polyline(points),axis=0),axis=1).sum())

def sample(points):
    p=polyline(points);u=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    t=np.r_[np.arange(0.,u[-1],.5),u[-1]]
    return np.column_stack([np.interp(t,u,p[:,i]) for i in [0,1]])

def extract_axes(mask):
    """This function accepts only a prediction mask. It cannot inspect references."""
    m=np.asarray(mask,dtype=bool)
    if m.ndim!=2:raise ValueError('2D native mask required')
    sk=skeletonize(m,method='zhang');lab,n=label(sk,structure=np.ones((3,3),dtype=int));axes=[];short=0
    for k in range(1,n+1):
        nodes=np.argwhere(lab==k);N=len(nodes)
        if N<2:short+=1;continue
        index={tuple(yx):i for i,yx in enumerate(nodes)};rr=[];cc=[];vv=[]
        for i,(y,x) in enumerate(nodes):
            for dy,dx in [(0,1),(1,-1),(1,0),(1,1)]:
                j=index.get((y+dy,x+dx))
                if j is not None:
                    w=float(np.hypot(dy,dx));rr.extend([i,j]);cc.extend([j,i]);vv.extend([w,w])
        graph=csr_matrix((vv,(rr,cc)),shape=(N,N));ends=np.flatnonzero(np.diff(graph.indptr)==1)
        eligible=ends if len(ends)>=2 else np.arange(N)
        best=(-1.,None)
        # Exact search, bounded distance-array memory; never approximate graph diameter.
        for start in range(0,len(eligible),16):
            sources=eligible[start:start+16]
            dist=dijkstra(graph,directed=False,indices=sources)
            for ii,source in enumerate(sources):
                targets=eligible[eligible>source]
                if not len(targets):continue
                dd=dist[ii,targets];jj=int(np.argmax(dd));target=int(targets[jj]);value=float(dd[jj])
                if not np.isfinite(value):raise RuntimeError('Disconnected skeleton graph')
                pair=(int(source),target)
                if value>best[0] or (value==best[0] and pair<best[1]):best=(value,pair)
        source,target=best[1];_,prev=dijkstra(graph,directed=False,indices=source,return_predecessors=True)
        route=[target]
        while route[-1]!=source:
            q=int(prev[route[-1]])
            if q<0 or len(route)>N:raise RuntimeError('Invalid graph route')
            route.append(q)
        route.reverse();points=nodes[route][:,::-1].astype(float)
        axes.append({'candidate_id':k,'points':points,'length':length(points)})
    return axes,{'skeleton_components':int(n),'short_components':int(short),'candidate_count':len(axes)}

def score_axes(axes,references):
    """References enter here only, after mask-only extraction has completed."""
    refs=sorted(references,key=lambda x:x['annotation_id']);nr=len(refs);nc=len(axes)
    cost=np.zeros((nr,nc+nr));cost[:,:nc]=1e6;details={}
    for i,ref in enumerate(refs):
        r=sample(ref['points']);rtree=cKDTree(r)
        for j,axis in enumerate(axes):
            q=sample(axis['points']);R=float((cKDTree(q).query(r)[0]<=3).mean());P=float((rtree.query(q)[0]<=3).mean())
            if R>=.9 and P>=.9:
                # Large cardinality reward, bounded secondary quality reward and lexicographic perturbation.
                cost[i,j]=-(nr+1)-.5*(R+P)+1e-10*(i*(nc+1)+j)
                details[i,j]=(R,P)
    row,col=linear_sum_assignment(cost);matched={i:j for i,j in zip(row,col) if j<nc and (i,j) in details}
    rows=[]
    for i,ref in enumerate(refs):
        L=length(ref['points']);j=matched.get(i)
        r={'annotation_id':ref['annotation_id'],'status':'SUCCESS' if j is not None else 'FAIL',
           'reference_length_px':L,'candidate_id':None,'reference_coverage':None,'axis_precision':None,
           'predicted_length_px':None,'signed_relative_error':None,'absolute_relative_error':None,'absolute_error_px':None}
        if j is not None:
            lp=axes[j]['length'];R,P=details[i,j]
            r.update(candidate_id=axes[j]['candidate_id'],reference_coverage=R,axis_precision=P,predicted_length_px=lp,
                signed_relative_error=(lp-L)/L,absolute_relative_error=abs(lp-L)/L,absolute_error_px=abs(lp-L))
        rows.append(r)
    return rows,{'eligible_references':nr,'successful_references':len(matched),
                 'success_rate':len(matched)/nr if nr else None,'candidate_count':nc,'unmatched_candidates':nc-len(matched)}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def self_test():
    reference=[{'annotation_id':'line','points':[[4.,10.],[24.,10.]]}]
    m=np.zeros((32,32),bool);m[10,4:25]=True
    a,_=extract_axes(m);r,s=score_axes(a,reference);assert s['success_rate']==1 and r[0]['absolute_relative_error']==0
    a,_=extract_axes(np.zeros_like(m));r,s=score_axes(a,reference);assert s['success_rate']==0 and r[0]['absolute_relative_error'] is None
    split=m.copy();split[10,13:17]=False;a,_=extract_axes(split);_,s=score_axes(a,reference);assert s['success_rate']==0
    a,_=extract_axes(m);_,s=score_axes(a,reference+[{'annotation_id':'duplicate','points':[[4.,10.],[24.,10.]]}]);assert s['successful_references']==1
    short=m.copy();short[10,23:25]=False;a,_=extract_axes(short);r,s=score_axes(a,reference)
    assert s['success_rate']==1 and abs(r[0]['signed_relative_error']+.1)<1e-10
    import inspect
    assert list(inspect.signature(extract_axes).parameters)==['mask']
    return {'status':'PASS','kind':'synthetic engineering tests only; no scientific observations scored',
        'checks':['perfect line','empty prediction counts failure','fragmented axis not joined','one-to-one duplicate reference',
                  'short axis signed length error','mask-only extraction interface'],
        'versions':{'python':sys.version.split()[0],'numpy':np.__version__,'scipy':scipy.__version__,'skimage':skimage.__version__}}

def evaluate(mask_manifest,output):
    here=Path(__file__).resolve().parent;lock=json.loads((here/'LOCKED_APPLICATION.json').read_text())
    assert sha(__file__)==lock['axis_evaluator_sha256'],'Extractor lock mismatch'
    assert sha(here/'REFERENCE_AXES.json')==lock['reference_axes_sha256'],'Reference lock mismatch'
    current={'python':sys.version.split()[0],'numpy':np.__version__,'scipy':scipy.__version__,'skimage':skimage.__version__}
    assert current==lock['engineering_runtime'],'Locked runtime differs'
    manifest=json.loads(Path(mask_manifest).read_text());records=manifest['masks'];run=manifest['run'];mode=manifest['mode']
    spec=next(x for x in lock['runs'] if x['name']==run)
    assert mode in ['validation_selected','fixed_0.5']
    assert mode!='validation_selected' or spec['selected_status']=='OK','Selected INFEASIBLE_NA has no mask scoring'
    threshold=spec['selected_threshold'] if mode=='validation_selected' else .5
    assert manifest['threshold']==threshold and manifest['checkpoint_sha256']==spec['checkpoint_sha256']
    assert manifest['segmentation_provenance'] in lock['segmentation_locks'],'Unknown segmentation provenance'
    assert sorted(x['observation'] for x in records)==lock['observations'] and len(records)==108
    refs=json.loads((here/'REFERENCE_AXES.json').read_text());result=[];filaments=[];summaries=[]
    # Validate the whole manifest before any scoring. Missing masks are engineering errors, never omitted.
    for rec in records:
        p=Path(rec['path']);assert p.is_file() and sha(p)==rec['sha256']
        arr=np.asarray(Image.open(p));assert arr.shape==(2048,2048) and set(np.unique(arr))<={0,255}
    for rec in sorted(records,key=lambda x:x['observation']):
        oid=rec['observation'];m=np.asarray(Image.open(rec['path']))>0;a,engineering=extract_axes(m)
        copies=[]
        for copy in refs[oid]:
            rows,summary=score_axes(a,copy['references']);copies.append(summary)
            for r in rows:filaments.append({'observation':oid,'sample':copy['sample'],**r})
            summaries.append({'observation':oid,'sample':copy['sample'],**summary})
        vals=[x['success_rate'] for x in copies if x['success_rate'] is not None]
        result.append({'observation':oid,'success_rate':float(np.mean(vals)) if vals else None,
                       'evaluable_copies':len(vals),'planned_copies':len(copies),**engineering})
    out=Path(output);assert not out.exists(),'Use a new output directory';out.mkdir(parents=True)
    for name,values in [('per_reference',filaments),('per_copy',summaries),('per_observation',result)]:
        (out/(name+'.json')).write_text(json.dumps(values,indent=2,allow_nan=False),encoding='utf-8')
    (out/'SCORING_METADATA.json').write_text(json.dumps({'run':run,'mode':mode,'threshold':threshold,
        'application_lock_sha256':sha(here/'LOCKED_APPLICATION.json'),'mask_manifest_sha256':sha(mask_manifest),
        'status':'SCORED; further cross-run frozen statistical analysis required'},indent=2),encoding='utf-8')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--self-test',action='store_true');ap.add_argument('--mask-manifest');ap.add_argument('--output')
    args=ap.parse_args()
    if args.self_test:print(json.dumps(self_test(),indent=2))
    elif args.mask_manifest and args.output:evaluate(args.mask_manifest,args.output)
    else:ap.error('Use --self-test or --mask-manifest with --output')
