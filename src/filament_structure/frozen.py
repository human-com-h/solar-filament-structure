"""Load frozen functions without executing historical command-line side effects.

CPU scoring needs no torch import. AST selection compiles the original function
bodies without textual edits; the source hashes remain independently checked.
"""
from pathlib import Path
from collections import defaultdict,Counter
from dataclasses import dataclass
import ast,csv,hashlib,importlib.util,json,platform,sys,time,threading
from .io import Path,REPO,read,sha,protocol,qc

def module(name,path):
    path=Path(path);spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def selected(path,names,namespace=None):
    path=Path(path);tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes}!=set(names):raise ValueError('Frozen definitions differ: '+str(path))
    ns={'__file__':str(path),**(namespace or {})}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    return ns
def axis():return module('_sabr_frozen_axis',REPO/'frozen_sources/axis/axis_evaluator.py')
def branch():return module('_sabr_frozen_branch',REPO/'frozen_sources/mechanism/branch_reference.py')
def target_functions(evidence_root):
    import numpy as np,cv2
    br=branch();excluded={r['annotation_id'] for r in qc(evidence_root)}
    ns=selected(REPO/'frozen_sources/mechanism/data_pipeline.py',['label_copy','balanced_weights'],dict(np=np,exclusions=lambda:excluded,branch_components_from_instance=br.branch_components_from_instance))
    render=selected(REPO/'frozen_sources/mechanism/annotation_helpers.py',['render_union_mask','deterministic_choice'],dict(np=np,cv2=cv2,hashlib=hashlib))
    return {**render,**ns}
def evaluator():
    import numpy as np,cv2
    from scipy.ndimage import distance_transform_edt
    from skimage.morphology import skeletonize
    names=['CopyGT','physical_id','rasterize','sample_spine','build_gt','longest_gap','evaluate_one']
    return selected(REPO/'frozen_sources/mechanism/frozen_evaluator.py',names,dict(np=np,cv2=cv2,dataclass=dataclass,defaultdict=defaultdict,Path=Path,skeletonize=skeletonize,distance_transform_edt=distance_transform_edt,EPS=1e-6,RADII=(1,3,5),METRICS=('dice','iou','precision','recall','cldice','msc','msgr')))
def threshold_functions():
    import numpy as np
    from scipy.ndimage import maximum_filter
    from types import SimpleNamespace
    fe=SimpleNamespace(**evaluator())
    thresholds=np.unique(np.r_[.01,np.arange(.05,.951,.025),.5,.99].round(6)).astype(np.float32)
    ns=selected(REPO/'frozen_sources/mechanism/threshold_curve.py',['counts','curve'],dict(np=np,THRESHOLDS=thresholds,maximum_filter=maximum_filter,fe=fe))
    return thresholds,ns['curve']
def score_one(mask,refs):
    import numpy as np,psutil
    ns=selected(REPO/'frozen_sources/application/run_application.py',['score_one'],dict(np=np,psutil=psutil,time=time,threading=threading,ax=axis(),LENGTH_FIELDS=['signed_relative_error','absolute_relative_error','absolute_error_px']))
    return ns['score_one'](mask,refs)
def boot():return module('_sabr_frozen_bootstrap',REPO/'frozen_sources/application/frozen_bootstrap.py').boot
def verify_sources():
    checks=read(REPO/'provenance/FROZEN_SOURCE_IDENTITIES.json')
    for x in checks:
        if sha(REPO/x['release_path'])!=x['sha256']:raise ValueError('Frozen source modified: '+x['release_path'])
    return len(checks)
