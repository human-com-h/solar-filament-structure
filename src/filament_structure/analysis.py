"""Accepted-score replay, frozen resampling and target census adapters."""
from pathlib import Path
from collections import defaultdict,Counter
import ast,csv,json,hashlib,sys,subprocess
from .io import *
from .frozen import module,selected,axis,boot

def replay(args):
    out=Path(args.output_root).resolve()
    if out.is_relative_to(REPO):raise ValueError('Replay output must be outside the source directory')
    if out.exists():raise FileExistsError('Use a new replay output directory: '+str(out))
    if args.models==21:
        cmd=[sys.executable,str(REPO/'frozen_sources/replay/replay21.py'),'--package',str(Path(args.evidence_root).resolve()/'02_evidence/original21'),'--output',str(out)]
    else:
        cmd=[sys.executable,str(REPO/'frozen_sources/replay/replay24.py'),'--workspace',str(Path(args.evidence_root).resolve()/REGION),'--output',str(out)]
    subprocess.run(cmd,check=True)
    return read(out/'REPLAY_QA.json')

def statistics(args):
    import numpy as np
    out=output(args.output_root)
    if list(out.iterdir()):raise FileExistsError('Use an empty statistics output directory')
    if args.arm in ('mechanism-region','mechanism-components'):
        if not args.receipts_root:raise ValueError('Original five-method statistics requires --receipts-root containing mechanism_locked_test/results and mechanism_branch_test; these receipts are outside the review subset.')
        rev=Path(args.receipts_root).resolve()
        lockpath=original(args.evidence_root,'project/mechanism_final_analysis/locked_test_protocol.json');lock=read(lockpath)
        meta={r['physical_observation_id']:r for r in manifest(args.evidence_root) if r['split']=='internal_test'}
        ns=selected(REPO/'frozen_sources/analysis/summarize_mechanism_test.py',['read','rows','write','boot','summarize'],dict(np=np,Path=Path,csv=csv,json=json,REV=rev,OUT=out,lock=lock,lockhash=sha(lockpath),methods=list(dict.fromkeys(r['method'] for r in lock['runs'])),seeds=sorted({r['seed'] for r in lock['runs']}),meta=meta))
        ns['summarize']('region' if args.arm=='mechanism-region' else 'branch')
        return dict(status='PASS',arm=args.arm,bootstrap_draws=100000,rng_offset=71,comparison_direction='SABR-minus-comparator',scope='original five-method frozen observation receipts')
    if args.arm=='segmentation21':
        import shutil
        src=Path(args.evidence_root)/ORIGINAL/'project/strong_baseline_final_analysis_20261001'
        rev=Path(args.evidence_root)/ORIGINAL/'project';old=rev/'mechanism_final_analysis';package=rev/'strong_baseline_final_eval/package_dual_gpu_v2'
        for name in ('ACCEPTANCE.json','test_observations.csv','new_per_observation_observed.csv'):
            shutil.copyfile(required(src/name,'accepted segmentation input'),out/name)
        paths=[src/n for n in ('ACCEPTANCE.json','test_observations.csv','new_per_observation_observed.csv')]+[package/'LOCKED_EVALUATION.json']+[old/n for n in ('locked_test_protocol.json','region_per_observation.csv','region_per_seed.csv','region_paired_intervals.csv')]
        identities={str(p):sha(required(p,'frozen statistics input')) for p in paths}
        def verify_inputs():
            for p,h in identities.items():verified(p,h,'frozen statistics input')
            return len(identities)
        module('frozen_bootstrap',REPO/'frozen_sources/application/frozen_bootstrap.py')
        ns=selected(REPO/'frozen_sources/analysis/audit_and_analyze.py',['sha','track','read','rows','dump','write','matrix','analyze'],dict(np=np,Path=Path,csv=csv,json=json,hashlib=hashlib,sys=sys,OUT=out,OLD=old,PACKAGE=package,INPUTS={},METRICS=['dice','iou','precision','recall','cldice','msc','msgr'],EXTRA=['foreground_fraction','false_positive_fraction','predicted_to_reference_area'],MODES=['fixed_0.5','validation_selected'],RADII=[1,3,5],NA_REASON='INFEASIBLE_NA: no feasible threshold in frozen validation grid satisfying same-seed precision floor and nonempty foreground; fixed-0.5 is diagnostic only',verify_inputs=verify_inputs))
        ns['analyze']()
        dump(out/'PORTABLE_INPUT_IDENTITIES.json',identities)
        return read(out/'STATISTICS_QA.json')
    if args.arm=='original21':
        src=Path(args.evidence_root)/ORIGINAL/'project/morphology_application_results_20261002'
        planned=read(required(src/'PLANNED_MODE_MATRIX.json','42 planned modes'))
        refs=read(original(args.evidence_root,'project/manuscript_revision_20261001/morphology_application/REFERENCE_AXES.json'))
        lock=read(original(args.evidence_root,'project/manuscript_revision_20261001/morphology_application/LOCKED_APPLICATION.json'))
        signature=read(src/'signature.json')
        def receipt(p):
            r=read(src/Path(p).relative_to(out));h=r.pop('payload_sha256')
            if hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=h:raise ValueError('Receipt integrity failed')
            if r['binding']['signature']!=signature:raise ValueError('Receipt signature differs')
            return r
        ns=selected(REPO/'frozen_sources/application/application_statistics.py',['table','clean_number','analyze'],dict(np=np,Path=Path,csv=csv,json=json,read=read,boot=boot(),read_receipt=receipt,APP=Path(args.evidence_root)/ORIGINAL/'project/manuscript_revision_20261001/morphology_application',LENGTH_FIELDS=['signed_relative_error','absolute_relative_error','absolute_error_px'],CONTROLS=['BCE_Dice','SRL','ResidualPixel','SRL_FG','FlatUNet_BCE','UNet_softDice_clDice'],SEEDS=[20260831,20260901,20260902],MODES=['validation_selected','fixed_0.5']))
        ns['analyze'](out,planned,refs,lock,signature)
        return dict(status='PASS',arm=args.arm,bootstrap_draws=100000,rng_offset=71,comparison_direction='SABR-minus-comparator',scope='accepted application score receipts; no new mask scoring')
    stage=Path(args.evidence_root)/REGION/'project/region_clDice_analysis_20261004'
    lock=read(region(args.evidence_root,'project/region_clDice_path_fix_20261004/evaluation_package/LOCKED_EVALUATION.json'))
    meta={r['physical_observation_id']:{'year':r['year'],'temporal_group':r['temporal_group']} for r in rows(original(args.evidence_root,'project/manuscript_revision_20261001/morphology_application/reference_manifest.csv'))}
    seg=rows(stage/'test_analysis/segmentation_per_observation.csv');app=rows(stage/'test_analysis/application/per_observation.csv')
    ns=selected(REPO/'frozen_sources/analysis/region_postprocess.py',['rows','table','paired_statistics','common_success_lengths'],dict(np=np,Path=Path,csv=csv,ROOT=Path(args.evidence_root)/ORIGINAL,boot=boot()))
    ns['paired_statistics'](out,lock,meta,seg,app)
    newref=rows(stage/'test_analysis/application/per_reference.csv')
    for r in newref:r['seed']=int(r['seed'])
    ns['common_success_lengths'](out,lock,newref)
    return dict(status='PASS',arm=args.arm,bootstrap_draws=100000,rng_offsets=dict(segmentation=7,application=71),comparison_direction='region+clDice-minus-comparator',scope='accepted score tables; no new training or mask scoring')

def s11(args):
    import numpy as np
    from .data import inputs
    out=output(args.output_root)
    if args.action=='census':
        inputs(args,False)
        ns=module('_sabr_census',REPO/'frozen_sources/s11/compute_target_quality.py')
        root=Path(args.evidence_root)/ORIGINAL
        ns.ROOT=root;ns.PKG=root/'project/mechanism_training';ns.OUT=out
        ns.ANN=safe(args.data_root,args.annotation)
        if args.labels_root:
            import shutil
            compatibility=out/'label_parity_inputs'
            labeldest=compatibility/'project/mechanism_branch_test/labels'
            for r in manifest(args.evidence_root):
                if r['split']!='internal_test':continue
                source=required(safe(args.labels_root,r['annotation_sample_id']+'.npz'),'frozen test label')
                dest=labeldest/source.name;dest.parent.mkdir(parents=True,exist_ok=True)
                if not dest.exists():shutil.copyfile(source,dest)
                elif sha(dest)!=sha(source):raise ValueError('Frozen test label identity differs')
            ns.ROOT=compatibility
        elif not (root/'project/mechanism_branch_test/labels').is_dir():
            raise FileNotFoundError('The review subset omits frozen label arrays. Supply --labels-root with the 170 original test NPZ labels. Full label parity has not run.')
        # The historical backup concerns manuscript-editing; this entry only
        # computes targets and writes new tables. All census code is unchanged.
        ns.backup=lambda:None
        ns.main()
        return read(out/'TARGET_QUALITY_SUMMARY.json')['parity']
    # Map the two evidence arms into the old workspace namespace. Only the
    # fixed documented sources are resolved; writes always target out.
    ns=module('_sabr_consolidate',REPO/'frozen_sources/s11/consolidate_evidence.py')
    real_rows=ns.rows
    def redirected(p):
        p=Path(p)
        if p.is_relative_to(out):return real_rows(p)
        rel=p.relative_to(ns.REV).as_posix()
        if rel.startswith('region_clDice_analysis_20261004/'):dest=region(args.evidence_root,'project/'+rel)
        else:dest=original(args.evidence_root,'project/'+rel)
        return real_rows(dest)
    ns.OUT=out;ns.rows=redirected
    # Provenance hashing inside the frozen implementation needs the same map;
    # a temporary compatibility layout supplies only its actual input closure.
    import shutil
    compat=out/'compatibility_inputs';compat.mkdir(exist_ok=True)
    rels=['mechanism_training/temporal_manifest.csv','strong_baseline_final_analysis_20261001/region_per_observation.csv','region_clDice_analysis_20261004/test_analysis/segmentation_per_observation.csv','mechanism_final_analysis/branch_per_observation.csv','morphology_application_results_20261002/per_observation.csv','region_clDice_analysis_20261004/test_analysis/application/per_observation.csv','morphology_application_results_20261002/comparison_sensitivity.csv','region_clDice_analysis_20261004/test_analysis/sensitivity.csv','mechanism_final_analysis/region_sensitivity.csv','mechanism_final_analysis/branch_sensitivity.csv','region_clDice_analysis_20261004/report_tables/all_eight_methods_application_per_seed.csv']
    for rel in rels:
        source=region(args.evidence_root,'project/'+rel) if rel.startswith('region_clDice') else original(args.evidence_root,'project/'+rel)
        dest=compat/'project'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():shutil.copyfile(source,dest)
        elif sha(dest)!=sha(source):raise ValueError('Compatibility input differs')
    ns.ROOT=compat;ns.REV=compat/'project';ns.rows=real_rows;ns.main()
    return read(out/'CONSOLIDATION_QA.json')

def diagnostic_functions():
    import numpy as np
    from scipy.spatial import cKDTree
    names=['mean','std','key','summarize_lengths','axes_json','diagnostic']
    return selected(REPO/'frozen_sources/diagnostics/analyze_calibration.py',names,dict(np=np,cKDTree=cKDTree,ax=axis(),LF=['signed_relative_error','absolute_relative_error','absolute_error_px']))

def diagnostics(args):
    """Reference-mask and unordered copy-pair diagnostics without new intervals."""
    if args.action=='full':return full_diagnostics(args)
    import numpy as np
    from itertools import combinations
    from .data import inputs
    from .frozen import evaluator,score_one
    _,rr,_,ims,anns=inputs(args,False)
    refs=read(original(args.evidence_root,'project/manuscript_revision_20261001/morphology_application/REFERENCE_AXES.json'))
    fe=evaluator();ax=axis();diag=diagnostic_functions()['diagnostic'];out=output(args.output_root)
    references=[];copies=[];observations=[];pairrows=[]
    for oid,cc in sorted(refs.items()):
        values=[]
        for c in cc:
            sid=c['sample'];mask=fe['rasterize'](ims[sid],anns[sid]);aa,eng=ax.extract_axes(mask);rs,ss=diag(aa,c['references'])
            result=score_one(mask,[c]);cr=result['per_copy'][0];copies.append(dict(observation=oid,**cr,**eng));values.append(cr)
            references.extend(dict(observation=oid,sample=sid,**r) for r in rs)
        observations.append(dict(observation=oid,success_rate=float(np.mean([r['success_rate'] for r in values]))))
        for left,right in combinations(cc,2):
            # Drawn spines are candidate curves here; matching remains the same
            # bidirectional one-to-one geometry, with this separate denominator.
            aa=[dict(candidate_id=i+1,points=np.asarray(r['points']),length=ax.length(r['points'])) for i,r in enumerate(left['references'])]
            _,ss=ax.score_axes(aa,right['references'])
            n=len(left['references'])+len(right['references'])
            pairrows.append(dict(observation=oid,left_sample=left['sample'],right_sample=right['sample'],matches=ss['successful_references'],agreement=2*ss['successful_references']/n if n else None))
    pairs=defaultdict(list)
    for r in pairrows:pairs[r['observation']].append(r['agreement'])
    if len(references)!=1368 or len(copies)!=170 or len(pairrows)!=84 or len(pairs)!=40:raise ValueError('Diagnostic denominators differ')
    table(out/'reference_per_reference.csv',references);table(out/'reference_per_copy.csv',copies);table(out/'reference_per_observation.csv',observations);table(out/'copy_pairs.csv',pairrows)
    summary=dict(reference_success_macro=float(np.mean([r['success_rate'] for r in observations])),reference_success_count=sum(r['status']=='SUCCESS' for r in references),reference_denominator=1368,copy_pair_agreement=float(np.mean([np.mean(v) for v in pairs.values()])),copy_pairs=84,pair_observations=40,new_bootstrap=0)
    dump(out/'DIAGNOSTICS.json',summary);return summary

def full_diagnostics(args):
    """Complete original yield, common-success and nine-run failure analysis."""
    import numpy as np,scipy,skimage,cv2,platform,time,shutil
    from datetime import datetime,timezone
    from itertools import combinations
    from scipy.spatial import cKDTree
    from PIL import Image
    if not args.masks_root or not args.diagnostic_protocol:raise ValueError('Full diagnostics requires --masks-root and --diagnostic-protocol; neither is redistributed here.')
    out=output(args.output_root);root=out/'compatibility_inputs';root.mkdir(exist_ok=True)
    freeze=read(REPO/'frozen_sources/diagnostics/ANALYSIS_FREEZE.json')
    def materialize(source,dest,h=None):
        source=verified(source,h,'diagnostic input') if h else required(source,'diagnostic input')
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():
            if sha(dest)!=sha(source):raise ValueError('Diagnostic compatibility input differs')
        else:shutil.copyfile(source,dest)
    for rel,h in freeze['input_sha256'].items():
        if rel.startswith('data/'):source=safe(args.data_root,rel.removeprefix('data/'))
        elif rel.endswith('application_calibration/PROTOCOL_CN.md'):source=args.diagnostic_protocol
        elif rel.startswith('output/'):source=safe(args.masks_root,'MASK_MANIFESTS.json')
        else:source=original(args.evidence_root,rel)
        materialize(source,root/rel,h)
    masksdest=root/'output/Kaggle_SABR_native_masks_output/mask_export'
    mm=read(Path(args.masks_root)/'MASK_MANIFESTS.json')
    focal=[m for m in mm if m['mode']=='validation_selected' and m['run'].split('_seed')[0] in ['SABR','SRL','BCE_Dice']]
    if len(focal)!=9:raise ValueError('The nine original focal selected manifests are required')
    for m in focal:
        source=safe(args.masks_root,m['path']);materialize(source,masksdest/m['path'],m['sha256'])
        for r in read(source)['masks']:materialize(safe(args.masks_root,r['path']),masksdest/r['path'],r['sha256'])
    for name in ('ANALYSIS_FREEZE.json','ANALYSIS_CODE_LOCK.json'):materialize(REPO/'frozen_sources/diagnostics'/name,out/name)
    funcs=['sha','read','dump','csvread','csvwrite','mean','std','key','summarize_lengths','axes_json','diagnostic','main']
    ns=selected(REPO/'frozen_sources/diagnostics/analyze_calibration.py',funcs,dict(Path=Path,datetime=datetime,timezone=timezone,defaultdict=defaultdict,Counter=Counter,combinations=combinations,ast=ast,csv=csv,hashlib=hashlib,json=json,platform=platform,sys=sys,time=time,np=np,scipy=scipy,skimage=skimage,cv2=cv2,Image=Image,cKDTree=cKDTree,ROOT=root,HERE=out,APP=root/'project/manuscript_revision_20261002/morphology_application',OLD=root/'project/morphology_application_results_20261002',MASKS=masksdest,ax=axis(),LF=['signed_relative_error','absolute_relative_error','absolute_error_px'],CATS=['SUCCESS','no_candidate','qualifying_edge_unassigned','reference_direction_only','candidate_direction_only','separate_direction_candidates','neither_direction_reaches_90pct']))
    ns['main']()
    return read(out/'ENGINEERING_QA.json')
