"""Run accepted figure code with explicit table, output and external QA roots."""
from pathlib import Path
import ast,sys,json,csv
from .io import *

def figures(args):
    import numpy as np,pandas as pd,matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    out=output(args.output_root)
    qa=Path(args.qa_tools_root).resolve()
    for name in ('audit_panel_alignment.py','audit_figure_collisions.py'):
        required(qa/name,'external figure QA code')
    for item in read(REPO/'provenance/EXTERNAL_TOOL_IDENTITIES.json')['plot_runtime']:
        verified(qa/item['file'],item['sha256'],'pinned external figure QA tool')
    sys.path.insert(0,str(qa))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    from audit_figure_collisions import audit_pdf
    originals=Path(args.evidence_root)/ORIGINAL/'project'
    regions=Path(args.evidence_root)/REGION/'project/region_clDice_analysis_20261004'
    common=dict(Path=Path,sys=sys,json=json,csv=csv,np=np,pd=pd,matplotlib=matplotlib,plt=plt,Line2D=Line2D,require_matplotlib_panel_alignment=require_matplotlib_panel_alignment,audit_pdf=audit_pdf)
    if args.family=='cases':
        case_figure(args,out,common)
    elif args.family=='control':
        path=REPO/'frozen_sources/figures/plot_control.py'
        tree=ast.parse(path.read_text())
        # Replace the three path-only setup assignments; every plotted value,
        # artist, dimension, font, interval and export call is unchanged.
        nodes=[]
        for n in tree.body:
            if isinstance(n,(ast.Import,ast.ImportFrom)):continue
            if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('H','F') for t in n.targets):continue
            if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func).startswith('sys.path.insert'):continue
            nodes.append(n)
        ns={**common,'H':regions,'F':out}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    else:
        name='build_segmentation_figures.py' if args.family=='segmentation' else 'build_application_figures.py'
        path=REPO/'frozen_sources/figures'/name;tree=ast.parse(path.read_text())
        ns={**common,'__file__':str(path),'__name__':'__main__','ROOT':originals.parent,'OUT':out,'FIG':out,'TAB':out/'tables','SRC':originals/('strong_baseline_final_analysis_20261001' if args.family=='segmentation' else 'morphology_application_results_20261002'),'OLD':originals/'strong_baseline_final_analysis_20261001/input_snapshot/old_final_analysis','HERE':originals/'morphology_application_execution_20261002'}
        ns['TAB'].mkdir(exist_ok=True)
        setup_paths={'ROOT','OUT','FIG','TAB','SRC','OLD','HERE'}
        nodes=[]
        for n in tree.body:
            if isinstance(n,(ast.Import,ast.ImportFrom)):continue
            # Historical manuscript copying is unrelated to figure generation.
            if isinstance(n,(ast.For,ast.Assert)):continue
            if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in setup_paths for t in n.targets):continue
            if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call):
                call=ast.unparse(n.value.func)
                if 'mkdir' in call or call.startswith('sys.path.insert'):continue
            nodes.append(n)
        from .frozen import selected
        display=selected(REPO/'frozen_sources/figures/display_labels.py',['DisplayOnly'],dict(ast=ast))['DisplayOnly']
        rendered=display().visit(ast.Module(body=nodes,type_ignores=[]));ast.fix_missing_locations(rendered)
        exec(compile(rendered,str(path),'exec'),ns)
    reports=[]
    for p in sorted(out.glob('*.pdf')):
        report=audit_pdf(p);dump(p.with_suffix('.collision-audit.json'),report)
        reports.append(dict(figure=p.name,verdict=report['verdict'],summary=report['summary']))
        if report['summary']['fail']:raise RuntimeError('Figure collision check failed: '+p.name)
    if not reports:raise RuntimeError('No figures were generated')
    dump(out/'FIGURE_EXPORTS.json',reports)
    return dict(status='PASS_EXPORT_WITH_QA',figures=len(reports),family=args.family)

def case_figure(args,out,common):
    """Run the original median-stratum case selection and native-image plate.

    Only the three selected observations are materialized outside the source
    tree. Whole-image masks are extracted before the original display crops.
    """
    import shutil,hashlib,cv2
    from collections import defaultdict
    from PIL import Image
    from .frozen import axis,selected
    from .data import inputs
    for field in ('masks_root','diagnostics_root','selection_protocol'):
        if not getattr(args,field):raise ValueError('Cases requires --'+field.replace('_','-'))
    inputs(args,False)
    root=out/'compatibility_inputs';here=root/'project/resolution_20261002/manuscript'
    here.mkdir(parents=True,exist_ok=True)
    cal=root/'project/resolution_20261002/application_calibration'
    app=root/'project/manuscript_revision_20261002/morphology_application'
    old=root/'project/morphology_application_results_20261002'
    def put(source,dest):
        source=required(source,'case-figure input');dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():
            if sha(source)!=sha(dest):raise ValueError('Case input differs')
        else:shutil.copyfile(source,dest)
    put(Path(args.diagnostics_root)/'reference_mask_per_reference.csv',cal/'reference_mask_per_reference.csv')
    put(original(args.evidence_root,'project/morphology_application_results_20261002/per_reference.csv'),old/'per_reference.csv')
    put(original(args.evidence_root,'project/manuscript_revision_20261001/morphology_application/REFERENCE_AXES.json'),app/'REFERENCE_AXES.json')
    put(REPO/'frozen_sources/axis/axis_evaluator.py',app/'axis_evaluator.py')
    put(REPO/'frozen_sources/mechanism/frozen_evaluator.py',root/'project/mechanism_training/frozen_evaluator.py')
    put(safe(args.data_root,args.annotation),root/'data/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json')
    put(args.selection_protocol,here.parent/'INTEGRATION_PROTOCOL_CN.md')
    path=REPO/'frozen_sources/figures/build_morphology_figure.py'
    common['matplotlib'].rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':7,'svg.fonttype':'none','pdf.fonttype':42})
    ns=selected(path,['sha','read','csvread','key','polygons','eligible','main'],dict(**common,hashlib=hashlib,ast=ast,defaultdict=defaultdict,cv2=cv2,Image=Image,mpl=common['matplotlib'],ROOT=root,HERE=here,CAL=cal,APP=app,OLD=old,OUT=out,ax=axis()))
    # Same eligibility, ordering and median tie rules as main(); preselection
    # is solely to copy the minimum licensed raw-image dependency closure.
    refrows=rows(cal/'reference_mask_per_reference.csv');pred={ns['key'](r):r for r in rows(old/'per_reference.csv') if r['run']=='SABR_seed20260831' and r['mode']=='validation_selected'}
    anns={str(a['id']):a for a in read(root/'data/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json')['annotations']};used=set()
    for rs,ps in [('SUCCESS','SUCCESS'),('SUCCESS','FAIL'),('FAIL','FAIL')]:
        pool=[r for r in refrows if r['status']==rs and pred[ns['key'](r)]['status']==ps and ns['eligible'](anns[r['annotation_id']],float(r['reference_length_px']))]
        pool.sort(key=lambda r:(float(r['reference_length_px']),ns['key'](r)))
        if not pool:raise ValueError('No eligible cases in a frozen stratum')
        order=sorted(range(len(pool)),key=lambda j:(abs(j-(len(pool)-1)/2),j))
        chosen=next((pool[j] for j in order if pool[j]['observation'] not in used),pool[order[0]])
        oid,sample,_=ns['key'](chosen);used.add(oid)
        put(safe(args.data_root,args.images+'/'+oid+'.jpeg'),root/'data/train/train_images'/(oid+'.jpeg'))
        put(Path(args.diagnostics_root)/'reference_mask_receipts'/(sample+'.json'),cal/'reference_mask_receipts'/(sample+'.json'))
        for method in ('SRL','SABR'):
            rel=method+'_seed20260831/validation_selected/'+oid+'.png'
            put(safe(args.masks_root,rel),root/'output/Kaggle_SABR_native_masks_output/mask_export'/rel)
    ns['main']()
