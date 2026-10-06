"""Revise display labels using existing Matplotlib functions and frozen data.

All observations, methods, seeds, numerical marks and uncertainty definitions
are retained. Scientific CSV/JSON sources are compared and never overwritten.
"""
from pathlib import Path
import ast, hashlib, json, re, shutil, sys, types
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

H=Path(__file__).resolve().parent
ROOT=H.parents[1]
OUT=H/'manuscript'
FIG=OUT/'figures'
SRC=ROOT/'project/strong_baseline_final_analysis_20261001'
OLD=SRC/'input_snapshot/old_final_analysis'
APP=ROOT/'project/morphology_application_results_20261002'
SKILL=ROOT/'external_figure_qa'
sys.path.insert(0,str(SKILL))
from audit_panel_alignment import require_matplotlib_panel_alignment

plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
 'font.size':7,'axes.titlesize':8,'axes.labelsize':7,'xtick.labelsize':7,'ytick.labelsize':7,
 'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,
 'axes.linewidth':.6,'savefig.dpi':600})
SEEDS=[20260831,20260901,20260902]
COLORS=['#667788','#007E87','#A54A36']; MARKERS=['o','s','^']
METHODS=['BCE_Dice','SRL','SABR','ResidualPixel','SRL_FG','FlatUNet_BCE','UNet_softDice_clDice']
COMP=CONTROL=[m for m in METHODS if m!='SABR']
LABEL={'BCE_Dice':'BCE–Dice','SRL':'SRL','SABR':'SABR','ResidualPixel':'ResidualPixel',
 'SRL_FG':'SRL–FG','FlatUNet_BCE':'Flat U-Net','UNet_softDice_clDice':'soft-clDice'}
MC=['#9B9B9B','#8B6AA5','#006B73','#AA7040','#486FA0','#BA6370','#343434']
MODE=['validation_selected','fixed_0.5']; TITLES=['Validation-selected','Fixed 0.5 diagnostic']
region=pd.read_csv(SRC/'region_per_seed.csv')
obs=pd.read_csv(SRC/'region_per_observation.csv')
branch=pd.read_csv(OLD/'branch_per_seed.csv')
run=pd.read_csv(APP/'per_seed.csv',float_precision='round_trip',low_memory=False)
ci=pd.read_csv(APP/'paired_bootstrap.csv',float_precision='round_trip',low_memory=False)
sensitivity=pd.read_csv(APP/'comparison_sensitivity.csv',float_precision='round_trip',low_memory=False)
segment=pd.read_csv(SRC/'region_per_seed.csv',float_precision='round_trip',low_memory=False)
seed_handles=[Line2D([],[],linestyle='none',marker=MARKERS[j],color=COLORS[j],label=str(s)) for j,s in enumerate(SEEDS)]
records=[]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def snapshot():
    dest=H/'history_before_figure_labels_v5'
    if dest.exists(): return
    dest.mkdir()
    for folder in ['manuscript','deliverables']:
        target=dest/folder; target.mkdir()
        src=H/folder
        for p in src.iterdir():
            if p.is_file(): shutil.copy2(p,target/p.name)
    shutil.copytree(FIG,dest/'manuscript/figures')
    shutil.copytree(OUT/'study_project',dest/'manuscript/study_project')
    for p in H.iterdir():
        if p.is_file() and p.suffix in ['.json','.md','.py','.csv']:
            shutil.copy2(p,dest/p.name)

def finish(fig,axes,name,source,caption,exclude=()):
    original=pd.read_csv(FIG/f'{name}_source.csv',float_precision='round_trip')
    drawn=pd.DataFrame(source)
    pd.testing.assert_frame_equal(original,drawn,check_dtype=False,check_exact=False,atol=1e-12,rtol=1e-12)
    before_hash=sha(FIG/f'{name}_source.csv')
    for i,a in enumerate(np.ravel(axes)):
        a.annotate(chr(97+i),xy=(0,1),xycoords='axes fraction',xytext=(-12,14),textcoords='offset points',fontweight='bold',fontsize=9)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=str(FIG/f'{name}.alignment.json'),
        overlay_svg=str(FIG/f'{name}.alignment.svg'),tolerance_pt=1.5,gutter_tolerance_pt=1.5,
        require_panel_labels=True,strict=True,exclude_axes=exclude)
    fig.savefig(FIG/f'{name}.pdf')
    fig.savefig(FIG/f'{name}.svg')
    fig.savefig(FIG/f'{name}.png',dpi=600)
    if (FIG/f'{name}.tiff').exists(): fig.savefig(FIG/f'{name}.tiff',dpi=600)
    assert before_hash==sha(FIG/f'{name}_source.csv')
    records.append({'figure':name,'rows':len(original),'plotted_numeric_rows_equivalent':True,
                    'source_sha256':before_hash,'source_bytes_unchanged':True})
    plt.close(fig)
    print(name,'rendered; all',len(original),'source rows unchanged',flush=True)

class DisplayOnly(ast.NodeTransformer):
    def visit_Expr(self,node):
        call=node.value
        if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and call.func.attr=='text':
            if any(isinstance(a,ast.Constant) and isinstance(a.value,str) and a.value=='soft-clDice: selected NA' for a in call.args):
                return None
        return self.generic_visit(node)
    def visit_If(self,node):
        self.generic_visit(node)
        if not node.body:node.body=[ast.Pass()]
        return node
    def visit_Call(self,node):
        self.generic_visit(node)
        if isinstance(node.func,ast.Attribute) and node.func.attr in ['text']:
            for arg in node.args:
                if isinstance(arg,ast.Constant) and isinstance(arg.value,str):
                    arg.value=arg.value.replace('gray = selected NA','gray = infeasible')
                    if arg.value in ['NA','FAIL_NA']:arg.value='—'
                elif isinstance(arg,ast.IfExp):
                    for val in [arg.body,arg.orelse]:
                        if isinstance(val,ast.Constant) and val.value in ['NA','FAIL_NA']:val.value='—'
        return node

def load_functions(path):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name not in ['finish','labels']]
    tree=DisplayOnly().visit(ast.Module(body=functions,type_ignores=[]))
    ast.fix_missing_locations(tree)
    exec(compile(tree,str(path),'exec'),globals())

def edit_captions():
    changes={
      '1':('Selected soft-clDice comparisons remain INFEASIBLE_NA.','A dash denotes a soft-clDice comparison with no feasible validation threshold.'),
      '2':('Selected soft-clDice runs are NA.','soft-clDice has no feasible validation threshold, so its selected points are absent.'),
      '4':('Infeasible selected soft-clDice modes remain NA and fixed-0.5 remains separate.','A dash denotes no feasible validation threshold for soft-clDice; fixed-0.5 remains separate.'),
      '5':('Selected soft-clDice remains INFEASIBLE_NA; zero-success numeric modes have FAIL_NA length.','A dash denotes no feasible validation threshold for soft-clDice.'),
      'S4':('selected soft-clDice remains NA.','soft-clDice has no feasible validation threshold, so its selected curves are absent.'),
      'S5':('soft-clDice selected remains infeasible NA; any numeric-mode zero-success length is explicitly FAIL_NA.','A dash in panel a denotes undefined conditional length error because soft-clDice has no successful matches; panels b,c retain the observed zero counts.'),
      'S6':('Selected soft-clDice remains NA.','A dash denotes no feasible validation threshold for soft-clDice.'),
      'S9':('Selected soft-clDice is explicitly NA.','soft-clDice has no feasible validation threshold, so its selected points are absent.')}
    for md in [OUT/'filament_structure_revised.md',OUT/'SUPPLEMENTARY_MATERIAL.md']:
        lines=md.read_text(encoding='utf-8').splitlines()
        for i,line in enumerate(lines):
            hit=re.match(r'^Figure (S?\d+)\.',line)
            if not hit:continue
            n=hit.group(1)
            if n in changes:
                a,b=changes[n];assert a in line,(n,a);line=line.replace(a,b)
            if n in ['S2','S3']:
                a='gray denotes the three infeasible soft-clDice selected runs, not zero differences.'
                b='gray cells denote no feasible validation threshold for the three soft-clDice runs.'
                assert a in line;line=line.replace(a,b)
            if n in ['S7','S8']:
                line=line.replace('gray cells are selected infeasible NA, not zero.','gray cells denote no feasible validation threshold for soft-clDice.')
            if n in ['5','S5']:
                line=line.replace('Failed lengths remain FAIL_NA.','Length errors are defined only for successful matches.')
                line=line.replace('Failed lengths remain FAIL_NA and are never zero imputed.','Length errors are defined only for successful matches.')
            lines[i]=line
            (FIG/f'Figure_{n}_caption.md').write_text(line+'\n',encoding='utf-8')
        md.write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':
    snapshot()
    load_functions(ROOT/'project/manuscript_revision_20261002/build_figures_20261001_FROZEN.py')
    load_functions(ROOT/'project/morphology_application_execution_20261002/build_application_figures.py')
    effects();tradeoff();sensitivities('year','Figure_S2');sensitivities('leave_one_temporal_group_out','Figure_S3');radius_plot()
    application_primary();conditional_lengths(MODE[0],'Figure_5');conditional_lengths(MODE[1],'Figure_S5');observation_intervals();coverage_to_application()
    edit_captions()
    (H/'FIGURE_LABEL_RENDER_QA.json').write_text(json.dumps({'status':'PASS_RENDER_AND_SOURCE_DATA_EQUIVALENCE','figures':records,'backend':'Python/Matplotlib','display':'em dash','new_training':False,'new_bootstrap_draws':False},indent=2)+'\n',encoding='utf-8')
