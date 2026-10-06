"""Python/Matplotlib only; complete frozen application and segmentation tables."""
from pathlib import Path
import json,sys
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OUT=HERE.parent/'manuscript_revision_20261002';FIG=OUT/'figures';FIG.mkdir(parents=True,exist_ok=True)
SRC=HERE.parent/'morphology_application_results_20261002'
assert json.loads((HERE/'FULL_APPLICATION_QA.json').read_text())['status'].startswith('PASS_')
sys.path.insert(0,str(ROOT/'external_figure_qa'))
from audit_panel_alignment import require_matplotlib_panel_alignment
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':7,
 'axes.titlesize':8,'axes.labelsize':7,'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42,'svg.fonttype':'none',
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'savefig.dpi':600})
METHODS=['BCE_Dice','SRL','SABR','ResidualPixel','SRL_FG','FlatUNet_BCE','UNet_softDice_clDice']
CONTROL=[m for m in METHODS if m!='SABR'];SEEDS=[20260831,20260901,20260902]
LABEL={'BCE_Dice':'BCE–Dice','SRL':'SRL','SABR':'SABR','ResidualPixel':'ResidualPixel','SRL_FG':'SRL–FG','FlatUNet_BCE':'Flat U-Net','UNet_softDice_clDice':'soft-clDice'}
COLORS=['#667788','#007E87','#A54A36'];MARKERS=['o','s','^'];MC=['#9B9B9B','#8B6AA5','#006B73','#AA7040','#486FA0','#BA6370','#343434']
MODE=['validation_selected','fixed_0.5'];TITLES=['Validation-selected','Fixed 0.5 diagnostic']
run=pd.read_csv(SRC/'per_seed.csv',float_precision='round_trip',low_memory=False);ci=pd.read_csv(SRC/'paired_bootstrap.csv',float_precision='round_trip',low_memory=False);sensitivity=pd.read_csv(SRC/'comparison_sensitivity.csv',float_precision='round_trip',low_memory=False)
segment=pd.read_csv(HERE.parent/'strong_baseline_final_analysis_20261001/region_per_seed.csv',float_precision='round_trip',low_memory=False)
source=[]
seed_handles=[Line2D([],[],linestyle='none',marker=MARKERS[i],color=COLORS[i],label=str(s)) for i,s in enumerate(SEEDS)]
def finish(fig,axes,name,rows,caption,exclude=()):
 for i,a in enumerate(np.ravel(axes)):a.annotate(chr(97+i),xy=(0,1),xycoords='axes fraction',xytext=(-12,14),textcoords='offset points',fontweight='bold',fontsize=9)
 fig.canvas.draw();require_matplotlib_panel_alignment(fig,json_out=str(FIG/(name+'.alignment.json')),overlay_svg=str(FIG/(name+'.alignment.svg')),
  tolerance_pt=1.5,gutter_tolerance_pt=1.5,require_panel_labels=True,strict=True,exclude_axes=exclude)
 fig.savefig(FIG/(name+'.pdf'))
 fig.savefig(FIG/(name+'.svg'))
 fig.savefig(FIG/(name+'.png'),dpi=600)
 fig.savefig(FIG/(name+'.tiff'),dpi=600)
 pd.DataFrame(rows).to_csv(FIG/(name+'_source.csv'),index=False)
 (FIG/(name+'_caption.md')).write_text(caption+'\n',encoding='utf-8');plt.close(fig);print(name,flush=True)
def dot_runs(a,mode,metric,scale=1):
 rows=[]
 for y,m in enumerate(METHODS):
  rr=run[(run.method==m)&(run['mode']==mode)].sort_values('seed');vals=[]
  for i,(_,r) in enumerate(rr.iterrows()):
   v=float(r[metric])*scale;rows.append({**r.to_dict(),'panel_metric':metric,'plotted_value':v})
   if np.isfinite(v):a.scatter(v,y+(i-1)*.16,s=18,marker=MARKERS[i],color=COLORS[i],zorder=3);vals.append(v)
  if len(vals)==3:a.errorbar(np.mean(vals),y,xerr=np.std(vals,ddof=1),fmt='|',ms=9,color='black',lw=.8,capsize=2,zorder=2)
  if not vals:a.text(.6,y,'NA' if rr.status.ne('OK').all() else 'FAIL_NA',transform=a.get_yaxis_transform(),ha='center',color='#777777',fontsize=7)
 a.set_yticks(range(7),[LABEL[m] for m in METHODS]);a.set_ylim(6.6,-.6);a.locator_params(axis='x',nbins=4)
 return rows
def forest(a,mode,unit):
 rows=[];a.axvline(0,color='#999999',lw=.7)
 for y,m in enumerate(CONTROL):
  rr=ci[(ci.control==m)&(ci['mode']==mode)&(ci.unit==unit)].sort_values('seed');vals=[]
  for i,(_,r) in enumerate(rr.iterrows()):
   rows.append(r.to_dict())
   if r.status=='OK':
    v=float(r.mean_difference);a.errorbar(v,y+(i-1)*.16,xerr=np.array([[v-r.ci_lower],[r.ci_upper-v]]),fmt=MARKERS[i],ms=3.6,color=COLORS[i],lw=.8,capsize=1.5,zorder=3);vals.append(v)
  if vals:a.scatter(np.mean(vals),y,marker='|',s=75,color='black',zorder=4)
  else:a.text(.6,y,'NA',transform=a.get_yaxis_transform(),ha='center',color='#777777')
 a.set_yticks(range(6),[LABEL[m] for m in CONTROL]);a.set_ylim(5.6,-.6);a.set_xlabel('SABR minus comparator');a.locator_params(axis='x',nbins=4)
 return rows
def application_primary():
 fig,axes=plt.subplots(2,2,figsize=(7.2047244,5.7480315))
 fig.subplots_adjust(left=.17,right=.98,bottom=.13,top=.91,hspace=.65,wspace=.48);rows=[]
 for j,mode in enumerate(MODE):
  rows+=dot_runs(axes[0,j],mode,'success_rate');axes[0,j].set_title(TITLES[j],pad=12);axes[0,j].set_xlabel('Main-axis extraction success rate')
  rows+=forest(axes[1,j],mode,'temporal_group');axes[1,j].set_title('Paired effect: 13-group bootstrap',pad=12)
  if j:axes[0,j].tick_params(labelleft=False);axes[1,j].tick_params(labelleft=False)
 upper=max(run.success_rate.max()*1.15,.01)
 for a in axes[0]:a.set_xlim(-upper*.04,upper)
 lo=min(ci[ci.unit=='temporal_group'].ci_lower.min(),-.003);hi=max(ci[ci.unit=='temporal_group'].ci_upper.max(),.003);pad=(hi-lo)*.1
 for a in axes[1]:a.set_xlim(lo-pad,hi+pad)
 fig.legend(handles=seed_handles+[Line2D([],[],marker='|',linestyle='none',color='black',label='Three-seed mean')],loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.57,.01))
 finish(fig,axes,'Figure_4',rows,'Figure 4. Complete main-axis application results for 21 frozen models. '
  'Panels a,b show all three seeds; black marks and horizontal bars give the arithmetic mean ± one sample SD of three run means. '
  'Panels c,d show paired SABR-minus-control effects with per-seed 95% percentile intervals from 100,000 original temporal-group bootstrap draws; '
  'black marks are three-seed means without averaged confidence intervals. Each run gives equal weight to 108 observations after copy aggregation, '
  'using 1,368 reference spines in 170 copies only as scoring instances. Higher success and positive differences favor SABR. '
  'The three infeasible selected soft-clDice modes remain NA; fixed-0.5 is diagnostic. The 13-group estimator remains observation weighted. '
  'Intervals are retrospective, exploratory, unadjusted and conditional on the trained models and thresholds; no significance claims or p values are made.')
def conditional_lengths(mode,name):
 fig,axes=plt.subplots(1,3,figsize=(7.2047244,4.1338583));fig.subplots_adjust(left=.17,right=.985,bottom=.24,top=.83,wspace=.43);rows=[]
 for a,metric,title,label,scale in zip(axes,['absolute_relative_error','successful_references','successful_observations'],['Conditional length error','Successful references','Observations with a success'],['Absolute relative error (%)','Count out of 1,368','Count out of 108'],[100,1,1]):
  rows+=dot_runs(a,mode,metric,scale);a.set_title(title,pad=12);a.set_xlabel(label)
  maxv=run[(run['mode']==mode)][metric].max()*scale
  a.set_xlim(-max(maxv,.1)*.04,max(maxv,.1)*1.2)
 for a in axes[1:]:a.tick_params(labelleft=False)
 fig.legend(handles=seed_handles+[Line2D([],[],marker='|',linestyle='none',color='black',label='Mean ± sample SD')],loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.58,.04))
 finish(fig,axes,name,rows,name.replace('_',' ')+'. Conditional projected length errors and yield at '+('selected operating points' if mode==MODE[0] else 'fixed 0.5 diagnostic points')+'. '
  'Every trained seed is retained. Black marks/bars represent the three-seed mean ± sample SD where all three conditional run values exist; '
  'they are not paired population-level length effects. Length error is averaged within successful references of a copy, across successful copies of an observation, '
  'and across observations with successes. Failed lengths remain FAIL_NA and are never zero imputed. Panel b reports successful reference counts out of 1,368, '
  'and panel c reports observations with at least one success out of 108; these yields are distinct from the primary copy/observation-macro success rate. '
  'Different methods have different successful populations, preventing a general claim of better length accuracy. soft-clDice selected remains infeasible NA; '
  'any numeric-mode zero-success length is explicitly FAIL_NA. The annotation reference is two-dimensional, without physical deprojection.')
def observation_intervals():
 fig,axes=plt.subplots(1,2,figsize=(7.2047244,3.9370079));fig.subplots_adjust(left=.17,right=.98,bottom=.24,top=.82,wspace=.43);rows=[]
 for a,mode,title in zip(axes,MODE,TITLES):rows+=forest(a,mode,'observation');a.set_title(title+': observation resampling',pad=12)
 axes[1].tick_params(labelleft=False)
 lo=min(ci[ci.unit=='observation'].ci_lower.min(),-.003);hi=max(ci[ci.unit=='observation'].ci_upper.max(),.003);pad=(hi-lo)*.1
 for a in axes:a.set_xlim(lo-pad,hi+pad)
 fig.legend(handles=seed_handles+[Line2D([],[],marker='|',linestyle='none',color='black',label='Three-seed mean')],loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.57,.04))
 finish(fig,axes,'Figure_S6',rows,'Figure S6. Complete paired application effects with 100,000 observation-bootstrap draws per seed/mode/comparator. '
  'All 108 observations are paired; reference spines are never independently resampled. Positive effects favor SABR. Three-seed means have no averaged interval. '
  'Selected soft-clDice remains NA. Compare Figure 4 for original 13-group resampling; both interval families are exploratory and unadjusted, without p values.')
def temporal(kind,name):
 rr=sensitivity[sensitivity.analysis==kind];columns=sorted(rr.year_or_omitted_group.astype(int).unique())
 fig,axes=plt.subplots(1,2,figsize=(7.2047244,5.1181102));fig.subplots_adjust(left=.205,right=.985,bottom=.27,top=.85,wspace=.2)
 rows=[];mats=[]
 for mode in MODE:
  data=np.full((18,len(columns)),np.nan)
  for i,m in enumerate(CONTROL):
   for j,s in enumerate(SEEDS):
    subset=rr[(rr.control==m)&(rr.seed==s)&(rr['mode']==mode)].set_index('year_or_omitted_group')
    for k,c in enumerate(columns):r=subset.loc[c];data[i*3+j,k]=r.subset_mean_difference;rows.append(r.to_dict())
  mats.append(data)
 extent=max(np.nanmax(np.abs(np.stack(mats))),.0001);cmap=plt.get_cmap('RdBu').copy();cmap.set_bad('#bdbdbd');caxes=[]
 for a,mode,title,data in zip(axes,MODE,TITLES,mats):
  image=a.imshow(np.ma.masked_invalid(data),aspect='auto',interpolation='nearest',cmap=cmap,vmin=-extent,vmax=extent)
  a.set_title(title,pad=12);a.set_yticks(range(18),[LABEL[m]+' / '+str(s)[-3:] for m in CONTROL for s in SEEDS]);a.tick_params(length=0,pad=3)
  labels=[]
  for c in columns:
   q=rr[rr.year_or_omitted_group==c].iloc[0];n=int(q.planned_subset_observations)
   labels.append(str(c)+'\n'+str(n))
  a.set_xticks(range(len(columns)),labels);a.set_xlabel('Year / n observations' if kind=='per_year' else 'Omitted group / remaining n',labelpad=8)
  for v in [2.5,5.5,8.5,11.5,14.5]:a.axhline(v,color='white',lw=.5)
  rect=a.get_position();ca=fig.add_axes([rect.x0,.13,rect.width,.02]);cb=fig.colorbar(image,cax=ca,orientation='horizontal');cb.set_label('SABR minus comparator success rate',labelpad=3);cb.ax.tick_params(labelsize=7);caxes.append(ca)
 axes[1].tick_params(labelleft=False)
 finish(fig,axes,name,rows,name.replace('_',' ')+'. '+('Year-specific' if kind=='per_year' else 'Leave-one-original-temporal-group-out')+' application sensitivity for every comparator, seed and mode. '
  'Rows ending 831, 901 and 902 identify seeds 20260831, 20260901 and 20260902. All numeric slices are shown; gray cells are selected infeasible NA, not zero. '
  'Both panels share one symmetric color scale, with positive values favoring SABR. Each subset retains copy-then-observation averaging. '
  'The labels give year/observation counts or omitted original group/remaining observation counts. The cohort has 108 observations and 13 temporal groups, '
  'including a group of 57 observations and singleton years. These descriptive means introduce no additional bootstrap or significance rule.',exclude=caxes)
def coverage_to_application():
 fig,axes=plt.subplots(1,2,figsize=(7.2047244,4.1338583));fig.subplots_adjust(left=.11,right=.985,bottom=.34,top=.83,wspace=.3);rows=[]
 ymax=max(run.success_rate.max()*1.15,.01)
 for a,mode,title in zip(axes,MODE,TITLES):
  for i,m in enumerate(METHODS):
   for j,s in enumerate(SEEDS):
    app=run[(run.method==m)&(run.seed==s)&(run['mode']==mode)].iloc[0]
    seg=segment[(segment.method==m)&(segment.seed==s)&(segment['mode']==mode)&(segment.radius==3)].iloc[0]
    rows.append({'method':m,'seed':s,'mode':mode,'status':app.status,'msc3':seg.msc,'main_axis_success_rate':app.success_rate})
    if app.status=='OK':a.scatter(seg.msc,app.success_rate,s=24,color=MC[i],marker=MARKERS[j])
  a.set_title(title,pad=12);a.set_xlabel('Manual-spine coverage at 3 native px');a.set_ylabel('Main-axis extraction success rate');a.set_ylim(-ymax*.04,ymax);a.set_xlim(-.025,1.025)
  if mode==MODE[0]:a.text(.5,-.24,'soft-clDice: selected NA',transform=a.transAxes,ha='center',color='#666666')
 handles=[Line2D([],[],linestyle='none',marker='o',color=c,label=LABEL[m]) for m,c in zip(METHODS,MC)]
 fig.legend(handles=handles,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.54,.04));fig.legend(handles=[Line2D([],[],linestyle='none',marker=MARKERS[j],color='black',label=str(s)) for j,s in enumerate(SEEDS)],loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.54,-.002))
 finish(fig,axes,'Figure_S9',rows,'Figure S9. Coverage and main-axis extraction are distinct endpoints. Every point is a trained-seed mean over the same 108 observations, '
  'after within-copy/observation aggregation. Colors encode seven methods and symbols encode all three seeds. Selected soft-clDice is explicitly NA. '
  'The fixed diagnostic retains its foreground-inflated coverage without substituting for selected inference. No curve, correlation test or causal relationship is fitted. '
  'Axis extraction is mask-only and references enter only at matching. Coverage improvements do not by themselves imply successful extraction or physical branch recovery.')
application_primary();conditional_lengths(MODE[0],'Figure_5');conditional_lengths(MODE[1],'Figure_S5');observation_intervals();temporal('per_year','Figure_S7');temporal('leave_one_group_out','Figure_S8');coverage_to_application()
