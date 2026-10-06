"""Read frozen tables only; no model inference, training or network calls."""
from pathlib import Path
import sys, json, shutil, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
FIG=OUT/'figures'; TAB=OUT/'supplement_tables'
FIG.mkdir(exist_ok=True); TAB.mkdir(exist_ok=True)
SRC=ROOT/'project/strong_baseline_final_analysis_20261001'
OLD=SRC/'input_snapshot/old_final_analysis'
sys.path.insert(0,str(ROOT/'external_figure_qa'))
from audit_panel_alignment import require_matplotlib_panel_alignment
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'font.size':7,'axes.titlesize':8,'axes.labelsize':7,'xtick.labelsize':7,'ytick.labelsize':7,
    'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,
    'axes.linewidth':.6,'savefig.dpi':600})
SEEDS=[20260831,20260901,20260902]
COLORS=['#667788','#007E87','#A54A36']; MARKERS=['o','s','^']
METHODS=['BCE_Dice','SRL','SABR','ResidualPixel','SRL_FG','FlatUNet_BCE','UNet_softDice_clDice']
COMP=[m for m in METHODS if m!='SABR']
LABEL={'BCE_Dice':'BCE–Dice','SRL':'SRL','SABR':'SABR','ResidualPixel':'ResidualPixel',
       'SRL_FG':'SRL–FG','FlatUNet_BCE':'Flat U-Net','UNet_softDice_clDice':'soft-clDice'}
region=pd.read_csv(SRC/'region_per_seed.csv')
obs=pd.read_csv(SRC/'region_per_observation.csv')
branch=pd.read_csv(OLD/'branch_per_seed.csv')
for p in [OUT/'historical_sources']:
    p.mkdir(exist_ok=True)
for name in ['filament_structure_working_draft.md','filament_structure_threshold_revision.md']:
    dest=OUT/'historical_sources'/name
    if not dest.exists():shutil.copy2(ROOT/'project'/name,dest)
for name in ['region_per_observation.csv','region_per_seed.csv','region_multiseed.csv',
             'new_paired_effects.csv','new_paired_multiseed.csv','new_paired_intervals.csv',
             'new_sensitivity.csv','accepted_runs.csv','test_observations.csv']:
    shutil.copy2(SRC/name,TAB/name)
for p in OLD.glob('*.csv'):
    shutil.copy2(p,TAB/('legacy_'+p.name))

def lookup(df,method,seed,mode,radius,metric):
    r=df[(df.method==method)&(df.seed==seed)&(df['mode']==mode)&(df.radius==radius)]
    assert len(r)==1,(method,seed,mode,radius)
    return float(r.iloc[0][metric])

def labels(axes):
    for i,a in enumerate(np.ravel(axes)):
        a.annotate(chr(97+i),xy=(0,1),xycoords='axes fraction',xytext=(-12,14),
                   textcoords='offset points',fontweight='bold',fontsize=9)

def finish(fig,axes,name,source,caption,exclude=()):
    labels(axes);fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=str(FIG/f'{name}.alignment.json'),
        overlay_svg=str(FIG/f'{name}.alignment.svg'),tolerance_pt=1.5,gutter_tolerance_pt=1.5,
        require_panel_labels=True,strict=True,exclude_axes=exclude)
    fig.savefig(FIG/f'{name}.pdf')
    fig.savefig(FIG/f'{name}.svg')
    fig.savefig(FIG/f'{name}.png',dpi=600)
    fig.savefig(FIG/f'{name}.tiff',dpi=600)
    pd.DataFrame(source).to_csv(FIG/f'{name}_source.csv',index=False)
    (FIG/f'{name}_caption.md').write_text(caption+'\n',encoding='utf-8')
    plt.close(fig)
    print(name,flush=True)

seed_handles=[Line2D([],[],linestyle='none',marker=MARKERS[j],color=COLORS[j],label=str(s)) for j,s in enumerate(SEEDS)]

def effects():
    fig,axes=plt.subplots(1,3,figsize=(7.2047244,3.9370079),sharey=True)
    fig.subplots_adjust(left=.18,right=.975,bottom=.24,top=.84,wspace=.28)
    source=[]
    for a,metric,title in zip(axes,['dice','msc','precision'],['Region fidelity','Spine coverage at 3 px','Precision']):
        a.axvline(0,color='#999999',lw=.7)
        for y,m in enumerate(COMP):
            ds=[]
            for j,s in enumerate(SEEDS):
                v=lookup(region,'SABR',s,'validation_selected',3,metric)-lookup(region,m,s,'validation_selected',3,metric)
                source.append(dict(comparator=m,seed=s,mode='validation_selected',radius=3,metric=metric,delta=v,
                                   status='INFEASIBLE_NA' if np.isnan(v) else 'OK'))
                if np.isfinite(v):
                    ds.append(v);a.scatter(v,y+(j-1)*.15,s=19,color=COLORS[j],marker=MARKERS[j],zorder=3)
            if ds:a.scatter(np.mean(ds),y,marker='|',s=95,color='black',linewidths=1.4)
            else:a.text(.5,y,'NA',transform=a.get_yaxis_transform(),ha='center',color='#777777')
        a.set_title(title,pad=12);a.set_yticks(range(6));a.set_yticklabels([LABEL[m] for m in COMP]);a.set_ylim(5.6,-.6)
        a.set_xlabel('SABR minus comparator',labelpad=8);a.locator_params(axis='x',nbins=4)
        lo,hi=a.get_xlim();a.set_xlim(min(lo,-.015),max(hi,.015))
    fig.legend(handles=seed_handles+[Line2D([],[],marker='|',linestyle='none',color='black',label='Three-seed mean')],
               loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.57,.02))
    finish(fig,axes,'Figure_1',source,
      'Figure 1. Paired differences at validation-selected operating points on the retrospective temporal-group test. '
      'Each colored symbol represents one trained seed and the black mark is the arithmetic mean of three paired differences, not a confidence interval. '
      'All six SABR comparisons and all three seeds are retained. soft-clDice is INFEASIBLE_NA in all selected comparisons; NA is not zero. '
      'Each run mean gives equal weight to 108 observations after within-copy and within-observation aggregation. '
      'MSC uses native-image radius 3 px. Positive differences favor SABR for each displayed metric. '
      'The plots do not establish universal superiority or a confirmed branch mechanism. No hypothesis test or multiplicity correction is applied.')

def tradeoff():
    fig,axes=plt.subplots(1,2,figsize=(7.2047244,4.1338583))
    fig.subplots_adjust(left=.09,right=.98,bottom=.34,top=.84,wspace=.30)
    mc=['#9B9B9B','#8B6AA5','#006B73','#AA7040','#486FA0','#BA6370','#343434'];source=[]
    for a,mode,title in zip(axes,['validation_selected','fixed_0.5'],['Validation-selected','Fixed 0.5 diagnostic']):
        for i,m in enumerate(METHODS):
            for j,s in enumerate(SEEDS):
                p=lookup(region,m,s,mode,3,'precision');c=lookup(region,m,s,mode,3,'msc')
                source.append(dict(method=m,seed=s,mode=mode,radius=3,precision=p,msc=c,
                                   status='INFEASIBLE_NA' if np.isnan(p) else 'OK'))
                if np.isfinite(p):a.scatter(p,c,color=mc[i],marker=MARKERS[j],s=25,alpha=.95,zorder=4 if m=='SABR' else 3)
        a.set_title(title,pad=12);a.set_xlabel('Precision');a.set_ylabel('Manual-spine coverage at 3 px')
        if mode=='validation_selected':
            a.set_xlim(.59,.79);a.set_ylim(.56,.84)
            a.text(.5,-.25,'soft-clDice: selected NA',transform=a.transAxes,ha='center',color='#666666')
        else:a.set_xlim(-.025,.79);a.set_ylim(.55,1.025)
    handles=[Line2D([],[],marker='o',linestyle='none',color=c,label=LABEL[m]) for m,c in zip(METHODS,mc)]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.52,.04),ncol=4,frameon=False)
    fig.legend(handles=[Line2D([],[],marker=MARKERS[j],linestyle='none',color='black',label=str(s)) for j,s in enumerate(SEEDS)],
               loc='lower center',bbox_to_anchor=(.52,-.002),ncol=3,frameon=False)
    finish(fig,axes,'Figure_2',source,
      'Figure 2. Precision and manual-spine coverage at the two frozen operating modes. All seven methods and three seeds are displayed; '
      'the three selected soft-clDice runs are explicitly NA. Panel ranges differ to display the extreme fixed-0.5 soft-clDice inflation. '
      'Every point is a 108-observation run mean, not an independent biological replicate. Colors identify methods and shapes identify seeds. '
      'These are discrete operating points rather than precision–recall curves or a demonstrated Pareto frontier. '
      'High spine coverage alone can be obtained with poor foreground selectivity. No intervals or hypothesis-test annotations are shown.')

def inflation():
    fig,axes=plt.subplots(1,3,figsize=(7.2047244,3.9370079))
    fig.subplots_adjust(left=.095,right=.98,bottom=.29,top=.84,wspace=.32)
    source=[];shown=['SABR','FlatUNet_BCE','UNet_softDice_clDice']
    for a,metric,title in zip(axes,['foreground_fraction','precision','msc'],['Predicted foreground fraction','Precision','Spine coverage at 3 px']):
        for i,m in enumerate(shown):
            for j,s in enumerate(SEEDS):
                d=obs[(obs.method==m)&(obs.seed==s)&(obs['mode']=='fixed_0.5')&(obs.radius==3)].sort_values('physical_observation_id')
                assert len(d)==108
                vals=d[metric].to_numpy();assert np.isfinite(vals).all()
                x=i*4+j; jitter=(np.arange(108)%11-5)*.035
                a.scatter(x+jitter,vals,s=3,color=COLORS[j],alpha=.42,linewidths=0,zorder=2)
                a.scatter(x,np.mean(vals),color='black',marker='_',s=90,linewidths=1.3,zorder=3)
                for oid,v in zip(d.physical_observation_id,vals):source.append(dict(method=m,seed=s,observation=oid,mode='fixed_0.5',radius=3,metric=metric,value=v))
        a.set_title(title,pad=12);a.set_xticks([1,5,9]);a.set_xticklabels(['SABR','Flat','soft-clDice'])
        a.set_xlim(-.6,10.6);a.set_ylim(-.045,1.045)
    fig.legend(handles=seed_handles+[Line2D([],[],marker='_',color='black',linestyle='none',label='Run mean')],
               loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.54,.03))
    finish(fig,axes,'Figure_3',source,
      'Figure 3. All observation-level fixed-0.5 diagnostics for SABR, Flat U-Net and soft-clDice. '
      'Every colored column contains all 108 observations for one seed; black horizontal marks indicate run means. '
      'Jitter is deterministic and only prevents overplotting. The 324 observations per method are repeated scores of the same 108 observations, '
      'not 324 independent biological samples. Foreground fraction uses native-image area; precision and MSC3 use the frozen evaluator. '
      'soft-clDice combines high coverage with large foreground extent and very low precision. No selected operating point was substituted, '
      'and no downstream axis-extraction outcome is inferred from these diagnostics.')

def branch_plot():
    fig,axes=plt.subplots(1,2,figsize=(7.2047244,3.7401575),sharey=True)
    fig.subplots_adjust(left=.19,right=.975,bottom=.25,top=.84,wspace=.3);source=[]
    for a,mode in zip(axes,['validation_selected','fixed_0.5']):
        a.axvline(0,color='#999999',lw=.7)
        for y,m in enumerate(COMP[:4]):
            ds=[]
            for j,s in enumerate(SEEDS):
                v=lookup(branch,'SABR',s,mode,0,'component_recall')-lookup(branch,m,s,mode,0,'component_recall')
                ds.append(v);a.scatter(v,y+(j-1)*.15,s=22,color=COLORS[j],marker=MARKERS[j])
                source.append(dict(comparator=m,seed=s,mode=mode,radius_grid=0,metric='component_recall',delta=v))
            a.scatter(np.mean(ds),y,marker='|',s=95,color='black')
        a.set_ylim(3.5,-.5);a.set_yticks(range(4));a.set_yticklabels([LABEL[m] for m in COMP[:4]])
        a.set_title('Validation-selected' if mode=='validation_selected' else 'Fixed 0.5 diagnostic',pad=12)
        a.set_xlabel('SABR minus comparator')
    fig.legend(handles=seed_handles,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.56,.04))
    finish(fig,axes,'Figure_S1',source,
      'Figure S1. Annotation-derived residual-component recall differences from the preserved five-method experiment. '
      'All four comparators and all three seeds are shown; black marks are three-seed means. The frozen proxy cohort comprises '
      '101 eligible observations, 157 annotation copies, 1,934 residual components and 17,019 pixels in 11 temporal groups. '
      'Radius is 0 at the 1024 training grid, not native-image radius 3. Positive values favor SABR. '
      'Selected SABR–SRL differences are negative for all seeds, whereas fixed-0.5 differences have a different pattern. '
      'No new-baseline branch evaluation was performed. This mask/spine-derived proxy does not validate physical solar barbs.')

def sensitivities(scope,name):
    old=pd.read_csv(OLD/'region_sensitivity.csv');old=old[old.scope==scope].copy();old['mode']='validation_selected';old['radius']=3
    new=pd.read_csv(SRC/'new_sensitivity.csv');new=new[(new.scope==scope)&(new['mode']=='validation_selected')&(new.radius==3)]
    data=pd.concat([old,new],ignore_index=True);source=[]
    meta=pd.read_csv(SRC/'test_observations.csv')
    key='year' if scope=='year' else 'temporal_group'
    counts=meta.groupby(key).size();labs=sorted(counts.index.astype(int))
    fig,axes=plt.subplots(1,3,figsize=(7.2047244,6.1023622),sharey=True)
    fig.subplots_adjust(left=.19,right=.98,bottom=.27,top=.88,wspace=.24)
    extra=[]
    for panel,(a,metric,title) in enumerate(zip(axes,['dice','msc','precision'],['Dice difference','MSC3 difference','Precision difference'])):
        matrix=np.full((18,len(labs)),np.nan)
        for i,m in enumerate(COMP):
            for j,s in enumerate(SEEDS):
                for k,l in enumerate(labs):
                    r=data[(data.comparison=='SABR-'+m)&(data.seed==s)&(data.metric==metric)&(data.label.astype(int)==l)]
                    assert len(r)==1,(scope,m,s,l,metric)
                    v=float(r.iloc[0].mean_delta);matrix[i*3+j,k]=v
                    source.append(dict(comparator=m,seed=s,scope=scope,label=l,metric=metric,mode='validation_selected',radius=3,delta=v,
                                       n_observations=int(r.iloc[0].n_observations),status='INFEASIBLE_NA' if np.isnan(v) else 'OK'))
        bound=max(.02,float(np.nanmax(np.abs(matrix))))
        cmap=plt.get_cmap('BrBG').copy();cmap.set_bad('#D9D9D9')
        im=a.imshow(matrix,aspect='auto',cmap=cmap,vmin=-bound,vmax=bound,interpolation='none',origin='upper')
        a.set_yticks(range(18));a.set_yticklabels([LABEL[m]+f' / {j+1}' for m in COMP for j in range(3)])
        a.set_xticks(range(len(labs)))
        if scope=='year':tick=[f'{l} ({counts.loc[l]})' for l in labs]
        else:tick=[str(l) for l in labs]
        a.set_xticklabels(tick,rotation=90,rotation_mode='anchor',ha='right',va='center')
        a.tick_params(axis='both',length=0,pad=5)
        a.set_title(title,pad=12);a.set_xlabel('Year (observation n)' if scope=='year' else 'Omitted temporal group',labelpad=9)
        for y in np.arange(2.5,17,3):a.axhline(y,color='white',lw=1)
        pos=a.get_position();cax=fig.add_axes([pos.x0,.12,pos.width,.018]);extra.append(cax)
        cb=fig.colorbar(im,cax=cax,orientation='horizontal');cb.locator=matplotlib.ticker.MaxNLocator(3);cb.update_ticks()
    fig.text(.57,.035,'SABR minus comparator; rows /1, /2, /3 are the three seeds; gray = selected NA',ha='center',fontsize=7)
    finish(fig,axes,name,source,
      f'{name.replace("_"," ")}. '+('Per-year' if scope=='year' else 'Leave-one-frozen-temporal-group-out')+
      ' selected-operating-point sensitivity of all six SABR comparisons and three seeds. '
      'Rows within each comparator follow seeds 20260831, 20260901 and 20260902. Positive values favor SABR for all three displayed metrics. '
      'Each panel has its own symmetric color scale; gray denotes the three infeasible soft-clDice selected runs, not zero differences. '
      'Year sample sizes are stated, including singleton years. Group deletion preserves the observation-weighted estimand and uses all 13 original groups. '
      'These descriptive views contain no bootstrap intervals, multiplicity correction or independent validation claim.',extra)

def radius_plot():
    fig,axes=plt.subplots(2,2,figsize=(7.2047244,5.3149606))
    fig.subplots_adjust(left=.10,right=.98,bottom=.22,top=.91,wspace=.30,hspace=.57);source=[]
    for row,mode in enumerate(['validation_selected','fixed_0.5']):
        for col,metric in enumerate(['msc','msgr']):
            a=axes[row,col];a.axhline(0,color='#999999',lw=.7)
            for i,m in enumerate(COMP[4:]):
                for j,s in enumerate(SEEDS):
                    vals=[]
                    for r in [1,3,5]:
                        v=lookup(region,'SABR',s,mode,r,metric)-lookup(region,m,s,mode,r,metric);vals.append(v)
                        source.append(dict(comparator=m,seed=s,mode=mode,radius=r,metric=metric,delta=v,
                                           status='INFEASIBLE_NA' if np.isnan(v) else 'OK'))
                    if np.isfinite(vals).all():a.plot([1,3,5],vals,color=COLORS[j],marker=MARKERS[j],ls='-' if i==0 else '--',lw=.8,ms=3)
            a.set_xticks([1,3,5]);a.set_xlim(.7,5.3);a.set_xlabel('Native-image radius (px)')
            a.set_title(('Selected' if row==0 else 'Fixed 0.5')+' / '+('MSC' if col==0 else 'MSGR'),pad=12)
            a.set_ylabel('SABR minus comparator')
            if row==0:a.text(.5,-.38,'soft-clDice: selected NA',transform=a.transAxes,ha='center',color='#777777')
    fig.legend(handles=seed_handles+[Line2D([],[],color='black',ls='-',label='Flat'),Line2D([],[],color='black',ls='--',label='soft-clDice')],
               loc='lower center',ncol=5,frameon=False,bbox_to_anchor=(.54,.02))
    finish(fig,axes,'Figure_S4',source,
      'Figure S4. Predefined native-radius sensitivity for the two added baselines. '
      'All three seeds are shown at radii 1, 3 and 5 using frozen selected and fixed-0.5 thresholds; selected soft-clDice remains NA. '
      'Each point is the difference between 108-observation run means. Larger MSC and smaller MSGR favor SABR, so favorable MSGR differences are negative. '
      'Lines connect only the predefined radius settings and do not denote fitted trends or uncertainty. No test-threshold retuning occurred.')

if __name__=='__main__':
    effects();tradeoff();inflation();branch_plot();sensitivities('year','Figure_S2');sensitivities('leave_one_temporal_group_out','Figure_S3');radius_plot()
