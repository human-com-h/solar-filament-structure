"""Plot accepted paired effects and frozen temporal-group intervals."""
from pathlib import Path
import csv, json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
H=Path(__file__).resolve().parent
F=H/'manuscript/figures'
sys.path.insert(0,str(H/'manuscript/figure_tools/qa_dependencies'))
sys.path.insert(0,str(H.parents[1]/'external_figure_qa'))
from audit_panel_alignment import require_matplotlib_panel_alignment
from audit_figure_collisions import audit_pdf
plt.rcParams.update({'font.family':'Arial','font.size':7,'axes.titlesize':8,'axes.labelsize':7,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.linewidth':.6,'lines.linewidth':.8,'savefig.facecolor':'white'})
rows=list(csv.DictReader((H/'report_tables/SABR_minus_region_clDice_group_intervals.csv').open(encoding='utf-8')))
source=[x for x in rows if x['metric'] in ['dice','msc','success_rate']]
assert len(source)==18
with (F/'Figure_7_source.csv').open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(source[0]));w.writeheader();w.writerows(source)
fig,axes=plt.subplots(1,3,figsize=(174/25.4,66/25.4))
fig.subplots_adjust(left=.10,right=.985,bottom=.31,top=.85,wspace=.58)
colors=['#0072B2','#D55E00','#009E73']
shapes=['o','s','^']
seeds=['20260831','20260901','20260902']
offsets=[.17,0,-.17]
for j,(ax,metric,title) in enumerate(zip(axes,['dice','msc','success_rate'],['Dice','MSC3','Main-axis success'])):
    for mode,y in [('validation_selected',1),('fixed_0.5',0)]:
        vals=[]
        for i,seed in enumerate(seeds):
            x=next(v for v in source if v['metric']==metric and v['mode']==mode and v['seed']==seed)
            center,lo,hi=[100*float(x[k]) for k in ['mean_difference','ci_low','ci_high']]
            vals.append(center)
            ax.errorbar(center,y+offsets[i],xerr=[[center-lo],[hi-center]],fmt=shapes[i],color=colors[i],markersize=3.5,capsize=1.5,elinewidth=.8,markeredgewidth=.6,zorder=3)
        ax.plot(np.mean(vals),y+.34,marker='D',markersize=3,color='black',zorder=4)
    ax.axvline(0,color='#777777',lw=.7,ls=(0,(3,2)),zorder=0)
    ax.set_ylim(-.40,1.56)
    ax.set_yticks([0,1],['Fixed 0.5','Selected'])
    ax.tick_params(axis='y',length=0,pad=4)
    ax.tick_params(axis='x',length=3,pad=3)
    ax.set_title(title,pad=9)
    ax.text(-.22,1.10,chr(97+j),transform=ax.transAxes,fontsize=9,fontweight='bold',ha='left',va='bottom')
    ax.set_xlabel('SABR − region+clDice (pp)',labelpad=6)
    for spine in ['top','right','left']:ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color('#666666')
axes[0].set_xlim(-2.2,9.3);axes[0].set_xticks([0,4,8])
axes[1].set_xlim(-2.2,14.5);axes[1].set_xticks([0,5,10])
axes[2].set_xlim(-5.2,10.5);axes[2].set_xticks([-5,0,5,10])
handles=[Line2D([],[],marker=shapes[i],color=colors[i],ls='none',markersize=4,label=seed) for i,seed in enumerate(seeds)]
handles.append(Line2D([],[],marker='D',color='black',ls='none',markersize=3.5,label='Three-seed mean'))
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.54,.015),ncol=4,frameon=False,handletextpad=.5,columnspacing=1.8)
fig.canvas.draw()
require_matplotlib_panel_alignment(fig,json_out=F/'Figure_7.alignment.json',overlay_svg=F/'Figure_7.alignment.svg',strict=True)
for ext in ['pdf','svg','png','tiff']:
    fig.savefig(F/('Figure_7.'+ext),dpi=600)
plt.close(fig)
audit=audit_pdf(F/'Figure_7.pdf')
(F/'Figure_7.collisions.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
print('Figure 7 collision audit',audit['verdict'],audit['summary'])
if audit['summary']['fail'] or audit['summary']['warn']:raise RuntimeError(audit['findings'])
print('Figure 7 exported: PDF/SVG/PNG/TIFF; all 18 paired effects and group intervals.')
