"""Frozen units/RNG, all seeds/modes/controls. Conditional lengths have no imputed intervals."""
from pathlib import Path
import csv,json
import numpy as np
from mask_acceptance import read
from frozen_bootstrap import boot
STAGE=Path(__file__).resolve().parent;ROOT=STAGE.parents[1];APP=ROOT/'project/manuscript_revision_20261001/morphology_application'
LENGTH_FIELDS=['signed_relative_error','absolute_relative_error','absolute_error_px']
CONTROLS=['BCE_Dice','SRL','ResidualPixel','SRL_FG','FlatUNet_BCE','UNet_softDice_clDice']
SEEDS=[20260831,20260901,20260902];MODES=['validation_selected','fixed_0.5']
def table(p,rows):
 assert rows;fields=list(dict.fromkeys(k for r in rows for k in r))
 with Path(p).open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
def clean_number(x):return None if x is None else float(x)
def read_receipt(p):
 from run_application import load_cached
 return load_cached(p)
def analyze(out,manifests,refs,lock,sig):
 out=Path(out);ids=lock['observations'];meta={}
 with (APP/'reference_manifest.csv').open(encoding='utf-8-sig') as f:
  for r in csv.DictReader(f):
   oid=r['physical_observation_id'];v={'year':r['year'],'temporal_group':r['temporal_group']}
   assert oid not in meta or meta[oid]==v;meta[oid]=v
 assert sorted(meta)==ids and len({v['temporal_group'] for v in meta.values()})==13
 perref=[];percopy=[];perobs=[];perrun=[];values={}
 for m in manifests:
  base={k:m[k] for k in ['run','method','seed','mode','status','na_reason','threshold']};obs=[]
  for oid in ids:
   common={**base,'observation':oid,**meta[oid]}
   if m['status']=='OK':
    rec=read_receipt(out/'receipts'/m['run']/m['mode']/(oid+'.json'));assert rec['binding']['signature']==sig
    result=rec['result'];summary=result['per_observation'];assert np.isfinite(summary['success_rate']) and 0<=summary['success_rate']<=1
    assert summary['planned_copies']==summary['evaluable_copies']==len(refs[oid])
    assert summary['eligible_references']==sum(len(c['references']) for c in refs[oid])
    assert len(result['per_reference'])==summary['eligible_references']
    for x in result['per_reference']:
     for field in LENGTH_FIELDS:
      assert (x[field] is None) if x['status']=='FAIL' else np.isfinite(x[field])
     perref.append({**common,'mode_status':base['status'],**x})
    for x in result['per_copy']:percopy.append({**common,**x})
   else:
    summary={'success_rate':None,'planned_copies':len(refs[oid]),'evaluable_copies':0,
     'eligible_references':sum(len(c['references']) for c in refs[oid]),'successful_references':None,
     'successful_copies':None,'length_status':'INFEASIBLE_NA',**{k:None for k in LENGTH_FIELDS}}
    for c in refs[oid]:
     percopy.append({**common,'sample':c['sample'],'eligible_references':len(c['references']),'success_rate':None,
      'successful_references':None,'length_status':'INFEASIBLE_NA',**{k:None for k in LENGTH_FIELDS}})
     for r in c['references']:perref.append({**common,'sample':c['sample'],'annotation_id':r['annotation_id'],
      'mode_status':'INFEASIBLE_NA','status':'INFEASIBLE_NA','reference_length_px':r['length_px'],'length_status':'INFEASIBLE_NA',**{k:None for k in LENGTH_FIELDS}})
   row={**common,**summary};perobs.append(row);obs.append(row)
  ok=m['status']=='OK';lc=[r for r in obs if r['length_status']=='OK']
  run={**base,'planned_observations':108,'evaluable_observations':108 if ok else 0,'planned_annotation_copies':170,'planned_references':1368,
   'success_rate':float(np.mean([r['success_rate'] for r in obs])) if ok else None,
   'successful_references':sum(r['successful_references'] for r in obs) if ok else None,
   'successful_copies':sum(r['successful_copies'] for r in obs) if ok else None,'successful_observations':len(lc) if ok else None,
   'length_status':'OK' if lc else 'FAIL_NA' if ok else 'INFEASIBLE_NA',**{k:float(np.mean([r[k] for r in lc])) if lc else None for k in LENGTH_FIELDS}}
  perrun.append(run);values[m['run'],m['mode']]=np.array([r['success_rate'] for r in obs],float) if ok else None
 assert len(perobs)==42*108 and len(percopy)==42*170 and len(perref)==42*1368
 multiseed=[]
 for method in ['BCE_Dice','SRL','SABR','ResidualPixel','SRL_FG','FlatUNet_BCE','UNet_softDice_clDice']:
  for mode in MODES:
   rows=[r for r in perrun if r['method']==method and r['mode']==mode];assert len(rows)==3 and {r['seed'] for r in rows}==set(SEEDS)
   result={'method':method,'mode':mode,'planned_seeds':3,'numeric_seeds':sum(r['status']=='OK' for r in rows),
    'status':'OK' if all(r['status']=='OK' for r in rows) else 'INFEASIBLE_NA','na_reason':next((r['na_reason'] for r in rows if r['na_reason']),''),
    'length_warning':'Conditional on each model/seed successes; populations differ. Not an overall length-accuracy comparison.'}
   for metric in ['success_rate']+LENGTH_FIELDS:
    v=[r[metric] for r in rows];complete=all(x is not None for x in v)
    result[metric+'_mean']=float(np.mean(v)) if complete else None;result[metric+'_sample_sd']=float(np.std(v,ddof=1)) if complete else None
    result[metric+'_numeric_seeds']=sum(x is not None for x in v)
   multiseed.append(result)
 comparisons=[];sens=[];diffs={}
 for control in CONTROLS:
  for seed in SEEDS:
   for mode in MODES:
    s=values[f'SABR_seed{seed}',mode];c=values[f'{control}_seed{seed}',mode];d=s-c if c is not None else None
    diffs[control,seed,mode]=d;reason='No feasible validation-selected soft-clDice threshold; fixed diagnostic not substituted' if d is None else ''
    base={'comparison':'SABR-minus-'+control,'control':control,'seed':seed,'mode':mode,'metric':'main_axis_success_rate',
     'status':'OK' if d is not None else 'INFEASIBLE_NA','na_reason':reason,'direction':'positive favors SABR',
     'planned_observations':108,'paired_observations':108 if d is not None else 0,'planned_temporal_groups':13,
     'mean_difference':float(d.mean()) if d is not None else None,'rng_seed':seed+71,'bootstrap_draws':100000,
     'interpretation':'Exploratory unadjusted descriptive intervals; not confirmatory significance'}
    for unit in ['observation','temporal_group']:
     ci=boot(d[:,None],ids if unit=='observation' else [meta[o]['temporal_group'] for o in ids],seed+71)[:,0] if d is not None else [None,None]
     comparisons.append({**base,'unit':unit,'ci_lower':clean_number(ci[0]),'ci_upper':clean_number(ci[1])})
    for kind,key in [('per_year','year'),('leave_one_group_out','temporal_group')]:
     for v in sorted({x[key] for x in meta.values()}):
      keep=np.array([meta[o][key]==v if kind=='per_year' else meta[o][key]!=v for o in ids]);n=int(keep.sum())
      sens.append({**base,'analysis':kind,'year_or_omitted_group':v,'planned_subset_observations':n,
       'paired_subset_observations':n if d is not None else 0,'subset_mean_difference':float(d[keep].mean()) if d is not None else None,
       'subset_sabr_success_rate':float(s[keep].mean()) if d is not None else None,'subset_control_success_rate':float(c[keep].mean()) if d is not None else None})
     # Sensitivity means carry no newly introduced bootstrap rules.
 diffsummary=[]
 for control in CONTROLS:
  for mode in MODES:
   v=[None if diffs[control,seed,mode] is None else float(diffs[control,seed,mode].mean()) for seed in SEEDS];ok=all(x is not None for x in v)
   diffsummary.append({'comparison':'SABR-minus-'+control,'control':control,'mode':mode,'status':'OK' if ok else 'INFEASIBLE_NA',
    'planned_seeds':3,'numeric_seeds':sum(x is not None for x in v),'delta_success_mean':float(np.mean(v)) if ok else None,
    'delta_success_sample_sd':float(np.std(v,ddof=1)) if ok else None,**{f'delta_seed{seed}':v[i] for i,seed in enumerate(SEEDS)},
    'shared_backbone_control':control in ['BCE_Dice','SRL','ResidualPixel','SRL_FG']})
 peryear=[];logo=[]
 for r in perrun:
  x=values[r['run'],r['mode']]
  for key,target,kind in [('year',peryear,'per_year'),('temporal_group',logo,'leave_one_group_out')]:
   for v in sorted({m[key] for m in meta.values()}):
    keep=np.array([meta[o][key]==v if kind=='per_year' else meta[o][key]!=v for o in ids]);target.append({
     'run':r['run'],'method':r['method'],'seed':r['seed'],'mode':r['mode'],'status':r['status'],'na_reason':r['na_reason'],
     'analysis':kind,'year_or_omitted_group':v,'planned_observations':int(keep.sum()),'evaluable_observations':int(keep.sum()) if x is not None else 0,
     'success_rate':float(x[keep].mean()) if x is not None else None})
 for name,rows in [('per_reference',perref),('per_copy',percopy),('per_observation',perobs),('per_seed',perrun),('three_seed_summary',multiseed),
  ('paired_bootstrap',comparisons),('comparison_three_seed_summary',diffsummary),('comparison_sensitivity',sens),('per_year',peryear),('leave_one_group_out',logo)]:table(out/(name+'.csv'),rows)
 lines=['# 冻结主轴应用结果','', '本报告来自完整21模型、42预定模式（39可计算、3 selected NA）的 native masks。主结果为108 observation的copy平均成功率；1368条spine仅为评分实例。所有区间为100000次原规则探索性描述，未校正多重性。','',
  '## 共享骨干对照','', '| 对照 | 模式 | 三seed平均成功率差 | 样本SD | 各seed差值 |','|---|---|---:|---:|---|']
 def f(x):return 'NA' if x is None else f'{x:.6f}'
 for r in diffsummary:
  if r['shared_backbone_control']:lines.append(f"| {r['control']} | {r['mode']} | {f(r['delta_success_mean'])} | {f(r['delta_success_sample_sd'])} | {', '.join(f(r[f'delta_seed{s}']) for s in SEEDS)} |")
 lines+=['','## 全部对照与边界','']
 for r in diffsummary:
  vals=[r[f'delta_seed{s}'] for s in SEEDS];pattern='全部seed正差' if all(v is not None and v>0 for v in vals) else '全部seed负差' if all(v is not None and v<0 for v in vals) else 'NA' if all(v is None for v in vals) else 'seed方向混合或零差'
  lines.append(f"- {r['comparison']} / {r['mode']}: {f(r['delta_success_mean'])} ± {f(r['delta_success_sample_sd'])}; {pattern}。")
 lines+=['','同骨干对照可检验额外spine监督是否有应用层面的标注一致性收益；Flat比较同时包含骨干差异。任何覆盖改善均不能自动替代主轴成功。逐seed/年份/剔组不利结果完整见表，主13组中一组57/108，部分年份仅1 observation，不能按正结果择报告。',
  '', '长度误差只在各方法成功样本上按copy→observation聚合，同时报告成功reference/copy/observation数量。FAIL_NA没有填零，没有长度配对区间；方法间成功群体不同，不能据条件均值宣称整体长度精度改善。',
  '', '不支持独立外部验证、三维真实物理分支恢复或因果太阳物理结论。三个soft-clDice selected严格NA，fixed-0.5仅诊断。公开归档及投稿尚未授权。']
 (out/'RESULTS_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
