"""Consolidate existing CPU score tables, preserving modes, NA and denominators."""
from pathlib import Path
from collections import defaultdict, Counter
import csv, hashlib, json, math
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
REV=ROOT/'project'
SEEDS=[20260831,20260901,20260902]

def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))

def write(name, data):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)

def number(s):
    try:
        x=float(s)
        return x if math.isfinite(x) else None
    except (ValueError,TypeError): return None

def main():
    manifest=rows(REV/'mechanism_training/temporal_manifest.csv')
    obsmeta={r['physical_observation_id']:r for r in manifest if r['split']=='internal_test'}
    eligible=rows(OUT/'target_quality_per_copy.csv')
    branchobs={r['observation_id'] for r in eligible if r['split']=='internal_test' and int(r['residual_valid'])}
    groups=[]
    for group in sorted({int(r['temporal_group']) for r in obsmeta.values()}):
        oo=[oid for oid,r in obsmeta.items() if int(r['temporal_group'])==group]
        cc=[r for r in eligible if r['split']=='internal_test' and int(r['temporal_group'])==group]
        groups.append(dict(temporal_group=group,observations=len(oo),observation_fraction=len(oo)/108,
            annotation_copies=len(cc),annotated_reference_spines=sum(int(r['annotated_instances']) for r in cc),
            residual_eligible_observations=sum(oid in branchobs for oid in oo),
            residual_eligible_copies=sum(int(r['residual_valid']) for r in cc),
            first_date=min(oid[:8] for oid in oo),last_date=max(oid[:8] for oid in oo),
            years='/'.join(sorted({str(obsmeta[oid]['year']) for oid in oo}))))
    write('temporal_group_inventory.csv',groups)
    source_paths=[REV/'strong_baseline_final_analysis_20261001/region_per_observation.csv',
                  REV/'region_clDice_analysis_20261004/test_analysis/segmentation_per_observation.csv',
                  REV/'mechanism_final_analysis/branch_per_observation.csv',
                  REV/'morphology_application_results_20261002/per_observation.csv',
                  REV/'region_clDice_analysis_20261004/test_analysis/application/per_observation.csv']
    buckets=defaultdict(dict)
    def add(endpoint,rr,metrics,oidfield):
        for r in rr:
            oid=r[oidfield]
            assert oid in obsmeta
            radius=int(r['radius']) if 'radius' in r else 3
            for metric in metrics:
                key=(endpoint,r['method'],int(r['seed']),r['mode'],radius,metric)
                assert oid not in buckets[key],(key,oid)
                buckets[key][oid]=number(r.get(metric))
    for p in source_paths[:2]:
        add('segmentation',rows(p),['dice','iou','precision','recall','cldice','msc','msgr'],'physical_observation_id')
    add('residual_component',rows(source_paths[2]),['component_recall','pixel_recall'],'physical_observation_id')
    for p in source_paths[3:]:
        add('application',rows(p),['success_rate'],'observation')
    print('Source method/mode coverage:',json.dumps({e:sorted({k[1] for k in buckets if k[0]==e}) for e in ['segmentation','residual_component','application']}),flush=True)
    assert {k[1] for k in buckets if k[0]=='application'} == {'BCE_Dice','SRL','SABR','ResidualPixel','SRL_FG','FlatUNet_BCE','UNet_softDice_clDice','UNet_region_clDice'}
    sensitive=[]
    for key,rr in sorted(buckets.items()):
        endpoint,method,seed,mode,radius,metric=key
        if method=='SABR': continue
        sabr=buckets[(endpoint,'SABR',seed,mode,radius,metric)]
        planned=branchobs if endpoint=='residual_component' else set(obsmeta)
        assert set(rr)==set(sabr)==planned,(key,len(rr),len(sabr),len(planned))
        subsets=[('full_cohort','all',planned)]
        subsets += [('per_year',str(y),{oid for oid in planned if int(obsmeta[oid]['year'])==y})
                    for y in sorted({int(obsmeta[oid]['year']) for oid in planned})]
        subsets += [('leave_one_group_out',str(g),{oid for oid in planned if int(obsmeta[oid]['temporal_group'])!=g})
                    for g in sorted({int(obsmeta[oid]['temporal_group']) for oid in planned})]
        for analysis,label,subset in subsets:
            valid=[oid for oid in sorted(subset) if rr[oid] is not None and sabr[oid] is not None]
            deltas=[sabr[oid]-rr[oid] for oid in valid]
            sensitive.append(dict(endpoint=endpoint,comparison='SABR-minus-'+method,control=method,seed=seed,mode=mode,
                radius=radius,metric=metric,analysis=analysis,subset_label=label,
                planned_observations=len(subset),paired_observations=len(valid),
                planned_temporal_groups=len({obsmeta[oid]['temporal_group'] for oid in subset}),
                status='OK' if valid else 'INFEASIBLE_NA',
                mean_difference=math.fsum(deltas)/len(deltas) if deltas else '',
                interpretation='Exploratory descriptive mean; fixed existing models and thresholds; no new bootstrap'))
    write('temporal_sensitivity_all_endpoints.csv',sensitive)
    focus=[r for r in sensitive if (r['endpoint']=='application' or
            (r['endpoint']=='segmentation' and r['radius']==3 and r['metric'] in ['dice','msc']) or
            (r['endpoint']=='residual_component' and r['radius']==0 and r['metric']=='component_recall'))
            and (r['analysis']=='full_cohort' or (r['analysis']=='leave_one_group_out' and r['subset_label']=='9'))]
    write('temporal_sensitivity_focus.csv',focus)
    # Cross-check every preserved original application sensitivity and every later-control sensitivity.
    lookup={(r['endpoint'],r['comparison'],r['seed'],r['mode'],r['radius'],r['metric'],r['analysis'],r['subset_label']):r for r in sensitive}
    parity=Counter()
    old=REV/'morphology_application_results_20261002/comparison_sensitivity.csv'
    for r in rows(old):
        k=('application',r['comparison'],int(r['seed']),r['mode'],3,'success_rate',r['analysis'],r['year_or_omitted_group'])
        dst=lookup[k]; src=number(r['subset_mean_difference']); got=number(dst['mean_difference'])
        assert src is None and got is None or src is not None and got is not None and abs(src-got)<1e-12,k
        assert int(r['planned_subset_observations'])==dst['planned_observations']
        parity['original_application_rows']+=1
    later=REV/'region_clDice_analysis_20261004/test_analysis/sensitivity.csv'
    for r in rows(later):
        if r['comparison']!='UNet_region_clDice-minus-SABR':continue
        metric=r['metric'];radius=int(r['radius']) if r['radius'] else 3
        k=(r['endpoint'],'SABR-minus-UNet_region_clDice',int(r['seed']),r['mode'],radius,metric,r['analysis'],r['year_or_omitted_group'])
        dst=lookup[k];src=number(r['mean_difference']);got=number(dst['mean_difference'])
        assert src is not None and got is not None and abs(src+got)<1e-12,k
        parity['later_control_rows']+=1
    for endpoint,path,radius in [('segmentation',REV/'mechanism_final_analysis/region_sensitivity.csv',3),
                                 ('residual_component',REV/'mechanism_final_analysis/branch_sensitivity.csv',0)]:
        for r in rows(path):
            if r['metric'] not in (['dice','iou','precision','recall','cldice','msc','msgr'] if endpoint=='segmentation' else ['component_recall','pixel_recall']):
                continue
            analysis={'year':'per_year','leave_one_temporal_group_out':'leave_one_group_out'}[r['scope']]
            k=(endpoint,r['comparison'].replace('SABR-','SABR-minus-',1),int(r['seed']),'validation_selected',radius,r['metric'],analysis,r['label'])
            dst=lookup[k];src=number(r['mean_delta']);got=number(dst['mean_difference'])
            assert src is not None and got is not None and abs(src-got)<1e-12,(k,src,got)
            assert int(r['n_observations'])==dst['paired_observations']
            parity['original_'+endpoint+'_rows']+=1
        source_paths.append(path)
    perseed=rows(REV/'region_clDice_analysis_20261004/report_tables/all_eight_methods_application_per_seed.csv')
    sabrsel=[r for r in perseed if r['method']=='SABR' and r['mode']=='validation_selected']
    assert len(sabrsel)==3
    copy_counts=Counter(r['physical_observation_id'] for r in manifest if r['split']=='internal_test')
    multico=len([x for x in copy_counts.values() if x>1])
    multi_copies=sum(x for x in copy_counts.values() if x>1)
    pair_count=sum(x*(x-1)//2 for x in copy_counts.values() if x>1)
    assert (multico,pair_count)==(40,84)
    denominators=[
        dict(endpoint='regional_and_spine_metrics',planned_observations=108,planned_copies=170,scoring_instances=1368,
             temporal_groups=13,aggregation='Pixels for regional scores; spines within copy for MSC/MSGR; copies within observation; equal observation weights',
             eligibility='All test observations and copies; no test QC exclusions',conditional='No'),
        dict(endpoint='residual_component_diagnostic',planned_observations=101,planned_copies=157,scoring_instances=1934,
             temporal_groups=11,aggregation='Component or residual-pixel recall within valid copy; copies within eligible observation; equal observation weights',
             eligibility='Only copies with retained residual targets; 17019 total residual pixels on 1024 grid',conditional='Target eligibility'),
        dict(endpoint='main_axis_success',planned_observations=108,planned_copies=170,scoring_instances=1368,
             temporal_groups=13,aggregation='Success/reference fraction within copy; copies within observation; equal observation weights',
             eligibility='All unmatched eligible references remain failures; selected infeasibility is NA',conditional='No'),
        dict(endpoint='reference_mask_diagnostic',planned_observations=108,planned_copies=170,scoring_instances=1368,
             temporal_groups=13,aggregation='Same reference/copy/observation hierarchy as main-axis success',
             eligibility='846 reference matches; lengths in 161 copies and 105 observations',conditional='Length only'),
        dict(endpoint='copy_pair_agreement',planned_observations=40,planned_copies=multi_copies,scoring_instances=84,
             temporal_groups='separate multi-copy subset',aggregation='2M/(Nleft+Nright) per unordered pair; pairs within observation; observations',
             eligibility='84 unordered pairs; 109 matched pair instances; local annotator identities unverified',conditional='Multiple-copy observations only')]
    for r in sorted(sabrsel,key=lambda x:int(x['seed'])):
        denominators.append(dict(endpoint='SABR_selected_conditional_length_seed'+r['seed'],
            planned_observations=int(r['successful_observations']),planned_copies=int(r['successful_copies']),
            scoring_instances=int(r['successful_references']),temporal_groups='successful population varies',
            aggregation='Successful references within successful copy; successful copies within observation; observations with success',
            eligibility='Model/mode/seed specific; FAIL_NA not zero; common-success estimates retain selected eligibility',conditional='Yes'))
    write('evaluation_denominators.csv',denominators)
    provenance={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths+[old,later]}
    (OUT/'CONSOLIDATION_QA.json').write_text(json.dumps(dict(source_sha256=provenance,
        sensitivity_rows=len(sensitive),focus_rows=len(focus),temporal_groups=groups,parity=dict(parity),
        source_score_tables_unchanged=True,new_bootstrap_runs=0,new_gpu_training_runs=0),indent=2),encoding='utf-8')
    print('Source sensitivity parity:',dict(parity),flush=True)
    print('Sensitivity rows:',len(sensitive),'focus rows:',len(focus),flush=True)
    print('SABR selected successful references/copies/observations:',[(r['seed'],r['successful_references'],r['successful_copies'],r['successful_observations']) for r in sabrsel],flush=True)

if __name__=='__main__':main()
