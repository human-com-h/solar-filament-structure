"""Lock all 3 validation operating points; never load test images."""
from pathlib import Path
import argparse,csv,json
from datetime import datetime,timezone
from methods import *
from runtime import *
from threshold_curve import THRESHOLDS

def main(roots,output):
    runs=[];code=verify_code()
    for seed in DESIGN['seeds']:
        options=[]
        for root in roots:
            options.extend(p.parent for p in Path(root).rglob('VALIDATION_COMPLETE.json') if p.parent.name==f'{METHODS[0]}_seed{seed}')
        assert len(options)==1,f'{seed}:需唯一完成验证run，找到{len(options)}份'
        run=options[0];status=json.loads((run/'STATUS.json').read_text());assert status['complete'] and not status['smoke']
        sig=json.loads((run/'VALIDATION_COMPLETE.json').read_text());cfg=sig['cfg']
        assert cfg['code_sha256']==code and not cfg['smoke'] and cfg['seed']==seed
        archive=run/'recovery.zip';assert sha(archive)==sig['archive_sha256']
        ck=load_archive(archive);assert ck['cfg']==cfg and ck['initial_model_digest']==json.loads((HERE/'INITIAL_DIGESTS.json').read_text())[str(seed)]
        assert ck['epoch']>=150 or ck['stale']>=20
        curves=run/'validation/curves.csv'
        with curves.open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
        assert len(rows)==len(THRESHOLDS)==39 and all(float(r['threshold'])==float(t) for r,t in zip(rows,THRESHOLDS))
        assert all(int(r['n_observations'])==107 for r in rows)
        floor=DESIGN['operating_point']['precision_floors'][str(seed)]
        candidates=[r for r in rows if float(r['precision'])>=floor and float(r['foreground_fraction'])>0]
        selected=max(candidates,key=lambda r:(float(r['msc']),-float(r['msgr']),float(r['dice']),float(r['threshold']))) if candidates else None
        runs.append({'method':METHODS[0],'seed':seed,'checkpoint_archive':str(archive.resolve()),'archive_sha256':sha(archive),
                     'best_epoch':ck['best_epoch'],'validation_curves_sha256':sha(curves),'precision_floor':floor,
                     'selected_feasible':selected is not None,'selected_threshold':float(selected['threshold']) if selected else None,
                     'selected_validation_metrics':selected,'fixed_threshold':.5,'code_sha256':code})
    dest=Path(output);assert not dest.exists(),'锁定文件已存在；不覆盖事前版本。'
    dump(dest,{'created_utc':datetime.now(timezone.utc).isoformat(),'design_sha256':sha(HERE/'FROZEN_DESIGN.json'),
               'scope':'Retrospective additional-control evaluation; old test already exposed; no claim of independent confirmation.',
               'original_21_models_modified':False,'application_endpoint_modified':False,'test_inference_started':False,'runs':runs})
    print('三个新增模型已统一validation锁定。该操作不运行test。',dest)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--roots',nargs='+',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();main(a.roots,a.output)
