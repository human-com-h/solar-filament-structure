"""Public commands. Dependencies are imported only for the requested operation."""
import argparse,json,sys
from pathlib import Path
from .io import REPO,read

METHODS=('BCE_Dice','SRL','SABR','ResidualPixel','SRL_FG','FlatUNet_BCE','UNet_softDice_clDice','UNet_region_clDice')
COMMANDS={
 'check-data':('data','check'), 'build-targets':('data','targets'),
 'stage-training':('training','stage'), 'prepare-cache':('training','prepare_cache'), 'train-preflight':('training','preflight'), 'train':('training','train'),
 'infer':('inference','infer'), 'select-threshold':('inference','thresholds'),
 'score-segmentation':('evaluation','segmentation'), 'score-axes':('evaluation','axes'), 'score-components':('evaluation','components'),
 'replay':('analysis','replay'), 'statistics':('analysis','statistics'), 's11':('analysis','s11'), 'diagnostics':('analysis','diagnostics'),
 'plot':('plotting','figures'), 'validate':('validation','validate'), 'verify-release':('validation','verify_release')}

def parser():
    p=argparse.ArgumentParser(description='Solar filament segmentation and main-axis extraction')
    subs=p.add_subparsers(dest='command',required=True)
    for cmd in COMMANDS:
        q=subs.add_parser(cmd)
        q.add_argument('--config',type=Path,help='JSON root configuration; relative paths resolve against this file')
        for key in ('data-root','weights-root','evidence-root','output-root'):q.add_argument('--'+key,type=Path)
        q.add_argument('--annotation',default='train/MAGFiLO_1.0_Annotations_kaggle2026_train.json')
        q.add_argument('--images',default='train/train_images')
        if cmd=='check-data':q.add_argument('--annotation-only',action='store_true')
        if cmd=='build-targets':
            q.add_argument('--splits',nargs='+',choices=['train','validation','internal_test'],default=['train','validation','internal_test']);q.add_argument('--sample-id')
        if cmd in ('stage-training','prepare-cache','train-preflight','train'):
            q.add_argument('--method',choices=METHODS,required=True);q.add_argument('--seed',type=int,choices=[20260831,20260901,20260902],default=20260831);q.add_argument('--cache',type=Path);q.add_argument('--gate',type=Path)
        if cmd=='train-preflight':q.add_argument('--resource-record',type=Path)
        if cmd in ('infer','select-threshold','score-components'):
            q.add_argument('--run',required=True);q.add_argument('--device',default='cuda:0')
        if cmd=='infer':
            q.add_argument('--mode',choices=['fixed_0.5','validation_selected'],default='validation_selected');q.add_argument('--split',choices=['validation','internal_test'],default='internal_test')
        if cmd in ('score-segmentation','score-axes'):q.add_argument('--mask-manifest',type=Path,required=True)
        if cmd=='score-components':q.add_argument('--targets-root',type=Path,required=True)
        if cmd=='replay':q.add_argument('--models',type=int,choices=[21,24],required=True)
        if cmd=='statistics':
            q.add_argument('--arm',choices=['original21','region24','segmentation21','mechanism-region','mechanism-components'],required=True)
            q.add_argument('--receipts-root',type=Path,help='Original five-method receipt root, containing mechanism_locked_test and mechanism_branch_test')
        if cmd=='s11':
            q.add_argument('--action',choices=['census','consolidate'],required=True);q.add_argument('--labels-root',type=Path,help='170 frozen test NPZ label files, supplied separately')
        if cmd=='diagnostics':
            q.add_argument('--action',choices=['reference','full'],default='reference');q.add_argument('--masks-root',type=Path,help='Original mask_export tree for the nine focal selected runs');q.add_argument('--diagnostic-protocol',type=Path,help='Original checksum-defined calibration PROTOCOL_CN.md')
        if cmd=='plot':
            q.add_argument('--family',choices=['segmentation','application','control','cases'],required=True);q.add_argument('--qa-tools-root',type=Path,required=True)
            q.add_argument('--masks-root',type=Path);q.add_argument('--diagnostics-root',type=Path);q.add_argument('--selection-protocol',type=Path)
        if cmd=='validate':q.add_argument('--with-models',action='store_true')
    return p
def main(argv=None):
    sys.dont_write_bytecode=True
    p=parser();a=p.parse_args(argv)
    if a.config:
        cfg=read(a.config)
        for key in ('data_root','weights_root','evidence_root','output_root'):
            if getattr(a,key) is None and cfg.get(key):
                path=Path(cfg[key]);setattr(a,key,path if path.is_absolute() else a.config.resolve().parent/path)
    needed=[]
    if a.command!='verify-release':needed+=['output_root']
    if a.command not in ('validate','verify-release'):needed+=['evidence_root']
    if a.command in ('check-data','build-targets','prepare-cache','train-preflight','train','infer','select-threshold','score-segmentation','score-components','diagnostics') or (a.command=='s11' and a.action=='census') or (a.command=='plot' and a.family=='cases'):needed+=['data_root']
    if a.command in ('infer','select-threshold','score-components'):needed+=['weights_root']
    if a.command in ('train','prepare-cache','train-preflight'):needed+=['cache']
    for key in needed:
        if getattr(a,key,None) is None:p.error('Missing --'+key.replace('_','-')+'; use explicit roots or --config.')
    try:
        from .frozen import verify_sources
        verify_sources()
        from importlib import import_module
        name,fn=COMMANDS[a.command];result=getattr(import_module('filament_structure.'+name),fn)(a)
        if isinstance(result,Path):result=dict(status='STAGED',path=str(result))
        print(json.dumps(result,indent=2,ensure_ascii=False,default=str))
    except (FileNotFoundError,FileExistsError,ValueError,RuntimeError,ImportError) as e:
        print(f'{a.command}: {e}',file=sys.stderr);return_code=2
        raise SystemExit(return_code)
if __name__=='__main__':main()
