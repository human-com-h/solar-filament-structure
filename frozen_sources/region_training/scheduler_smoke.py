"""Selected seed, completed/export-only recovery and the shared session clock."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import argparse, json, tempfile, time, subprocess, sys
import kaggle_entry as entry
from methods import DESIGN
from runtime import dump, verify_code, run_process


def scenario(root, seed, receipt, spent=0):
    calls = []
    a = SimpleNamespace(seed=seed, input='SYNTHETIC', output=str(root), session_started=time.time()-spent)
    def launch(action, args, annotation, images, cache, out, deadline):
        assert args.seed == seed
        calls.append((action, args.seed))
        run = out/f'{entry.METHOD}_seed{seed}'
        run.mkdir(parents=True,exist_ok=True)
        if action == 'train':
            dump(run/'STATUS.json', {'complete':True})
        else:
            dump(run/'VALIDATION_COMPLETE.json', {'purpose':'controller test only'})
    with patch.object(entry.torch.cuda,'device_count',return_value=2), \
         patch.object(entry.torch.cuda,'is_available',return_value=False), \
         patch.object(entry.torch.cuda,'get_device_name',return_value='Tesla T4'), \
         patch.object(entry.psutil,'virtual_memory',return_value=SimpleNamespace(available=8*1024**3)), \
         patch.object(entry,'disk_check'), patch.object(entry,'discover',return_value=('annotation','images')), \
         patch.object(entry,'prepare'), patch.object(entry,'embedded_checks',return_value={'status':'CPU_MOCK'}), \
         patch.object(entry,'restore',return_value={str(seed):receipt}), \
         patch.object(entry,'distributed_call',side_effect=launch):
        result=entry.run_session(a)
    assert result['runs']==[(entry.METHOD,seed)] and result['selected_seed']==seed
    assert result['execution_deadline_unix']==a.session_started+42300
    assert result['platform_limit_seconds']==43200 and result['final_collection_reserve_seconds']==900
    return calls, result


def main(output):
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    checks={}
    try:
        run_process([sys.executable,'-c','import time; time.sleep(30)'],timeout=.2)
    except subprocess.TimeoutExpired:
        checks['timed_out_owned_process_is_reaped_before_return']=True
    else:
        raise AssertionError('Process timeout did not fire')
    with tempfile.TemporaryDirectory(prefix='single_seed_',dir=root) as temporary:
        temp=Path(temporary)
        for seed in DESIGN['seeds']:
            calls,result=scenario(temp/str(seed),seed,{'training_complete':False,'validation_complete':False})
            assert calls==[('train',seed),('export',seed)] and result['complete']
        checks['only_configured_seed_runs_on_both_GPUs']=True
        calls,_=scenario(temp/'export_only',DESIGN['seeds'][0],{'training_complete':True,'validation_complete':False})
        assert calls==[('export',DESIGN['seeds'][0])]
        checks['completed_training_without_STATUS_recovery_exports_only']=True
        calls,_=scenario(temp/'completed',DESIGN['seeds'][0],{'training_complete':True,'validation_complete':True})
        assert not calls
        checks['completed_validation_is_skipped']=True
        calls,result=scenario(temp/'spent',DESIGN['seeds'][0],{'training_complete':False,'validation_complete':False},spent=42000)
        assert not calls and not result['complete']
        checks['spent_check_cache_budget_is_not_reset']=True
        try:
            scenario(temp/'bad',123,{'training_complete':False,'validation_complete':False})
        except AssertionError:
            checks['unknown_seed_rejected']=True
        else:
            raise AssertionError('Unknown seed accepted')
    report={'status':'SINGLE_SEED_CONTROLLER_SMOKE_PASSED','code_sha256':verify_code(),
            'checks':checks,'GPU_training_started':False,'limits':'Controller calls mocked; no speed or target-resource claim'}
    dump(root/'SCHEDULER_CPU_SMOKE.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    main(parser.parse_args().output)
