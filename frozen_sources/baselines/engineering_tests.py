"""Meaningful failure-path tests without any model training."""
from pathlib import Path
import tempfile,shutil,subprocess,sys,json
from types import SimpleNamespace
from runtime import *
from methods import HERE,CONFIG
from data import audit_cache
from engine import require_release
def run(cache,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);results={}
    with tempfile.TemporaryDirectory(dir=output) as temp:
        temp=Path(temp)
        copy=temp/'cache';shutil.copytree(cache,copy)
        (copy/'images/UNEXPECTED_TEST_SENTINEL.png').write_bytes(b'not an image')
        try:audit_cache(copy,True)
        except AssertionError:results['unexpected_cache_file_rejected']=True
        else:raise AssertionError('Poisoned cache accepted')
        # A drifted child package must fail before a GPU model is constructed.
        package=temp/'package';shutil.copytree(HERE,package,ignore=shutil.ignore_patterns('__pycache__','evidence'))
        with (package/'methods.py').open('a') as f:f.write('\n# intentional code drift test\n')
        p=subprocess.run([sys.executable,'-c','from runtime import verify_code;verify_code()'],cwd=package,capture_output=True,text=True)
        assert p.returncode and 'drift' in p.stderr;results['code_drift_rejected']=True
        (temp/'gate.json').write_text(json.dumps({'status':'NOT_READY_FOR_EXECUTION'}))
        cfg={'environment':{'torch':'2.10.0+cu128','cuda':'12.8','numpy':'2.0.2','gpu':'Tesla T4'}}
        try:require_release(SimpleNamespace(smoke=False,gate=str(temp/'gate.json')),cfg)
        except AssertionError:results['formal_start_rejected_without_release']=True
        else:raise AssertionError('Formal start accepted without gates')
        bad={**cfg,'environment':{**cfg['environment'],'torch':'2.2.2'}}
        try:require_release(SimpleNamespace(smoke=False,gate=str(temp/'gate.json')),bad)
        except AssertionError:results['wrong_target_torch_rejected']=True
        else:raise AssertionError('Wrong environment accepted')
        # Force quota failure without allocating a real 20GB file.
        try:disk_check(temp,20_000_000_000)
        except RuntimeError as e:assert 'RESOURCE_GATE_FAILED disk' in str(e);results['disk_budget_rejected']=True
        else:raise AssertionError('Disk capacity guard did not fire')
    dump(output/'ENGINEERING_TESTS.json',results);print(json.dumps(results,indent=2))
if __name__=='__main__':run(sys.argv[1],sys.argv[2])
