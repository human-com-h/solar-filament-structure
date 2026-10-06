"""Run on Kaggle/T4 before target smoke or any formal training."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json,subprocess,sys
from pathlib import Path
import torch
from methods import *
from runtime import *
def init_worker(method,seed):
    seed_all(seed);m=factory(method)
    print(json.dumps({'method':method,'seed':seed,'initial_digest':model_digest(m)}))
def preflight(output,selected_seed):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    report={'status':'TARGET_ENV_PREFLIGHT_FAILED','environment':environment(),'code_sha256':verify_code(),'checks':[]}
    report['environment_sha256']=digest(report['environment']);dump(output/'TARGET_ENV_PREFLIGHT.json',report)
    try:
        assert sys.platform=='linux' and Path('/kaggle/input').exists(),'Actual Kaggle Linux runtime required'
        require_runtime_spec(report['environment'])
        assert torch.__version__=='2.10.0+cu128' and torch.version.cuda=='12.8','Requires archived PyTorch 2.10.0+cu128 / CUDA12.8; no automatic version substitution'
        assert np.__version__=='2.0.2','Requires archived numpy 2.0.2'
        assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0),'Actual Tesla T4 required'
        expected=json.loads((HERE/'INITIAL_DIGESTS.json').read_text())
        for method in METHODS:
            for seed in [selected_seed]:
                hashes=[]
                for repeat in range(2):
                    p=subprocess.run([sys.executable,str(Path(__file__)),'--worker',method,'--seed',str(seed)],capture_output=True,text=True,timeout=600,check=True)
                    hashes.append(json.loads(p.stdout.strip().splitlines()[-1])['initial_digest'])
                assert hashes[0]==hashes[1],(method,seed,'independent initialization differs')
                if method=='UNet_region_clDice':assert hashes[0]==expected[str(seed)],(seed,'archived initialization mismatch')
                report['checks'].append({'method':method,'seed':seed,'two_independent_processes':hashes,'passed':True})
                dump(output/'TARGET_ENV_PREFLIGHT.json',report)
        from local_smoke import invariant_tests
        report['invariants']=invariant_tests(output/'invariants')
        assert all(x['matches'] for x in report['invariants']['initialization'].values())
        report['status']='TARGET_ENV_PREFLIGHT_PASSED'
    except Exception as e:report['error']=repr(e)
    dump(output/'TARGET_ENV_PREFLIGHT.json',report)
    if report['status']!='TARGET_ENV_PREFLIGHT_PASSED':raise RuntimeError(report.get('error','Target environment preflight failed'))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',choices=METHODS);p.add_argument('--seed',type=int);p.add_argument('--output');a=p.parse_args()
    if a.worker:init_worker(a.worker,a.seed)
    else:preflight(a.output,a.seed)
