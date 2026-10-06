"""Fail closed: all required evidence must belong to this exact code revision."""
from pathlib import Path
import argparse,json
from methods import METHODS,DESIGN,HERE
from runtime import verify_code,digest,environment,dump,sha
def release(local,preflight,resource,target,output):
    paths={'local':Path(local),'preflight':Path(preflight),'resource':Path(resource),'target':Path(target)}
    docs={k:json.loads(p.read_text()) for k,p in paths.items()};code=verify_code();env=digest(environment())
    checks={
        'LOCAL_SMOKE_PASSED':docs['local']['status']=='LOCAL_SMOKE_PASSED' and all(docs['local']['methods'].get(m,{}).get('status')=='PASS' for m in METHODS),
        'TARGET_ENV_PREFLIGHT_PASSED':docs['preflight']['status']=='TARGET_ENV_PREFLIGHT_PASSED',
        'MORDEN_EXCLUSION_VERIFIED':docs['resource']['status']=='RESOURCE_GATE_FAILED' and docs['resource']['limiting_resource']=='GPU_MEMORY' and sha(paths['resource'])==DESIGN['resource_amendment']['checkpoint_failure_sha256'],
        'TARGET_SMOKE_PASSED':docs['target']['status']=='TARGET_SMOKE_PASSED' and all(docs['target']['methods'].get(m,{}).get('status')=='PASS' for m in METHODS),
        'code_consistency':all(v['code_sha256']==code for k,v in docs.items() if k!='resource'),
        'target_environment_consistency':all(v.get('environment_sha256',digest(v['environment']) if 'environment' in v else '')==env for k,v in docs.items() if k not in ('local','resource'))}
    passed=all(checks.values());report={'status':'READY_FOR_EXECUTION' if passed else 'NOT_READY_FOR_EXECUTION','checks':checks,'code_sha256':code,'environment_sha256':env,'passed_methods':list(METHODS) if passed else [],'evidence_sha256':{k:sha(p) for k,p in paths.items()}}
    dump(output,report);return report
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('local','preflight','resource','target','output'):p.add_argument('--'+k,required=True)
    print(json.dumps(release(**vars(p.parse_args())),indent=2))
