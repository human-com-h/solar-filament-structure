"""Two-GPU task scheduler; training/validation only, budget-aware, resumable output.zip."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time,zipfile
from types import SimpleNamespace
import torch
from data_pipeline import HERE,protocol,prepare,dump,sha
from trainer import train
from validation_export import export

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--json-path',required=True);ap.add_argument('--image-dir',required=True)
    ap.add_argument('--cache',required=True);ap.add_argument('--output',required=True);ap.add_argument('--seeds',nargs='+',type=int,default=[20260831])
    ap.add_argument('--workers',type=int,default=2);ap.add_argument('--wall-hours',type=float,default=9)
    ap.add_argument('--worker-task');ap.add_argument('--deadline',type=float,default=0);ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args();cfg=protocol()
    if args.worker_task:
        method,seed=args.worker_task.split(':');args.method=method;args.seed=int(seed);args.stop_after_epoch=0
        finished=train(args)
        if finished:export(args)
        return
    assert set(args.seeds)<=set(cfg['seeds']) and len(set(args.seeds))==len(args.seeds)
    if not torch.cuda.is_available():raise RuntimeError('GPU is required; select T4 x2 in Kaggle.')
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    prepare(args.json_path,args.image_dir,args.cache,limit=4 if args.smoke else 0)
    tasks=[(m,s) for s in args.seeds for m in cfg['methods']]
    deadline=time.time()+args.wall_hours*3600;active={};errors=[];next_task=0;ngpu=min(2,torch.cuda.device_count());last_notice=0
    dump(out/'REQUEST.json',{'seeds_requested':args.seeds,'full_protocol_seeds':cfg['seeds'],'methods':cfg['methods'],'smoke':args.smoke,'protocol_sha256':sha(HERE/'protocol.json'),'deadline_unix':deadline,'test_evaluated':False})
    while next_task<len(tasks) or active:
        for gpu in range(ngpu):
            if gpu in active or next_task>=len(tasks) or time.time()+240>=deadline:continue
            method,seed=tasks[next_task];next_task+=1
            env=dict(os.environ);env['CUDA_VISIBLE_DEVICES']=str(gpu);env['OMP_NUM_THREADS']='2';env['OPENBLAS_NUM_THREADS']='2'
            cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker-task',f'{method}:{seed}','--json-path',args.json_path,'--image-dir',args.image_dir,
                 '--cache',args.cache,'--output',str(out),'--workers',str(args.workers),'--deadline',str(deadline)]
            if args.smoke:cmd+=['--smoke']
            logfile=(out/f'{method}_seed{seed}.log').open('a',encoding='utf-8')
            active[gpu]=(subprocess.Popen(cmd,env=env,stdout=logfile,stderr=subprocess.STDOUT),logfile,method,seed)
            print('Started GPU',gpu,method,seed,flush=True)
        for gpu,(process,logfile,method,seed) in list(active.items()):
            status=process.poll()
            if status is not None:
                logfile.close();del active[gpu]
                if status:errors.append({'method':method,'seed':seed,'exit_code':status})
                print('Finished process',method,seed,'exit',status,flush=True)
        if not active and (next_task==len(tasks) or time.time()+240>=deadline):break
        if time.time()-last_notice>30:
            print('Active tasks:',[(g,m,s) for g,(_,_,m,s) in active.items()],'queued:',len(tasks)-next_task,flush=True);last_notice=time.time()
        time.sleep(2)
    statuses=[]
    for m,s in tasks:
        folder=out/f'{m}_seed{s}'
        statuses.append({'method':m,'seed':s,'training_complete':(folder/'TRAINING_COMPLETE.json').exists(),'validation_complete':(folder/'VALIDATION_COMPLETE.json').exists(),'resumable':(folder/'last.pt').exists()})
    completed=all(r['training_complete'] and (args.smoke or r['validation_complete']) for r in statuses)
    dump(out/'SUITE_STATUS.json',{'requested_stage_complete':completed,'full_study_complete':completed and set(args.seeds)==set(cfg['seeds']) and not args.smoke,
                                'tasks':statuses,'errors':errors,'test_evaluated':False,'next_step':'Return output.zip even if partial. Preserve last.pt for resuming the same protocol.'})
    archive=out.parent/'output.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for path in out.rglob('*'):
            if path.is_file() and path.suffix!='.tmp':z.write(path,path.relative_to(out))
    print('Output:',archive,'requested stage complete:',completed,'errors:',errors,flush=True)
if __name__=='__main__':main()
