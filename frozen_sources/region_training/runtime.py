"""Strict signatures, process locks and transactional self-contained archives."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import contextlib,hashlib,json,platform,random,shutil,threading,time,zipfile,signal,subprocess
import numpy as np
import torch,psutil,PIL,scipy,skimage
HERE=Path(__file__).resolve().parent
GIB=1024**3

def require_runtime_spec(env):
    expected=json.loads((HERE/'RUNTIME_SPEC.json').read_text())['expected']
    mismatches={key:{'expected':value,'actual':env.get(key)} for key,value in expected.items() if env.get(key)!=value}
    assert not mismatches, 'Historical numerical runtime mismatch: '+json.dumps(mismatches)

def run_process(command,timeout=None,**kwargs):
    # A timed-out torchrun must terminate its two ranks before collection.
    process=subprocess.Popen(command,start_new_session=os.name!='nt',**kwargs)
    def stop():
        if process.poll() is not None:return
        if os.name=='nt':process.terminate()
        else:
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:return
        try:process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            if os.name=='nt':process.kill()
            else:
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
            process.wait(timeout=15)
    old_term=None
    if os.name!='nt' and threading.current_thread() is threading.main_thread():
        old_term=signal.getsignal(signal.SIGTERM)
        def terminate(signum,frame):raise SystemExit(128+signum)
        signal.signal(signal.SIGTERM,terminate)
    try:
        code=process.wait(timeout=timeout)
    except BaseException:
        stop()
        raise
    finally:
        if old_term is not None:signal.signal(signal.SIGTERM,old_term)
    if code:raise subprocess.CalledProcessError(code,command)
    return code
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def dump(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:json.dump(data,f,indent=2,ensure_ascii=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def environment():
    return {'python':platform.python_version(),'platform':platform.platform(),'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'numpy':np.__version__,'pillow':PIL.__version__,'scipy':scipy.__version__,'skimage':skimage.__version__,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'gpu_bytes':torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,'deterministic':True,'amp':False,'tf32':False,'optimizer':'AdamW default foreach/fused; lr=1e-4 wd=1e-5','torch_build':torch.__config__.show()}
def verify_code():
    expected=json.loads((HERE/'CODE_LOCK.json').read_text())
    actual={str(p.relative_to(HERE)).replace('\\','/'):sha(p) for p in HERE.rglob('*') if p.is_file() and ('legacy' in p.parts or 'vendor' in p.parts or p.parent==HERE) and p.suffix in ('.py','.csv','.json') and p.name not in ('CODE_LOCK.json','ENGINEERING_STATUS.json')}
    assert actual==expected,'Code/protocol/manifest/method configuration drift'
    return digest(expected)
def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True);torch.set_num_threads(2)
def rng():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}
def restore_rng(s):
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch']);torch.cuda.set_rng_state_all(s['cuda'])
def model_digest(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()
def size_tree(root):return sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file())
@contextlib.contextmanager
def run_lock(path,blocking=False):
    # OS advisory lock releases on process death; no stale PID deletion race.
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as f:
        if os.name=='nt':
            import msvcrt
            f.seek(0);f.write(b'0');f.flush();f.seek(0)
            while True:
                try:msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1);break
                except OSError:
                    if not blocking:raise RuntimeError('Run already owned: '+str(path))
                    time.sleep(.1)
        else:
            import fcntl
            try:fcntl.flock(f,fcntl.LOCK_EX if blocking else fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise RuntimeError('Run already owned: '+str(path))
        try:yield
        finally:
            if os.name=='nt':f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(f,fcntl.LOCK_UN)
class Monitor:
    def __init__(self,root):self.root=Path(root);self.done=threading.Event();self.ram=0;self.system_used=0;self.disk=0
    def sample(self):
        self.ram=max(self.ram,psutil.Process().memory_info().rss);self.system_used=max(self.system_used,psutil.virtual_memory().used)
        self.disk=max(self.disk,size_tree(self.root))
    def __enter__(self):
        self.sample()
        def loop():
            while not self.done.wait(.1):self.sample()
        self.thread=threading.Thread(target=loop,daemon=True);self.thread.start();return self
    def __exit__(self,*exc):self.done.set();self.thread.join();self.sample()
def disk_check(root,additional):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    # Conservative bytes, not an optimistic compression ratio; shared session cap.
    used=size_tree(root);free=shutil.disk_usage(root).free
    if free<additional+GIB or used+additional+GIB>20_000_000_000:
        raise RuntimeError(f'RESOURCE_GATE_FAILED disk: used={used}, required_additional={additional}, free={free}, cap=20000000000, reserve={GIB}')
def save_archive(state,run,root,fail_before_replace=False):
    run=Path(run);run.mkdir(parents=True,exist_ok=True);dest=run/'recovery.zip';tmp=run/'recovery.zip.partial'
    if Path('/kaggle/working').exists():root=Path('/kaggle/working')
    # Serialize directly inside ZIP: no simultaneous last.pt/best.pt/zip copies.
    payload=sum(t.numel()*t.element_size() for t in state['model'].values())*4+32*1024**2
    started=time.perf_counter()
    with run_lock(Path(root)/'.archive.lock',blocking=True):
        if tmp.exists():tmp.unlink() # previous incomplete write only, complete archive untouched
        disk_check(root,payload)
        with Monitor(root) as mon:
            try:
                with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
                    with z.open('state.pt','w',force_zip64=True) as f:torch.save(state,f)
                    z.writestr('identity.json',json.dumps(state['cfg'],sort_keys=True))
                    z.writestr('epoch.json',json.dumps({'epoch':state.get('epoch',0)}))
                with zipfile.ZipFile(tmp) as z:
                    assert z.testzip() is None
                    raw=z.getinfo('state.pt').file_size
                with tmp.open('r+b') as f:os.fsync(f.fileno())
                mon.sample() # Capture old ZIP + fully written new ZIP before atomic replacement.
                if fail_before_replace:raise RuntimeError('injected archive interruption')
                os.replace(tmp,dest)
            except BaseException:
                if tmp.exists():tmp.unlink()
                raise
    return {'checkpoint_bytes':raw,'zip_bytes':dest.stat().st_size,'peak_workspace_bytes':mon.disk,'peak_process_rss_bytes':mon.ram,'peak_system_used_bytes':mon.system_used,'archive_seconds':time.perf_counter()-started}
def load_archive(path):
    # ZipExtFile supports seeking; deserialize on CPU, retaining full optimizer and RNG.
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        with z.open('state.pt') as f:return torch.load(f,map_location='cpu',weights_only=False)
