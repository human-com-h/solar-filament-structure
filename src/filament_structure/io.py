"""Root resolution and immutable-input checks shared by command line entries."""
from pathlib import Path as PlainPath
import csv,hashlib,json

class Path(type(PlainPath())):
    """Resolve a canonical workspace component in an existing evidence tree.

    A directory named ``project`` is used directly. Otherwise its parent may
    contain exactly one scientific workspace identified by the fixed evidence
    layout. Discovery changes only path resolution; inputs are still checked
    against their recorded hashes. Ambiguous layouts require an explicit
    canonical directory rather than choosing a workspace arbitrarily.
    """
    def __truediv__(self, key):
        result=self
        for part in PlainPath(key).parts:
            candidate=type(PlainPath()).__truediv__(result,part)
            if part=='project' and not candidate.exists() and result.is_dir():
                choices=[]
                for child in result.iterdir():
                    if not child.is_dir():continue
                    names={p.name for p in child.iterdir()}
                    if 'mechanism_training' in names or any(n.startswith('region_clDice_') for n in names):choices.append(child)
                if len(choices)>1:raise ValueError('Ambiguous scientific workspace under '+str(result))
                if len(choices)==1:candidate=choices[0]
            result=candidate
        return result

REPO=Path(__file__).resolve().parents[2]
ORIGINAL='02_evidence/original21/payload'
REGION='02_evidence/region/workspace'
SEEDS=(20260831,20260901,20260902)

def source_files(root=REPO):
    """Enumerate sources, excluding only the root Git metadata directory/file.

    Nested Git metadata and unexpected generated files remain in the enumeration
    so the release file-set check rejects them.
    """
    root=Path(root);pending=[root]
    while pending:
        directory=pending.pop()
        for p in sorted(directory.iterdir()):
            if directory==root and p.name=='.git':continue
            if p.is_symlink():yield p
            elif p.is_dir():pending.append(p)
            elif p.is_file():yield p

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def dump(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.partial')
    tmp.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(p)
def table(p,rr):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if not rr:raise ValueError('No records to export: '+p.name)
    fields=list(dict.fromkeys(k for r in rr for k in r))
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rr)
def safe(root,rel):
    root=Path(root).resolve();p=(root/str(rel).replace('\\','/')).resolve()
    if not p.is_relative_to(root):raise ValueError('Path escapes configured root: '+str(rel))
    return p
def required(p,role):
    p=Path(p)
    if not p.is_file():raise FileNotFoundError(f'Missing {role}: {p}. See docs/INPUTS.md.')
    return p
def verified(p,h,role):
    p=required(p,role)
    if sha(p)!=h:raise ValueError('Checksum mismatch for '+role+': '+str(p))
    return p
def output(p):
    p=Path(p).resolve()
    if p.is_relative_to(REPO):raise ValueError('Keep computation outputs outside the repository source directory.')
    p.mkdir(parents=True,exist_ok=True);return p
def model(run):
    mm=read(REPO/'configs/paper24/models.json')
    found=[m for m in mm if m['run']==run]
    if len(found)!=1:raise ValueError('Unknown paper24 run: '+run)
    return found[0]
def evidence(root,rel):return required(safe(root,rel),'external evidence')
def original(root,rel):return evidence(root,ORIGINAL+'/'+rel)
def region(root,rel):return evidence(root,REGION+'/'+rel)
def protocol():return read(REPO/'configs/paper24/protocol.json')
def manifest(root):
    p=original(root,'project/mechanism_training/temporal_manifest.csv')
    return rows(verified(p,protocol()['manifest_sha256'],'fixed split manifest'))
def qc(root):
    p=original(root,'project/mechanism_training/manual_spine_qc_exclusions.csv')
    return rows(verified(p,protocol()['source_provenance']['manual_spine_qc_exclusions.csv']['sha256'],'spine QC exclusions'))
