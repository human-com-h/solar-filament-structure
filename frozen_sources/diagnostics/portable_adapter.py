"""Portable file-locator adapter for immutable exploratory calibration source."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,ast,copy,hashlib,json
HERE=Path(__file__).resolve().parent
CORE=HERE/'analyze_calibration.py'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def adapt(workspace):
 code=CORE.read_text(encoding='utf-8-sig');tree=ast.parse(code,filename=str(CORE));updated=copy.deepcopy(tree)
 found=[]
 for i,node in enumerate(tree.body):
  if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='ROOT':found.append(i)
 assert len(found)==1,'Ambiguous file locator'
 i=found[0];assert ast.unparse(tree.body[i].value)=="Path('E:/SABR')",'Unexpected immutable source locator'
 # Only the literal workspace path in the original top-level ROOT assignment changes.
 updated.body[i].value=ast.Call(func=ast.Name(id='Path',ctx=ast.Load()),args=[ast.Constant(value=str(workspace))],keywords=[])
 before=copy.deepcopy(tree);after=copy.deepcopy(updated)
 before.body[i].value=ast.Constant(value='FILE_LOCATOR_REMOVED_FOR_EQUALITY_QA')
 after.body[i].value=ast.Constant(value='FILE_LOCATOR_REMOVED_FOR_EQUALITY_QA')
 scientific_ast_before=ast.dump(before,include_attributes=False);scientific_ast_after=ast.dump(after,include_attributes=False)
 assert scientific_ast_before==scientific_ast_after,'Adapter changed scientific source AST'
 h=hashlib.sha256(scientific_ast_before.encode()).hexdigest()
 return ast.fix_missing_locations(updated),{'core_source_sha256':sha(CORE),'scientific_ast_excluding_workspace_locator_sha256':h,'all_nonlocator_ast_identical':True,'single_change':'top-level ROOT Path string literal only','source_file_unchanged':True}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--workspace',type=Path);ap.add_argument('--verify-only',action='store_true');a=ap.parse_args()
 workspace=(a.workspace or HERE.parents[2]).resolve();tree,qa=adapt(workspace)
 assert sha(CORE)==read(HERE/'ANALYSIS_CODE_LOCK.json')['sha256']
 qa['workspace']=str(workspace);qa['adapter_sha256']=sha(Path(__file__))
 if a.verify_only:print(json.dumps(qa,indent=2));return
 # Replay in the same new directory, using only matching bound receipts.
 scientific_files=sorted(list(HERE.glob('*.csv'))+list(HERE.glob('*_SUMMARY.json'))+list((HERE/'reference_mask_receipts').glob('*.json'))+list((HERE/'prediction_diagnostic_receipts').rglob('*.json')))
 before={p.relative_to(HERE).as_posix():sha(p) for p in scientific_files}
 first=HERE/'FIRST_EXECUTION_ENGINEERING_QA.json'
 if not first.exists():first.write_bytes((HERE/'ENGINEERING_QA.json').read_bytes())
 qa['started_at_utc']=datetime.now(timezone.utc).isoformat();namespace={'__file__':str(CORE),'__name__':'__main__'}
 exec(compile(tree,str(CORE),'exec'),namespace)
 after={p.relative_to(HERE).as_posix():sha(p) for p in scientific_files}
 assert before==after,'Scientific output changed in portable replay'
 qa.update(status='PASS_TRUE_LOCAL_PORTABLE_ADAPTER_REPLAY',all_scientific_tables_and_bound_receipts_identical=True,scientific_files_checked=len(before),completed_at_utc=datetime.now(timezone.utc).isoformat(),scientific_output_sha256=after)
 assert sha(CORE)==read(HERE/'ANALYSIS_CODE_LOCK.json')['sha256']
 dump(HERE/'PORTABLE_ADAPTER_QA.json',qa)
 print(json.dumps({k:v for k,v in qa.items() if k!='scientific_output_sha256'},indent=2))
if __name__=='__main__':main()
