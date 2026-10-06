"""Local acceptance and frozen statistics only. No torch, inference or remote calls."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'
from pathlib import Path
import ast, csv, hashlib, json, re, shutil, sys, zipfile
from collections import Counter
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
REV = ROOT / 'project'
SOURCE = ROOT / 'output/Kaggle_SABR_six_final_evaluation_dual_gpu_v2_output'
PACKAGE = REV / 'strong_baseline_final_eval/package_dual_gpu_v2'
OLD = REV / 'mechanism_final_analysis'
METRICS = ['dice', 'iou', 'precision', 'recall', 'cldice', 'msc', 'msgr']
EXTRA = ['foreground_fraction', 'false_positive_fraction', 'predicted_to_reference_area']
MODES = ['fixed_0.5', 'validation_selected']
RADII = [1, 3, 5]
NA_REASON = 'INFEASIBLE_NA: no feasible threshold in frozen validation grid satisfying same-seed precision floor and nonempty foreground; fixed-0.5 is diagnostic only'
REQUESTED = [
    'strong_baseline_audit/REPORT_CN.md', 'strong_baseline_audit/FROZEN_DESIGN.json',
    'strong_baseline_six_execution/FROZEN_DESIGN.json',
    'strong_baseline_six_resume_audit/REPORT_CN.md', 'strong_baseline_six_resume_audit/FINAL_RUN_MANIFEST.json',
    'strong_baseline_final_eval/package_dual_gpu_v2/LOCKED_EVALUATION.json',
    'strong_baseline_final_eval/package_dual_gpu_v2/EVAL_CODE_LOCK.json',
    'strong_baseline_final_eval/delivery_dual_gpu_v2/README_CN.md',
    'mechanism_final_analysis/RESULTS_CN.md', 'mechanism_final_analysis/DECISION_CN.md',
    'mechanism_final_analysis/locked_test_protocol.json',
]
INPUTS = {}
CHECKS = []

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''): h.update(block)
    return h.hexdigest()

def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def track(path):
    p = Path(path).resolve()
    INPUTS[str(p)] = {'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)}
    return p

def read(path):
    return json.loads(track(path).read_text(encoding='utf-8-sig'))

def rows(path):
    with track(path).open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))

def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def write(name, rr):
    with (OUT / name).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rr[0])); w.writeheader(); w.writerows(rr)

def check(condition, label, **evidence):
    CHECKS.append({'check': label, 'passed': bool(condition), **evidence})
    if not condition: raise AssertionError(label)

def verify_record(record, oid, expected):
    assert record['observation_id'] == oid
    rr = record['rows']
    assert len(rr) == len(expected)
    assert {(r['mode'], r['radius']) for r in rr} == set(expected)
    for r in rr:
        assert r['physical_observation_id'] == oid
        assert r['threshold'] == expected[r['mode'], r['radius']]
        for m in METRICS + EXTRA:
            assert isinstance(r[m], (int, float)) and np.isfinite(r[m]), (oid, m)
            assert r[m] >= 0
            if m != 'predicted_to_reference_area': assert r[m] <= 1 + 1e-12
        assert r['false_positive_fraction'] <= r['foreground_fraction'] + 1e-12
    for mode in {r['mode'] for r in rr}:
        byradius = {r['radius']: r for r in rr if r['mode'] == mode}
        assert set(byradius) == set(RADII)
        for r in (3, 5):
            for m in METRICS[:5] + EXTRA: assert byradius[r][m] == byradius[1][m]
        assert byradius[1]['msc'] <= byradius[3]['msc'] + 1e-12 <= byradius[5]['msc'] + 2e-12
        assert byradius[1]['msgr'] + 2e-12 >= byradius[3]['msgr'] + 1e-12 >= byradius[5]['msgr']
    if len(expected) == 6 and expected['fixed_0.5', 1] == expected['validation_selected', 1]:
        for radius in RADII:
            a = next(r for r in rr if r['mode'] == 'fixed_0.5' and r['radius'] == radius)
            b = next(r for r in rr if r['mode'] == 'validation_selected' and r['radius'] == radius)
            assert all(a[m] == b[m] for m in METRICS + EXTRA)
    return rr

def audit():
    lock = read(PACKAGE / 'LOCKED_EVALUATION.json')
    code = read(PACKAGE / 'EVAL_CODE_LOCK.json')
    oldlock = read(OLD / 'locked_test_protocol.json')
    manifest = read(REV / 'strong_baseline_six_resume_audit/FINAL_RUN_MANIFEST.json')
    design = read(REV / 'strong_baseline_six_execution/FROZEN_DESIGN.json')
    for p in REQUESTED: track(REV / p)
    track(OUT / 'ENGINEERING_RULES_CN.md')
    for rel, expected_hash in code.items():
        p = track(PACKAGE / rel); assert INPUTS[str(p)]['sha256'] == expected_hash, rel
    training_code = read(PACKAGE / 'training_code/CODE_LOCK.json')
    actual = {p.relative_to(PACKAGE / 'training_code').as_posix(): sha(p)
              for p in (PACKAGE / 'training_code').rglob('*')
              if p.is_file() and ('legacy' in p.parts or 'vendor' in p.parts or p.parent == PACKAGE / 'training_code')
              and p.suffix in ('.py', '.csv', '.json') and p.name not in ('CODE_LOCK.json', 'ENGINEERING_STATUS.json')}
    check(actual == training_code and digest(training_code) == lock['training_code_sha256'], 'evaluation and training code locks', evaluation_files=len(code), training_files=len(training_code))
    for rel, expected_hash in design['authority_sha256'].items():
        assert sha(track(ROOT / rel)) == expected_hash, rel
    check(sha(OLD / 'locked_test_protocol.json') == lock['old_locked_protocol_sha256']
          and sha(REV / 'strong_baseline_six_resume_audit/FINAL_RUN_MANIFEST.json') == lock['final_manifest_sha256']
          and sha(REV / 'strong_baseline_six_execution/FROZEN_DESIGN.json') == lock['frozen_design_sha256'], 'scientific authority hashes')
    check(lock['test_ids'] == oldlock['test_ids'] and lock['radii'] == RADII and lock['metrics'] == METRICS
          and lock['spine_qc_exclusions'] == oldlock['spine_qc_exclusions']
          and lock['annotation_sha256'] == oldlock['annotation_sha256']
          and lock['manifest_sha256'] == oldlock['manifest_sha256']
          and lock['test_selection_prohibited'] and not lock['external_evaluation_released'], 'unchanged observations, annotation, QC, metrics and external boundary')
    check(lock['selection_rule'] == design['operating_point'], 'selection rule unchanged')
    runmap = {r['run']: r for r in manifest['runs']}
    for run in lock['runs']:
        r = runmap[run['name']]
        assert sha(track(Path(r['archive_path']))) == run['source_archive_sha256'] == r['archive_sha256']
        assert r['best_epoch'] == run['best_epoch'] and r['epochs'] == run['training_epochs']
        assert sha(track(Path(r['archive_path']).parent / 'validation/curves.csv')) == run['validation_curve_sha256']
        assert sha(track(PACKAGE / run['checkpoint'])) == run['sha256']
        assert run['validation_precision_floor'] == design['operating_point']['precision_floors'][str(run['seed'])]
        if run['method'] == 'UNet_softDice_clDice':
            assert run['selected_status'] == 'INFEASIBLE_NA' and run['threshold'] is None and run['selected_validation_metrics'] is None
        else:
            assert run['selected_status'] == 'FEASIBLE' and run['threshold'] == {20260831: .5, 20260901: float(np.float32(.675)), 20260902: float(np.float32(.7))}[run['seed']]
    check(len(runmap) == len(lock['runs']) == 6, 'all six final model sources, checkpoints, epochs and threshold statuses')
    candidates = sorted(p.parent for p in (ROOT / 'output').rglob('signature.json') if p.parent.name == 'eval_test')
    check(candidates == [SOURCE / 'eval_test'], 'unique extracted final test output', candidates=[str(p) for p in candidates])
    normalized = []
    smokeerrors = []
    accepted = []
    assignments = [[lock['runs'][i]['name'] for i in ii] for ii in [(0, 3, 4), (1, 2, 5)]]
    for part, mode in [('eval_smoke', 'smoke'), ('eval_test', 'test')]:
        folder = SOURCE / part
        files = sorted(p for p in folder.rglob('*') if p.is_file())
        for p in files: track(p)
        with zipfile.ZipFile(track(SOURCE / (part + '.zip'))) as z:
            names = [n for n in z.namelist() if not n.endswith('/')]
            assert len(names) == len(set(names)) and z.testzip() is None
            assert all(not Path(n.replace('\\', '/')).is_absolute() and '..' not in Path(n.replace('\\', '/')).parts for n in names)
            zm = {n.replace('\\', '/'): z.read(n) for n in names}
            assert set(zm) == {p.relative_to(folder).as_posix() for p in files}
            assert all(zm[p.relative_to(folder).as_posix()] == p.read_bytes() for p in files)
        check(True, part + ' ZIP CRC, unique safe members and byte parity', files=len(files))
        sig = read(folder / 'signature.json'); complete = read(folder / 'COMPLETE.json'); status = read(folder / 'STATUS.json')
        assert sig['lock_sha256'] == sha(PACKAGE / 'LOCKED_EVALUATION.json') == complete['lock_sha256']
        assert sig['code_lock_sha256'] == sha(PACKAGE / 'EVAL_CODE_LOCK.json') == complete['code_lock_sha256']
        assert digest(sig['environment']) == lock['target_environment_sha256'] == complete['environment_sha256']
        assert sig['mode'] == mode and sig['local_smoke'] is False
        assert sig['execution'] == {'version': 'dual_gpu_v2', 'assignments': [[0, 3, 4], [1, 2, 5]], 'independent_processes': True, 'DataParallel': False}
        env = sig['environment']
        assert env['torch'] == '2.10.0+cu128' and env['cuda'] == '12.8' and env['gpu'] == 'Tesla T4'
        assert env['deterministic'] and not env['amp'] and not env['tf32']
        assert complete['status'] == ('TARGET_EVAL_SMOKE_PASSED' if mode == 'smoke' else 'SIX_MODEL_TEST_EVALUATION_COMPLETE')
        assert complete['complete'] and complete['exit_codes'] == [0, 0]
        assert complete['physical_gpus'] == [0, 1] and complete['independent_single_gpu_processes']
        assert complete['assignment'] == {str(i): assignments[i] for i in (0, 1)}
        assert status == {'complete': True, 'paused': False}
        ids = [lock['runs'][0]['validation_smoke_id']] if mode == 'smoke' else lock['test_ids']
        assert complete['observations_per_run'] == len(ids) and complete['runs'] == 6
        checks = []
        for rank in (0, 1):
            worker = read(folder / f'worker_gpu{rank}_COMPLETE.json')
            assert worker['complete'] and worker['rank'] == rank and worker['physical_gpu'] == str(rank)
            assert worker['signature'] == sig and worker['assigned_runs'] == assignments[rank]
            assert worker['observations_per_run'] == len(ids)
            checks.extend(worker['checks'])
            log = (folder / f'worker_gpu{rank}.log').read_text(encoding='utf-8-sig')
            assert not re.search(r'Traceback|CUDA out of memory|AssertionError|RuntimeError', log)
            for name in assignments[rank]:
                assert f'physical GPU {rank} {name} {len(ids)} / {len(ids)}' in log
            seen = re.findall(r'physical GPU (\d+) (\S+) (\d+) / (\d+)', log)
            assert seen and all(int(r) == rank and n in assignments[rank] and int(t) == len(ids) for r, n, v, t in seen)
        assert checks == complete['checks']
        assert {p.name for p in folder.iterdir() if p.is_dir()} == {r['name'] for r in lock['runs']}
        assert not any(p.name.endswith(('.tmp', '.partial')) for p in files)
        for run in lock['runs']:
            rf = folder / run['name']
            assert {p.stem for p in rf.glob('*.json')} == set(ids)
            modes = [('fixed_0.5', .5)]
            if mode == 'test' and run['threshold'] is not None: modes.append(('validation_selected', run['threshold']))
            expected = {(m, r): t for m, t in modes for r in RADII}
            for oid in ids:
                rr = verify_record(read(rf / (oid + '.json')), oid, expected)
                if mode == 'smoke':
                    r = next(r for r in rr if r['radius'] == 3)
                    fields = ['dice', 'iou', 'precision', 'recall', 'msc', 'msgr', 'foreground_fraction']
                    error = max(abs(r[f] - v) for f, v in zip(fields, run['validation_smoke_scores']))
                    c = next(c for c in checks if c['run'] == run['name'])
                    assert c['observation_id'] == oid and c['max_error'] == error and c['tolerance'] == 1e-7 and error <= 1e-7
                    smokeerrors.append({'run': run['name'], 'max_error': error, 'tolerance': 1e-7})
                else:
                    normalized.extend({**r, 'method': run['method'], 'seed': run['seed'], 'status': 'OK', 'na_reason': '', 'provenance': 'new_dual_gpu_v2'} for r in rr)
            if mode == 'test': accepted.append({'run': run['name'], 'seed': run['seed'], 'observations': len(ids), 'rows': len(ids) * len(expected), 'selected_status': run['selected_status'], 'threshold': run['threshold'], 'best_epoch': run['best_epoch']})
        assert len(checks) == (6 if mode == 'smoke' else 0)
        check(True, part + ' target environment, locks, both workers, logs, unique coverage and all finite rows', elapsed_seconds=complete['elapsed_seconds'], observations_per_run=len(ids))
    write('new_per_observation_observed.csv', normalized)
    write('smoke_parity.csv', smokeerrors)
    write('accepted_runs.csv', accepted)
    original_rows = rows(OLD / 'region_per_observation.csv')
    index = {}
    for r in original_rows:
        key = (r['method'], int(r['seed']), r['physical_observation_id'], r['mode'], int(r['radius']))
        assert key not in index
        index[key] = r
    assert len(index) == 9720
    oldcomplete = read(REV / 'mechanism_locked_test/results/COMPLETE.json')
    assert oldcomplete['protocol_sha256'] == sha(OLD / 'locked_test_protocol.json') and not oldcomplete['limited_pilot']
    for run in oldlock['runs']:
        for oid in oldlock['test_ids']:
            expected = {(m, r): t for m, t in [('fixed_0.5', .5), ('validation_selected', run['threshold'])] for r in RADII}
            rr = verify_record(read(REV / 'mechanism_locked_test/results' / run['name'] / (oid + '.json')), oid, expected)
            for r in rr:
                oldr = index[run['method'], run['seed'], oid, r['mode'], r['radius']]
                assert all(float(oldr[k]) == r[k] for k in METRICS + EXTRA + ['threshold'])
    check(True, 'old 15 runs reused, CSV exactly equals all 1620 frozen observation JSONs', rows=9720)
    meta = {}
    copies = Counter()
    for r in rows(REV / 'mechanism_training/temporal_manifest.csv'):
        if r['split'] != 'internal_test': continue
        oid = r['physical_observation_id']; copies[oid] += 1
        if oid in meta: assert (meta[oid]['temporal_group'], meta[oid]['year']) == (r['temporal_group'], r['year'])
        meta[oid] = r
    assert set(meta) == set(lock['test_ids']) and len(set(r['temporal_group'] for r in meta.values())) == 13
    write('test_observations.csv', [{'physical_observation_id': oid, 'year': meta[oid]['year'], 'temporal_group': meta[oid]['temporal_group'], 'annotation_copies': copies[oid]} for oid in lock['test_ids']])
    check(True, 'original 108 observation / 13 temporal group mapping', annotation_copies=sum(copies.values()), years=dict(Counter(r['year'] for r in meta.values())), groups=dict(Counter(r['temporal_group'] for r in meta.values())))
    for p in OLD.iterdir():
        if p.is_file():
            track(p); target = OUT / 'input_snapshot/old_final_analysis' / p.name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, target)
    for rel in REQUESTED:
        p = REV / rel; target = OUT / 'input_snapshot/authorities' / rel; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, target)
    stats = track(REV / 'scripts/summarize_mechanism_test.py')
    tree = ast.parse(stats.read_text(encoding='utf-8-sig'))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'boot')
    function_text = ast.get_source_segment(stats.read_text(encoding='utf-8-sig'), fn)
    (OUT / 'frozen_bootstrap.py').write_text('import numpy as np\n\n' + function_text + '\n', encoding='utf-8')
    dump('ACCEPTANCE.json', {'status': 'ALL_EVALUATION_CHECKS_PASSED', 'checks': CHECKS, 'runs': accepted, 'smoke': smokeerrors, 'source': str(SOURCE), 'target_environment_sha256': lock['target_environment_sha256'], 'lock_sha256': sha(PACKAGE / 'LOCKED_EVALUATION.json'), 'code_lock_sha256': sha(PACKAGE / 'EVAL_CODE_LOCK.json'), 'test_files': 648, 'test_numeric_rows': 2916, 'soft_cldice_selected': 'INFEASIBLE_NA for all three seeds; explicit NA added by analysis adapter', 'native_grid_evidence': 'locked evaluate.py nearest binary remapping and byte-identical frozen_evaluator.py: native GT shapes, distance_transform_edt at native resolution, RADII 1/3/5', 'kaggle_run_required': False, 'external_inference_performed': False})
    write('INPUT_SHA256.csv', sorted(INPUTS.values(), key=lambda r: r['path']))
    print('Acceptance passed: 648 test observations, 2916 finite numeric rows; old 9720 rows exact.', flush=True)

def matrix(rr, method, seed, mode, radius, ids):
    selected = [r for r in rr if r['method'] == method and int(r['seed']) == seed and r['mode'] == mode and int(r['radius']) == radius]
    byid = {r['physical_observation_id']: r for r in selected}
    assert len(byid) == len(selected) == len(ids) and set(byid) == set(ids)
    return np.array([[float(byid[oid][m]) for m in METRICS + EXTRA] for oid in ids])

def verify_inputs():
    rr = rows(OUT / 'INPUT_SHA256.csv')
    assert all(Path(r['path']).stat().st_size == int(r['bytes']) and sha(r['path']) == r['sha256'] for r in rr)
    return len(rr)

def analyze():
    assert read(OUT / 'ACCEPTANCE.json')['status'] == 'ALL_EVALUATION_CHECKS_PASSED'
    input_count = verify_inputs()
    from frozen_bootstrap import boot
    lock = read(PACKAGE / 'LOCKED_EVALUATION.json'); oldlock = read(OLD / 'locked_test_protocol.json')
    ids = lock['test_ids']; seeds = sorted({r['seed'] for r in lock['runs']})
    methods = list(dict.fromkeys(r['method'] for r in oldlock['runs'] + lock['runs']))
    metadata = {r['physical_observation_id']: r for r in rows(OUT / 'test_observations.csv')}
    groups = [metadata[i]['temporal_group'] for i in ids]; years = [metadata[i]['year'] for i in ids]
    observed = rows(OLD / 'region_per_observation.csv')
    observed = [{**r, 'status': 'OK', 'na_reason': '', 'provenance': 'old_frozen_reused'} for r in observed] + rows(OUT / 'new_per_observation_observed.csv')
    allrows = list(observed)
    arr = {}; points = []
    oldpoints = {(r['method'], int(r['seed']), r['mode'], int(r['radius'])): r for r in rows(OLD / 'region_per_seed.csv')}
    for method in methods:
        for seed in seeds:
            for mode in MODES:
                missing = method == 'UNet_softDice_clDice' and mode == 'validation_selected'
                for radius in RADII:
                    key = (method, seed, mode, radius)
                    point = {'method': method, 'seed': seed, 'mode': mode, 'radius': radius, 'status': 'INFEASIBLE_NA' if missing else 'OK', 'na_reason': NA_REASON if missing else '', 'n_observations': 0 if missing else 108, 'planned_observations': 108}
                    if missing:
                        point.update({m: 'NA' for m in METRICS + EXTRA})
                        allrows.extend({'mode': mode, 'threshold': 'NA', 'radius': radius, 'physical_observation_id': oid, **{m: 'NA' for m in METRICS + EXTRA}, 'method': method, 'seed': seed, 'status': 'INFEASIBLE_NA', 'na_reason': NA_REASON, 'provenance': 'locked_structural_NA'} for oid in ids)
                    else:
                        a = matrix(observed, method, seed, mode, radius, ids); assert np.isfinite(a).all(); arr[key] = a
                        point.update(dict(zip(METRICS + EXTRA, map(float, a.mean(0)))))
                        if key in oldpoints: assert all(abs(point[m] - float(oldpoints[key][m])) <= 1e-15 for m in METRICS + EXTRA)
                    points.append(point)
    write('region_per_observation.csv', allrows); write('region_per_seed.csv', points)
    multi = []
    for method in methods:
        for mode in MODES:
            missing = method == 'UNet_softDice_clDice' and mode == 'validation_selected'
            for radius in RADII:
                entry = {'method': method, 'mode': mode, 'radius': radius, 'status': 'INFEASIBLE_NA' if missing else 'OK', 'na_reason': NA_REASON if missing else '', 'n_seeds': 0 if missing else 3, 'planned_seeds': 3, 'n_observations': 0 if missing else 108, 'planned_observations': 108}
                a = None if missing else np.array([arr[method, s, mode, radius].mean(0) for s in seeds])
                for j, m in enumerate(METRICS + EXTRA):
                    entry[m + '_mean'] = 'NA' if missing else float(a[:, j].mean())
                    entry[m + '_sd'] = 'NA' if missing else float(a[:, j].std(ddof=1))
                multi.append(entry)
    write('region_multiseed.csv', multi)
    parity = []
    legacy_ci = rows(OLD / 'region_paired_intervals.csv')
    for seed in seeds:
        d = arr['SABR', seed, 'validation_selected', 3] - arr['BCE_Dice', seed, 'validation_selected', 3]
        for unit, labels in [('observation', ids), ('temporal_group', groups)]:
            ci = boot(d, labels, seed + 71)
            for j, m in enumerate(METRICS):
                r = next(r for r in legacy_ci if int(r['seed']) == seed and r['comparison'] == 'SABR-BCE_Dice' and r['metric'] == m and r['resampling_unit'] == unit)
                error = max(abs(ci[0, j] - float(r['ci95_low'])), abs(ci[1, j] - float(r['ci95_high'])), abs(d[:, j].mean() - float(r['mean_delta'])))
                assert error <= 1e-14
                parity.append({'seed': seed, 'metric': m, 'resampling_unit': unit, 'max_abs_error': float(error)})
    write('legacy_bootstrap_parity.csv', parity)
    intervals = []; sensitivity = []; effects = []
    for seed in seeds:
        for comparator in ['FlatUNet_BCE', 'UNet_softDice_clDice']:
            for mode in MODES:
                missing = comparator == 'UNet_softDice_clDice' and mode == 'validation_selected'
                for radius in RADII:
                    d = None if missing else arr['SABR', seed, mode, radius][:, :7] - arr[comparator, seed, mode, radius][:, :7]
                    base = {'seed': seed, 'comparison': 'SABR-' + comparator, 'mode': mode, 'radius': radius, 'status': 'INFEASIBLE_NA' if missing else 'OK', 'na_reason': NA_REASON if missing else ''}
                    for j, m in enumerate(METRICS): effects.append({**base, 'metric': m, 'mean_delta': 'NA' if missing else float(d[:, j].mean()), 'n_observations': 0 if missing else 108, 'planned_observations': 108})
                    for unit, labels in [('observation', ids), ('temporal_group', groups)]:
                        ci = None if missing else boot(d, labels, seed + 71)
                        for j, m in enumerate(METRICS): intervals.append({**base, 'metric': m, 'resampling_unit': unit, 'n_units': 0 if missing else len(set(labels)), 'planned_units': len(set(labels)), 'mean_delta': 'NA' if missing else float(d[:, j].mean()), 'ci95_low': 'NA' if missing else float(ci[0, j]), 'ci95_high': 'NA' if missing else float(ci[1, j]), 'resamples': 0 if missing else 100000, 'planned_resamples': 100000, 'rng_seed': seed + 71})
                    for scope, labels in [('year', years), ('leave_one_temporal_group_out', groups)]:
                        for label in sorted(set(labels)):
                            keep = np.array(labels) == label
                            if scope.startswith('leave'): keep = ~keep
                            for j, m in enumerate(METRICS): sensitivity.append({**base, 'scope': scope, 'label': label, 'n_observations': 0 if missing else int(keep.sum()), 'planned_observations': int(keep.sum()), 'metric': m, 'mean_delta': 'NA' if missing else float(d[keep, j].mean())})
                    print(f'Frozen statistics completed: {seed} {comparator} {mode} native radius {radius}', flush=True)
    write('new_paired_intervals.csv', intervals); write('new_sensitivity.csv', sensitivity); write('new_paired_effects.csv', effects)
    multieffects = []
    for comparator in ['FlatUNet_BCE', 'UNet_softDice_clDice']:
        for mode in MODES:
            missing = comparator == 'UNet_softDice_clDice' and mode == 'validation_selected'
            for radius in RADII:
                for m in METRICS:
                    rr = [r for r in effects if r['comparison'] == 'SABR-' + comparator and r['mode'] == mode and r['radius'] == radius and r['metric'] == m]
                    assert len(rr) == 3
                    a = None if missing else np.array([r['mean_delta'] for r in rr])
                    multieffects.append({'comparison': 'SABR-' + comparator, 'mode': mode, 'radius': radius, 'metric': m, 'status': 'INFEASIBLE_NA' if missing else 'OK', 'na_reason': NA_REASON if missing else '', 'n_seeds': 0 if missing else 3, 'planned_seeds': 3, 'delta_mean': 'NA' if missing else float(a.mean()), 'delta_sample_sd': 'NA' if missing else float(a.std(ddof=1))})
    write('new_paired_multiseed.csv', multieffects)
    assert len(allrows) == 13608 and sum(r['status'] == 'OK' for r in allrows) == 12636
    assert len(points) == 126 and len(multi) == 42 and len(intervals) == 504
    assert sum(r['status'] == 'INFEASIBLE_NA' for r in intervals) == 126
    unchanged = verify_inputs()
    assert input_count == unchanged
    dump('STATISTICS_QA.json', {'status': 'FROZEN_STATISTICS_COMPLETE', 'all_21_models_included': True, 'no_old_result_overwritten': True, 'old_mean_parity_tolerance': 1e-15, 'old_bootstrap_checks': len(parity), 'old_bootstrap_max_error': max(r['max_abs_error'] for r in parity), 'bootstrap_resamples': 100000, 'rng': 'numpy.default_rng(seed+71), original unmodified boot function', 'python': sys.version, 'numpy': np.__version__, 'observation_rows': len(allrows), 'numeric_observation_rows': 12636, 'NA_observation_rows': 972, 'per_seed_rows': len(points), 'multiseed_rows': len(multi), 'interval_rows': len(intervals), 'numeric_interval_rows': 378, 'NA_interval_rows': 126, 'sensitivity_rows': len(sensitivity), 'input_files_rechecked_unchanged': unchanged, 'selected_soft_cldice_filled_from_fixed': False, 'confirmatory_significance_claimed': False, 'external_inference_performed': False})
    print('Statistics complete; all original input hashes unchanged.', flush=True)

if __name__ == '__main__':
    stage = sys.argv[1]
    if stage == 'audit':
        try: audit()
        except Exception as e:
            dump('ACCEPTANCE_FAILED.json', {'status': 'ACCEPTANCE_FAILED', 'error': str(e), 'checks': CHECKS}); raise
    elif stage == 'analyze': analyze()
    else: raise ValueError(stage)
