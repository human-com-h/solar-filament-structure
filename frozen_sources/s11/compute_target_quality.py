"""CPU-only census of the unchanged frozen annotation-derived targets."""
from pathlib import Path
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
import ast, csv, hashlib, json, sys, platform, shutil
import numpy as np
import cv2, scipy, skimage
from skimage.morphology import skeletonize, dilation, diamond

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PKG = ROOT / 'project/mechanism_training'
MANUSCRIPT = ROOT / 'project/region_clDice_analysis_20261004/manuscript'
ANN = ROOT / 'data/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'
cv2.setNumThreads(1)

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read_csv(p):
    with Path(p).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def write_csv(name, rows):
    if not rows:
        return
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

def function_namespace(path, names, namespace):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in selected} == set(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace

def backup():
    dest = OUT / 'before'
    if dest.exists():
        return
    dest.mkdir()
    for name in ['filament_structure_revised.md', 'SUPPLEMENTARY_MATERIAL.md', 'REFERENCE_METADATA.json', 'build_sources.py', 'compile_and_render.py']:
        shutil.copy2(MANUSCRIPT / name, dest / name)
    working = ROOT / 'project/filament_structure_working_draft.md'
    shutil.copy2(working, dest / working.name)
    (dest / 'study_project').mkdir()
    for name in ['filament_structure_main.tex', 'filament_structure_main.pdf', 'filament_structure_supplement.tex', 'filament_structure_supplement.pdf', 'references.bib']:
        shutil.copy2(MANUSCRIPT / 'study_project' / name, dest / 'study_project' / name)
    snapshot = {str(p.relative_to(dest)): sha(p) for p in dest.rglob('*') if p.is_file()}
    (dest / 'SNAPSHOT.json').write_text(json.dumps(snapshot, indent=2), encoding='utf-8')

def describe(values):
    a = np.asarray(values, dtype=float)
    return dict(n=int(len(a)), minimum=float(a.min()) if len(a) else None,
                q25=float(np.quantile(a,.25)) if len(a) else None,
                median=float(np.median(a)) if len(a) else None,
                q75=float(np.quantile(a,.75)) if len(a) else None,
                p95=float(np.quantile(a,.95)) if len(a) else None,
                maximum=float(a.max()) if len(a) else None,
                mean=float(a.mean()) if len(a) else None)

def main():
    backup()
    cfg = json.loads((PKG / 'protocol.json').read_text())
    assert sha(ANN) == cfg['annotation_sha256']
    assert sha(PKG / 'temporal_manifest.csv') == cfg['manifest_sha256']
    for name, info in cfg['source_provenance'].items():
        assert sha(PKG / name) == info['sha256'], name
    sys.path.insert(0, str(PKG))
    import branch_reference as br
    reasons = ['missing_spine','missing_polygon','degenerate_bbox','skeleton_under_two_pixels',
               'collapsed_endpoints','route_leaves_skeleton_or_under_two_pixels']
    lines = (PKG / 'branch_reference.py').read_text().splitlines()
    ri = 0
    for i, line in enumerate(lines):
        if line.strip() == 'return []':
            lines[i] = line.replace('return []', 'return [], ' + repr(reasons[ri])); ri += 1
        elif line.strip() == 'return comps':
            lines[i] = line.replace('return comps', "return comps, 'accepted_path'")
    assert ri == len(reasons)
    instrumented = {}
    exec(compile('\n'.join(lines), 'instrumented_frozen_branch_reference', 'exec'), instrumented)
    frozen = function_namespace(PKG / 'data_pipeline.py', ['label_copy', 'exclusions'],
                                dict(np=np, HERE=PKG, csv=csv, branch_components_from_instance=br.branch_components_from_instance))
    rendering = function_namespace(PKG / 'annotation_helpers.py', ['render_union_mask'], dict(np=np, cv2=cv2))
    qc_rows = read_csv(PKG / 'manual_spine_qc_exclusions.csv')
    qc = {r['annotation_id'] for r in qc_rows}
    manifest = read_csv(PKG / 'temporal_manifest.csv')
    data = json.loads(ANN.read_text())
    images = {str(im['id']): im for im in data['images']}
    anns = defaultdict(list)
    for a in data['annotations']:
        anns[str(a['image_id'])].append(a)
    copies, instances, components = [], [], []

    def process(r):
        sid = r['annotation_sample_id']; im = images[sid]
        meta = dict(sample_id=sid, observation_id=r['physical_observation_id'],
                    split=r['split'], year=int(r['year']), temporal_group=int(r['temporal_group']))
        labels = np.zeros((1024,1024), np.uint16); lid = 0
        instrows, comprows = [], []
        for a in anns[sid]:
            ir = dict(**meta, annotation_id=str(a['id']), status='QC_EXCLUDED',
                      raw_components=0, preownership_eligible_components=0, retained_components=0,
                      small_components_rejected=0, overlap_components_dropped=0, overlap_pixels_removed=0)
            if str(a['id']) not in qc:
                comps, reason = instrumented['branch_components_from_instance'](a,int(im['width']),int(im['height']),1024)
                ir['status'] = reason
                ir['raw_components'] = len(comps)
                for ci, comp in enumerate(comps):
                    size = len(comp)
                    cr = dict(**meta, annotation_id=str(a['id']), raw_component_index=ci,
                              raw_size=size, retained_size=0, final_label=0, status='SIZE_REJECTED')
                    if size < 5:
                        ir['small_components_rejected'] += 1
                    else:
                        ir['preownership_eligible_components'] += 1
                        yy, xx = comp[:,0], comp[:,1]; free = labels[yy,xx] == 0
                        yy, xx = yy[free], xx[free]
                        ir['overlap_pixels_removed'] += size - len(yy)
                        if not len(yy):
                            cr['status'] = 'OWNERSHIP_DROPPED'; ir['overlap_components_dropped'] += 1
                        else:
                            lid += 1; labels[yy,xx] = lid
                            cr.update(retained_size=int(len(yy)), final_label=lid, status='RETAINED')
                            ir['retained_components'] += 1
                    comprows.append(cr)
            instrows.append(ir)
        # Execute the original function, unmodified, and compare all copy label arrays.
        np.testing.assert_array_equal(labels, frozen['label_copy'](im, anns[sid]))
        if r['split'] == 'internal_test':
            with np.load(ROOT / 'project/mechanism_branch_test/labels' / (sid + '.npz')) as f:
                np.testing.assert_array_equal(labels, f['labels'])
        mask = rendering['render_union_mask'](sid,images,anns,1024).astype(bool)
        srl = dilation(skeletonize(mask), diamond(2)) & mask
        cr = dict(**meta, annotated_instances=len(anns[sid]),
                  nonexcluded_instances=sum(x['status']!='QC_EXCLUDED' for x in instrows),
                  srl_valid=int(srl.any()), residual_valid=int(lid>0), retained_components=lid,
                  retained_pixels=int((labels>0).sum()), regional_foreground_pixels=int(mask.sum()),
                  accepted_paths=sum(x['status']=='accepted_path' for x in instrows),
                  instances_with_retained_components=sum(x['retained_components']>0 for x in instrows))
        return cr, instrows, comprows

    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, (cr, ir, brs) in enumerate(pool.map(process, manifest)):
            copies.append(cr); instances.extend(ir); components.extend(brs)
            if (i+1)%50 == 0 or i+1 == len(manifest):
                print(f'CPU target census: {i+1}/{len(manifest)} copies; exact frozen-label parity passed', flush=True)
    observations = []
    groups = defaultdict(list)
    for r in copies: groups[r['observation_id']].append(r)
    for oid, rs in sorted(groups.items()):
        observations.append(dict(observation_id=oid,split=rs[0]['split'],year=rs[0]['year'],
            temporal_group=rs[0]['temporal_group'],copies=len(rs),
            copies_with_residual=sum(r['residual_valid'] for r in rs),
            any_residual=int(any(r['residual_valid'] for r in rs)),
            all_copies_residual=int(all(r['residual_valid'] for r in rs)),
            residual_copy_fraction=sum(r['residual_valid'] for r in rs)/len(rs),
            components_across_copies=sum(r['retained_components'] for r in rs),
            pixels_across_copies=sum(r['retained_pixels'] for r in rs)))
    for r in qc_rows:
        source = next(x for x in manifest if x['annotation_sample_id']==r['annotation_sample_id']) if 'annotation_sample_id' in r else None
        if source is None:
            source = next(x for x in manifest if x['annotation_sample_id']==r['sample_id'])
        r.update(split=source['split'], temporal_group=source['temporal_group'])
    summary = {}
    for split in ['train','validation','internal_test','all']:
        cc = [r for r in copies if split=='all' or r['split']==split]
        ii = [r for r in instances if split=='all' or r['split']==split]
        bb = [r for r in components if split=='all' or r['split']==split]
        oo = [r for r in observations if split=='all' or r['split']==split]
        retained = [r for r in bb if r['status']=='RETAINED']
        summary[split] = dict(observations=len(oo), copies=len(cc), annotated_instances=len(ii),
            temporal_groups=len({r['temporal_group'] for r in cc}),
            srl_valid_copies=sum(r['srl_valid'] for r in cc),
            residual_valid_copies=sum(r['residual_valid'] for r in cc),
            residual_valid_copy_fraction=np.mean([r['residual_valid'] for r in cc]).item(),
            observations_any_residual=sum(r['any_residual'] for r in oo),
            observations_all_copies_residual=sum(r['all_copies_residual'] for r in oo),
            observation_macro_residual_copy_fraction=np.mean([r['residual_copy_fraction'] for r in oo]).item(),
            instance_status_counts=dict(Counter(r['status'] for r in ii)),
            instances_with_retained_components=sum(r['retained_components']>0 for r in ii),
            raw_components=len(bb), small_components_rejected=sum(r['status']=='SIZE_REJECTED' for r in bb),
            eligible_components_before_ownership=sum(r['raw_size']>=5 for r in bb),
            overlap_components_dropped=sum(r['status']=='OWNERSHIP_DROPPED' for r in bb),
            overlap_pixels_removed=sum(r['overlap_pixels_removed'] for r in ii),
            retained_components=len(retained), retained_pixels=sum(r['retained_size'] for r in retained),
            retained_components_under_five=sum(r['retained_size']<5 for r in retained),
            retained_component_sizes=describe([r['retained_size'] for r in retained]),
            eligible_component_sizes_before_ownership=describe([r['raw_size'] for r in bb if r['raw_size']>=5]),
            components_per_residual_valid_copy=describe([r['retained_components'] for r in cc if r['residual_valid']]),
            nonempty_regional_mask_copies=sum(r['regional_foreground_pixels']>0 for r in cc))
    test = summary['internal_test']
    assert (test['observations_any_residual'],test['residual_valid_copies'],test['retained_components'],test['retained_pixels']) == (101,157,1934,17019)
    # Original diagnostic cohort additionally contains exactly eleven temporal groups.
    assert len({r['temporal_group'] for r in copies if r['split']=='internal_test' and r['residual_valid']}) == 11
    write_csv('target_quality_per_copy.csv', copies)
    write_csv('target_quality_per_instance.csv', instances)
    write_csv('target_quality_per_component.csv', components)
    write_csv('target_quality_per_observation.csv', observations)
    write_csv('qc_exclusions_with_partition.csv',qc_rows)
    input_paths = [ANN, PKG/'temporal_manifest.csv',PKG/'protocol.json', PKG/'branch_reference.py',
                   PKG/'data_pipeline.py',PKG/'annotation_helpers.py',PKG/'manual_spine_qc_exclusions.csv']
    payload = dict(summary=summary, input_sha256={str(p):sha(p) for p in input_paths},
        parity=dict(frozen_label_copies=len(copies),existing_test_label_copies=170,failures=0,
                    branch_diagnostic_counts=[101,157,11,1934,17019]),
        environment=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,
                         scipy=scipy.__version__,opencv=cv2.__version__,skimage=skimage.__version__),
        definitions=dict(resolution=1024,connectivity=8,min_size_before_ownership=5,
                         path_neighbourhood='diamond(1)',srl_neighbourhood='diamond(2)',
                         quantiles='numpy default linear interpolation',
                         no_new_gpu_training=True, no_changes_to_frozen_implementations=True,
                         description='Annotation-derived supervision eligibility census, not validated physical branch quality.'))
    (OUT/'TARGET_QUALITY_SUMMARY.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)

if __name__ == '__main__':
    main()
