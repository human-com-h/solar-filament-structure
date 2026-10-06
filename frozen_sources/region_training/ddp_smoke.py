"""Embedded selected-seed T4x2 checks, then the same notebook trains that seed."""
from pathlib import Path
import argparse, gc, json, subprocess, sys, time
from methods import HERE, METHODS
from runtime import *
from smoke import scientific_state


def torchrun(script, arguments, log, timeout):
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
               '--nproc-per-node=2', str(HERE/script), *map(str, arguments)]
    print('双卡内嵌检查', script, flush=True)
    with Path(log).open('w', encoding='utf-8') as stream:
        run_process(command, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)


def main(a):
    root = Path(a.output)
    root.mkdir(parents=True, exist_ok=True)
    report = {'status': 'NOT_READY', 'code_sha256': verify_code(), 'environment': environment(),
              'environment_sha256': digest(environment()), 'formal_training_started': False,
              'execution_backend': 'DDP2_SyncBatchNorm_global_loss', 'selected_seed': a.seed}
    dump(root/'SMOKE_REPORT.json', report)
    torchrun('ddp_parity.py', ['--output',root/'parity'], root/'parity.log', a.timeout)
    report['global_batch_parity'] = json.loads((root/'parity/GLOBAL_BATCH_PARITY.json').read_text())
    method = METHODS[0]
    runid = f'{method}_seed{a.seed}'
    reference = None
    for directory, stop, action in [('continuous',0,'train'), ('resumed',1,'train'),
                                   ('resumed',0,'train'), ('resumed',0,'export')]:
        arguments = [action,'--method',method,'--seed',a.seed,'--smoke','--json-path',a.json_path,
                     '--image-dir',a.image_dir,'--cache',a.cache,'--output',root/directory,
                     '--stop-after-epoch',stop]
        torchrun('ddp_engine.py', arguments, root/f'{directory}_{action}_{stop}.log', a.timeout)
        if directory == 'continuous':
            saved = load_archive(root/directory/runid/'recovery.zip')
            reference = scientific_state(saved)
            from smoke import fingerprint
            reference['rng_by_rank'] = fingerprint(saved['rng_by_rank'])
            dump(root/'continuous_scientific_fingerprint.json',reference)
            del saved
            gc.collect()
            # Remove only this disposable synthetic-engineering comparison archive.
            (root/directory/runid/'recovery.zip').unlink()
    saved = load_archive(root/'resumed'/runid/'recovery.zip')
    actual = scientific_state(saved)
    from smoke import fingerprint
    actual['rng_by_rank'] = fingerprint(saved['rng_by_rank'])
    assert actual == reference, 'Two-rank full-state resume differs'
    report.update({'status': 'TARGET_SMOKE_PASSED', 'passed_methods': list(METHODS),
                   'resume_model_optimizer_two_rank_rng_best_history_exact': True,
                   'history_timing_not_compared': True, 'validation_export': True,
                   'resources': json.loads((root/'resumed'/runid/'archive_resources.json').read_text()),
                   'history': saved['history']})
    del saved
    gc.collect()
    dump(root/'SMOKE_REPORT.json', report)
    print('同一seed双卡训练/恢复/导出检查通过', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('json-path','image-dir','cache','output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--timeout', type=int, default=3600)
    main(parser.parse_args())
