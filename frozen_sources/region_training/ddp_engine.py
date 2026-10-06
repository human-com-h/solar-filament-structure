"""One selected seed, two GPU ranks, global batch2 and synchronized BatchNorm."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from datetime import timedelta
from pathlib import Path
import gc, json, time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from PIL import Image
from methods import HERE, DESIGN, METHODS, factory, criterion, probability
from data import Dataset, old
from runtime import *
from distributed_math import rank_loader, global_loss, any_rank, gather_rng
from engine import identity, require_release, parser


@torch.no_grad()
def validate(model, cache, method, smoke, rank, device):
    model.eval()
    values = []
    groups = old.validation_groups(2 if smoke else 0)
    for index, (oid, aids) in enumerate(groups):
        if index % 2 != rank:
            continue
        with Image.open(Path(cache) / 'images' / f'{oid}.png') as im:
            image = np.asarray(im, dtype=np.float32) / 255.0
        p = probability(model(torch.from_numpy(image[None, None]).to(device)), method)
        assert torch.isfinite(p).all() and p.min() >= 0 and p.max() <= 1
        pred = (p[0, 0] >= 0.5).cpu().numpy()
        scores = []
        for aid in aids:
            with np.load(Path(cache) / 'targets' / f'{aid}.npz') as archive:
                target = archive['mask']
            scores.append(float((2 * (pred & target).sum() + 1e-6) /
                                (pred.sum() + target.sum() + 1e-6)))
        values.append(float(np.mean(scores)))
    sums = torch.tensor([sum(values), len(values)], device=device, dtype=torch.float64)
    dist.all_reduce(sums)
    assert int(sums[1]) == len(groups)
    return float((sums[0] / sums[1]).item())


def train(a, rank, device):
    cfg = identity(a)
    require_release(a, cfg)
    run = Path(a.output) / f'{a.method}_seed{a.seed}'
    run.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        cp = run / 'config.json'
        if cp.exists():
            assert json.loads(cp.read_text()) == cfg, 'Resume identity/config drift'
        else:
            dump(cp, cfg)
    dist.barrier()
    seed_all(a.seed)
    raw = factory(a.method).to(device)
    initial = model_digest(raw)
    matches = initial == json.loads((HERE / 'INITIAL_DIGESTS.json').read_text())[str(a.seed)]
    if not a.smoke:
        assert matches, 'Historical initialization mismatch'
    raw = torch.nn.SyncBatchNorm.convert_sync_batchnorm(raw)
    model = DDP(raw, device_ids=[rank], output_device=rank,
                broadcast_buffers=True, gradient_as_bucket_view=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    loss_fn = criterion(a.method)
    best, stale, history, start, best_state, best_epoch = -1.0, 0, [], 1, None, 0
    archive = run / 'recovery.zip'
    if archive.exists():
        saved = load_archive(archive)
        assert saved['cfg'] == cfg and saved['initial_model_digest'] == initial
        assert len(saved['rng_by_rank']) == 2
        raw.load_state_dict(saved['model'])
        opt.load_state_dict(saved['optimizer'])
        for state in opt.state.values():
            for key, value in state.items():
                if torch.is_tensor(value) and key != 'step':
                    state[key] = value.to(device)
        best, stale, history, start = saved['best'], saved['stale'], saved['history'], saved['epoch'] + 1
        best_state, best_epoch = saved['best_model'], saved['best_epoch']
        restore_rng(saved['rng_by_rank'][rank])
        del saved
        gc.collect()
    dataset = Dataset(a.cache, a.seed, a.smoke)
    paused, resources = False, None
    for epoch in range(start, cfg['max_epochs'] + 1):
        if stale >= 20:
            break
        last = history[-1]['seconds'] if history else float(a.epoch_estimate)
        reserve = max(900, 2 * float((resources or {}).get('archive_seconds', 0)))
        if any_rank(a.deadline and time.time() + last * 1.25 + reserve >= a.deadline, device):
            paused = True
            break
        loader = rank_loader(dataset, a.seed, epoch, rank)
        model.train()
        losses, steps = [], []
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        with Monitor(a.output) as monitor:
            for step, (x, y, oids, aids) in enumerate(loader, 1):
                if any_rank(a.deadline and time.time() + reserve >= a.deadline, device):
                    paused = True
                    break
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                torch.cuda.synchronize(device)
                t = time.perf_counter()
                opt.zero_grad(set_to_none=True)
                loss = global_loss(model(x), y, loss_fn)
                if any_rank(not torch.isfinite(loss).item(), device):
                    raise RuntimeError('Non-finite global loss')
                loss.backward()
                gradients = [p.grad for p in model.parameters() if p.requires_grad]
                missing = any(g is None for g in gradients)
                bad = missing or not torch.stack([torch.isfinite(g).all() for g in gradients]).all().item()
                if any_rank(bad, device):
                    raise RuntimeError('Missing/non-finite distributed gradient')
                opt.step()
                torch.cuda.synchronize(device)
                steps.append(time.perf_counter() - t)
                losses.append(float(loss.detach()))
                if rank == 0 and (step == 1 or step % 50 == 0 or step == len(loader)):
                    print(f'seed={a.seed} epoch={epoch} step={step}/{len(loader)} loss={losses[-1]:.6f} elapsed={time.perf_counter()-started:.1f}s', flush=True)
                del x, y, loss
            if paused:
                break
            train_seconds = time.perf_counter() - started
            val_started = time.perf_counter()
            score = validate(raw, a.cache, a.method, a.smoke, rank, device)
            val_seconds = time.perf_counter() - val_started
        peaks = [None, None]
        dist.all_gather_object(peaks, {
            'rank': rank, 'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(device),
            'peak_gpu_reserved_bytes': torch.cuda.max_memory_reserved(device),
            'peak_process_rss_bytes': monitor.ram, 'peak_system_used_bytes': monitor.system_used})
        elapsed = torch.tensor(time.perf_counter() - started, device=device, dtype=torch.float64)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        row = {'epoch': epoch, 'train_loss': float(np.mean(losses)), 'val_physical_macro_dice': score,
               'seconds': float(elapsed.item()), 'training_seconds': train_seconds,
               'validation_seconds': val_seconds, 'step_seconds': steps, 'resources_by_rank': peaks}
        history.append(row)
        if score > best + 1e-6:
            best, stale, best_epoch = score, 0, epoch
            if rank == 0:
                best_state = {k: v.detach().cpu().clone() for k, v in raw.state_dict().items()}
        else:
            stale += 1
        all_rng = gather_rng(rng())
        if rank == 0:
            state = {'model': raw.state_dict(), 'optimizer': opt.state_dict(), 'best_model': best_state,
                     'best': best, 'best_epoch': best_epoch, 'stale': stale, 'history': history, 'epoch': epoch,
                     'cfg': cfg, 'rng': all_rng[0], 'rng_by_rank': all_rng,
                     'initial_model_digest': initial, 'initialization_matches_archived': matches}
            resources = save_archive(state, run, a.output)
            dump(run / 'archive_resources.json', resources)
            dump(run / 'history.json', history)
            print(f'seed={a.seed} epoch={epoch} train={train_seconds:.1f}s val={val_seconds:.1f}s saved={resources["archive_seconds"]:.1f}s best_epoch={best_epoch}', flush=True)
        box = [resources]
        dist.broadcast_object_list(box, src=0)
        resources = box[0]
        dist.barrier()
        if a.stop_after_epoch and epoch >= a.stop_after_epoch:
            paused = True
            break
    complete = bool(history) and (stale >= 20 or history[-1]['epoch'] >= cfg['max_epochs'])
    if rank == 0:
        dump(run / 'STATUS.json', {'complete': complete, 'paused': paused, 'epochs': len(history),
                                  'initialization_matches_archived': matches, 'smoke': a.smoke,
                                  'execution_backend': 'DDP2_SyncBatchNorm_global_loss'})
    dist.barrier()
    return complete


def export(a, rank, device):
    import frozen_evaluator as fe
    from threshold_curve import THRESHOLDS, FIELDS, curve, reference_check
    seed_all(a.seed)
    cfg = identity(a)
    run = Path(a.output) / f'{a.method}_seed{a.seed}'
    saved = load_archive(run / 'recovery.zip')
    assert saved['cfg'] == cfg and (saved['epoch'] >= cfg['max_epochs'] or saved['stale'] >= 20)
    model = factory(a.method).to(device)
    model.load_state_dict(saved['best_model'])
    del saved
    gc.collect()
    model.eval()
    ids = [x[0] for x in old.validation_groups(2 if a.smoke else 0)]
    coco = json.loads(Path(a.json_path).read_text())
    excluded = old.exclusions()
    allowed = {im['id'] for im in coco['images'] if fe.physical_id(im['file_name']) in set(ids)}
    coco['images'] = [im for im in coco['images'] if im['id'] in allowed]
    coco['annotations'] = [item for item in coco['annotations'] if item['image_id'] in allowed]
    for item in coco['annotations']:
        if str(item['id']) in excluded:
            item['spine'] = []
    gt = fe.build_gt(coco, set(ids))
    paths = {fe.physical_id(im['file_name']): Path(a.image_dir) / im['file_name'] for im in coco['images']}
    dataset = fe.Images(ids, paths)
    out = run / 'validation'
    out.mkdir(exist_ok=True)
    sig = {'cfg': cfg, 'archive_sha256': sha(run / 'recovery.zip'), 'ids': ids,
           'thresholds': [float(t) for t in THRESHOLDS]}
    if rank == 0:
        if (out / 'signature.json').exists():
            assert json.loads((out / 'signature.json').read_text()) == sig
        else:
            dump(out / 'signature.json', sig)
    dist.barrier()
    for index in range(rank, len(dataset), 2):
        if a.deadline and time.time() + 900 >= a.deadline:
            break
        x, oid, size = dataset[index]
        dest = out / f'{oid}.json'
        if dest.exists():
            assert json.loads(dest.read_text())['observation_id'] == oid
            continue
        with torch.inference_mode():
            p = probability(model(x[None].to(device)), a.method).cpu().numpy()[0, 0]
        assert np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1
        native = np.asarray(Image.fromarray(p).resize(size, Image.Resampling.NEAREST))
        scores = curve(native, gt[oid])
        parity = reference_check(oid, native, gt[oid], scores) if index == 0 else None
        dump(dest, {'observation_id': oid, 'scores': scores.tolist(), 'reference_check': parity})
        print(f'validation export seed={a.seed} rank={rank} observation={index+1}/{len(ids)}', flush=True)
    dist.barrier()
    complete = all((out / f'{oid}.json').exists() for oid in ids)
    if rank == 0 and complete:
        means = np.mean([json.loads((out / f'{oid}.json').read_text())['scores'] for oid in ids], axis=0)
        rows = [{'threshold': float(t), 'n_observations': len(ids), **dict(zip(FIELDS, map(float, means[i])))}
                for i, t in enumerate(THRESHOLDS)]
        fe.write_csv(out / 'curves.csv', rows, list(rows[0]))
        dump(run / 'VALIDATION_COMPLETE.json', sig)
    dist.barrier()
    return complete


def main():
    a = parser().parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    assert int(os.environ['WORLD_SIZE']) == 2
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    dist.init_process_group('nccl', timeout=timedelta(minutes=20))
    run = Path(a.output) / f'{a.method}_seed{a.seed}'
    # Only rank0 owns the run; peers synchronize after ownership is established.
    try:
        if rank == 0:
            with run_lock(run / '.run.lock'):
                dist.barrier()
                (train if a.action == 'train' else export)(a, rank, device)
        else:
            dist.barrier()
            (train if a.action == 'train' else export)(a, rank, device)
    except BaseException as error:
        dump(run / f'FAILURE_rank{rank}.json', {'error': repr(error), 'type': type(error).__name__, 'rank': rank})
        raise
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
