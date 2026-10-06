"""Synthetic value/gradient parity against the canonical global batch2 loss."""
from pathlib import Path
import gc
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from methods import METHODS, factory, criterion
from runtime import seed_all, dump
from distributed_math import global_loss


def simple_model():
    return torch.nn.Sequential(torch.nn.Conv2d(1, 4, 3, padding=1), torch.nn.Tanh(),
                               torch.nn.Conv2d(4, 1, 1))


def probe(rank, device, output, full_unet=False):
    side = 1024 if full_unet else 32
    loss_fn = criterion(METHODS[0])
    cases = ['asymmetric_mixed'] if full_unet else ['asymmetric_mixed', 'empty', 'full']
    results = []
    for case in cases:
        generator = torch.Generator().manual_seed(816)
        images = torch.rand(2, 1, side, side, generator=generator)
        targets = torch.zeros_like(images)
        if case == 'full':
            targets.fill_(1)
        elif case != 'empty':
            targets[0, :, side//5:side//2, side//4:side//3] = 1
            targets[1, :, side//7:4*side//5, side//2:3*side//4] = 1
        reference = None
        if rank == 0:
            seed_all(20260831)
            model = (factory(METHODS[0]) if full_unet else simple_model()).to(device)
            model.train()
            logits = model(images.to(device))
            loss = loss_fn(logits, targets.to(device))
            loss.backward()
            reference = {
                'loss': float(loss.detach()), 'logits': logits.detach().cpu(),
                'gradients': {name: p.grad.detach().cpu().clone() for name, p in model.named_parameters()},
                'buffers': {name: v.detach().cpu().clone() for name, v in model.named_buffers()}}
            del model, logits, loss
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        dist.barrier()
        seed_all(20260831)
        raw = (factory(METHODS[0]) if full_unet else simple_model()).to(device)
        if full_unet:
            raw = torch.nn.SyncBatchNorm.convert_sync_batchnorm(raw)
        model = DDP(raw, device_ids=[rank] if device.type == 'cuda' else None)
        raw.train()
        logits = model(images[rank:rank+1].to(device))
        loss = global_loss(logits, targets[rank:rank+1].to(device), loss_fn)
        loss.backward()
        gathered = [torch.empty_like(logits) for _ in range(2)]
        dist.all_gather(gathered, logits.detach())
        if rank == 0:
            tolerance = {'rtol': 5e-3, 'atol': 1e-4} if full_unet else {'rtol': 2e-5, 'atol': 2e-6}
            assert abs(float(loss.detach()) - reference['loss']) <= (5e-5 if full_unet else 2e-6)
            actual_logits = torch.cat(gathered).cpu()
            torch.testing.assert_close(actual_logits, reference['logits'], **tolerance)
            gradient_error = 0.0
            for name, parameter in raw.named_parameters():
                actual, expected = parameter.grad.detach().cpu(), reference['gradients'][name]
                torch.testing.assert_close(actual, expected, **tolerance)
                gradient_error = max(gradient_error, float((actual - expected).abs().max()))
            buffer_error = 0.0
            for name, value in raw.named_buffers():
                actual, expected = value.detach().cpu(), reference['buffers'][name]
                torch.testing.assert_close(actual, expected, **tolerance)
                buffer_error = max(buffer_error, float((actual - expected).abs().max()))
            results.append({'case': case, 'global_loss_abs_error': abs(float(loss.detach()) - reference['loss']),
                            'logit_max_abs_error': float((actual_logits - reference['logits']).abs().max()),
                            'parameter_gradient_max_abs_error': gradient_error,
                            'buffer_max_abs_error': buffer_error, 'tolerance': tolerance,
                            'global_batch_size': 2, 'per_rank_batch_size': 1})
        dist.barrier()
        del model, raw, loss, logits, gathered, reference
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    if rank == 0:
        dump(Path(output) / 'GLOBAL_BATCH_PARITY.json', {
            'status': 'PASS_GLOBAL_BATCH_VALUE_GRADIENT_PARITY', 'cases': results,
            'full_unet': full_unet, 'shape': [2, 1, side, side],
            'synchronized_batch_norm_checked': full_unet, 'real_data_used': False,
            'bitwise_single_GPU_trajectory_claimed': False})


if __name__ == '__main__':
    import os, argparse
    from datetime import timedelta
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', timeout=timedelta(minutes=20))
    try:
        probe(rank, torch.device('cuda', rank), args.output, full_unet=True)
    finally:
        dist.destroy_process_group()
