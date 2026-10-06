"""Two real CPU/Gloo ranks: global loss gradients, full-state resume and sampling."""
from pathlib import Path
from datetime import timedelta
import argparse, gc, json, tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from methods import METHODS, DESIGN, criterion
from runtime import *
from distributed_math import global_batches, global_loss, gather_rng
from ddp_parity import probe, simple_model
from local_smoke import same


def cpu_step(model, optimizer, epoch):
    generator = torch.Generator().manual_seed(884 + epoch)
    images = torch.rand(2, 1, 32, 32, generator=generator)
    targets = (torch.rand(2, 1, 32, 32, generator=generator) > 0.82).float()
    rank = dist.get_rank()
    optimizer.zero_grad(set_to_none=True)
    loss = global_loss(model(images[rank:rank+1]), targets[rank:rank+1], criterion(METHODS[0]))
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def worker(rank, uri, output):
    dist.init_process_group('gloo', init_method=uri, rank=rank, world_size=2, timeout=timedelta(minutes=5))
    try:
        probe(rank, torch.device('cpu'), output)
        seed_all(20260831)
        raw = simple_model()
        model = DDP(raw)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        first_loss = cpu_step(model, optimizer, 1)
        states = gather_rng(rng())
        if rank == 0:
            save_archive({'model': raw.state_dict(), 'optimizer': optimizer.state_dict(),
                          'cfg': {'purpose': 'CPU synthetic DDP resume'}, 'epoch': 1,
                          'rng_by_rank': states}, Path(output)/'resume', output)
        dist.barrier()
        second_loss = cpu_step(model, optimizer, 2)
        reference = {'model': {k:v.clone() for k,v in raw.state_dict().items()},
                     'optimizer': optimizer.state_dict(), 'rng': rng()}
        del model, raw, optimizer
        gc.collect()
        saved = load_archive(Path(output)/'resume/recovery.zip')
        raw = simple_model()
        model = DDP(raw)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        raw.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        restore_rng(saved['rng_by_rank'][rank])
        resumed_loss = cpu_step(model, optimizer, 2)
        assert second_loss == resumed_loss
        assert same(raw.state_dict(), reference['model'])
        assert same(optimizer.state_dict(), reference['optimizer'])
        assert same(rng(), reference['rng'])
        digest_by_rank = [None, None]
        dist.all_gather_object(digest_by_rank, model_digest(raw))
        assert digest_by_rank[0] == digest_by_rank[1]
        if rank == 0:
            dump(Path(output)/'CPU_DDP_RESUME.json', {
                'status': 'PASS_TWO_RANK_STATE_RESUME', 'model_optimizer_rng_loss_exact': True,
                'two_ranks_final_model_equal': True, 'first_loss': first_loss,
                'second_loss': second_loss, 'real_data_used': False})
    finally:
        dist.destroy_process_group()


def main(output):
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    sampling = {}
    for seed in DESIGN['seeds']:
        for epoch in (1, 2, 150):
            batches = global_batches(492, seed, epoch)
            loader = torch.utils.data.DataLoader(range(492), batch_size=2, shuffle=True,
                generator=torch.Generator().manual_seed(seed + epoch*100003), num_workers=0)
            assert batches == [b.tolist() for b in loader]
            shards = [[b[rank] for b in batches] for rank in range(2)]
            reconstructed = [[a,b] for a,b in zip(*shards)]
            assert reconstructed == batches and sorted(sum(batches, [])) == list(range(492))
            sampling[f'{seed}:{epoch}'] = digest(batches)
    with tempfile.TemporaryDirectory(prefix='ddp_cpu_', dir=root) as temporary:
        workspace = Path(temporary)
        uri = (workspace/'store').as_uri()
        mp.spawn(worker, args=(uri, str(workspace)), nprocs=2, join=True)
        parity = json.loads((workspace/'GLOBAL_BATCH_PARITY.json').read_text())
        resume = json.loads((workspace/'CPU_DDP_RESUME.json').read_text())
    report = {'status': 'DDP_CPU_SMOKE_PASSED', 'code_sha256': verify_code(),
              'two_real_CPU_Gloo_ranks': True, 'global_loss_value_gradient_parity': parity,
              'full_state_resume': resume, 'historical_global_batch_order': sampling,
              'no_training_padding_or_drops': True, 'formal_GPU_training_started': False,
              'limits': 'Full U-Net SyncBatchNorm at 1024 and NCCL speed/resources checked only inside the normal Kaggle notebook'}
    dump(root/'DDP_CPU_SMOKE.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    main(parser.parse_args().output)
