"""Global-batch semantics for two ranks; never average separate clDice ratios."""
import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_reduce
from methods import bce_dice_loss


def global_batches(size, seed, epoch):
    assert size > 0 and size % 2 == 0, 'Global batch2 requires an even training set; no padding or dropping'
    generator = torch.Generator().manual_seed(seed + epoch * 100003)
    # Use the real historical DataLoader, including its base-seed consumption.
    loader = torch.utils.data.DataLoader(range(size), batch_size=2, shuffle=True,
                                        generator=generator, num_workers=0)
    return [batch.tolist() for batch in loader]


def rank_loader(dataset, seed, epoch, rank):
    dataset.epoch = epoch
    batches = global_batches(len(dataset), seed, epoch)
    indices = [batch[rank] for batch in batches]
    return torch.utils.data.DataLoader(dataset, batch_size=1, sampler=indices,
                                      generator=torch.Generator().manual_seed(seed + epoch * 100003),
                                      num_workers=0, pin_memory=torch.cuda.is_available())


def global_loss(logits, target, loss_module):
    assert dist.is_initialized() and dist.get_world_size() == 2
    assert logits.shape == target.shape and logits.shape[0] == 1
    assert logits.dtype == target.dtype == torch.float32
    probability = torch.sigmoid(logits)
    skeleton = loss_module.structure.soft_skeletonize
    predicted = skeleton(probability)
    reference = skeleton(target)
    stats = torch.stack((bce_dice_loss(logits, target),
                         (predicted * target).sum(), predicted.sum(),
                         (reference * probability).sum(), reference.sum()))
    # Autograd-aware SUM is required. Backward sums the two identical global
    # objectives, and DDP's gradient average cancels that factor of two.
    total = all_reduce(stats, op=dist.ReduceOp.SUM)
    precision = (total[1] + 1.0) / (total[2] + 1.0)
    sensitivity = (total[3] + 1.0) / (total[4] + 1.0)
    topology = 1.0 - 2.0 * precision * sensitivity / (precision + sensitivity)
    return total[0] / 2.0 + 0.05 * topology


def any_rank(flag, device):
    value = torch.tensor(int(bool(flag)), device=device, dtype=torch.int32)
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return bool(value.item())


def gather_rng(local):
    result = [None] * dist.get_world_size()
    dist.all_gather_object(result, local)
    return result
