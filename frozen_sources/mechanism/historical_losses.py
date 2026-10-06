
import torch
import torch.nn.functional as F


def soft_dice_score(logits, target, eps=1e-6):
    p = torch.sigmoid(logits)
    dims = tuple(range(1, p.ndim))
    inter = (p * target).sum(dims)
    den = p.sum(dims) + target.sum(dims)
    return ((2 * inter + eps) / (den + eps)).mean()


def bce_dice_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    return 0.5 * bce + 0.5 * (1.0 - soft_dice_score(logits, target))


def binary_dice(pred, target, eps=1e-6):
    pred = pred.float(); target = target.float()
    dims = tuple(range(1, pred.ndim))
    inter = (pred * target).sum(dims)
    den = pred.sum(dims) + target.sum(dims)
    return (2 * inter + eps) / (den + eps)


def soft_recall_per_sample(logits, structure_target, eps=1e-6):
    """Return B-vector loss and B-vector validity mask."""
    p = torch.sigmoid(logits)
    dims = tuple(range(1, p.ndim))
    den = structure_target.sum(dims)
    num = (p * structure_target).sum(dims)
    valid = den > 0
    loss = torch.zeros_like(den)
    loss[valid] = 1.0 - (num[valid] + eps) / (den[valid] + eps)
    return loss, valid


def branch_recall_per_sample(logits, label_maps, aids, branch_weights, agreement_aware=False, eps=1e-6):
    """Equal-per-branch recall.

    label_maps: Bx1xHxW integer component ids.
    branch_weights: dict[(annotation_sample_id, component_id)] -> agreement support [0,1].

    Unweighted SABR: every extracted lateral branch component receives equal weight.
    Agreement-aware SABR: branch components are weighted by independent-annotator
    spatial support. If a sample has no positively supported branch, its branch term
    is marked invalid and training falls back to pure SRL for that sample.
    """
    p = torch.sigmoid(logits)
    B = p.shape[0]
    sample_losses = p.new_zeros((B,))
    valid = torch.zeros(B, dtype=torch.bool, device=p.device)

    for b in range(B):
        labels = torch.unique(label_maps[b])
        labels = labels[labels > 0]
        recalls = []
        weights = []
        aid = str(aids[b])
        for lab in labels:
            lab_i = int(lab.item())
            m = (label_maps[b] == lab).float()
            den = m.sum()
            if den <= 0:
                continue
            recall = ((p[b] * m).sum() + eps) / (den + eps)
            w = float(branch_weights.get((aid, lab_i), 0.0)) if agreement_aware else 1.0
            if w > 0:
                recalls.append(recall)
                weights.append(w)

        if recalls:
            rr = torch.stack(recalls)
            ww = torch.as_tensor(weights, dtype=rr.dtype, device=rr.device)
            sample_losses[b] = 1.0 - (rr * ww).sum() / ww.sum().clamp_min(eps)
            valid[b] = True

    return sample_losses, valid


def mixed_srl_branch_loss(
    logits,
    srl_target,
    branch_labels,
    aids,
    branch_weights,
    branch_mix_alpha=0.5,
    agreement_aware=False,
):
    """Fixed total structural budget.

    For samples with valid branch supervision:
        L_conn = (1-alpha)*L_SRL + alpha*L_branch
    Otherwise:
        L_conn = L_SRL

    Thus adding branch supervision does not automatically double the structural
    regularization weight compared with the SRL baseline.
    """
    srl_loss, srl_valid = soft_recall_per_sample(logits, srl_target)
    branch_loss, branch_valid = branch_recall_per_sample(
        logits, branch_labels, aids, branch_weights, agreement_aware=agreement_aware
    )

    conn = srl_loss.clone()
    use_branch = branch_valid & srl_valid
    if use_branch.any():
        a = float(branch_mix_alpha)
        conn[use_branch] = (1.0-a)*srl_loss[use_branch] + a*branch_loss[use_branch]

    valid = srl_valid
    if valid.any():
        return conn[valid].mean(), branch_valid.float().mean()
    return logits.sum()*0.0, branch_valid.float().mean()
