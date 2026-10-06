"""One additive structure-budget control, keeping the historical region loss."""
from pathlib import Path
import sys,json
import torch
from torch import nn
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'legacy'))
from unet import UNet
from historical_losses import bce_dice_loss
from vendor.cldice.cldice import soft_cldice
DESIGN=json.loads((HERE/'FROZEN_DESIGN.json').read_text())
CONFIG={x['id']:x for x in DESIGN['methods']}
METHODS=tuple(CONFIG)

def factory(method):
    assert method in METHODS
    m=UNet(**CONFIG[method]['configuration'])
    assert sum(p.numel() for p in m.parameters())==7762465
    return m

def probability(output,method):
    assert method in METHODS
    return torch.sigmoid(output)

class RegionClDice(nn.Module):
    def __init__(self):
        super().__init__()
        self.structure=soft_cldice(iter_=10,smooth=1.,exclude_background=False)
        assert self.structure.soft_skeletonize.num_iter==10
    def forward(self,logits,target):
        return bce_dice_loss(logits,target)+.05*self.structure(target,torch.sigmoid(logits))

def criterion(method):
    assert method in METHODS
    return RegionClDice()

def loss_value(output,target,method,loss):
    assert method in METHODS and output.shape==target.shape
    assert output.ndim==4 and output.shape[1]==1
    assert output.dtype==target.dtype==torch.float32
    return loss(output,target)
