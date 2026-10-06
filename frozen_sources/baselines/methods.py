import sys,json
from pathlib import Path
import torch
from torch import nn
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'legacy'))
sys.path.insert(0,str(HERE/'vendor/morden'))
from unet import UNet
from vendor.flat.fs_model import Flat_Unet
from segmentation.utils.models.filament_seg import FilamentSeg
from segmentation.utils.criteria.focal_loss import FocalLoss
from vendor.cldice.cldice import soft_dice_cldice
DESIGN=json.loads((HERE/'FROZEN_DESIGN.json').read_text())
CONFIG={m['id']:m for m in DESIGN['methods']}
METHODS=tuple(CONFIG)
def factory(method):
    c=CONFIG[method]
    if method=='FlatUNet_BCE':m=Flat_Unet(**c['configuration'])
    elif method=='MORDEN_Focal':m=FilamentSeg(**c['configuration'])
    else:m=UNet(**c['configuration'])
    assert sum(p.numel() for p in m.parameters())==c['parameters']
    return m
def probability(output,method):
    # MORDEN forward already contains its only sigmoid.
    return output if method=='MORDEN_Focal' else torch.sigmoid(output)
def criterion(method):
    if method=='FlatUNet_BCE':return nn.BCEWithLogitsLoss(reduction='mean')
    if method=='MORDEN_Focal':return FocalLoss(gamma=2,alpha=.66,reduction='mean')
    loss=soft_dice_cldice(iter_=10,alpha=.5,smooth=1.,exclude_background=False)
    assert loss.soft_skeletonize.num_iter==10
    return loss
def loss_value(output,target,method,loss):
    assert output.shape==target.shape and output.ndim==4 and output.shape[1]==1
    assert output.dtype==target.dtype==torch.float32
    if method=='UNet_softDice_clDice':return loss(target,probability(output,method))
    return loss(output,target)
