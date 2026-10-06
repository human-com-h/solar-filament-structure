"""Factories using the frozen U-Net, Flat U-Net and official clDice bodies."""
from .io import REPO
from .frozen import module,selected

METHODS=('BCE_Dice','SRL','SABR','ResidualPixel','SRL_FG','FlatUNet_BCE','UNet_softDice_clDice','UNet_region_clDice')
def factory(method):
    if method not in METHODS:raise ValueError('Unknown method: '+method)
    if method=='FlatUNet_BCE':
        net=module('_sabr_flat',REPO/'frozen_sources/vendor/flat/fs_model.py').Flat_Unet(simp_list=[1,1,1,1,0],num_layers=4,flat_channels=32,out_channels=1)
        expected=258081
    else:
        net=module('_sabr_unet',REPO/'frozen_sources/mechanism/unet.py').UNet(base=32);expected=7762465
    if sum(p.numel() for p in net.parameters())!=expected:raise ValueError('Architecture differs')
    return net
def loss_module(method):
    import sys,types,torch
    if method=='FlatUNet_BCE':return torch.nn.BCEWithLogitsLoss(reduction='mean')
    package=types.ModuleType('_sabr_vendor_cldice');package.__path__=[str(REPO/'frozen_sources/vendor/cldice')];sys.modules[package.__name__]=package
    cl=module('_sabr_vendor_cldice.cldice',REPO/'frozen_sources/vendor/cldice/cldice.py')
    if method=='UNet_softDice_clDice':return cl.soft_dice_cldice(iter_=10,alpha=.5,smooth=1.,exclude_background=False)
    if method=='UNet_region_clDice':return cl.soft_cldice(iter_=10,smooth=1.,exclude_background=False)
    return None
def loss(logits,batch,method):
    import torch
    historical=module('_sabr_historical_loss',REPO/'frozen_sources/mechanism/historical_losses.py')
    if method in METHODS[:5]:
        f=selected(REPO/'frozen_sources/mechanism/objectives.py',['objective'],dict(torch=torch,METHODS=METHODS[:5],bce_dice_loss=historical.bce_dice_loss,soft_recall_per_sample=historical.soft_recall_per_sample))['objective']
        return f(logits,batch,method)[0]
    criterion=loss_module(method)
    if method=='FlatUNet_BCE':return criterion(logits,batch['y'])
    if method=='UNet_softDice_clDice':return criterion(batch['y'],torch.sigmoid(logits))
    return historical.bce_dice_loss(logits,batch['y'])+.05*criterion(batch['y'],torch.sigmoid(logits))
