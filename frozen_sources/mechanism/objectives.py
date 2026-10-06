"""Fixed-budget mechanism controls; same region loss and branch eligibility."""
import torch
from historical_losses import bce_dice_loss,soft_recall_per_sample

METHODS=('BCE_Dice','SRL','SABR','ResidualPixel','SRL_FG')
def objective(logits,batch,method):
    assert method in METHODS
    region=bce_dice_loss(logits,batch['y']);zero=logits.sum()*0
    if method=='BCE_Dice':return region,region,zero
    srl,svalid=soft_recall_per_sample(logits,batch['srl'])
    if method=='SRL':conn=srl
    else:
        eligible=(batch['residual'].flatten(1).sum(1)>0)&svalid
        if method=='SABR':
            p=torch.sigmoid(logits)
            special=1-(p*batch['weights']).flatten(1).sum(1)-batch['offset']
        elif method=='ResidualPixel':special,_=soft_recall_per_sample(logits,batch['residual'])
        else:special,_=soft_recall_per_sample(logits,batch['y'])
        conn=torch.where(eligible,.75*srl+.25*special,srl)
    structural=conn[svalid].mean() if svalid.any() else zero
    return region+.05*structural,region,structural
