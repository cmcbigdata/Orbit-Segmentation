import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        
    def forward(self, pred, gt, weight, smooth=1.0):
        #flatten label and prediction tensors
        pred = pred.view(-1)
        gt = gt.view(-1)
        
        intersection = (pred * gt).sum()           
        dice = -1 * weight * (2.0*intersection + smooth)/(pred.sum() + gt.sum() + smooth)  
        
        return dice    

