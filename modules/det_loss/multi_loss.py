# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.PostProcess import _tranpose_and_gather_feature,_gather_feature
from modules.det_loss.giou import giou_wh_loss

#criterion = nn.BCEWithLogitsLoss(reduction='none')
criterion=nn.BCELoss(reduction='none')

def __focal(target, actual, alpha=1, gamma=2):
    focal = alpha * torch.pow(torch.abs(target - actual), gamma)
    return focal


def heatmap_loss(pred, gt):
    #print(pred.shape,gt.shape)
    pred = torch.clamp(torch.sigmoid(pred), min=1e-6, max=1 - 1e-6)
    
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    
    neg_weights = torch.pow(1 - gt, 4)
    
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
    
    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    
    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss







class Multi_inds_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def wh_loss(self,pred_wh, label_wh, mask):
        mask = mask[:, :, None].expand_as(label_wh).float()
        loss = F.l1_loss(pred_wh * mask, label_wh * mask, reduction='sum') / (mask.sum())
        # loss=F.smooth_l1_loss(pred_wh* mask, label_wh * mask, reduction='sum') / (mask.sum())
        return loss
    def wh_reg_loss(self,pred_wh, label_wh, mask):
        mask = mask[:, :, None].expand_as(label_wh).float()
        pred_reg=pred_wh[...,2:4]
        pred_wh=pred_wh[...,0:2]
        label_reg=label_wh[...,2:4]
        label_wh=label_wh[...,0:2]
        mask = mask.float()
        loss_wh=giou_wh_loss(pred_wh, label_wh )*mask
        loss_wh=loss_wh.sum()/mask.sum()
        mask = mask[:, :, None].expand_as(label_reg)
        loss_reg=F.smooth_l1_loss(pred_reg* mask, label_reg * mask, reduction='sum') / mask.sum()
        loss=loss_wh+loss_reg
        return loss
    
    def forward(self, predictions,label_hmap,label_wh,ind):
        pred_hmap=predictions['hmap']
        pred_wh=predictions['wh']
        loss_hmap=heatmap_loss(pred_hmap,label_hmap)
        ind=ind.long()
        sel_pred_wh=_tranpose_and_gather_feature(pred_wh,ind)
        ind_mask=torch.where(ind>0,torch.ones_like(ind),torch.zeros_like(ind))
        loss_wh=self.wh_loss(sel_pred_wh,label_wh,ind_mask)
        #loss_wh = self.wh_reg_loss(sel_pred_wh, label_wh, ind_mask)
        losses = {}
        losses['hmap']=loss_hmap
        losses['wh']=loss_wh

        return losses


