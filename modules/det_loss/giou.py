#!/usr/bin/env python
# encoding: utf-8
'''
@author: xpf
@license: (C) Copyright 2019-2027, Node Supply Chain Manager Corporation Limited.
@contact: me
@software: garner
@file: giou.py
@time: 2019/8/6 下午5:32
@desc:
'''

import  torch

def giou_loss(preds,targets):
    '''
    Args:
    preds: [n,4] ltrb
    targets: [n,4]
    '''
    lt_min=torch.min(preds[:,:2],targets[:,:2])
    rb_min=torch.min(preds[:,2:],targets[:,2:])
    wh_min=(rb_min+lt_min).clamp(min=0)
    overlap=wh_min[:,0]*wh_min[:,1]#[n]
    area1=(preds[:,2]+preds[:,0])*(preds[:,3]+preds[:,1])
    area2=(targets[:,2]+targets[:,0])*(targets[:,3]+targets[:,1])
    union=(area1+area2-overlap)
    iou=overlap/union

    lt_max=torch.max(preds[:,:2],targets[:,:2])
    rb_max=torch.max(preds[:,2:],targets[:,2:])
    wh_max=(rb_max+lt_max).clamp(0)
    G_area=wh_max[:,0]*wh_max[:,1]#[n]

    giou=iou-(G_area-union)/G_area.clamp(1e-10)
    loss=1.-giou
    return loss.sum()

def giou_wh_loss(preds,targets):
    '''
    Args:
    preds: [n,4] ltrb
    targets: [n,4]
    '''
    w_min=torch.min(preds[...,0],targets[...,0]).clamp(0)
    h_min=torch.min(preds[...,1],targets[...,1]).clamp(0)
    overlap=w_min*h_min
    
    area1=preds[...,0]*preds[...,1]
    area2=targets[...,0]*targets[...,1]
    union=(area1+area2-overlap)
    iou=overlap/union.clamp(1e-10)

    w_max = torch.max(preds[..., 0], targets[..., 0]).clamp(0)
    h_max = torch.max(preds[..., 1], targets[..., 1]).clamp(0)
    
    G_area=w_max*h_max

    giou=iou-(G_area-union)/G_area.clamp(1e-10)
    loss=1.-giou
    return loss

if __name__ == "__main__":

    print("aaaaa")

    box1 = torch.Tensor([[10,10]])
    box2 = torch.Tensor([[20,5]])

    m = giou_wh_loss(box1,box2)
    print(m)