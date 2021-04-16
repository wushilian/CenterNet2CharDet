import torch
import torch.nn.functional as F

import numpy as np

def _tranpose_and_gather_feature(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feature(feat, ind)
    return feat


def flip_tensor(x):
    return torch.flip(x, [3])


def _gather_feature(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _nms(heat, kernel=3):
    #hmax = F.max_pool2d(heat, kernel, stride=1, padding=(kernel - 1) // 2)
    hmax = F.max_pool2d(heat,(9,3), stride=1, padding=(4,1))
    keep = (hmax == heat).float()
    return heat * keep


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()
    # height=height.cuda()
    # width=width.cuda()
    
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)
    
    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feature(topk_inds.view(
        batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feature(topk_ys.view(
        batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feature(topk_xs.view(
        batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def ctdet_8_decode(hmap, w_h_, K=100):
    batch, cat, height, width = hmap.shape
    hmap = torch.sigmoid(hmap)

    # if flip test
    if batch > 1:
        hmap = (hmap[0:1] + flip_tensor(hmap[1:2])) / 2
        w_h_ = (w_h_[0:1] + flip_tensor(w_h_[1:2])) / 2

    batch = 1

    hmap = _nms(hmap)  # perform nms on heatmaps

    scores, inds, clses, ys, xs = _topk(hmap, K=K)

    xs = xs.view(batch, K, 1)
    ys = ys.view(batch, K, 1)
    w_h_ = _tranpose_and_gather_feature(w_h_, inds)

    w_h_ = w_h_.view(batch, K, 8)

    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)

    num = 4
    bboxes = torch.cat([xs * num - w_h_[..., 0:1] * num,
                        ys * num - w_h_[..., 1:2] * num,
                        xs * num + w_h_[..., 2:3] * num,
                        ys * num - w_h_[..., 3:4] * num,
                        xs * num + w_h_[..., 4:5] * num,
                        ys * num + w_h_[..., 5:6] * num,
                        xs * num - w_h_[..., 6:7] * num,
                        ys * num + w_h_[..., 7:8] * num
                        ], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)
    return [bboxes, scores]
def ctdet_decode(hmap, w_h_, K=500):
    batch, cat, height, width = hmap.shape
    hmap = torch.sigmoid(hmap)

    # if flip test
    if batch > 1:
        hmap = (hmap[0:1] + flip_tensor(hmap[1:2])) / 2
        w_h_ = (w_h_[0:1] + flip_tensor(w_h_[1:2])) / 2

    batch = 1

    hmap = _nms(hmap,3)  # perform nms on heatmaps

    scores, inds, clses, ys, xs = _topk(hmap, K=K)

    xs = xs.view(batch, K, 1) + 0.5
    ys = ys.view(batch, K, 1) + 0.5

    xs *= 4
    ys *= 4

    w_h_ = _tranpose_and_gather_feature(w_h_, inds)
    w_h_ = w_h_.view(batch, K, 4)
    reg_=w_h_[...,2:]*4
    w_h_=w_h_[...,0:2]
    #w_h_ *= 4
    
    xs=xs+reg_[...,0:1]
    ys=ys+reg_[...,1:]
    
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs - w_h_[..., 0:1] / 2,
                        ys - w_h_[..., 1:2] / 2,
                        xs + w_h_[..., 0:1] / 2,
                        ys + w_h_[..., 1:2] / 2], dim=2)
    detections = torch.cat([bboxes, scores], dim=2)
    return detections
    #return [bboxes, scores]


def yolact_decode(hmap, w_h_, proto_p, coef_p, K=500):
    batch, cat, height, width = hmap.shape
    hmap = torch.sigmoid(hmap)
    mask_p = mask_decode(proto_p, coef_p)
    # if flip test
    if batch > 1:
        hmap = (hmap[0:1] + flip_tensor(hmap[1:2])) / 2
        w_h_ = (w_h_[0:1] + flip_tensor(w_h_[1:2])) / 2

    batch = 1

    hmap = _nms(hmap)  # perform nms on heatmaps

    scores, inds, clses, ys, xs = _topk(hmap, K=K)

    xs = xs.view(batch, K, 1) + 0.5
    ys = ys.view(batch, K, 1) + 0.5
    res_mask = []
    xind = xs[0, :, 0].int().cpu().numpy()
    yind = ys[0, :, 0].int().cpu().numpy()
    #res_mask=mask_p[...,yind+width+xind]
    for i in range(K):
        index=yind[i] * width + xind[i]
        cur_mask = mask_p[...,index]
        #print(coef_p[:,index,:])
        #print('111',cur_mask.shape,mask_p[..., index].shape,mask_p[...,0].shape)
        res_mask.append(cur_mask)
    xs *= cfg.center_downsample
    ys *= cfg.center_downsample

    w_h_ = _tranpose_and_gather_feature(w_h_, inds)
    w_h_ = w_h_.view(batch, K, 2)
    w_h_ *= cfg.center_downsample
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs - w_h_[..., 0:1] / 2,
                        ys - w_h_[..., 1:2] / 2,
                        xs + w_h_[..., 0:1] / 2,
                        ys + w_h_[..., 1:2] / 2], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)
    res_mask = torch.cat(res_mask, dim=0)
    return detections, res_mask
def yolact_fast_decode(hmap, w_h_, proto_p, coef_p, K=500):
    '''
    
    :param hmap: (batch,1,centersize,centersize)
    :param w_h_: (batch,2,centersize,centersize)
    :param proto_p: (batch,masksize,masksize,coefdim)
    :param coef_p: (batch,centersize*centersize,coefdim)
    :param K:
    :return:
    '''
    batch, cat, height, width = hmap.shape
    hmap = torch.sigmoid(hmap)
    
    hmap = _nms(hmap)  # perform nms on heatmaps

    scores, inds, clses, ys, xs = _topk(hmap, K=K)
    top_coef=_gather_feature(coef_p,inds)
    mask_p = mask_decode(proto_p, top_coef)
    mask_p=mask_p.permute(0,3,1,2)
    xs = xs.view(batch, K, 1) + 0.5
    ys = ys.view(batch, K, 1) + 0.5
    xs *= cfg.center_downsample
    ys *= cfg.center_downsample

    w_h_ = _tranpose_and_gather_feature(w_h_, inds)
    w_h_ = w_h_.view(batch, K, 2)
    w_h_ *= cfg.center_downsample
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs - w_h_[..., 0:1] / 2,
                        ys - w_h_[..., 1:2] / 2,
                        xs + w_h_[..., 0:1] / 2,
                        ys + w_h_[..., 1:2] / 2], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)
    return detections, mask_p



def mask_decode(proto_p, coef_p):
    coef_p = coef_p.permute(0, 2, 1)
    n, h, w, c = proto_p.shape
    proto_p = torch.reshape(proto_p, (n, h * w, c))
    mask_p = torch.bmm(proto_p, coef_p)
    mask_p = torch.sigmoid(mask_p)
    mask_p = torch.reshape(mask_p, (n, h, w, -1))
    return mask_p
