import torch
import numpy as np
from utils.PostProcess import ctdet_decode
from glob import glob
from utils.read_data import read_labelme
import os
from tqdm import tqdm
import cv2
from utils.metrics import DetEval
import shutil
def evaluate(model,data_dirs):
    my_eval=DetEval(0.5)
    model.eval()
    score_thresh=0.3
    device=torch.device('cuda')
    img_files=[]
    bboxes=[]
    for img_dir in data_dirs:
        json_files = glob(img_dir + '*.json')
        # print(json_files)
        for json_file in json_files:
            bbox = read_labelme(json_file)
            img_types = ['jpg', 'JPG', 'png']
            for img_type in img_types:
                im_path = os.path.join(img_dir, json_file.replace('json', img_type))
                if os.path.exists(im_path):
                    img_files.append(json_file.replace('json', img_type))
                    bboxes.append(bbox)
                    break
            
    
    for i in tqdm(range(len(img_files))):
        img = cv2.imread(img_files[i])
        h, w, c = img.shape
        H = 48
        img = cv2.resize(img, (int(w / h * H), H))
        cur_bbox=bboxes[i]
        cur_bbox=cur_bbox.astype(np.float)
        cur_bbox*=H/h
        
        img = img.astype(np.float32)
        ori_img=img.copy()
        img /= 256.
        img = (img - 0.5) / 0.5
        img = img[np.newaxis, :, :, :].transpose(0, 3, 1, 2)
        img = torch.from_numpy(img.astype(np.float32))
        img = img.to(device)
        with torch.no_grad():
            predict = model(img)
        res = ctdet_decode(predict['hmap'], predict['wh'], K=50)
        dets, scores = res[:, :, 0:4], res[:, :, 4]
        scores = scores.detach().cpu().numpy().reshape(-1, 1)
        
        dets = dets.detach().cpu().numpy().reshape(-1, dets.shape[2])
        index=scores>score_thresh
        
        dets=dets[index[:,0]]
        cur_bbox=cur_bbox.reshape(-1,8)
        x0, y0, x1, y1 = np.min(cur_bbox[:, [0,2,4,6]],axis=1), np.min(cur_bbox[:,[1,3,5,7]],axis=1), np.max(cur_bbox[:,[0,2,4,6]],axis=1), np.max(cur_bbox[:,[1,3,5,7]],axis=1)
        gt_bbox=np.concatenate((x0[:,None],y0[:,None],x1[:,None],y1[:,None]),axis=1)
        #------------------------------viz gt---------------------------------
        # gt_bbox=gt_bbox.astype(np.int)
        # outimg=ori_img.astype(np.uint8)
        # for j in range(len(gt_bbox)):
        #     outimg = cv2.rectangle(img=outimg, pt1=(gt_bbox[j][0], gt_bbox[j][1]), pt2=(gt_bbox[j][2], gt_bbox[j][3]), color=(255,0,0),
        #                            thickness=1, )
        # cv2.imshow('a',outimg)
        # cv2.waitKey(0)
        #-----------------------------------------------------------------------
        fn_bbox=my_eval(dets,gt_bbox)
        fn_bbox=fn_bbox.astype(np.int)
        if len(fn_bbox)!=0:
            outimg=ori_img.astype(np.uint8)
            for j in range(len(fn_bbox)):
                outimg = cv2.rectangle(img=outimg, pt1=(fn_bbox[j][0], fn_bbox[j][1]), pt2=(fn_bbox[j][2], fn_bbox[j][3]), color=(0,0,255),thickness=1, )
            dets=dets.astype(np.int)
            for j in range(len(dets)):
                outimg = cv2.rectangle(img=outimg, pt1=(dets[j][0], dets[j][1]),
                                       pt2=(dets[j][2], dets[j][3]), color=(255, 0, 0), thickness=1, )
            #cv2.imwrite(os.path.join('error',os.path.basename(img_files[i])),outimg)
            shutil.copy(img_files[i],os.path.join('error',os.path.basename(img_files[i])))
            # json_path=img_files[i].split('.')[0]+'.json'
            #shutil.copy(json_path,os.path.join('error',os.path.basename(json_path)))
    my_eval.eval()
    

if __name__=='__main__':
    from modules.build_yolact import Yolact

    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    net = Yolact(freeze_bn=False)
    net.load_state_dict(torch.load('CKPT/labelmodel_coor.pkl'), strict=True)
    net.eval()
    net = net.to(device)
    evaluate(net,['/media/wsl/SB@data/dataset/hoke/wp/dataset/单字符检测/train/','/media/wsl/SB@data/字符识别/数据集/标注/lab/'])