import time
import torch
from modules.build_centernet import CenterNet
from utils.PostProcess import ctdet_8_decode,ctdet_decode
import cv2
import numpy as np
from matplotlib import pyplot as plt
from glob import glob
import os
COLORS = [ [244, 67, 54], [233, 30, 99], [156, 39, 176], [103, 58, 183], [100, 30, 60],
                   [63, 81, 181], [33, 150, 243], [3, 169, 244], [0, 188, 212], [20, 55, 200],
                   [0, 150, 136], [76, 175, 80], [139, 195, 74], [205, 220, 57], [70, 25, 100],
                   [255, 235, 59], [255, 193, 7], [255, 152, 0], [255, 87, 34], [90, 155, 50],
                   [121, 85, 72], [158, 158, 158], [96, 125, 139], [15, 67, 34], [98, 55, 20],
                   [21, 82, 172], [58, 128, 255], [196, 125, 39], [75, 27, 134], [90, 125, 120],
                   [121, 82, 7], [158, 58, 8], [96, 25, 9], [115, 7, 234], [8, 155, 220],
                   [221, 25, 72], [188, 58, 158], [56, 175, 19], [215, 67, 64], [198, 75, 20],
                   [62, 185, 22], [108, 70, 58], [160, 225, 39], [95, 60, 144], [78, 155, 120],
                   [101, 25, 142], [48, 198, 28], [96, 225, 200], [150, 167, 134], [18, 185, 90],
                   [21, 145, 172], [98, 68, 78], [196, 105, 19], [215, 67, 84], [130, 115, 170],
                   [255, 0, 255], [255, 255, 0], [196, 185, 10], [95, 167, 234], [18, 25, 190],
                   [0, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], [155, 0, 0],
                   [0, 155, 0], [0, 0, 155], [46, 22, 130], [255, 0, 155], [155, 0, 255],
                   [255, 155, 0], [155, 255, 0], [0, 155, 255], [0, 255, 155], [18, 5, 40],
                   [120, 120, 255], [255, 58, 30], [60, 45, 60], [75, 27, 244], [128, 25, 70]]

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
np.set_printoptions(threshold=9999999999)


cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

net = CenterNet(freeze_bn=False)
net.load_state_dict(torch.load('CKPT/pretrain.pkl'),strict=True)
net.eval()
net=net.to(device)

target_dir='result'
if not os.path.exists(target_dir):
    os.mkdir(target_dir)
def test_detect():
    from sklearn.utils import shuffle
    #img_dirs = '/media/wsl/SB@data/dataset/瓶盖分类/dataset/synth/train1031'
    #img_dirs='/media/wsl/SB@data/dataset/瓶盖分类/dataset/unsup/xuanzhuan/'
    img_dirs='/media/wsl/SB@data/dataset/hoke/seg/9'
    #img_dirs = '/media/wsl/SB@data/text-recognition-benchmark/CurvedSynthText/img/0.img/0001'
    #img_dirs='/media/wsl/SB@data/all_dataset/plate_images/single_yellow'
    #img_dirs='/media/wsl/SB@data/字符识别/数据集/文本行图片/crop'
    #img_dirs='/media/wsl/SB@data/OCR/MyDatagen/im/labelme_chinese/'
    files = os.listdir(img_dirs)
    for file in files:
        if 'json' in file:
            continue
        print(file)
        file=os.path.join(img_dirs,file)
        img=cv2.imread(file)
        h,w,c=img.shape
        
        H=48
        img = cv2.resize(img, (int(w/h*H), H))
        outimg = img.copy()
        img=img.astype(np.float32)
        
        img/=256.
        img=(img-0.5)/0.5
        img=img[np.newaxis,:,:,:].transpose(0,3,1,2)
        img = torch.from_numpy(img.astype(np.float32))
        img = img.to(device)
        with torch.no_grad():
            predict=net(img)
        heat = torch.sigmoid(predict['hmap']).cpu().numpy()
        heat = heat[0, 0, :, :]
        threash = 0.3
        heat=cv2.resize(heat,(outimg.shape[1],outimg.shape[0]))*255
        heat=cv2.applyColorMap(heat.astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imshow('heat',heat)
        res=ctdet_decode(predict['hmap'],predict['wh'],K=50)
        dets, scores=res[:,:,0:4],res[:,:,4]
        scores = scores.detach().cpu().numpy().reshape(-1, 1)
        dets = dets.detach().cpu().numpy().reshape(-1, dets.shape[2])
        print(dets.shape)
        num=0
        for det, score in zip(dets, scores):
            if score < threash:
                continue
            det = det.reshape(2, 2).astype(np.int)
            #print(det)
            #print((COLORS[num][0],COLORS[num][1],COLORS[num][2]))
            outimg=cv2.rectangle(img=outimg,pt1=(det[0][0],det[0][1]),pt2=(det[1][0],det[1][1]),color=COLORS[num],thickness=2,)
            num+=1
            #outimg = cv2.polylines(outimg, [det.astype(np.int)], isClosed=True, color=(255, 0, 0), thickness=2)

        cv2.imshow('a',outimg)
        cv2.waitKey(0)
        cv2.imwrite(os.path.join(target_dir,os.path.basename(file)),outimg)

if __name__=='__main__':
    test_detect()