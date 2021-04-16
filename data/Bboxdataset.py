#!/usr/bin/python
# encoding: utf-8
import os
import cv2
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
from PIL import Image,ImageEnhance,ImageOps
import numpy as np
import json
import data.trans as trans
from data.draw_guassian import *
import math
from numba import jit
from glob import glob
from utils.read_data import read_labelme
from data.augmentation import AugPoly
debug_idx = 0
debug = False

crop = trans.Crop(probability=0.1)
crop2 = trans.Crop2(probability=1.1)
random_contrast = trans.RandomContrast(probability=0.1)
random_brightness = trans.RandomBrightness(probability=0.1)
random_color = trans.RandomColor(probability=0.1)
random_sharpness = trans.RandomSharpness(probability=0.1)
compress = trans.Compress(probability=0.3)
exposure = trans.Exposure(probability=0.1)
rotate = trans.Rotate(probability=0.1)
blur = trans.Blur(probability=0.3)
motion_blur=trans.MotionBlur(probability=0.3)
salt = trans.Salt(probability=0.1)
adjust_resolution = trans.AdjustResolution(probability=0.1)
stretch = trans.Stretch(probability=0.1)
random_line=trans.RandomLine(probability=0.3)
crop.setparam()
crop2.setparam()
random_contrast.setparam()
random_brightness.setparam()
random_color.setparam()
random_sharpness.setparam()
compress.setparam()
exposure.setparam()
rotate.setparam()
blur.setparam()
motion_blur.setparam()
salt.setparam()
adjust_resolution.setparam()
stretch.setparam()

def randomColor(image):
    """
    对图像进行颜色抖动
    :param image: PIL的图像image
    :return: 有颜色色差的图像image
    """
    random_factor = np.random.randint( 0, 31 ) / 10.  # 随机因子
    color_image = ImageEnhance.Color( image ).enhance( random_factor )  # 调整图像的饱和度
    random_factor = np.random.randint( 10, 21 ) / 10.  # 随机因子
    brightness_image = ImageEnhance.Brightness( color_image ).enhance( random_factor )  # 调整图像的亮度
    random_factor = np.random.randint( 10, 21 ) / 10.  # 随机因1子
    contrast_image = ImageEnhance.Contrast( brightness_image ).enhance( random_factor )  # 调整图像对比度
    random_factor = np.random.randint( 0, 31 ) / 10.  # 随机因子
    return ImageEnhance.Sharpness( contrast_image ).enhance( random_factor )  # 调整图像锐度

def randomGaussian(image, mean=0.2, sigma=0.3):
    """
     对图像进行高斯噪声处理
    :param image:
    :return:
    """

    def gaussianNoisy(im, mean=0.2, sigma=0.3):
        """
        对图像做高斯噪音处理
        :param im: 单通道图像
        :param mean: 偏移量
        :param sigma: 标准差
        :return:
        """
        for _i in range( len( im ) ):
            im[_i] += random.gauss( mean, sigma )
        return im

    # 将图像转化成数组
    img = np.asarray( image )
    img.flags.writeable = True  # 将数组改为读写模式
    width, height = img.shape[:2]
    img_r = gaussianNoisy( img[:, :, 0].flatten(), mean, sigma )
    img_g = gaussianNoisy( img[:, :, 1].flatten(), mean, sigma )
    img_b = gaussianNoisy( img[:, :, 2].flatten(), mean, sigma )
    img[:, :, 0] = img_r.reshape( [width, height] )
    img[:, :, 1] = img_g.reshape( [width, height] )
    img[:, :, 2] = img_b.reshape( [width, height] )
    return Image.fromarray( np.uint8( img ) )

def inverse_color(image):
    if np.random.random()<0.4:
        image = ImageOps.invert(image)
    return image

# def data_tf(img):
#     img = randomColor(img)
#     # img = randomGaussian(img)
#     img = inverse_color(img)
#     return img

def data_tf(img):
    img = crop.process(img)
    img = random_contrast.process(img)
    img = random_brightness.process(img)
    img = random_color.process(img)
    img = random_sharpness.process(img)
    img=random_line.process(img)

    if img.size[1]>=32:
        img = compress.process(img)
        #img = adjust_resolution.process(img)
        img = motion_blur.process(img)
        img = blur.process(img)
    img = exposure.process(img)
    img = rotate.process(img)
    img = salt.process(img)
    img = inverse_color(img)
    #img = stretch.process(img)

    if debug and np.random.random() < 0.001:
        global debug_idx
        img.save('debug_files/{:05}.jpg'.format(debug_idx))
        debug_idx += 1
        if debug_idx == 10000:
            debug_idx = 0
    return img

def data_tf_fullimg(img,loc):
    left, top, right, bottom = loc
    img = crop2.process([img, left, top, right, bottom])
    img = random_contrast.process(img)
    img = random_brightness.process(img)
    img = random_color.process(img)
    img = random_sharpness.process(img)
    img = compress.process(img)
    img = exposure.process(img)
    # img = rotate.process(img)
    img = blur.process(img)
    img = salt.process(img)
    # img = inverse_color(img)
    img = adjust_resolution.process(img)
    img = stretch.process(img)
    return img

def FullToHalf(s):
    n = []
    for char in s:
        num = ord(char)
        if num == 0x3000:
            num = 32
        elif 0xFF01 <= num <= 0xFF5E:
            num -= 0xfee0
        num = chr(num)
        n.append(num)
    return ''.join(n)



class JsonDataset(Dataset):
    def __init__(self, data_dirs=[''], train=True, transform=data_tf, target_transform=None):
        super(Dataset, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.augmenter=AugPoly()
        print('start read annotation')
        self.img_files = []
        self.bboxes = []
        print('start read annotation')
        for img_dir in data_dirs:
            json_files = glob(img_dir + '*.json')
            # print(json_files)
            for json_file in json_files:
                bbox = read_labelme(json_file)
                img_types = ['jpg', 'JPG', 'png']
                for img_type in img_types:
                    im_path = os.path.join(img_dir, json_file.replace('json', img_type))
                    if os.path.exists(im_path):
                        self.img_files.append(json_file.replace('json', img_type))
                        self.bboxes.append(bbox)
                        break
                
        print('loaded annotation')
        assert len(self.bboxes)==len(self.img_files)
        print('total img:%d'%len(self.img_files))
    def name(self):
        return 'MyDataset'
    
    def __getitem__(self, index):
        bbox =self.bboxes[index]
        bbox = np.array(bbox)
        #print(bbox.shape)
        #bbox = np.transpose(bbox, (2, 1, 0))
        # print(bbox.shape)
        path = self.img_files[index]
        img = Image.open(path)
        img=np.array(img)
        h,w,c=img.shape
        bbox[:,:,0]=np.clip(bbox[:,:,0],0,w-1)
        bbox[:, :, 1] = np.clip(bbox[:, :, 1], 0, h - 1)
        
        #----------------------------debug----------------------------------
        # print(bbox.shape,path)
        # viz_img=cv2.polylines(img,bbox.astype(np.int),isClosed=True,color=(255,0,0),thickness=1)
        # cv2.imshow('a',viz_img)
        # cv2.waitKey(0)
        #------------------------------------------------------------------
        img,bbox=self.augmenter.aug(img,bbox,viz=False)
        img=Image.fromarray(img)
        # if self.transform is not None:
        #     img = self.transform(img)
        
        return (img, bbox)
    
    def __len__(self):
        return len(self.img_files)



class resizeNormalize(object):
    def __init__(self, size, interpolation=Image.LANCZOS,is_test=False):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
        self.is_test = is_test

    def __call__(self, img,label):
        w,h = self.size
        w0 = img.size[0]
        h0 = img.size[1]
        if w<=(w0/h0*h):
            img = img.resize(self.size, self.interpolation)
            label[...,0]=label[...,0]*w/w0
            label[..., 1] = label[..., 1] * h / h0
            img = self.toTensor(img)
            img.sub_(0.5).div_(0.5)
        else:
            w_real = int(w0/h0*h)
            img = img.resize((w_real,h), self.interpolation)

            label[..., 0] = label[..., 0] * w_real / w0
            label[..., 1] = label[..., 1] * h / h0
            img = self.toTensor(img)
            img.sub_(0.5).div_(0.5)
            start = random.randint(0,w-w_real-1)
            if self.is_test:
                start = 5
                w+=10
            #print(img.shape[0], h, w)
            tmp = torch.zeros([img.shape[0], h, w])+0.5
            tmp[:,:,start:start+w_real] = img
            label[..., 0]=label[..., 0]+start
            img = tmp
        return img,label

class randomSequentialSampler(sampler.Sampler):
    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.range(0, self.batch_size - 1)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.range(0, tail - 1)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples


class alignCollate(object):
    def __init__(self, imgH=32, imgW=100, keep_ratio=True, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio
        self.random_scale=[1,1.2,1.5]
    def __call__(self, batch):
        images, labels = zip(*batch)
        scale=random.choice(self.random_scale)
        imgH = int(self.imgH*scale)
        imgW = int(self.imgW*scale)
        #self.keep_ratio=np.random.random()<0.5
        if self.keep_ratio:
            ratios = []
            for image in images:
                w, h = image.size
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            imgW = int(np.floor(max_ratio * imgH))
            imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW
            imgW=min(imgW,self.imgW)
            #print(imgW)
        transform = resizeNormalize((imgW, imgH))
        res_img=[]
        res_label=[]
        heatmaps=[]
        label_whs=[]
        inds=[]
        for image,label in zip(images,labels):
            a,b=transform(image,label)
            res_img.append(a)
            heat,lwh,lind=create_label(label,(imgH,imgW),4,30)
            heatmaps.append(torch.FloatTensor(heat))
            label_whs.append(torch.FloatTensor(lwh))
            inds.append(torch.FloatTensor(lind))
            # res_label.append(b)
        images = torch.cat([t.unsqueeze(0) for t in res_img], 0)
        return images, torch.stack(heatmaps,0),torch.stack(label_whs,0),torch.stack(inds,0)

def create_label( bboxes,img_size,down_sample_rate=4,max_objects=256):
    h,w=img_size
    heatmap = np.zeros((math.ceil(h/down_sample_rate), math.ceil(w/down_sample_rate), 1),dtype=np.float32)
    #---------------------wh and reg------------------------------
    label_wh = np.zeros((max_objects, 4),dtype=np.float32)
    inds=np.zeros(shape=(max_objects,),dtype=np.int64)
    k=0
    for bbox in bboxes:
        # print(bbox)
        bbox=bbox/down_sample_rate
        #print(bbox)
        bbox[:,0]=np.clip(bbox[:,0], 0, w//down_sample_rate - 1)
        bbox[:, 1] = np.clip(bbox[:, 1], 0, h // down_sample_rate - 1)
        
        
        x0,y0,x1,y1=np.min(bbox[:,0]),np.min(bbox[:,1]),np.max(bbox[:,0]),np.max(bbox[:,1])
        xind_float,yind_float=np.mean(bbox[:,0]),np.mean(bbox[:,1])
        xind,yind=np.int(xind_float),np.int(yind_float)
        #print(xind_float,xind,yind_float,yind)
        c_w,c_h=x1-x0,y1-y0
        ra, rb = gaussian_radius_1([c_w, c_h], 0.7)
        # rb, ra = gaussian_radius_1([c_w,c_h], 0.7)
        ra = max(0, int(ra))
        rb = max(0, int(rb))
        heatmap[...,0] = draw_umich_gaussian_1(heatmap[..., 0], (xind, yind), ra, rb)
        
        predict_r = 0.0
        for y in range(yind-int(c_h*predict_r),yind+int(c_h*predict_r)+1):
            for x in range(xind-int(c_w*predict_r),xind+int(c_w*predict_r)+1):
                # relate_box=bbox.reshape(8,).copy()
                # relate_box[[0, 2, 4, 6]] = np.abs(
                #     relate_box[[0, 2, 4, 6]] -x)
                # relate_box[[1, 3, 5, 7]] = np.abs(
                #     relate_box[[1, 3, 5, 7]] - y)
                #label_wh[k,:]=relate_box.reshape(8,)
                label_wh[k,0]=c_w*down_sample_rate
                label_wh[k,1]=c_h*down_sample_rate
                label_wh[k,2]=xind_float-xind
                label_wh[k,3]=yind_float-yind
                inds[k]=y*(w//down_sample_rate)+x
                k+=1
    return heatmap.transpose(2, 0, 1), label_wh,inds

def test_dataset():
    from matplotlib import pyplot as plt
    #mydata=MyDataset('label.npy')
    mydata = JsonDataset(['/media/wsl/SB@data/dataset/hoke/seg/lab/'])
    randomSequentialSampler(mydata, 32)
    myloader=torch.utils.data.DataLoader(mydata, batch_size=32,shuffle=True, num_workers=1,collate_fn=alignCollate(imgH=48, imgW=800, keep_ratio=True))
    # for bb in myloader:
    #     print(bb[0].shape)
    train_iter = iter(myloader)
    a,b,c,d=train_iter.__next__()
    a=a.cpu().numpy()
    b=b.cpu().numpy()

    img=a*128+128
    heat=b[:,0,:,:]
    for a,b in zip(img,heat):
        a=np.transpose(a,(1,2,0))
        # plt.imshow(b)
        # plt.show()
        # cv2.imshow('a',a.astype(np.uint8))
        # cv2.imshow('b',b)
        # cv2.waitKey(0)
    # print(a.shape,b.shape)

if __name__ == '__main__':
    
    test_dataset()
 