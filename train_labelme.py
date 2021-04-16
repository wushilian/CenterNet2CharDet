import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

import os

from modules.build_centernet import CenterNet
from modules.det_loss.multi_loss import Multi_inds_Loss as Multi_Loss
from data.Bboxdataset import JsonDataset,alignCollate
from eval import evaluate

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import logging
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler('log.txt')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

console = logging.StreamHandler()
console.setLevel(logging.INFO)

logger.addHandler(handler)
logger.addHandler(console)




def set_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def data_to_device(datum):

    im,heatmap, label_wh,inds = datum

    if cuda:
        im=im.cuda()
        heatmap=heatmap.cuda()
        label_wh=label_wh.cuda()
        
        inds=inds.cuda()


    else:
        pass

    return im,heatmap, label_wh,inds

cuda = torch.cuda.is_available()


net = CenterNet(freeze_bn=False)
net.train()
#net.load_state_dict(torch.load('./CKPT/labelmodel.pkl'),strict=False)


print('\nTraining from begining, weights initialized.\n')




optimizer=optim.Adam(net.parameters(),lr=0.0001)
criterion = Multi_Loss()
best_mask_ap=0
best_bbox_ap=0
batch_size=32
warmup_epoch=1

if cuda:
    cudnn.benchmark = True
    net=net.cuda()
    criterion = criterion.cuda()
    # net = nn.DataParallel(net).cuda()
    # criterion = nn.DataParallel(criterion).cuda()


# dataset =JsonDataset(['/media/wsl/SB@data/dataset/瓶盖分类/dataset/单字检测/聪明盖/','/media/wsl/SB@data/dataset/瓶盖分类/dataset/单字检测/黄色/'])
dataset =JsonDataset([#'/media/wsl/SB@data/OCR/MyDatagen/im/labelme/',
                      '/media/wsl/SB@data/dataset/hoke/wp/dataset/单字符检测/train/',
                      #'/media/wsl/SB@data/OCR/MyDatagen/im/labelme_complex/',
                      '/media/wsl/SB@data/字符识别/数据集/标注/lab/',
                      '/media/wsl/SB@data/OCR/MyDatagen/im/labelme_fuhao/',
                      '/media/wsl/SB@data/OCR/MyDatagen/im/labelme_chinese/'
                      ])
val_dirs=['/media/wsl/SB@data/dataset/hoke/wp/dataset/单字符检测/train/','/media/wsl/SB@data/字符识别/数据集/标注/lab/']

data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                       collate_fn=alignCollate(imgH=48, imgW=800, keep_ratio=True),)


iter_per_epoch = len(dataset)//batch_size

#warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * warmup_epoch)
train_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9) #CosineAnnealingLR(optimizer_ft, 200)

for ep in range(500):
    net.train()
    for i, datum in enumerate(data_loader):
        im, heatmap, label_wh, wh_mask,=data_to_device(datum)
        
        # if cuda:
        #     torch.cuda.synchronize()
        predictions = net(im)
        losses = criterion(predictions,heatmap, label_wh, wh_mask)
        losses = {k: v.mean() for k, v in losses.items()}  # Mean here because Dataparallel.
        loss=losses['hmap']+1*losses['wh']
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if ep <= warmup_epoch:
        #     warmup_scheduler.step()
        if i%50==0:
            for key in losses.keys():
                print(key,losses[key].item(),ep,i)
            print('*'*20)
        
    if ep > warmup_epoch:
        train_scheduler.step(ep)
    if ep%10==0:
        evaluate(net,val_dirs)
        torch.save(net.state_dict(),'CKPT/model.pkl')
