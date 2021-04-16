import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.backone.ResNet_backbone import get_resnet_backbone
from modules.backone.Simple_backbone import get_simple_backbone
from utils.PostProcess import ctdet_decode
from modules.neck.fpn import FPN
from modules.head.centernet_head import PredictionModule





    

class CenterNet(nn.Module):
    def __init__(self,back_bone='resnet_18',freeze_bn=False,export_onnx=False):
        super().__init__()
        self.backbone_layers=int(back_bone.split('_')[1])
        self.backbone,self.out_size = get_resnet_backbone(self.backbone_layers)
        #self.backbone, self.out_size = get_simple_backbone()
        self.freeze_bn_flag=freeze_bn
        self.export_onnx=export_onnx
        if freeze_bn:
            self.freeze_bn()
        
        
        in_channels=128
        # self.fpn=FPN_Transpose(out=self.out_size,fpn_out=in_channels)
        self.fpn = FPN(out=self.out_size, fpn_out=in_channels)
        self.up_feature=nn.Sequential(nn.Conv2d(in_channels,128, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(128),
                                    nn.ReLU(inplace=True))
        # self.up_feature=CoordConv(in_channels=in_channels,out_channels=128,kernel_size=3, padding=1)

        self.prediction_layers = PredictionModule(in_channels=in_channels)


    def load_weights(self, path, cuda):
        if cuda:
            state_dict = torch.load(path)
        else:
            state_dict = torch.load(path, map_location='cpu')

        for key in list(state_dict.keys()):
            if key.startswith('fpn.downsample_layers.'):
                if int(key.split('.')[2]) >= 2:
                    del state_dict[key]

        self.load_state_dict(state_dict)

    def init_weights(self):
        self.backbone.init_weights(self.backbone_layers)
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d) and module not in self.backbone.backbone_modules:
                if 'hmap' in name:
                    print(name)
                    continue
                nn.init.xavier_uniform_(module.weight.data)
                if module.bias is not None:
                    module.bias.data.zero_()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_bn_flag:
            self.freeze_bn()

    def freeze_bn(self):
        """ Adapted from https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8 """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                module.weight.requires_grad = False
                module.bias.requires_grad = False

    def forward(self, x):
        outs = self.backbone(x)
        p2,p3,p4,p5=self.fpn(outs)
        p2 = self.up_feature(p2)
        predictions = self.prediction_layers(p2)
        if self.export_onnx:
            return ctdet_decode(predictions['hmap'],predictions['wh'],K=30)
        else:
            return predictions



if __name__=='__main__':
    model=Yolact()
    model.init_weights()
    x=torch.zeros(size=(1,3,256,256))
    x=model(x)
    # for key in x.keys():
    #     print(key,x[key].shape)
