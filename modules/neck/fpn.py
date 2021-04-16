import torch
import torch.nn as nn
import torch.nn.functional as F


class Concat(nn.Module):
    def __init__(self, nets, extra_params):
        super().__init__()
        
        self.nets = nn.ModuleList(nets)
        self.extra_params = extra_params
    
    def forward(self, x):
        return torch.cat([net(x) for net in self.nets], dim=1, **self.extra_params)


class InterpolateModule(nn.Module):
    """
    A module version of F.interpolate.
    """
    
    def __init__(self, *args, **kwdargs):
        super().__init__()
        
        self.args = args
        self.kwdargs = kwdargs
    
    def forward(self, x):
        return F.interpolate(x, *self.args, **self.kwdargs)


class FPN(nn.Module):
    def __init__(self, out, fpn_out=256):
        super().__init__()
        conv_out = fpn_out
        inplace = True
        # Top layer
        self.toplayer = nn.Sequential(nn.Conv2d(out[3], conv_out, kernel_size=1, stride=1, padding=0),
                                      nn.BatchNorm2d(conv_out),
                                      nn.ReLU(inplace=inplace)
                                      )
        # Lateral layers
        self.latlayer1 = nn.Sequential(nn.Conv2d(out[2], conv_out, kernel_size=1, stride=1, padding=0),
                                       nn.BatchNorm2d(conv_out),
                                       nn.ReLU(inplace=inplace)
                                       )
        self.latlayer2 = nn.Sequential(nn.Conv2d(out[1], conv_out, kernel_size=1, stride=1, padding=0),
                                       nn.BatchNorm2d(conv_out),
                                       nn.ReLU(inplace=inplace)
                                       )
        self.latlayer3 = nn.Sequential(nn.Conv2d(out[0], conv_out, kernel_size=1, stride=1, padding=0),
                                       nn.BatchNorm2d(conv_out),
                                       nn.ReLU(inplace=inplace)
                                       )
        
        # Smooth layers
        self.smooth1 = nn.Sequential(nn.Conv2d(conv_out, conv_out, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(conv_out),
                                     nn.ReLU(inplace=inplace)
                                     )
        self.smooth2 = nn.Sequential(nn.Conv2d(conv_out, conv_out, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(conv_out),
                                     nn.ReLU(inplace=inplace)
                                     )
        self.smooth3 = nn.Sequential(nn.Conv2d(conv_out, conv_out, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(conv_out),
                                     nn.ReLU(inplace=inplace)
                                     )
    
    def _upsample_add(self, x, y):
        # return F.interpolate(
        #     x, size=y.size()[2:], mode='bilinear', align_corners=False) + y
        return F.interpolate(
            x, size=y.size()[2:], mode='nearest') + y
    
    def forward(self, x):
        assert len(x) == 4
        c2, c3, c4, c5 = x
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p4 = self.smooth1(p4)
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p3 = self.smooth2(p3)
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        p2 = self.smooth3(p2)
        return p2, p3, p4, p5


class FPN_Transpose(nn.Module):
    def __init__(self, out, fpn_out=256):
        super().__init__()
        conv_out = fpn_out
        inplace = True
        # Top layer
        self.toplayer = nn.Sequential(nn.Conv2d(out[3], conv_out, kernel_size=1, stride=1, padding=0),
                                      nn.BatchNorm2d(conv_out),
                                      nn.ReLU(inplace=inplace)
                                      )
        # Lateral layers
        self.latlayer1 = nn.Sequential(nn.Conv2d(out[2], conv_out, kernel_size=1, stride=1, padding=0),
                                       nn.BatchNorm2d(conv_out),
                                       nn.ReLU(inplace=inplace)
                                       )
        self.latlayer2 = nn.Sequential(nn.Conv2d(out[1], conv_out, kernel_size=1, stride=1, padding=0),
                                       nn.BatchNorm2d(conv_out),
                                       nn.ReLU(inplace=inplace)
                                       )
        self.latlayer3 = nn.Sequential(nn.Conv2d(out[0], conv_out, kernel_size=1, stride=1, padding=0),
                                       nn.BatchNorm2d(conv_out),
                                       nn.ReLU(inplace=inplace)
                                       )
        
        # Smooth layers
        self.smooth1 = nn.Sequential(nn.Conv2d(conv_out, conv_out, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(conv_out),
                                     nn.ReLU(inplace=inplace)
                                     )
        self.smooth2 = nn.Sequential(nn.Conv2d(conv_out, conv_out, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(conv_out),
                                     nn.ReLU(inplace=inplace)
                                     )
        self.smooth3 = nn.Sequential(nn.Conv2d(conv_out, conv_out, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(conv_out),
                                     nn.ReLU(inplace=inplace)
                                     )
        self.trans_conv1 = nn.ConvTranspose2d(fpn_out, fpn_out, kernel_size=2, stride=2)
        self.trans_conv2 = nn.ConvTranspose2d(fpn_out, fpn_out, kernel_size=2, stride=2)
        self.trans_conv3 = nn.ConvTranspose2d(fpn_out, fpn_out, kernel_size=2, stride=2)
    
    def forward(self, x):
        assert len(x) == 4
        c2, c3, c4, c5 = x
        p5 = self.toplayer(c5)
        p4 = self.trans_conv1(p5) + self.latlayer1(c4)
        p4 = self.smooth1(p4)
        p3 = self.trans_conv2(p4) + self.latlayer2(c3)
        p3 = self.smooth2(p3)
        p2 = self.trans_conv3(p3) + self.latlayer3(c2)
        p2 = self.smooth3(p2)
        return p2, p3, p4, p5
