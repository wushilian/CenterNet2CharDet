from torch import nn
import torch
class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
                in_chan, out_chan, kernel_size=ks, stride=stride,
                padding=padding, dilation=dilation,
                groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv(x)
        feat = self.bn(feat)
        feat = self.relu(feat)
        return feat
    
class SimpleBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.S1 = nn.Sequential(
            ConvBNReLU(3, 64, 3, stride=2),
            ConvBNReLU(64, 64, 3, stride=1),
        )
        self.S2 = nn.Sequential(
            ConvBNReLU(64, 64, 3, stride=2),
            ConvBNReLU(64, 64, 3, stride=1),
            #ConvBNReLU(64, 64, 3, stride=1),
        )
        self.S3 = nn.Sequential(
            ConvBNReLU(64, 128, 3, stride=2),
            ConvBNReLU(128, 128, 3, stride=1),
            #ConvBNReLU(128, 128, 3, stride=1),
        )
        self.S4 = nn.Sequential(
            ConvBNReLU(128, 128, 3, stride=2),
            ConvBNReLU(128, 128, 3, stride=1),
            #ConvBNReLU(128, 128, 3, stride=1),
        )
        self.S5 = nn.Sequential(
            ConvBNReLU(128, 128, 3, stride=2),
            ConvBNReLU(128, 128, 3, stride=1),
           # ConvBNReLU(128, 128, 3, stride=1),
        )

        self.backbone_modules = [m for m in self.modules() if isinstance(m, nn.Conv2d)]
        
    def forward(self,x):
        c1=self.S1(x)
        c2=self.S2(c1)
        c3=self.S3(c2)
        c4=self.S4(c3)
        c5=self.S5(c4)
        return (c2,c3,c4,c5)
    
    def init_weights(self,layers):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                if hasattr(module, 'last_bn') and module.last_bn:
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

def get_simple_backbone():
    model=SimpleBackbone()
    out=[64,128,128,128]
    return model,out

if __name__ == "__main__":
    from time import time
    from torchvision import models
    """Testing
    """
    #model=models.resnet18(pretrained=False)
    model=SimpleBackbone()
    #print(model)
    net=model.cuda()
    dummy_input = torch.randn(1, 3, 128, 128, device='cpu')
    dummy_input = dummy_input.cuda()
    for i in range(100):
        t1 = time()
        b = net(dummy_input)
        t2 = time()
        print(t2-t1)
        