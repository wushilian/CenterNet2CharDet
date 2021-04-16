import torch
import torch.nn as nn


class PredictionModule(nn.Module):
    def __init__(self, in_channels, num_classes=1):
        super().__init__()
        
        self.num_classes = num_classes
        self.hmap = nn.Conv2d(in_channels=in_channels, out_channels=num_classes, kernel_size=1, bias=True)
        self.w_h_ = nn.Conv2d(in_channels=in_channels, out_channels=2, kernel_size=1, bias=True)
        self.reg_ = nn.Conv2d(in_channels=in_channels, out_channels=2, kernel_size=1, bias=True)
        self.hmap.bias.data.fill_(-2.19)
        self.hmap.weight.data.fill_(0)
        
        self.reg_.bias.data.fill_(-2.19)
        self.reg_.weight.data.fill_(0)
    
    def forward(self, x):
        hmap = self.hmap(x)
        wh = torch.exp_(self.w_h_(x))
        reg = self.reg_(x)
        wh = torch.cat([wh, reg], dim=1)
        return {'hmap': hmap, 'wh': wh}