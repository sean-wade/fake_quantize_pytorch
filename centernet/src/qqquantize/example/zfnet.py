import torch
import torch.nn as nn
from torch.quantization.stubs import QuantStub, DeQuantStub

__all__ = ['ZFNet']

class ConvBnReLU(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(*args, **kwargs)
        self.bn = nn.BatchNorm2d(self.conv.out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class LinearBnReLU(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.linear = nn.Linear(*args, **kwargs)
        self.bn = nn.BatchNorm1d(self.linear.out_features)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ZFNet(nn.Module):
    def __init__(self, cf=1.0):
        super(ZFNet, self).__init__()
        self.conv = nn.Sequential(
            # 第一层
            ConvBnReLU(3, int(96*cf), 7, 2, bias=False),
            ConvBnReLU(int(96*cf), int(160*cf), 5, 2, bias=False),
            # 第三层
            ConvBnReLU(int(160*cf), int(256*cf), 3, 1, 1, bias=False),
            # 第四层
            ConvBnReLU(int(256*cf), int(256*cf), 3, 1, 0, bias=False),
            # 第五层
            ConvBnReLU(int(256*cf), int(256*cf), 3, 1, 0, bias=False),
        )
        # 全连接层
        self.fc = nn.Sequential(
            LinearBnReLU(int(256*cf), int(256*cf), bias=False),
            nn.Linear(int(256*cf), 10, bias=False),
        )
    def forward(self, img):
        BS = img.shape[0]
        feature = self.conv(img)
        feature = feature.reshape([BS, -1])
        output = self.fc(feature)
        return output

def fuse_zfnet(net, inplace=False):
    from qqquantize.utils import fuse_conv_bn
    import copy
    if not inplace:
        net = copy.deepcopy(net)
    for mod in net.modules():
        if isinstance(mod, ConvBnReLU):
            mod.conv = fuse_conv_bn(mod.conv, mod.bn)
            mod.bn = nn.Identity()
        if isinstance(mod, LinearBnReLU):
            mod.linear = fuse_conv_bn(mod.linear, mod.bn)
            mod.bn = nn.Identity()
    return net