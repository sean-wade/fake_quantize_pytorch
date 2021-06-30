import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class DarknetConv2D_BN_Leaky(nn.Module):
    def __init__(self, numIn, numOut, ksize, stride=1, padding=1):
        super(DarknetConv2D_BN_Leaky, self).__init__()
        self.conv1 = nn.Conv2d(numIn, numOut, ksize, stride, padding, bias=True)  # regularizer': l2(5e-4)
        self.bn1 = nn.BatchNorm2d(numOut)
        self.leakyReLU = nn.LeakyReLU(0.125)
        # self.conv_bn_relu = ConvBnReLU2d(numIn, numOut, ksize, stride, padding)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leakyReLU(x)
        # x = self.conv_bn_relu(x)
        return x


class Yolov3Tiny_Random(nn.Module):
    def __init__(self, channels: List[int], numAnchor, numClass, img_dim=(512, 512)):
        super(Yolov3Tiny_Random, self).__init__()
        self.conv1 = DarknetConv2D_BN_Leaky(3, channels[0], 3)
        self.conv2 = DarknetConv2D_BN_Leaky(channels[0], channels[1], 3)
        self.conv3 = DarknetConv2D_BN_Leaky(channels[1], channels[2], 3)
        self.conv4 = DarknetConv2D_BN_Leaky(channels[2], channels[3], 3)
        self.conv5 = DarknetConv2D_BN_Leaky(channels[3], channels[4], 3)
        self.conv6 = DarknetConv2D_BN_Leaky(channels[4], channels[5], 3)
        self.conv7 = DarknetConv2D_BN_Leaky(channels[5], channels[6], 3)
        self.maxpoo1_s = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpoo2_s = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpoo3_s = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpoo4_s = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpoo5_s = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpoo1_ns = torch.nn.MaxPool2d(kernel_size=2, stride=1)
        self.pad = torch.nn.ConstantPad2d((0, 1, 0, 1), 0.)
        # self.pad = torch.nn.ZeroPad2d((0, 1, 0, 1))

        self.lastconv1 = DarknetConv2D_BN_Leaky(channels[6], channels[7], 1, padding=0)
        self.lastconv2 = DarknetConv2D_BN_Leaky(channels[7], channels[8], 3)
        self.lastconv3 = nn.Conv2d(channels[8], numAnchor * (numClass + 5), 1, bias=True, padding=0)
        self.lastconv4 = DarknetConv2D_BN_Leaky(channels[7], channels[9], 1, padding=0)
        self.lastconv5 = DarknetConv2D_BN_Leaky(channels[4] + channels[9], channels[10], 3)
        self.lastconv6 = nn.Conv2d(channels[10], numAnchor * (numClass + 5), 1, bias=True, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.normal_(m.weight)
                nn.init.normal_(m.bias)

    def forward(self, x, target=None):  # [2, 3, 416, 416]
        x = self.conv1(x)
        x = self.maxpoo1_s(x)
        x = self.conv2(x)
        x = self.maxpoo2_s(x)
        x = self.conv3(x)
        x = self.maxpoo3_s(x)
        x = self.conv4(x)
        x = self.maxpoo4_s(x)
        y1 = self.conv5(x)
        x = self.maxpoo5_s(y1)
        x = self.conv6(x)
        x = self.pad(x)
        x = self.maxpoo1_ns(x)
        y2 = self.conv7(x)

        branch = self.lastconv1(y2)
        tmp = self.lastconv2(branch)
        tmp1 = self.lastconv3(tmp)

        tmp = self.lastconv4(branch)
        tmp = F.interpolate(tmp, scale_factor=2)
        tmp = torch.cat((tmp, y1), 1)
        tmp = self.lastconv5(tmp)
        tmp2 = self.lastconv6(tmp)
        return tmp1, tmp2