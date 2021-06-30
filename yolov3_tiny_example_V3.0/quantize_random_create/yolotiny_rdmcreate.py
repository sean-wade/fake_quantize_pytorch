import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class DarknetConv2D_BN_Leaky(nn.Module):
    def __init__(self, numIn, numOut, ksize, stride=1, padding=1):
        super(DarknetConv2D_BN_Leaky, self).__init__()
        self.conv1 = nn.Conv2d(numIn, numOut, ksize, stride, padding, bias=True)  # regularizer': l2(5e-4)
        self.bn1 = nn.BatchNorm2d(numOut)
        self.leakyReLU = nn.LeakyReLU(0.125)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leakyReLU(x)
        return x


class Yolov3Tiny(nn.Module):
    def __init__(self, numAnchor, numClass, img_dim=(512, 512)):
        super(Yolov3Tiny, self).__init__()
        self.conv1 = DarknetConv2D_BN_Leaky(3, 16, 3)
        self.conv2 = DarknetConv2D_BN_Leaky(16, 32, 3)
        self.conv3 = DarknetConv2D_BN_Leaky(32, 64, 3)
        self.conv4 = DarknetConv2D_BN_Leaky(64, 128, 3)
        self.conv5 = DarknetConv2D_BN_Leaky(128, 256, 3)
        self.conv6 = DarknetConv2D_BN_Leaky(256, 512, 3)
        self.conv7 = DarknetConv2D_BN_Leaky(512, 1024, 3)
        self.maxpoo1_s = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpoo2_s = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpoo3_s = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpoo4_s = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpoo5_s = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpoo1_ns = torch.nn.MaxPool2d(kernel_size=2, stride=1)
        self.pad = torch.nn.ConstantPad2d((0, 1, 0, 1), 0.)
        # self.pad = torch.nn.ZeroPad2d((0, 1, 0, 1))

        self.lastconv1 = DarknetConv2D_BN_Leaky(1024, 256, 1, padding=0)
        self.lastconv2 = DarknetConv2D_BN_Leaky(256, 512, 3)
        self.lastconv3 = nn.Conv2d(512, numAnchor * (numClass + 5), 1, bias=True, padding=0)
        self.lastconv4 = DarknetConv2D_BN_Leaky(256, 128, 1, padding=0)
        self.lastconv5 = DarknetConv2D_BN_Leaky(384, 256, 3)
        self.lastconv6 = nn.Conv2d(256, numAnchor * (numClass + 5), 1, bias=True, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.normal_(m.weight)
                nn.init.normal_(m.bias)


    def forward(self, x):  # [2, 3, 416, 416]
        nix = []
        nix_class = []
        nix.append(x.size(-1))
        x = self.conv1(x)
        x = self.maxpoo1_s(x)
        nix.append(x.size(-1))
        x = self.conv2(x)
        x = self.maxpoo2_s(x)
        nix.append(x.size(-1))
        x = self.conv3(x)
        x = self.maxpoo3_s(x)
        nix.append(x.size(-1))
        x = self.conv4(x)
        x = self.maxpoo4_s(x)
        nix.append(x.size(-1))
        y1 = self.conv5(x)
        x = self.maxpoo5_s(y1)
        nix.append(x.size(-1))
        x = self.conv6(x)
        x = self.pad(x)
        x = self.maxpoo1_ns(x)
        nix.append(x.size(-1))
        y2 = self.conv7(x)
        nix.append(y2.size(-1))

        branch = self.lastconv1(y2)
        nix.append(branch.size(-1))
        tmp = self.lastconv2(branch)
        nix_class.append(tmp.size(-1))
        tmp1 = self.lastconv3(tmp)

        tmp = self.lastconv4(branch)
        tmp = F.interpolate(tmp, scale_factor=2)
        tmp = torch.cat((tmp, y1), 1)
        nix.append(tmp.size(-1))
        tmp = self.lastconv5(tmp)
        nix_class.append(tmp.size(-1))
        tmp2 = self.lastconv6(tmp)
        return nix, nix_class


def Random_chnl_crt(input_random):
    img = torch.zeros(1, 3, 32 * input_random, 32 * input_random)

    model = Yolov3Tiny(3, 80, (32 * input_random, 32 * input_random))
    out, _ = model(img)

    random_channel = []

    # for i in range(10):
    #     if (out[i] < 32):
    #         out[i] = 32
        
    #     if (i != 4) and (i != 8):
    #         channel_max = min(1024, math.floor(78643 / out[i]))
    #         channel = np.random.randint(4, channel_max + 1)
    #         random_channel.append(channel)
    #     elif i == 8:
    #         channel_max1 = min(1024, math.floor(78643 / out[i]))
    #         channel1 = np.random.randint(4, channel_max1 + 1)
    #         random_channel.append(channel1)
    #         channel_max2 = min(512, math.floor(78643 / out[i]))
    #         channel2 = np.random.randint(4, channel_max2 + 1)
    #         random_channel.append(channel2)
    #     else:
    #         channel_max = min(512, math.floor(78643 / out[i]))
    #         channel = np.random.randint(4, channel_max + 1)
    #         random_channel.append(channel)

    channel_max = 16
    
    for i in range(7):
        channel = np.random.randint(4, channel_max + 1)
        random_channel.append(channel)

        channel_max *= 2
    
    random_channel.append(np.random.randint(4, 257))
    random_channel.append(np.random.randint(4, 513))
    random_channel.append(np.random.randint(4, 129))
    random_channel.append(np.random.randint(4, 257))
    
    return random_channel


def Random_class_crt(input_random):
    # img = torch.zeros(1, 3, 32 * input_random, 32 * input_random)

    # model = Yolov3Tiny(3, 80, (32 * input_random, 32 * input_random))
    # _, out = model(img)

    # random_class = []

    # for i in range(2):
    #     if (out[i] < 32):
    #         out[i] = 32
        
    #     random_class.append(out[i])
    
    # cmpr = max(random_class[0], random_class[1])
    # channel_ = min(2048, math.floor(78643 / cmpr))
    # class_max = math.floor(channel_ / 3 - 5)
    clss = np.random.randint(10, 81)

    return clss


if __name__ == "__main__":
    # img = torch.zeros(1, 3, 416, 416)
    # model = Yolov3Tiny(3, 80, (416, 416))
    # out = model(img)
    # print(out)
    class_ = Random_class_crt(16)
    print(class_)

    # chnl = Random_chnl_crt(12)
    # print(chnl)

    # channel_max = min(2048, math.floor(78643 / out[6]))
    # print(channel_max)
    # for i in range(7):
    #     print(out[i])