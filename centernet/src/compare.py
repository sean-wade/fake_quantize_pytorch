from __future__ import division
from lib.models.networks.msra_resnet import get_pose_net

import argparse
import cv2
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
import copy

import qqquantize
import qqquantize.qqquantize.qmodules as qm
# from qqquantize import utils as qutils
from qqquantize.qqquantize.qconfig import COMPARE_QAT_MODULE_MAPPING
from qqquantize.qqquantize.quantize import ModelConverter
from qqquantize.qqquantize.observers.histogramobserver import HistObserver, swap_minmax_to_hist
from qqquantize.qqquantize.observers.minmaxchannelsobserver import MinMaxChannelsObserver, MovingAverageMinMaxChannelsObserver
from qqquantize.qqquantize.observers.minmaxobserver import MinMaxObserver, MovingAverageMinMaxObserver
from qqquantize.qqquantize.observers.fake_quantize import (
    FakeQuantize,
    Fake_quantize_per_channel,
    Fake_quantize_per_tensor,
    enable_fake_quant,
    disable_fake_quant,
    enable_observer,
    disable_observer,
    enable_calc_qparams,
    disable_calc_qparams,
    calc_qparams,
)
from qqquantize.qqquantize.savehook import register_intermediate_hooks

from easydict import EasyDict as edict
from collections import OrderedDict

CKPT_PATH = '/home/yxchen/CenterNet/centernet-perchannel.pt'
input_npy_path = '/home/yxchen/CenterNet/compare-data/centernet_input_8bit.npy'
output0_npy_path = '/home/yxchen/CenterNet/compare-data/centernet_output0_8bit.npy'
output1_npy_path = '/home/yxchen/CenterNet/compare-data/centernet_output1_8bit.npy'
output2_npy_path = '/home/yxchen/CenterNet/compare-data/centernet_output2_8bit.npy'


def input_quant(ckpt, x, bits):
    model = torch.load(ckpt)
    input_scale = model['inputStub.act_quant.scale'].numpy()
    xq = np.floor(x / input_scale)
    xq = np.clip(xq, - 2 ** (bits-1), 2 ** (bits-1) - 1)
    xqf = xq * input_scale
    return xqf


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--num_layers", type=int, default=50, help="the layer size of the resnet backbone")
    parser.add_argument("--head_conv", type=int, default=256, help="channel size of the centernet head")
    parser.add_argument("--num_classes", type=int, default=20, help="the number of object classes")
    parser.add_argument("--img_size", type=int, default=[224, 224], help="size of each image dimension")
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()

    model = get_pose_net(opt.num_layers,
                         {'hm': opt.num_classes, 'wh': 2, 'hps': 34},
                         opt.head_conv).to(opt.device)

    qconfig = edict({
        'activation': FakeQuantize.with_args(
            observer=MovingAverageMinMaxObserver,
            quantize_func=Fake_quantize_per_tensor,
            bits=8
        ),
        'weight': FakeQuantize.with_args(
            observer=MinMaxChannelsObserver,
            quantize_func=Fake_quantize_per_channel,
            bits=8
        ),
        'bias': FakeQuantize.with_args(bits=8),
    })

    mapping = copy.deepcopy(COMPARE_QAT_MODULE_MAPPING)
    qconverter = ModelConverter(qconfig, mapping, '', ['f', 'i'])
    model = qconverter(model)
    enable_observer(model)

    test_img = torch.ones(opt.batch_size, 3, *opt.img_size).to(opt.device)
    test_img = test_img.type(torch.DoubleTensor)
    test_out = model(test_img)

    calc_qparams(model)
    enable_fake_quant(model)

    ckpt = torch.load(CKPT_PATH, map_location=opt.device)
    model.load_state_dict(ckpt)

    disable_calc_qparams(model)
    disable_observer(model)
    hook = register_intermediate_hooks(model)

    feat = model(test_img)
    hook_data = hook.output_data()

    x = np.random.rand(*opt.img_size, 3)
    xq = input_quant(CKPT_PATH, x, 8)
    np.save(input_npy_path, xq.astype(np.double))
    # x = np.load(input_npy_path)

    x_tsr = transforms.ToTensor()(x)
    x_batch = x_tsr.unsqueeze(0)
    input_img = Variable(x_batch).to(opt.device)
    input_img = input_img.type(torch.DoubleTensor)

    model.double()
    model.eval()

    def forward_hooker(name):
        def hook_func(module, input, output):
            if isinstance(module, (qm.QConv2d_d, qm.QBatchNorm2d_d, qm.QConvTranspose2d_d, qm.QReLU)):
                np.save('/home/yxchen/CenterNet/compare-data/'
                        + name + '_out.npy', output.detach().cpu().numpy().astype(np.double))
        return hook_func
    
    for name, module in model.named_modules():
        module.register_forward_hook(forward_hooker(name))

    with torch.no_grad():
        output = model(input_img)
    output0 = output[0].numpy()
    output1 = output[1].numpy()
    output2 = output[2].numpy()
    np.save(output0_npy_path, output0.astype(np.double))
    np.save(output1_npy_path, output1.astype(np.double))
    np.save(output2_npy_path, output2.astype(np.double))