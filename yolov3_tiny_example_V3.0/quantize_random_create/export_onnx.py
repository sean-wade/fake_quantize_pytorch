from __future__ import division
import sys
sys.path.append('../')

from yolotiny_rdmcreate import Random_chnl_crt, Random_class_crt
from yolotiny_random import Yolov3Tiny_Random

import argparse
import cv2
import numpy as np
from pathlib import Path
import torch
import copy

import qqquantize
import qqquantize.qmodules as qm
# from qqquantize import utils as qutils
from qqquantize.qconfig import DEFAULT_QAT_MODULE_MAPPING, COMPARE_QAT_MODULE_MAPPING
from qqquantize.quantize import ModelConverter
from qqquantize.observers.histogramobserver import HistObserver, swap_minmax_to_hist
from qqquantize.observers.minmaxchannelsobserver import MinMaxChannelsObserver, MovingAverageMinMaxChannelsObserver
from qqquantize.observers.minmaxobserver import MinMaxObserver, MovingAverageMinMaxObserver, FixObserver
from qqquantize.observers.fake_quantize import (
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
from qqquantize.savehook import register_intermediate_hooks

from easydict import EasyDict as edict
from collections import OrderedDict
import torch.onnx
from torchvision import transforms
from torch.autograd import Variable


CKPT_PATH = './yolotiny_8bit_onnx/v3tiny.pt'
ONNX_PATH = './yolotiny_8bit_onnx/yolotiny_8bit_rdtest%d.onnx'


def fix_state_dict(stat_dict):
    new_dict = OrderedDict()
    for k, v in stat_dict.items():
        if k.startswith('module.'):
            k = k[7:]
            new_dict[k] = v
    return new_dict


if __name__ == "__main__":
    for i in range(2):
        parser = argparse.ArgumentParser()
        parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
        parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        opt = parser.parse_args()

        yolotiny_r_in = np.random.randint(4, 14)
        chnl = Random_chnl_crt(yolotiny_r_in)
        clss = Random_class_crt(yolotiny_r_in)

        model = Yolov3Tiny_Random(chnl, 3, clss, (32 * yolotiny_r_in, 32 * yolotiny_r_in)).to(opt.device)

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

        mapping = copy.deepcopy(DEFAULT_QAT_MODULE_MAPPING)
        qconverter = ModelConverter(qconfig, mapping, '', ['f', 'i'])
        model = qconverter(model)
        enable_observer(model)

        test_img = torch.ones(opt.batch_size, 3, 32 * yolotiny_r_in, 32 * yolotiny_r_in).to(opt.device)
        test_out = model(test_img)

        calc_qparams(model)
        enable_fake_quant(model)
        disable_calc_qparams(model)
        disable_observer(model)
        hook = register_intermediate_hooks(model)

        feat = model(test_img)
        torch.save(model.state_dict(), CKPT_PATH)
        
        hook_data = hook.output_data()
        new_hook_data = fix_state_dict(hook_data)

        new_model = Yolov3Tiny_Random(chnl, 3, clss, (32 * yolotiny_r_in, 32 * yolotiny_r_in)).to(opt.device)
        new_state_dict = new_model.state_dict()
        for k in list(new_state_dict.keys()):
            if 'bn' in k:
                if 'running_mean' not in k:
                    if 'running_var' not in k:
                        if 'num_batches_tracked' not in k:
                            v = torch.tensor(new_hook_data[f'{k}_quant']['values'][0])
                            new_state_dict[k] = v.view(-1)
            elif 'downsample.1' in k:
                if 'running_mean' not in k:
                    if 'running_var' not in k:
                        if 'num_batches_tracked' not in k:
                            v = torch.tensor(new_hook_data[f'{k}_quant']['values'][0])
                            new_state_dict[k] = v.view(-1)
            else:
                if 'running_mean' not in k:
                    if 'running_var' not in k:
                        if 'num_batches' not in k:
                            if 'anchors' not in k:
                                if 'anchor_grid' not in k:
                                    v = torch.tensor(new_hook_data[f'{k}_quant']['values'][0])
                                    new_state_dict[k] = v
        new_model.load_state_dict(new_state_dict)
        new_model.eval()

#         f = '/home/yxchen/ProjectRandom_II/yolotiny_random/yolotiny_8bit_onnx/yolotiny_8bit_rdtest' + str(i) + '.onnx'
        img = torch.zeros(opt.batch_size, 3, 32 * yolotiny_r_in, 32 * yolotiny_r_in).to(opt.device)
        torch.onnx.export(new_model, 
                          test_img, 
                          ONNX_PATH%i, 
                          verbose=False, 
                          opset_version=10, 
                          input_names=['images'],
                          output_names=['scores'], 
                          training=True)
