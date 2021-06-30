from __future__ import division
from qqquantize.qqquantize.qmodules.qstub import QStub
from resnet import resnet18, resnet50

import argparse
import cv2
import numpy as np
from pathlib import Path
import torch
import copy

import qqquantize
import qqquantize.qqquantize.qmodules as qm
# from qqquantize import utils as qutils
from qqquantize.qqquantize.qconfig import DEFAULT_QAT_MODULE_MAPPING,COMPARE_QAT_MODULE_MAPPING
from qqquantize.qqquantize.quantize import ModelConverter
from qqquantize.qqquantize.observers.histogramobserver import HistObserver, swap_minmax_to_hist
from qqquantize.qqquantize.observers.minmaxchannelsobserver import MinMaxChannelsObserver, MovingAverageMinMaxChannelsObserver
from qqquantize.qqquantize.observers.minmaxobserver import MinMaxObserver, MovingAverageMinMaxObserver, FixObserver
from qqquantize.qqquantize.observers.fixobserver import FixActObserver, FixWtObserver
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
import torch.onnx
from torchvision import transforms
from torch.autograd import Variable


CKPT_PATH = './resnet18-fix-perchannel.pt'
input_npy_path = './compare-data/resnet18_input_8bit.npy'
output_npy_path = './compare-data/resnet18_output_8bit.npy'


def fix_state_dict(stat_dict):
    new_dict = OrderedDict()
    for k, v in stat_dict.items():
        if k.startswith('module.'):
            k = k[7:]
            new_dict[k] = v
    return new_dict


def input_quant(ckpt, x, bits):
    model = torch.load(ckpt)
    input_scale = model['inputStub.act_quant.scale'].numpy()
    xq = np.floor(x / input_scale)
    xq = np.clip(xq, - 2 ** (bits-1), 2 ** (bits-1) - 1)
    xqf = xq * input_scale
    return xqf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--num_classes", type=int, default=10, help="the number of object classes")
    parser.add_argument("--img_size", type=int, default=[224, 224], help="size of each image dimension")
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()

    model = resnet18(pretrained=False).to(opt.device)
    # model = resnet50(pretrained=False).to(opt.device)

#     qconfig = edict({
#         'activation': FakeQuantize.with_args(
#             observer=FixObserver,
#             quantize_func=Fake_quantize_per_tensor,
#             bits=8
#         ),
#         'weight': FakeQuantize.with_args(
#             observer=MinMaxChannelsObserver,
#             quantize_func=Fake_quantize_per_channel,
#             bits=8
#         ),
#         'bias': FakeQuantize.with_args(bits=8),
#     })

    qconfig = edict({
    'activation': FakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        quantize_func=Fake_quantize_per_tensor,
        bits=8  #BITS
    ),
    'weight': FakeQuantize.with_args(
        observer=MinMaxChannelsObserver,
        quantize_func=Fake_quantize_per_channel,
        bits=8  #BITS
    ),
    'bias': FakeQuantize.with_args(
        bits=16  #BITS
        ),
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
    disable_calc_qparams(model)
    disable_observer(model)
    hook = register_intermediate_hooks(model)

    feat = model(test_img)
    torch.save(model.state_dict(), CKPT_PATH)

    hook_data = hook.output_data()
    new_hook_data = fix_state_dict(hook_data)

    new_model = resnet18(pretrained=False).to(opt.device)
    # new_model = resnet50(pretrained=False).to(opt.device)
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
    new_model.double()
    new_model.eval()

    f = "./resnet18_fix_perchannel.onnx"
    img = torch.zeros(opt.batch_size, 3, *opt.img_size).to(opt.device)
    torch.onnx.export(new_model, test_img, f, verbose=False, opset_version=10, input_names=['images'],
                      output_names=['scores'], training=True)

    # m = onnx.load('./resnet18_8bit_test.onnx')
    # model_simp, check = simplify(m)
    # onnx.save(model_simp, 
    #           './resnet18_8bit.onnx')
    # assert check, "Simplified ONNX model could not be validated"

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
            if isinstance(module, (qm.QConv2d_d, qm.QBatchNorm2d_d,
                          qm.QLinear_d, qm.QStub, qm.QReLU, qm.QAvgPooling2d)):
                np.save('./compare-data/'
                        + name + '_out.npy', output.detach().cpu().numpy().astype(np.double))
        return hook_func

    for name, module in model.named_modules():
        module.register_forward_hook(forward_hooker(name))

    with torch.no_grad():
        output = model(input_img)
    output = output.numpy()
    np.save(output_npy_path, output.astype(np.double))
