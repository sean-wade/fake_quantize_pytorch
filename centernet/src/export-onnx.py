from __future__ import division
from lib.models.networks.msra_resnet import get_pose_net

import argparse
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from torch.autograd import Variable
import copy

import qqquantize
import qqquantize.qqquantize.qmodules as qm
# from qqquantize import utils as qutils
from qqquantize.qqquantize.qconfig import DEFAULT_QAT_MODULE_MAPPING, COMPARE_QAT_MODULE_MAPPING
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


CKPT_PATH = '/home/yxchen/CenterNet/centernet-perchannel-16bit.pt'
input_npy_path = '/home/yxchen/CenterNet/compare-data-16bit/centernet_input_16bit.npy'
output0_npy_path = '/home/yxchen/CenterNet/compare-data-16bit/centernet_output0_16bit.npy'
output1_npy_path = '/home/yxchen/CenterNet/compare-data-16bit/centernet_output1_16bit.npy'
output2_npy_path = '/home/yxchen/CenterNet/compare-data-16bit/centernet_output2_16bit.npy'
filename = '/home/yxchen/CenterNet/images/16004479832_a748d55f21_k.jpg'


def input_quant(ckpt, x, bits):
    model = torch.load(ckpt)
    input_scale = model['inputStub.act_quant.scale'].numpy()
    xq = np.floor(x / input_scale)
    xq = np.clip(xq, - 2 ** (bits-1), 2 ** (bits-1) - 1)
    xqf = xq * input_scale
    return xqf


def fix_state_dict(stat_dict):
    new_dict = OrderedDict()
    for k, v in stat_dict.items():
        if k.startswith('module.'):
            k = k[7:]
            new_dict[k] = v
    return new_dict


def bn_state_dict(stat_dict):
    new_dict = OrderedDict()
    for k in list(stat_dict.keys()):
        if 'running_mean' in k:
            k = k[:-13]
            new_dict[k] = 0
    return new_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--num_layers", type=int, default=50, help="the layer size of the resnet backbone")
    parser.add_argument("--head_conv", type=int, default=256, help="channel size of the centernet head")
    parser.add_argument("--num_classes", type=int, default=20, help="the number of object classes")
    parser.add_argument("--img_size", type=int, default=[224, 224], help="size of each image dimension")
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()

    model = get_pose_net(opt.num_layers, {'hm': opt.num_classes, 'wh': 2, 'hps': 34}, opt.head_conv).to(opt.device)

    qconfig = edict({
        'activation': FakeQuantize.with_args(
            observer=FixActObserver,
            quantize_func=Fake_quantize_per_tensor,
            bits=16
        ),
        'weight': FakeQuantize.with_args(
            observer=FixWtObserver,
            quantize_func=Fake_quantize_per_tensor,
            bits=16
        ),
        'bias': FakeQuantize.with_args(
            observer=FixActObserver,
            quantize_func=Fake_quantize_per_tensor,
            bits=16
        ),
    })

    mapping = copy.deepcopy(COMPARE_QAT_MODULE_MAPPING)
    qconverter = ModelConverter(qconfig, mapping, '', ['f', 'i'])
    model = qconverter(model)
    model.double()
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

    new_model = get_pose_net(opt.num_layers,
                             {'hm': opt.num_classes, 'wh': 2, 'hps': 34},
                             opt.head_conv).to(opt.device)
    new_state_dict = new_model.state_dict()
    bn_keys = bn_state_dict(new_state_dict)
    for k in list(new_state_dict.keys()):
        if k[:-7] in list(bn_keys.keys()):
            v = torch.tensor(new_hook_data[f'{k}_quant']['values'][0])
            new_state_dict[k] = v.view(-1)
        elif k[:-5] in list(bn_keys.keys()):
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

    f = "./centernet_fix_perchannel_16bit.onnx"
    img = torch.zeros(opt.batch_size, 3, *opt.img_size).to(opt.device)
    torch.onnx.export(new_model, test_img, f, verbose=False, opset_version=10, input_names=['images'],
                      training=True)

    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    x = input_tensor.unsqueeze(0)

    # x = np.random.rand(*opt.img_size, 3)
    xq = input_quant(CKPT_PATH, input_tensor.numpy(), 16)
    np.save(input_npy_path, xq.astype(np.double))
    # x = np.load(input_npy_path)

    # x_tsr = transforms.ToTensor()(input_tensor)
    # x_batch = x_tsr.unsqueeze(0)
    input_img = Variable(x).to(opt.device)
    input_img = input_img.type(torch.DoubleTensor)

    model.double()
    model.eval()

    def forward_hooker(name):
        def hook_func(module, input, output):
            if isinstance(module, (qm.QConv2d_d, qm.QBatchNorm2d_d,
                                   qm.QConvTranspose2d_d, qm.QReLU, qm.QStub)):
                np.save('/home/yxchen/CenterNet/compare-data-16bit/'
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
