from __future__ import division
from resnet import resnet18, resnet50

import argparse
import numpy as np
import torch
import copy

import qqquantize
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
import onnx

CKPT_PATH = './resnet18-fix-perchannel.pt'
txt_path = "./resnet18_fix_perchannel_quanttable.txt"

if __name__ == '__main__':
    onnx_model = onnx.load('./resnet18_fix_perchannel.onnx')
    graph = onnx_model.graph
    node = graph.node

    conv_input = []
    conv_output = []
    bn_input = []
    bn_output = []
    elt_input = []
    elt_output = []
    avgpool_input = []
    avgpool_output = []
    fc_input = []
    fc_output = []

    for i in range(len(node)):
        if 'Conv' in node[i].name:
            conv_input.append(node[i].input[0])
            conv_output.append(node[i].output[0])
        if 'BatchNormalization' in node[i].name:
            bn_input.append(node[i].input[0])
            bn_output.append(node[i].output[0])
        if 'Add' in node[i].name:
            elt_input.append(node[i].input[0])
            elt_input.append(node[i].input[1])
            elt_output.append(node[i].output[0])
        if 'GlobalAveragePool' in node[i].name:
            avgpool_input.append(node[i].input[0])
            avgpool_output.append(node[i].output[0])
        if 'Gemm' in node[i].name:
            fc_input.append(node[i].input[0])
            fc_output.append(node[i].output[0])

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
    model.double()
    enable_observer(model)

    test_img = torch.zeros(opt.batch_size, 3, *opt.img_size).to(opt.device)
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

    conv_act = []
    conv_wt = []
    bn_act = []
    bn_wt = []
    bn_bias = []
    elt_act = []
    avgpool_act = []
    fc_act = []
    fc_wt = []
    fc_bias = []
    input_act = []

    for k in hook_data.keys():
        scale = hook_data[k]['scale']
        # assert scale.size == 1
        float_bit = []
        for i in range(len(scale)):
            float_bit.append(- np.log2(scale[i]))

        if 'conv' in k:
            if 'act' in k:
                conv_act.append(float_bit)
            if 'weight' in k:
                conv_wt.append(float_bit)
            # conv_bias.append(None)

        if 'downsample.0' in k:
            if 'act' in k:
                conv_act.append(float_bit)
            if 'weight' in k:
                conv_wt.append(float_bit)

        if 'bn' in k:
            if 'act' in k:
                bn_act.append(float_bit)
            if 'weight' in k:
                bn_wt.append(float_bit)
            if 'bias' in k:
                bn_bias.append(float_bit)

        if 'eltwise' in k:
            if 'act' in k:
                elt_act.append(float_bit)

        if 'downsample.1' in k:
            if 'act' in k:
                bn_act.append(float_bit)
            if 'weight' in k:
                bn_wt.append(float_bit)
            if 'bias' in k:
                bn_bias.append(float_bit)

        if 'avgpool' in k:
            if 'act' in k:
                avgpool_act.append(float_bit)

        if 'fc' in k:
            if 'act' in k:
                fc_act.append(float_bit)
            if 'weight' in k:
                fc_wt.append(float_bit)
            if 'bias' in k:
                fc_bias.append(float_bit)

        if 'input' in k:
            if 'act' in k:
                input_act.append(float_bit)

    with open(txt_path, 'a+') as fp:
        fp.write('input_act:' + '\t' + str(input_act[0]) + '\n')

    for i in range(len(bn_bias)):
        with open(txt_path, 'a+') as fp:
            fp.write('conv_' + str(i) + ':' + '\t' + 'act: ' + str(conv_act[i]) + '\t' + 'weight: ' + str(conv_wt[i])
                     + '\t' + 'input: ' + str(conv_input[i]) + '\t' + 'output: ' + str(conv_output[i]) + '\n')
        with open(txt_path, 'a+') as fp:
            fp.write('bn_' + str(i) + ':' + '\t' + 'act: ' + str(bn_act[i]) + '\t' + 'weight: ' + str(bn_wt[i]) + '\t'
                     + 'bias: ' + str(bn_bias[i]) + '\t' + 'input: ' + str(bn_input[i]) + '\t'
                     + 'output: ' + str(bn_output[i]) + '\n')

    for i in range(len(elt_act)):
        with open(txt_path, 'a+') as fp:
            fp.write('elt_' + str(i) + ':' + '\t' + 'act: ' + str(elt_act[i]) + '\t'
                     + 'input: ' + str(elt_input[2 * i]) + '\t' + str(elt_input[2 * i + 1]) + '\t'
                     + 'output: ' + str(elt_output[i]) + '\n')

    for i in range(len(avgpool_act)):
        with open(txt_path, 'a+') as fp:
            fp.write('avgpool_' + str(i) + ':' + '\t' + 'act: ' + str(avgpool_act[i]) + '\t'
                     + 'input: ' + str(avgpool_input[i]) + '\t' + 'output: ' + str(avgpool_output[i]) + '\n')

    for i in range(len(fc_act)):
        with open(txt_path, 'a+') as fp:
            fp.write('fc_' + str(i) + ':' + '\t' + 'act: ' + str(fc_act[i]) + '\t'
                     + 'weight: ' + str(fc_wt[i]) + '\t'  + 'bias: ' + str(fc_bias[i]) + '\t'
                     + 'input: ' + str(fc_input[i]) + '\t' + 'output: ' + str(fc_output[i]) + '\n')
