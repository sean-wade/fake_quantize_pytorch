from __future__ import division
from lib.models.networks.msra_resnet import get_pose_net

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

CKPT_PATH = '/home/yxchen/CenterNet/centernet-perchannel-16bit.pt'
txt_path = "/home/yxchen/CenterNet/centernet_fix_perchannel_16bit_quanttable.txt"


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
    onnx_model = onnx.load('/home/yxchen/CenterNet/src/centernet_fix_perchannel_16bit.onnx')
    graph = onnx_model.graph
    node = graph.node

    conv_input = []
    conv_output = []
    bn_input = []
    bn_output = []
    elt_input = []
    elt_output = []
    convtrans_input = []
    convtrans_output = []

    for i in range(len(node)):
        if 'Conv' in node[i].name and 'Transpose' not in node[i].name:
            conv_input.append(node[i].input[0])
            conv_output.append(node[i].output[0])
        if 'BatchNormalization' in node[i].name:
            bn_input.append(node[i].input[0])
            bn_output.append(node[i].output[0])
        if 'Add' in node[i].name:
            elt_input.append(node[i].input[0])
            elt_input.append(node[i].input[1])
            elt_output.append(node[i].output[0])
        if 'ConvTranspose' in node[i].name:
            convtrans_input.append(node[i].input[0])
            convtrans_output.append(node[i].output[0])
    
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
    
    new_model = get_pose_net(opt.num_layers,
                             {'hm': opt.num_classes, 'wh': 2, 'hps': 34},
                             opt.head_conv).to(opt.device)
    new_state_dict = new_model.state_dict()
    bn_keys = bn_state_dict(new_state_dict)

    conv_act = []
    conv_wt = []
    conv_bias = []
    bn_act = []
    bn_wt = []
    bn_bias = []
    elt_act = []
    convtrans_act = []
    convtrans_wt = []
    convtrans_bias = []
    input_act = []

    for k in hook_data.keys():
        scale = hook_data[k]['scale']
        # assert scale.size == 1
        float_bit = []
        for i in range(len(scale)):
            float_bit.append(- np.log2(scale[i]))

        # if 'conv' in k:
        #     if 'up' not in k:
        #         if 'act' in k:
        #             conv_act.append(float_bit)
        #         if 'weight' in k:
        #             conv_wt.append(float_bit)
        #         if 'bias' in k:
        #             conv_bias.append(float_bit)
        if 'input' in k:
            if 'act' in k:
                input_act.append(float_bit)
        
        elif k[7:-13] in list(bn_keys.keys()):
            bn_wt.append(float_bit)
        
        elif k[7:-10] in list(bn_keys.keys()):
            bn_act.append(float_bit)
        
        elif k[7:-11] in list(bn_keys.keys()):
            bn_bias.append(float_bit)
        
        elif 'eltwise' in k:
            if 'act' in k:
                elt_act.append(float_bit)
        
        elif 'deconv' in k:
            if 'act' in k:
                convtrans_act.append(float_bit)
            if 'weight' in k:
                convtrans_wt.append(float_bit)
            if 'bias' in k:
                convtrans_bias.append(float_bit)
        
        else:
            if 'act' in k:
                conv_act.append(float_bit)
            if 'weight' in k:
                conv_wt.append(float_bit)
            if 'bias' in k:
                conv_bias.append(float_bit)

    with open(txt_path, 'a+') as fp:
        fp.write('input_act:' + '\t' + str(input_act[0]) + '\n')
    
    for i in range(len(conv_act) - 6):
        with open(txt_path, 'a+') as fp:
            fp.write('conv_' + str(i) + ':' + '\t' + 'act: ' + str(conv_act[i]) + '\t' + 'weight: ' + str(conv_wt[i])
                     + '\t' + 'input: ' + str(conv_input[i]) + '\t' + 'output: ' + str(conv_output[i]) + '\n')
    
    for i in range(2):
        with open(txt_path, 'a+') as fp:
            fp.write('conv_' + str(len(conv_act) - 6 + i) + ':' + '\t' + 'act: ' + str(conv_act[len(conv_act) - 6 + i])
                     + '\t' + 'weight: ' + str(conv_wt[len(conv_act) - 6 + i]) + '\t' + 'bias: ' + str(conv_bias[i])
                     + '\t' + 'input: ' + str(conv_input[len(conv_act) - 6 + i]) + '\t'
                     + 'output: ' + str(conv_output[len(conv_act) - 6 + i]) + '\n')
    
    for i in range(2):
        with open(txt_path, 'a+') as fp:
            fp.write('conv_' + str(len(conv_act) - 4 + i) + ':' + '\t' + 'act: ' + str(conv_act[len(conv_act) - 4 + i])
                     + '\t' + 'weight: ' + str(conv_wt[len(conv_act) - 4 + i]) + '\t' + 'bias: ' + str(conv_bias[i])
                     + '\t' + 'input: ' + str(conv_input[len(conv_act) - 2 + i]) + '\t'
                     + 'output: ' + str(conv_output[len(conv_act) - 2 + i]) + '\n')
    
    for i in range(2):
        with open(txt_path, 'a+') as fp:
            fp.write('conv_' + str(len(conv_act) - 2 + i) + ':' + '\t' + 'act: ' + str(conv_act[len(conv_act) - 2 + i])
                     + '\t' + 'weight: ' + str(conv_wt[len(conv_act) - 2 + i]) + '\t' + 'bias: ' + str(conv_bias[i])
                     + '\t' + 'input: ' + str(conv_input[len(conv_act) - 4 + i]) + '\t'
                     + 'output: ' + str(conv_output[len(conv_act) - 4 + i]) + '\n')

    for i in range(len(bn_act)):
        with open(txt_path, 'a+') as fp:
            fp.write('bn_' + str(i) + ':' + '\t' + 'act: ' + str(bn_act[i]) + '\t' + 'weight: ' + str(bn_wt[i]) + '\t' 
                     + 'bias: ' + str(bn_bias[i]) + '\t' + 'input: ' + str(bn_input[i]) + '\t'
                     + 'output: ' + str(bn_output[i]) + '\n')
    
    for i in range(len(elt_act)):
        with open(txt_path, 'a+') as fp:
            fp.write('elt_' + str(i) + ':' + '\t' + 'act: ' + str(elt_act[i]) + '\t' 
                     + 'input: ' + str(elt_input[2 * i]) + '\t' + str(elt_input[2 * i + 1]) + '\t'
                     + 'output: ' + str(elt_output[i]) + '\n')
    
    for i in range(len(convtrans_act)):
        with open(txt_path, 'a+') as fp:
            fp.write('convtranspose_' + str(i) + ':' + '\t' + 'act: ' + str(convtrans_act[i]) + '\t' 
                     + 'weight: ' + str(convtrans_wt[i]) + '\t' + 'input: ' + str(convtrans_input[i])
                     + '\t' + 'output: ' + str(convtrans_output[i]) + '\n')