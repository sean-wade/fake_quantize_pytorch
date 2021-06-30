'''
这是 风兴 的量化模型导出 onnx/txt/npy 代码，需要注意的事项有：
    
    1、这个代码与 quantize_export_onnx_txt_compare_v3tiny_V2.py 的区别是：
        这个代码多了一步从训练好的checkpoint(ckpt)中加载的过程，仅此而已！
        因此这个代码可以导出训好的模型！！！
    2、注意：
        qconfig 的设置，应该与训练的 qconfig 一致才行！！！
    3、其他：
        这个代码还没进行充分测试，准确性存疑
'''

import os
import onnx
import torch
import argparse
import torch.onnx
import numpy as np
from torchvision import transforms
from torch.autograd import Variable
from easydict import EasyDict as edict

from models.yolo import Model

from quantize_export_onnx import recreate_model
from qqquantize.savehook import register_intermediate_hooks
from quantize_utils import torch2quant, load_yolo_from_q_ckpt, fix_state_dict, input_quant
from quantize_utils import (
    FakeQuantize,
    Fake_quantize_per_channel,
    Fake_quantize_per_tensor,
    MinMaxObserver, 
    MovingAverageMinMaxObserver,
    MinMaxChannelsObserver,
    FixObserver,
    FixActObserver, 
    FixWtObserver,
    COMPARE_QAT_MODULE_MAPPING,
    enable_fake_quant,
    enable_observer,
    disable_observer,
    disable_calc_qparams,
    calc_qparams,
    qm,
)


BITS = 8
SAVE_PATH = "./compare/"
os.system("rm %s/* -rf" % SAVE_PATH)

NPY_PATH  = SAVE_PATH + "compare_data/"
os.makedirs(NPY_PATH, exist_ok=True)

TXT_PATH  = SAVE_PATH + "yolotiny_fix_%sbit.txt" % BITS
CKPT_PATH = SAVE_PATH + "yolotiny_fix_%sbit.pt" % BITS
ONNX_PATH = SAVE_PATH + "yolotiny_fix_%sbit.onnx" % BITS

input_npy_path   = NPY_PATH + 'yolov3tiny_input_%sbit.npy' % BITS
output1_npy_path = NPY_PATH + 'yolov3tiny_output1_%sbit.npy' % BITS
output2_npy_path = NPY_PATH + 'yolov3tiny_output2_%sbit.npy' % BITS


def export_v3tiny_txt(onnx_ckpt, hook_data, txt_path):
    onnx_model = onnx.load(onnx_ckpt)
    graph = onnx_model.graph
    node = graph.node

    conv_input, conv_output, bn_input, bn_output, concat_input, concat_output = [], [], [], [], [], []

    for i in range(len(node)):
        if 'Conv' in node[i].name:
            conv_input.append(node[i].input[0])
            conv_output.append(node[i].output[0])
        if 'BatchNormalization' in node[i].name:
            bn_input.append(node[i].input[0])
            bn_output.append(node[i].output[0])

    conv_act, conv_wt, conv_bias, bn_act, bn_wt, bn_bias, elwise_act, input_act  = [], [], [], [], [], [], [], []
    conv_act_type, conv_wt_type = [], []
    
    for k in hook_data.keys():
        scale = hook_data[k]['scale']
#         print(k, ":", hook_data[k]['quantize_type'])
        # assert scale.size == 1
        float_bit = []
        for i in range(len(scale)):
            float_bit.append(- np.log2(scale[i]))

        if 'conv' in k:
            q_type = "_pt" if int(hook_data[k]['quantize_type']) == 0 else "_pc"
            if 'act' in k:
                conv_act.append(float_bit)
                conv_act_type.append(q_type)
            if 'weight' in k:
                conv_wt.append(float_bit)
                conv_wt_type.append(q_type)
            # conv_bias.append(None)
        
        if '20' in k:
            q_type = "_pt" if int(hook_data[k]['quantize_type']) == 0 else "_pc"
            if 'act' in k:
                conv_act.append(float_bit)
                conv_act_type.append(q_type)
            if 'weight' in k:
                conv_wt.append(float_bit)
                conv_wt_type.append(q_type)
            if 'bias' in k:
                conv_bias.append(float_bit)
        
        if 'bn' in k:
            if 'act' in k:
                bn_act.append(float_bit)
            if 'weight' in k:
                bn_wt.append(float_bit)
            if 'bias' in k:
                bn_bias.append(float_bit)
        
        # if '18' in k:
        #     if 'act' in k:
        #         elwise_act.append(float_bit)
        
        if 'input' in k:
            if 'act' in k:
                input_act.append(float_bit)


    with open(txt_path, 'a+') as fp:
        fp.write('input_act:' + '\t' + str(input_act[0]) + '\n')

    # for i in range(len(elwise_act)):
    #     with open(txt_path, 'a+') as fp:
    #         fp.write('elwise_act' + ':' + '\t' + str(elwise_act[0]) + '\n')
    
    for i in range(len(bn_bias)):
        with open(txt_path, 'a+') as fp:
            fp.write('conv_' + str(i) + ':' + '\t' + 
                     'act%s: '%(conv_act_type[i]) + str(conv_act[i]) + '\t' + 
                     'weight%s: '%(conv_wt_type[i]) + str(conv_wt[i]) + '\t' + 
                     'input: ' + str(conv_input[i]) + '\t' + 
                     'output: ' + str(conv_output[i]) + '\n')
            
        with open(txt_path, 'a+') as fp:
            fp.write('bn_' + str(i) + ':' + '\t' 
                     + 'act: ' + str(bn_act[i]) + '\t' 
                     + 'weight: ' + str(bn_wt[i]) + '\t' 
                     + 'bias: ' + str(bn_bias[i]) + '\t' 
                     + 'input: ' + str(bn_input[i]) + '\t'
                     + 'output: ' + str(bn_output[i]) + '\n')
    
    for i in range(len(conv_bias)):
        with open(txt_path, 'a+') as fp:
            fp.write('conv_' + str(i + 11) + ':' + '\t'
                     + 'act%s: '%(conv_act_type[i+11]) + str(conv_act[i + 11]) + '\t' 
                     + 'weight%s: '%(conv_wt_type[i+11]) + str(conv_wt[i + 11]) + '\t'  
                     + 'bias: ' + str(conv_bias[i]) + '\t'
                     + 'input: ' + str(conv_input[i + 11]) + '\t' 
                     + 'output: ' + str(conv_output[i + 11]) + '\n')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='models/yolov3-tiny_half.yaml', help='model.yaml path')
    parser.add_argument('--ckpt', type=str, default='ir_quant_best_zh.pt', help='checkpoint path')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument("--img_size", type=int, default=[416, 416], help="size of each image dimension")
    opt = parser.parse_args()

#     model = Model(opt.cfg, ch=3, nc=20).to(opt.device)
#     model.model[-1].export = True

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

#     model = torch2quant(model, qconfig, COMPARE_QAT_MODULE_MAPPING)
    model = load_yolo_from_q_ckpt(opt.ckpt, 
                                  qconfig, 
                                  COMPARE_QAT_MODULE_MAPPING, 
                                  opt.device, 
                                  opt.cfg, 
                                  ch=3, 
                                  nc=2)
    
    model.double()
    enable_observer(model)

    test_img = torch.zeros(1, 3, *opt.img_size).to(opt.device)
    test_img = test_img.type(torch.DoubleTensor)
    _ = model(test_img)

    calc_qparams(model)
    enable_fake_quant(model)
    disable_calc_qparams(model)
    disable_observer(model)
    hook = register_intermediate_hooks(model)

    _ = model(test_img)
    torch.save(model.state_dict(), CKPT_PATH)

    hook_data = hook.output_data()
    new_hook_data = fix_state_dict(hook_data)

    new_model = recreate_model(new_hook_data, opt.cfg, ch=3, nc=2, device=opt.device)
    new_model.double()

    torch.onnx.export(new_model, 
                      test_img, 
                      ONNX_PATH, 
                      verbose=False, 
                      opset_version=10, 
                      input_names=['image'], 
                      output_names=['layer1', 'layer2'], 
                      training=True)
    
    # m = onnx.load('/home/yxchen/models/Quant_test_project/quantized_yolotiny_ori.onnx')
    # model_simp, check = simplify(m)
    # onnx.save(model_simp, 
    #           '/home/yxchen/models/Quant_test_project/quantized_yolotiny.onnx')
    # assert check, "Simplified ONNX model could not be validated"

    x = np.random.rand(*opt.img_size, 3)
    xq = input_quant(CKPT_PATH, x, BITS)
    np.save(input_npy_path, xq.astype(np.double))
    x = np.load(input_npy_path)

    x_tsr = transforms.ToTensor()(x)
    x_batch = x_tsr.unsqueeze(0)
    input_img = Variable(x_batch).to(opt.device)
    input_img = input_img.type(torch.FloatTensor)

    model.double()
    model.eval().to(opt.device)

    hook = register_intermediate_hooks(model)
    _ = model(input_img)
    hook_data2 = hook.output_data()
    export_v3tiny_txt(ONNX_PATH, hook_data2, TXT_PATH)    # zhanghao

    def forward_hooker(name):
        def hook_func(module, input, output):
            if isinstance(module, (qm.QConv2d_d, 
                                   qm.QBatchNorm2d_d, 
                                   qm.QLinear_d, 
                                   qm.QStub, 
                                   qm.QReLU, 
                                   qm.QAvgPooling2d, 
                                   qm.QConvTranspose2d_d)):
                np.save(NPY_PATH + name + '_out.npy', 
                        output.detach().cpu().numpy().astype(np.double))
        return hook_func
    
    for name, module in model.named_modules():
        module.register_forward_hook(forward_hooker(name))

    with torch.no_grad():
        output1, output2 = model(input_img)
    output1 = output1.numpy()
    output2 = output2.numpy()
    np.save(output1_npy_path, output1.astype(np.double))
    np.save(output2_npy_path, output2.astype(np.double))

    # model_0_bn_weight = hook_data['module.model.0.bn.weight_quant']['values'][0]
    # model_0_bn_bias = hook_data['module.model.0.bn.bias_quant']['values'][0]

    # bn1_weight_npy_path = '/home/yxchen/models/Quant_test_project/compare-data-v3tiny/yolov3tiny_bn1wt_8bit.npy'
    # bn1_bias_npy_path = '/home/yxchen/models/Quant_test_project/compare-data-v3tiny/yolov3tiny_bn1bias_8bit.npy'

    # np.save(bn1_weight_npy_path, model_0_bn_weight.astype(np.double))
    # np.save(bn1_bias_npy_path, model_0_bn_bias.astype(np.double))
