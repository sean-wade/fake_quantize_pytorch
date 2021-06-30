'''
这是 风兴 的量化的一个常用的函数和工具代码，需要注意的事项有：

    1、45行左右的 qconfig，作用是描述将普通pytorch模型转换为量化模型时的 weight/bias/activation 各采用什么方法/位宽等
    2、torch2quant函数：根据 qconfig 和 Qmapping 将pytorch模型转换为量化模型
    3、input_quant函数：将输入的 numpy 矩阵，根据模型存储的input量化位宽进行量化激活操作
    4、fix_state_dict函数：修改模型字典的一些键值，返回新字典
    5、load_yolo_from_q_ckpt：加载训练好的yolo量化模型，（不是很完善，不同的yolo可能要对函数内部某些参数修改）
'''
import torch
import copy
import numpy as np
from collections import OrderedDict

from easydict import EasyDict as edict
import pickle
import qqquantize
from qqquantize.qconfig import DEFAULT_QAT_MODULE_MAPPING, COMPARE_QAT_MODULE_MAPPING
from qqquantize.quantize import ModelConverter
from qqquantize.observers.histogramobserver import HistObserver, swap_minmax_to_hist
from qqquantize.observers.minmaxchannelsobserver import MinMaxChannelsObserver, MovingAverageMinMaxChannelsObserver, MovingAverageMinMaxActChannelsObserver
from qqquantize.observers.minmaxobserver import MinMaxObserver, MovingAverageMinMaxObserver, FixObserver
from qqquantize.observers.fixobserver import FixActObserver, FixWtObserver
from qqquantize.observers.fake_quantize import (
    FakeQuantize,
    Fake_quantize_per_channel,
    Fake_quantize_per_tensor,
    enable_fake_quant,
    disable_fake_quant,
    enable_observer,
    disable_observer,
    enable_calc_qparams,
    enable_calc_qparams_IP,
    disable_calc_qparams,
    calc_qparams,
)
from qqquantize.savehook import register_intermediate_hooks
from qqquantize.toolbox import fxquantize
import qqquantize.qmodules as qm
from qqquantize.toolbox.fxquantize import fx_adjust_bits_IP1, fx_adjust_bits_IP2, fx_adjust_bits_Bias


BITS = 16
###quantize, MovingAverageMinMaxChannelsObserver, MinMaxChannelsObserver, Fake_quantize_per_channel
# MinMaxObserver, MovingAverageMinMaxObserver, Fake_quantize_per_tensor
qconfig = edict({
    'activation': FakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
#         observer=FixObserver,
        quantize_func=Fake_quantize_per_tensor,
        bits=8
    ),
    'weight': FakeQuantize.with_args(
#         observer=MinMaxChannelsObserver,
#         quantize_func=Fake_quantize_per_channel,
        observer=MinMaxObserver,
        quantize_func=Fake_quantize_per_tensor,
        bits=8
    ),
    'bias': FakeQuantize.with_args(bits=16),
})


def torch2quant(torch_model, Qconfig=qconfig, Qmapping=DEFAULT_QAT_MODULE_MAPPING):
    """
    Qmapping: 
        if compare: 
            COMPARE_QAT_MODULE_MAPPING
        else:       
            DEFAULT_QAT_MODULE_MAPPING 
            if yolov3:  
                from models.common import Concat
                mapping[Concat] = qm.QConcat
    """
    print("Start convert pytorch to quantize, QCONFIG is:\n[%s]\nQMAP is:\n[%s]\n"%(Qconfig, Qmapping))
    mapping = copy.deepcopy(Qmapping)
    qconverter = ModelConverter(Qconfig, mapping, '', ['f', 'i'])
    Qmodel = qconverter(torch_model)
    return Qmodel


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


def load_yolo_from_q_ckpt(ckpt_path, 
                          Qconfig=qconfig, 
                          Qmapping=DEFAULT_QAT_MODULE_MAPPING, 
                          device="cuda", 
                          cfg="models/yolov3-tiny.yaml", 
                          ch=3, 
                          nc=20):
    ckpt = torch.load(ckpt_path, map_location=device)  # load checkpoint
    
    from models.yolo import Model, Concat
    model = Model(cfg, ch, nc).to(device)
    model.model[-1].export = True
    mapping = copy.deepcopy(Qmapping)
    mapping[Concat] = qm.QConcat

    qmodel = torch2quant(model, Qconfig, mapping)
    enable_observer(qmodel)

    is_compare = (Qmapping == COMPARE_QAT_MODULE_MAPPING)
    test_img = torch.zeros(1, 3, 416, 416).to(device)
    if is_compare:
        test_img = test_img.type(torch.DoubleTensor)
    test_out = qmodel(test_img)

    calc_qparams(qmodel)
    enable_fake_quant(qmodel)
    qmodel.load_state_dict(ckpt['model'], strict=False)
    disable_calc_qparams(qmodel)
    disable_observer(qmodel)

    print("load_yolo_from_q_ckpt finished, getting quant model...")
    return qmodel


if __name__ == "__main__":
    from torch import nn
    
    class ModelTest(nn.Module):
        def __init__(self):
            super(ModelTest,self).__init__()
            self.conv1 = nn.Conv2d(3, 64, 3)
            self.conv2 = nn.ConvTranspose2d(3, 12, 3)

        def forward(self, x):
            y1 = self.conv1(x)
            y2 = self.conv2(x)
            return y1, y2
        
    net = ModelTest()
    print(net)
    print("-*"*50)
    qmodel = torch2quant(net)
    print(qmodel)
      
    test_img = torch.zeros(1, 3, 32, 32)
    torch.onnx.export(qmodel, 
                      test_img, 
                      "quant_test.onnx", 
                      verbose=False, 
                      opset_version=10, 
                      input_names=['image'], 
                      output_names=['conv_out', "convtrans_out"], 
                      training=True)


    model = load_yolo_from_q_ckpt("weights/voc_tiny_quant_fix2_perchannel_0.416.pt", cfg="models/yolov3-tiny.yaml", ch=3, nc=20)
    test_img = torch.zeros(1, 3, 416, 416).to("cuda")
    test_out = model(test_img)
    print(test_out)

