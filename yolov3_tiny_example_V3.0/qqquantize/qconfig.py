'''
这是 风兴 的量化代码 里的两个mapping，需要注意的事项有：
    
    1、DEFAULT_QAT_MODULE_MAPPING 是正常训练等使用的mapping
    2、COMPARE_QAT_MODULE_MAPPING 是导出txt/onnx/npy和软件部比对时，使用的double精度的mapping
        
'''
import torch.nn as nn
import qqquantize
import qqquantize.qmodules as qm

DEFAULT_QAT_MODULE_MAPPING = {
    nn.Linear: qm.QLinear,
    nn.Conv2d: qm.QConv2d,
    nn.ConvTranspose2d: qm.QConvTranspose2d,
    nn.BatchNorm2d: qm.QBatchNorm2d,
    nn.ReLU: qm.QReLU,
    qm.InputStub: qm.QStub,
}

COMPARE_QAT_MODULE_MAPPING = {
    nn.Linear: qm.QLinear_d,
    nn.Conv2d: qm.QConv2d_d,
    nn.ConvTranspose2d: qm.QConvTranspose2d_d,
    nn.BatchNorm2d: qm.QBatchNorm2d_d,
    nn.ReLU: qm.QReLU,
    qm.InputStub: qm.QStub,
    nn.AdaptiveAvgPool2d: qm.QAvgPooling2d,
}
