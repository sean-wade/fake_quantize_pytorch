import torch.nn as nn
import qqquantize
import qqquantize.qqquantize.qmodules as qm

DEFAULT_QAT_MODULE_MAPPING = {
    nn.Linear: qm.QLinear,
    nn.Conv2d: qm.QConv2d,
    nn.BatchNorm2d: qm.QBatchNorm2d,
    nn.ReLU: qm.QReLU,
    qm.InputStub: qm.QStub,
}

COMPARE_QAT_MODULE_MAPPING = {
    nn.Linear: qm.QLinear_d,
    nn.Conv2d: qm.QConv2d_d,
    nn.BatchNorm2d: qm.QBatchNorm2d_d,
    nn.ReLU: qm.QReLU,
    qm.InputStub: qm.QStub,
    nn.AdaptiveAvgPool2d: qm.QAvgPooling2d,
}
