import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['QLinear']

class QLinear(nn.Linear):
    _FLOAT_MODULE = nn.Linear

    def __init__(self, in_features, out_features, bias=True,
                 qconfig=None):
        super().__init__(in_features, out_features, bias)
        assert qconfig, 'qconfig must be provided for QAT module'
        self.qconfig = qconfig
        self.act_quant = qconfig.activation()
        self.weight_quant = qconfig.weight()
        if bias:
            self.bias_quant = qconfig.bias()

    def forward(self, input):
        weight = self.weight_quant(self.weight)
        if isinstance(self.bias, torch.Tensor):
            bias = self.bias_quant(self.bias)
        else:
            bias = None
        return self.act_quant(
            F.linear(input, weight, bias)
        )

    @classmethod
    def from_float(cls, mod):
        assert type(mod) == cls._FLOAT_MODULE, ' qat.' + cls.__name__ + '.from_float only works for ' + \
            cls._FLOAT_MODULE.__name__
        qconfig = mod.qconfig
        qat_linear = cls(mod.in_features, mod.out_features, bias=mod.bias is not None, qconfig=qconfig)
        qat_linear.weight = mod.weight
        if mod.bias is not None:
            qat_linear.bias = mod.bias
        return qat_linear
