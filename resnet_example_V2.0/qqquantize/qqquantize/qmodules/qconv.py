import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['QConv2d']

class QConv2d(nn.Conv2d):
    _FLOAT_MODULE = nn.Conv2d

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', qconfig=None):
        super().__init__(in_channels, out_channels, kernel_size,
                                     stride=stride, padding=padding, dilation=dilation,
                                     groups=groups, bias=bias, padding_mode=padding_mode)
        assert qconfig, 'qconfig must be provided for QAT module'
        self.qconfig = qconfig
        self.act_quant = qconfig.activation()
        self.weight_quant = qconfig.weight()
        if bias:
            self.bias_quant = qconfig.bias()

    def forward(self, inputs):
        weight = self.weight_quant(self.weight)
        
        if isinstance(self.bias, torch.Tensor):
            bias = self.bias_quant(self.bias)
        else:
            bias = None
        
        if self.padding_mode != 'zeros':
            paddings = (0, 0)
        else:
            paddings = self.padding
        
        return self.act_quant(
            F.conv2d(inputs, weight, bias, self.stride,
                        paddings, self.dilation, self.groups)
            )

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict

            Args: `mod` a float module, either produced by torch.quantization utilities
            or directly from user
        """
        assert type(mod) == cls._FLOAT_MODULE, 'qat.' + cls.__name__ + '.from_float only works for ' + \
            cls._FLOAT_MODULE.__name__
        qconfig = mod.qconfig
        qat_conv = cls(mod.in_channels, mod.out_channels, mod.kernel_size,
                       stride=mod.stride, padding=mod.padding, dilation=mod.dilation,
                       groups=mod.groups, bias=mod.bias is not None,
                       padding_mode=mod.padding_mode, qconfig=qconfig)
        qat_conv.weight = torch.nn.Parameter(mod.weight)
        if mod.bias is not None:
            qat_conv.bias =torch.nn.Parameter(mod.bias)
        return qat_conv
