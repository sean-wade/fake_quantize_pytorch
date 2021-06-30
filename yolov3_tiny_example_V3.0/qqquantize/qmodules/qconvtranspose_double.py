import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.utils import _pair
from typing import Optional, List

__all__ = ['QConvTranspose2d_d']


class QConvTranspose2d_d(nn.modules.conv._ConvTransposeNd):
    _FLOAT_MODULE = nn.ConvTranspose2d

    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=0,
                 output_padding=0, groups: int = 1, bias: bool = True, dilation: int = 1,
                 padding_mode: str = 'zeros', qconfig=None):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        output_padding = _pair(output_padding)
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                         True, output_padding, groups, bias, padding_mode)
        assert qconfig, 'qconfig must be provided for QAT module'
        self.qconfig = qconfig
        self.act_quant = qconfig.activation()
        self.weight_quant = qconfig.weight()
        if bias:
            self.bias_quant = qconfig.bias()

    def forward(self, input: Tensor, output_size: Optional[List[int]] = None) -> Tensor:
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose2d')

        output_padding = self._output_padding(
            input, output_size, self.stride, self.padding, self.kernel_size, self.dilation)
        
        weight = self.weight_quant(self.weight.float()).double()
        
        if isinstance(self.bias, torch.Tensor):
            bias = self.bias_quant(self.bias.float()).double()
        else:
            bias = None
        
        return self.act_quant(
            F.conv_transpose2d(input, weight, bias, self.stride, self.padding,
                               output_padding, self.groups, self.dilation)
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
        qat_convtranspose = cls(mod.in_channels, mod.out_channels, mod.kernel_size,
                                stride=mod.stride, padding=mod.padding, dilation=mod.dilation,
                                output_padding=mod.output_padding, groups=mod.groups,
                                bias=mod.bias is not None, padding_mode=mod.padding_mode, 
                                qconfig=qconfig)
        qat_convtranspose.weight = torch.nn.Parameter(mod.weight)
        if mod.bias is not None:
            qat_convtranspose.bias = torch.nn.Parameter(mod.bias)
        return qat_convtranspose
