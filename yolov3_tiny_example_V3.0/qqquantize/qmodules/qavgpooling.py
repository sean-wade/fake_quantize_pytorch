import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['QAvgPooling2d']

class QAvgPooling2d(nn.AdaptiveAvgPool2d):
    _FLOAT_MODULE = nn.AdaptiveAvgPool2d

    def __init__(self, output_size, qconfig=None):
        super().__init__(output_size)
        assert qconfig, 'qconfig must be provided for QAT module'
        self.qconfig = qconfig
        self.act_quant = qconfig.activation()

    def forward(self, input):
        
        return self.act_quant(
            F.adaptive_avg_pool2d(input, self.output_size)
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
        qat_avgpool = cls(mod.output_size, qconfig=qconfig)
        return qat_avgpool
        