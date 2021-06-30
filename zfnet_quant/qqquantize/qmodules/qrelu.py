import torch.nn as nn
import torch.nn.functional as F

__all__ = ['QReLU']

class QReLU(nn.ReLU):
    _FLOAT_MODULE = nn.ReLU
    
    def forward(self, x):
        out = F.relu(x, self.inplace)
        # because there is no fake_quant module, mannual add scale
        if hasattr(x, 'scale'):
            out.scale = x.scale
            out.zero_point = x.zero_point
        return out
    
    @classmethod
    def from_float(cls, mod):
        assert type(mod) == cls._FLOAT_MODULE, ' qat.' + cls.__name__ + '.from_float only works for ' + \
            cls._FLOAT_MODULE.__name__
        return cls(mod.inplace)
