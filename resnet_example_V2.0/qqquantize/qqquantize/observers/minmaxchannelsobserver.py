import torch
import math

from .observerbase import ObserverBase

class MinMaxChannelsObserver(ObserverBase):
    def __init__(self, bits=8, c_dim=0):
        super().__init__()
        self.register_buffer('bits', torch.tensor([bits], dtype=torch.int))
        self.register_buffer('min_val', torch.tensor([]))
        self.register_buffer('max_val', torch.tensor([]))
        self.qmin = - 2 ** (bits-1)
        self.qmax = 2 ** (bits-1) - 1
        self.c_dim = c_dim

    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``."""
        x = x_orig.detach()  # avoid keeping autograd tape
        x = x.to(self.min_val.dtype)
        permute_idx = list(range(len(x.shape)))
        permute_idx[0] = self.c_dim
        permute_idx[self.c_dim] = 0
        c_num = x.shape[self.c_dim]
        x = x.permute(permute_idx).reshape([c_num, -1])

        min_val = self.min_val
        max_val = self.max_val
        if min_val.numel() == 0 or max_val.numel() == 0:
            min_val = torch.min(x, dim=1).values
            max_val = torch.max(x, dim=1).values
        else:
            min_val = torch.min(torch.min(x, dim=1).values, min_val)
            max_val = torch.max(torch.max(x, dim=1).values, max_val)
        self.min_val.resize_(min_val.shape)
        self.max_val.resize_(max_val.shape)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig

    @torch.jit.export
    def calculate_qparams(self):
        """quant_x = int(float_x / scale)
        scale is the unit size
        zero_point is always 0"""
        min_val, max_val = self.min_val, self.max_val
        if min_val.numel() == 0 or max_val.numel() == 0:
            raise ValueError(
                "must run observer before calling calculate_qparams.\
                                    Returning default scale and zero point "
            )
        max_val = torch.max(min_val.abs(), max_val.abs())
        scale = max_val / max(abs(self.qmin), abs(self.qmax))
        scale = torch.max(scale, torch.zeros_like(scale).fill_(1e-8))
        scale = 2 ** torch.ceil(torch.log2(scale))
        zero_point = torch.zeros_like(scale)
        if torch.sum(scale.abs()) == 0:
            raise ValueError('scale is 0')
        return scale, zero_point
        
    @torch.jit.export
    def extra_repr(self):
        return "min_val={}, max_val={}".format(self.min_val, self.max_val)

class MovingAverageMinMaxChannelsObserver(MinMaxChannelsObserver):
    def __init__(self, averaging_constant=0.01, bits=8):
        super().__init__(bits)
        self.averaging_constant = averaging_constant

    def forward(self, x_orig):
        x = x_orig.detach()  # avoid keeping autograd tape
        x = x.to(self.min_val.dtype)
        permute_idx = list(range(len(x.shape)))
        permute_idx[0] = self.c_dim
        permute_idx[self.c_dim] = 0
        c_num = x.shape[self.c_dim]
        x = x.permute(permute_idx).reshape([c_num, -1])

        min_val = self.min_val
        max_val = self.max_val
        if min_val.numel() == 0 or max_val.numel() == 0:
            min_val = torch.min(x, dim=1).values
            max_val = torch.max(x, dim=1).values
        else:
            min_val = min_val + self.averaging_constant * (torch.min(x, dim=1).values - min_val)
            max_val = max_val + self.averaging_constant * (torch.max(x, dim=1).values - max_val)
        self.min_val.resize_(min_val.shape)
        self.max_val.resize_(max_val.shape)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig

class MinMaxActChannelsObserver(ObserverBase):
    def __init__(self, bits=8, c_dim=1):
        super().__init__()
        self.register_buffer('bits', torch.tensor([bits], dtype=torch.int))
        self.register_buffer('min_val', torch.tensor([]))
        self.register_buffer('max_val', torch.tensor([]))
        self.qmin = - 2 ** (bits-1)
        self.qmax = 2 ** (bits-1) - 1
        self.c_dim = c_dim

    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``."""
        x = x_orig.detach()  # avoid keeping autograd tape
        x = x.to(self.min_val.dtype)
        permute_idx = list(range(len(x.shape)))
        permute_idx[0] = self.c_dim
        permute_idx[self.c_dim] = 0
        c_num = x.shape[self.c_dim]
        x = x.permute(permute_idx).reshape([c_num, -1])

        min_val = self.min_val
        max_val = self.max_val
        if min_val.numel() == 0 or max_val.numel() == 0:
            min_val = torch.min(x, dim=1).values
            max_val = torch.max(x, dim=1).values
        else:
            min_val = torch.min(torch.min(x, dim=1).values, min_val)
            max_val = torch.max(torch.max(x, dim=1).values, max_val)
        self.min_val.resize_(min_val.shape)
        self.max_val.resize_(max_val.shape)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig

    @torch.jit.export
    def calculate_qparams(self):
        """quant_x = int(float_x / scale)
        scale is the unit size
        zero_point is always 0"""
        min_val, max_val = self.min_val, self.max_val
        if min_val.numel() == 0 or max_val.numel() == 0:
            raise ValueError(
                "must run observer before calling calculate_qparams.\
                                    Returning default scale and zero point "
            )
        max_val = torch.max(min_val.abs(), max_val.abs())
        scale = max_val / max(abs(self.qmin), abs(self.qmax))
        scale = torch.max(scale, torch.zeros_like(scale).fill_(1e-8))
        scale = 2 ** torch.ceil(torch.log2(scale))
        zero_point = torch.zeros_like(scale)
        if torch.sum(scale.abs()) == 0:
            raise ValueError('scale is 0')
        return scale, zero_point
        
    @torch.jit.export
    def extra_repr(self):
        return "min_val={}, max_val={}".format(self.min_val, self.max_val)

class MovingAverageMinMaxActChannelsObserver(MinMaxActChannelsObserver):
    def __init__(self, averaging_constant=0.01, bits=8):
        super().__init__(bits)
        self.averaging_constant = averaging_constant

    def forward(self, x_orig):
        x = x_orig.detach()  # avoid keeping autograd tape
        x = x.to(self.min_val.dtype)
        permute_idx = list(range(len(x.shape)))
        permute_idx[0] = self.c_dim
        permute_idx[self.c_dim] = 0
        c_num = x.shape[self.c_dim]
        x = x.permute(permute_idx).reshape([c_num, -1])

        min_val = self.min_val
        max_val = self.max_val
        if min_val.numel() == 0 or max_val.numel() == 0:
            min_val = torch.min(x, dim=1).values
            max_val = torch.max(x, dim=1).values
        else:
            min_val = min_val + self.averaging_constant * (torch.min(x, dim=1).values - min_val)
            max_val = max_val + self.averaging_constant * (torch.max(x, dim=1).values - max_val)
        self.min_val.resize_(min_val.shape)
        self.max_val.resize_(max_val.shape)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig