import torch
import math

from .observerbase import ObserverBase

class FixActObserver(ObserverBase):
    def __init__(self, bits=16, max_factor=1.0):
        super().__init__()
        self.register_buffer('bits', torch.tensor([bits], dtype=torch.int))
        self.register_buffer('max_factor', torch.tensor([max_factor]))
        self.register_buffer('min_val', torch.tensor([]))
        self.register_buffer('max_val', torch.tensor([]))
        self.qmin = - 2 ** (bits-1)
        self.qmax = 2 ** (bits-1) - 1

    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``."""
        x = x_orig.detach()  # avoid keeping autograd tape
        x = x.to(self.min_val.dtype)
        min_val = self.min_val
        max_val = self.max_val
        if min_val.numel() == 0 or max_val.numel() == 0:
            min_val = torch.min(x)
            max_val = torch.max(x)
        else:
            min_val = torch.min(torch.min(x), min_val)
            max_val = torch.max(torch.max(x), max_val)
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
        scale = 0.5 ** 8
        zero_point = 0
        if scale == 0:
            raise ValueError('scale is 0')
        return torch.tensor([scale]), torch.tensor([zero_point])
        
    @torch.jit.export
    def extra_repr(self):
        return "min_val={}, max_val={}".format(self.min_val, self.max_val)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super(FixActObserver, self)._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'min_val'] = self.min_val
        destination[prefix + 'max_val'] = self.max_val

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

        local_state = ['min_val', 'max_val']
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                setattr(self, name, val)
            elif strict:
                missing_keys.append(key)
        super(FixActObserver, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                                          missing_keys, unexpected_keys, error_msgs)

class FixWtObserver(ObserverBase):
    def __init__(self, bits=16, max_factor=1.0):
        super().__init__()
        self.register_buffer('bits', torch.tensor([bits], dtype=torch.int))
        self.register_buffer('max_factor', torch.tensor([max_factor]))
        self.register_buffer('min_val', torch.tensor([]))
        self.register_buffer('max_val', torch.tensor([]))
        self.qmin = - 2 ** (bits-1)
        self.qmax = 2 ** (bits-1) - 1

    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``."""
        x = x_orig.detach()  # avoid keeping autograd tape
        x = x.to(self.min_val.dtype)
        min_val = self.min_val
        max_val = self.max_val
        if min_val.numel() == 0 or max_val.numel() == 0:
            min_val = torch.min(x)
            max_val = torch.max(x)
        else:
            min_val = torch.min(torch.min(x), min_val)
            max_val = torch.max(torch.max(x), max_val)
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
        scale = 0.5 ** 11
        zero_point = 0
        if scale == 0:
            raise ValueError('scale is 0')
        return torch.tensor([scale]), torch.tensor([zero_point])
        
    @torch.jit.export
    def extra_repr(self):
        return "min_val={}, max_val={}".format(self.min_val, self.max_val)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super(FixWtObserver, self)._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'min_val'] = self.min_val
        destination[prefix + 'max_val'] = self.max_val

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

        local_state = ['min_val', 'max_val']
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                setattr(self, name, val)
            elif strict:
                missing_keys.append(key)
        super(FixWtObserver, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                                          missing_keys, unexpected_keys, error_msgs)
