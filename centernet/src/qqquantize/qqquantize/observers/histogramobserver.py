import torch
import math
import numpy as np
import re

from .observerbase import ObserverBase, calculate_qparams_symmetric
from .minmaxobserver import MinMaxObserver, MovingAverageMinMaxObserver

"""
Note: this observer cannot directly in qconfig
must use swap_minmax_to_hist
because when enable_observer, calculate_qparams is execute in every inference
but you have to have min_val and max_val before call calculate_qparams.
"""
class HistObserver(ObserverBase):
    _default_ratio = 0.999
    def __init__(self, bin_num=2048, bits=8, min_val=0, max_val=0, ratio=_default_ratio):
        super().__init__()
        self.register_buffer('bits', torch.tensor([bits], dtype=torch.int))
        self.register_buffer('bin_num', torch.tensor([bin_num]))
        self.register_buffer('min_val', torch.tensor([min_val]))
        self.register_buffer('max_val', torch.tensor([max_val]))
        self.register_buffer('bins', torch.zeros([bin_num], dtype=torch.int64))
        self.register_buffer('min_val_calib', torch.tensor([0.]))
        self.register_buffer('max_val_calib', torch.tensor([0.]))
        self.qmin = - 2 ** (bits-1)
        self.qmax = 2 ** (bits-1) - 1
        self.ratio = ratio
    
    def forward(self, x):
        x_hist = torch.histc(x.float(), bins=self.bin_num.item(), min=self.min_val.item(), max=self.max_val.item())
        self.bins += x_hist.int()
        if torch.max(self.bins) > 2**60:
            raise Exception('bins is going overflow')
        return x
    
    def calculate_qparams(self, **kwargs):
        # find proper val_calib
        min_val, max_val = self.min_val.item(), self.max_val.item()
        ticks = np.arange(min_val, max_val, (max_val-min_val)/self.bin_num.item())
        ticks_abs = np.abs(ticks)
        sort_idxs = np.argsort(ticks_abs)
        amount = torch.sum(self.bins)
        rate = 0
        for i in sort_idxs:
            rate += self.bins[i]
            if rate / amount >= self.ratio:
                break
        q = ticks_abs[i]
        self.min_val_calib[0] = -q
        self.max_val_calib[0] = q

        # normal calucate scale and zero_point
        scale, zero_point = calculate_qparams_symmetric(
            min_val = -q,
            max_val = q,
            qmin = self.qmin,
            qmax = self.qmax
        )
        return (
            torch.tensor(scale, device=self.bits.device),
            torch.tensor(zero_point, device=self.bits.device)
        )
        
    @torch.jit.export
    def extra_repr(self):
        return "min_val={}, max_val={}, min_calib={}, max_calib={}".format(
            self.min_val, self.max_val,
            self.min_val_calib, self.max_val_calib
        )
    
    @classmethod
    def from_minmaxobserver(cls, ob, bin_num=1024, ratio=_default_ratio):
        device = list(ob.buffers())[0].device
        if ob.min_val == ob.max_val:
            raise Warning('minmaxobesrver min==max')
        return cls(bin_num, ob.bits.item(), ob.min_val, ob.max_val).to(device)


# model global function
def swap_minmax_to_hist(model, name_pattern=''):
    for name, mod in list(model._modules.items()):
        if isinstance(mod, MinMaxObserver) and re.search(name_pattern, name):
            model._modules[name] = HistObserver.from_minmaxobserver(mod)
        else:
            swap_minmax_to_hist(mod)

def collect_hist(model):
    hist = {}
    hist_range = {}
    for name, mod in model.named_modules():
        if isinstance(mod, HistObserver):
            hist[name] = mod.bins
            hist_range[name] = (mod.min_val.item(), mod.max_val.item())

    return {
        'hist': hist, 
        'hist_range': hist_range
    }
