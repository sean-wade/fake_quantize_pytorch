import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['QConcat']

class QConcat(nn.Module):
    """There is not _FLOAT_MODULE because it may come from outsides"""
    '''zhanghao add: I think this is only for {U-yolov3/5-common.py: Concat}'''
    def __init__(self, dim, qconfig):
        super().__init__()
        self.d = dim
        self.qconfig = qconfig
        self.act_quant = qconfig.activation()
    
    def forward(self, tensors):
        return self.act_quant(torch.cat(tensors, dim=self.d))

    @classmethod
    def from_float(cls, mod):
        qconfig = mod.qconfig
        qconcat = cls(mod.d, qconfig)
        return qconcat
