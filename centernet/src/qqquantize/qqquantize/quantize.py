import torch.quantization
import copy, re
from easydict import EasyDict as edict
from qqquantize.qqquantize.utils import get_unique_devices_
import torch.nn as nn
import qqquantize.qqquantize.qmodules as qm
from qqquantize.qqquantize.qmodules import QStubWrapper, InputStub
from qqquantize.qqquantize.qconfig import DEFAULT_QAT_MODULE_MAPPING
from qqquantize.qqquantize.observers.minmaxchannelsobserver import MinMaxConvTChannelsObserver
from qqquantize.qqquantize.observers.minmaxobserver import MovingAverageMinMaxObserver
from qqquantize.qqquantize.observers.fake_quantize import (
    FakeQuantize,
    Fake_quantize_ConvT_per_channel,
    Fake_quantize_per_tensor
)

__all__ = ['ModelConverter']

class ModelConverter:
    def __init__(self, qconfig, mapping=None, pattern='', extra_attr=None):
        self.qconfig = qconfig
        self.mapping = mapping if mapping is not None else DEFAULT_QAT_MODULE_MAPPING
        self.pattern = pattern # used by propaget_qconfig
        # assert isinstance(extra_attr, list)
        self.extra_attr = extra_attr # used by swapmodule
    
    def __call__(self, model):
        model = QStubWrapper(model)
        device = list(get_unique_devices_(model))[0]
        self._propagate_qconfig(model)
        self._convert(model, device)
        return model.to(device)
    
    def _propagate_qconfig(self, model):
        r"""Propagate qconfig through the module hierarchy and assign `qconfig`
        attribute on each leaf module
        """
        add_cfg_lst =  list(self.mapping.keys()) + [InputStub]
        for name, mod in model.named_modules():
            if any([isinstance(mod, valid_type) for valid_type in add_cfg_lst]):
                if re.search(self.pattern, name):
                    mod.qconfig = copy.deepcopy(self.qconfig)
            # if 'deconv_layers.0' in name or 'deconv_layers.3' in name or 'deconv_layers.6' in name:
            #     mod.qconfig.weight = FakeQuantize.with_args(
            #                             observer=MinMaxConvTChannelsObserver,
            #                             quantize_func=Fake_quantize_ConvT_per_channel,
            #                             bits=16
            #                         ) 
    
    def _convert(self, model, device):
        reassign = {}

        swappable_modules = list(self.mapping.keys())

        for name, mod in model.named_children():
            if type(mod) not in swappable_modules or not hasattr(mod, 'qconfig'):
                self._convert(mod, device)
            else:
                reassign[name] = swap_module(mod, self.mapping, self.extra_attr).to(device)

        for key, value in reassign.items():
            model._modules[key] = value

        return model

def swap_module(mod, mapping, extra_attr=None):
    r"""Swaps the module if it has a quantized counterpart and it has an
    `observer` attached.

    Args:
        mod: input module
        mapping: a dictionary that maps from nn module to nnq module

    Return:
        The corresponding quantized module of `mod`
    """
    new_mod = mod
    # Always replace dequantstub with dequantize
    if hasattr(mod, 'qconfig') and mod.qconfig is not None:
        if type(mod) in mapping:
            # respect device affinity when swapping modules
            devices = get_unique_devices_(mod)
            assert len(devices) <= 1, (
                "swap_module only works with cpu or single-device CUDA modules, "
                "but got devices {}".format(devices)
            )
            new_mod = mapping[type(mod)].from_float(mod)
        if extra_attr is not None:
            for attr in extra_attr:
                if hasattr(mod, attr):
                    new_mod.__setattr__(attr, mod.__dict__['f'])
    return new_mod
