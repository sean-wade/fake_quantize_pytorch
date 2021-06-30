'''
这是 风兴 的量化代码 里至关重要的 ModelConverter 类的实现，需要注意的事项有：
    1、功能及使用：
        根据 qconfig Qmapping 实例化 ModelConverter 对象
        通过__call__方法将 pytorch 的模型进行转换
    
    2、_propagate_qconfig 作用：
        该方法递归地将 qconfig 赋予 pytorch 模型的每个子module
        因此，如果需要对 pytorch 模型的某个子模块指定单独的 qconfig，
        可以参考60行左右的写法，即：
            根据 module 的 唯一 name 进行判断，匹配的 name，嵌入指定的 qconfig
        如：本代码中示例是 yolov3-tiny 的一些修改方法（详见代码的注释内容）
'''
import torch.quantization
import copy, re
from easydict import EasyDict as edict
from qqquantize.utils import get_unique_devices_
import torch.nn as nn
from qqquantize.qmodules import QStubWrapper, InputStub
from qqquantize.qconfig import DEFAULT_QAT_MODULE_MAPPING
from qqquantize.observers.minmaxchannelsobserver import MovingAverageMinMaxActChannelsObserver, MinMaxChannelsObserver, MovingAverageMinMaxChannelsObserver, MinMaxActChannelsObserver
from qqquantize.observers.minmaxobserver import MovingAverageMinMaxObserver, FixObserver
from qqquantize.observers.fixobserver import FixMaxBiasObserver
from qqquantize.observers.fake_quantize import (
    FakeQuantize,
    Fake_quantize_Act_per_channel,
    Fake_quantize_per_tensor,
    Fake_quantize_per_channel
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
            # 以下都是额外的处理，如果模型没有额外要求，全局 qconfig 可以满足要求的话，下面58-94行都不需要了
            if ".bn" in name:
                # 硬件要求，bn层的 act/weight/bias 分别为 8/16/16 bit，和全局配置不同，因此在这里单独修改
                mod.qconfig.activation = FakeQuantize.with_args(
                                         observer=MovingAverageMinMaxObserver,
                                         quantize_func=Fake_quantize_per_tensor,
                                         bits=8
                                         )
                mod.qconfig.weight = FakeQuantize.with_args(
                                         observer=MinMaxChannelsObserver,
                                         quantize_func=Fake_quantize_per_channel,
                                         bits=16  #BITS
                                         )
                mod.qconfig.bias = FakeQuantize.with_args(
                                         bits=16
                                         )
            if 'm.' in name:
                # yolo层需要做 per-channel，收敛性更好（陈昱贤的经验，未验证）
                mod.qconfig.activation = FakeQuantize.with_args(
                                        observer=MinMaxActChannelsObserver, #MovingAverageMinMaxActChannelsObserver,
                                        quantize_func=Fake_quantize_Act_per_channel,
                                        bits=8
                                    ) 
                # add by zhanghao
                # 为包含 bias 的 conv层，专门写的 FixMaxBiasObserver，作用是在后期训练时统计 weight_bit 和 input_bit,
                # 保证 weight_bit + input_bit 和 bias_bit 满足某些关系，方法是强行修改 bias 的 observer 的 scale 值
                mod.qconfig.bias = FakeQuantize.with_args(
                                        observer=FixMaxBiasObserver,
                                        bits=16
                                    ) 
            if ('8.bn' in name) or ('16.bn' in name):
                print("Modify Concat Layer to FixObserver :", name)
                # 针对 yolov3-tiny 需要做 concat 的两个层, 8.bn & 16.bn 设置为相同的位宽（FixObserver）
                mod.qconfig.activation = FakeQuantize.with_args(
                                        observer=FixObserver,
                                        quantize_func=Fake_quantize_per_channel,
                                        bits=8    #16
                                    ) 
    
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
