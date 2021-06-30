'''
这是 风兴 的量化代码 里至关重要的 FakeQuantize 类的实现，需要注意的事项有：
    1、功能及使用：
        a. pytorch 的模型通过 ModelConverter 转换成量化模型后，会在每个子module中嵌入 FakeQuantize 模块
        b. FakeQuantize 中两个重要的成员
            self.observer : 如何统计数据，获取scale
            self.quantize_func : 如何对 input 进行量化操作(pertensor or perchannel等)
        c. 一些 enable/disable 等开关，控制是否开启统计/量化运算等操作
    
    2、注意：
        Fake_quantize_per_channel 和 Fake_quantize_Act_per_channel 的区别：
            参数中 c_dim 不同，0/1 取决于是权重/激活层

        155行左右，是后期我新增的一个属性，用一个buffer记录本层是 perchannel 还是 pertensor，在后期导出txt时用，
        如果后面不需要或者对其他地方有影响，可以去掉。。。
'''
import torch
import torch.nn as nn
from torch.autograd.function import InplaceFunction
from .minmaxobserver import MovingAverageMinMaxObserver
from .observerbase import _with_args
import re

__all__ = [
    'Fake_quantize_per_tensor',
    'Fake_quantize_per_channel',
    'FakeQuantize',
    'enable_fake_quant',
    'disable_fake_quant',
    'enable_observer',
    'disable_observer',
    'calc_qparams',
]

class Fake_quantize_per_tensor(InplaceFunction):
    """return a quantized and dequantized a float tensor"""
    @staticmethod
    def forward(ctx, X, scale, zero_point, qmin, qmax, inplace=False):
        ctx.inplace = inplace
        if ctx.inplace:
            ctx.mark_dirty(X)
        else:
            X = X.clone()
        ctx.save_for_backward(
            X,
            torch.tensor([qmin], device=X.device),
            torch.tensor([qmax], device=X.device)
        )

        with torch.no_grad():
            Xq = torch.floor(X / scale + zero_point)
            Xq = torch.clip(Xq, qmin, qmax)
            Xqf = (Xq - zero_point) * scale
            Xqf.scale = scale
            Xqf.zero_point = zero_point
            return Xqf

    @staticmethod
    def backward(ctx, grad_output):
        X, qmin, qmax = ctx.saved_tensors
        grad_input = grad_output.detach().clone()
        m0 = torch.logical_and(X<qmin, grad_input>0)
        m1 = torch.logical_and(X>qmax, grad_input<0)
        m = torch.logical_or(m0, m1)
        grad_input[m] = 0
        return grad_input.clone(), None, None, None, None, None

class Fake_quantize_per_channel(InplaceFunction):
    """return a quantized and dequantized a float tensor"""
    @staticmethod
    def forward(ctx, X, scale, zero_point, qmin, qmax, c_dim=0, inplace=False):
        ctx.inplace = inplace
        if ctx.inplace:
            ctx.mark_dirty(X)
        else:
            X = X.clone()
        ctx.save_for_backward(
            X,
            torch.tensor([qmin], device=X.device),
            torch.tensor([qmax], device=X.device)
        )
        
        shape = [1] * len(X.shape)
        shape[c_dim] = -1
        scale = scale.reshape(shape)
        zero_point = zero_point.reshape(shape)

        with torch.no_grad():
            Xq = torch.floor(X / scale + zero_point)
            Xq = torch.clip(Xq, qmin, qmax)
            Xqf = (Xq - zero_point) * scale
            Xqf.scale = scale
            Xqf.zero_point = zero_point
            return Xqf

    @staticmethod
    def backward(ctx, grad_output):
        X, qmin, qmax = ctx.saved_tensors
        grad_input = grad_output.detach().clone()
        m0 = torch.logical_and(X<qmin, grad_input>0)
        m1 = torch.logical_and(X>qmax, grad_input<0)
        m = torch.logical_or(m0, m1)
        grad_input[m] = 0
        return grad_input, None, None, None, None, None

class Fake_quantize_Act_per_channel(InplaceFunction):
    """return a quantized and dequantized a float activation tensor"""
    @staticmethod
    def forward(ctx, X, scale, zero_point, qmin, qmax, c_dim=1, inplace=False):
        ctx.inplace = inplace
        if ctx.inplace:
            ctx.mark_dirty(X)
        else:
            X = X.clone()
        ctx.save_for_backward(
            X,
            torch.tensor([qmin], device=X.device),
            torch.tensor([qmax], device=X.device)
        )
        
        shape = [1] * len(X.shape)
        shape[c_dim] = -1
        scale = scale.reshape(shape)
        zero_point = zero_point.reshape(shape)

        with torch.no_grad():
            Xq = torch.floor(X / scale + zero_point)
            Xq = torch.clip(Xq, qmin, qmax)
            Xqf = (Xq - zero_point) * scale
            Xqf.scale = scale
            Xqf.zero_point = zero_point
            return Xqf

    @staticmethod
    def backward(ctx, grad_output):
        X, qmin, qmax = ctx.saved_tensors
        grad_input = grad_output.detach().clone()
        m0 = torch.logical_and(X<qmin, grad_input>0)
        m1 = torch.logical_and(X>qmax, grad_input<0)
        m = torch.logical_or(m0, m1)
        grad_input[m] = 0
        return grad_input, None, None, None, None, None

class FakeQuantize(nn.Module):
    def __init__(self, observer=MovingAverageMinMaxObserver, quantize_func=Fake_quantize_per_tensor, **observer_kwargs):
        super().__init__()
        self.register_buffer('fake_quant_enabled', torch.tensor([0], dtype=torch.uint8))
        self.register_buffer('observer_enabled', torch.tensor([0], dtype=torch.uint8))
        self.register_buffer('calc_qparams', torch.tensor([0], dtype=torch.uint8))
        self.observer = observer(**observer_kwargs)
        self.register_buffer('scale', torch.tensor([1.0]))
        self.register_buffer('zero_point', torch.tensor([0]))
        self.quantize_func = quantize_func
        # zhanghao add
        self.register_buffer('quantize_type', 
                             torch.tensor([0]) if self.quantize_func==Fake_quantize_per_tensor else torch.tensor([1]))

    @torch.jit.export
    def enable_fake_quant(self):
        self.fake_quant_enabled[0] = 1
        return self

    @torch.jit.export
    def disable_fake_quant(self):
        self.fake_quant_enabled[0] = 0
        return self

    @torch.jit.export
    def enable_observer(self):
        self.observer_enabled[0] = 1
        return self

    @torch.jit.export
    def disable_observer(self):
        self.observer_enabled[0] = 0
        return self
    
    @torch.jit.export
    def enable_calc_qparams(self):
        self.calc_qparams[0] = 1
        return self

    @torch.jit.export
    def enable_calc_qparams_IP(self):
        self.calc_qparams[0] = 2
        return self
    
    @torch.jit.export
    def disable_calc_qparams(self):
        self.calc_qparams[0] = 0
        return self

    @torch.jit.export
    def calculate_qparams(self, inplace=False):
        _scale, _zero_point = self.observer.calculate_qparams()
        _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
        if inplace:
            self.scale.resize_(_scale.shape)
            self.scale.copy_(_scale)
            self.zero_point.resize_(_zero_point.shape)
            self.zero_point.copy_(_zero_point)
        else:
            return _scale, _zero_point

    def forward(self, X):
        if self.observer_enabled[0] == 1:
            self.observer(X.detach())
        
        if self.calc_qparams[0] == 1:
            self.calculate_qparams(inplace=True)
        
        if self.calc_qparams[0] == 2:
            self.calculate_qparams(inplace=False)

        if self.fake_quant_enabled[0] == 1:
            X = self.quantize_func.apply(
                X, self.scale, self.zero_point,
                self.observer.qmin, self.observer.qmax
            )
        return X

    with_args = classmethod(_with_args)

    @torch.jit.export
    def extra_repr(self):
        return 'fake_quant_enabled={}, observer_enabled={},\
            scale={}, zero_point={}'.format(
            self.fake_quant_enabled, self.observer_enabled,
            self.scale, self.zero_point)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # We cannot currently register scalar values as buffers, so need to manually
        # specify serialization here.
        super(FakeQuantize, self)._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'scale'] = self.scale
        destination[prefix + 'zero_point'] = self.zero_point

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Removing this function throws an error that the the size of the loaded tensor does not match the original size
        # i.e., These buffers start out with numel 0 and become numel 1 once they have their first forward pass.
        local_state = ['scale', 'zero_point']
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                setattr(self, name, val)
            elif strict:
                missing_keys.append(key)
        super(FakeQuantize, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                                        missing_keys, unexpected_keys, error_msgs)



def enable_fake_quant(module):
    for mod in module.modules():
        if hasattr(mod, 'enable_fake_quant'):
            mod.enable_fake_quant()

def disable_fake_quant(module):
    for mod in module.modules():
        if hasattr(mod, 'disable_fake_quant'):
            mod.disable_fake_quant()

def enable_observer(module, pattern=''):
    for name, mod in module.named_modules():
        if hasattr(mod, 'enable_observer') and re.search(pattern, name):
            mod.enable_observer()

def disable_observer(module):
    for mod in module.modules():
        if hasattr(mod, 'disable_observer'):
            mod.disable_observer()

def enable_calc_qparams(module):
    for mod in module.modules():
        if hasattr(mod, 'enable_calc_qparams'):
            mod.enable_calc_qparams()

def enable_calc_qparams_IP(module):
    for mod in module.modules():
        if hasattr(mod, 'enable_calc_qparams'):
            mod.enable_calc_qparams_IP()

def disable_calc_qparams(module):
    for mod in module.modules():
        if hasattr(mod, 'disable_calc_qparams'):
            mod.disable_calc_qparams()

def calc_qparams(module):
    for mod in module.modules():
        if isinstance(mod, FakeQuantize):
            mod.calculate_qparams(inplace=True)