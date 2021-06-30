import torch
from qqquantize.observers.fake_quantize import FakeQuantize
from qqquantize.qconfig import DEFAULT_QAT_MODULE_MAPPING
import qqquantize.qmodules as qm
from qqquantize.qtensor import QTensor
import math

"""
Get each layer's input weight and act float bits
"""
class GetBitsHook:
    def __init__(self, mod_name):
        self.mod_name = mod_name
        self.bits = {}
    def __call__(self, module, inputs, outputs):
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs, )
            for i, out in enumerate(outputs):
                if isinstance(out, torch.Tensor) and hasattr(out, 'scale'):
                    b = -math.log2(out.scale)
                    self.bits.setdefault('out', []).append(b)
        for i, inp in enumerate(inputs):
            if isinstance(inp, torch.Tensor) and hasattr(inp, 'scale') and inp.scale is not None:
                b = -math.log2(inp.scale)
                self.bits.setdefault('inp', []).append(b)
        for name, module in module.named_children():
            if isinstance(module, FakeQuantize):
                b = -math.log2(module.scale)
                self.bits[name] = b

def get_layer_bits(model, fake_input, qmodules=None):
    bits_dict = {}
    if qmodules is None:
        qmodules = list(DEFAULT_QAT_MODULE_MAPPING.values())

    for mod_name, mod in model.named_modules():
        if type(mod) in qmodules:
            mod.register_forward_hook(GetBitsHook(mod_name))
    
    _ = model(fake_input)

    for mod_name, mod in model.named_modules():
        for h_name, h in list(mod._forward_hooks.items()):
            if isinstance(h, GetBitsHook):
                bits_dict[mod_name] = h.bits
            mod._forward_hooks.pop(h_name)
    return bits_dict

def conv_bit_adjust_IP1(module, i_bit):
    assert isinstance(module, qm.QConv2d)
    device = module.act_quant.scale.device
    w_bit = -math.log2(module.weight_quant.scale)
    a_bit = -math.log2(module.act_quant.scale)
    new_a_bit = a_bit
    if i_bit + w_bit - a_bit < 0:
        new_a_bit = i_bit + w_bit
    if i_bit + w_bit - a_bit > 7:
        new_a_bit = i_bit + w_bit - 7
    module.weight_quant.scale.fill_(0.5**new_a_bit)
    # if new_a_bit != a_bit:
    #     return True
    # else:
    #     return False

def conv_bit_adjust_IP2(module, i_bit):
    assert isinstance(module, qm.QConv2d)
    device = module.weight_quant.scale.device
    w_bit = -math.log2(module.weight_quant.scale)
    a_bit = -math.log2(module.act_quant.scale)
    new_a_bit = a_bit
    if i_bit + w_bit - a_bit < 2:
        new_a_bit = a_bit - i_bit + 2
    if i_bit + w_bit - a_bit > 7:
        new_a_bit = a_bit - i_bit + 7
    module.weight_quant.scale.fill_(0.5**new_a_bit)
    if module.bias is not None:
        b_bit = -math.log2(module.bias_quant.scale)
        new_a_bit = b_bit
        if i_bit + w_bit - b_bit <= 0:
            new_a_bit = i_bit + w_bit
            module.weight_quant.scale.fill_(0.5**new_a_bit)
    # if new_a_bit != a_bit:
    #     return True
    # else:
    #     return False

def bn_bit_adjust_IP2(module, i_bit):
    assert isinstance(module, qm.QBatchNorm2d)
    device = module.weight_quant.scale.device
    w_bit = -math.log2(module.weight_quant.scale)
    b_bit = -math.log2(module.bias_quant.scale)
    new_a_bit = b_bit
    if i_bit + w_bit - b_bit <= 0:
        new_a_bit = i_bit + w_bit
    module.weight_quant.scale.fill_(0.5**new_a_bit)
    # if new_a_bit != a_bit:
    #     return True
    # else:
    #     return False

def global_bit_adjust(model, TYPELIST):
    _fbits = []
    for name, mod in model.named_modules():
        if type(mod) in TYPELIST:
            for m in mod.modules():
                if isinstance(m, FakeQuantize):
                    _fbits.append(-math.log2(m.scale))
    min_fbits = min(_fbits)
    max_fbits = max(_fbits)
    if max_fbits - min_fbits > 7:
        min_fbits = int((min_fbits + max_fbits) / 2 - 3.5)
        max_fbits = min_fbits + 7
    
    for name, mod in model.named_modules():
        if type(mod) in TYPELIST:
            for m in mod.modules():
                if isinstance(m, FakeQuantize):
                    if -math.log2(m.scale) < min_fbits:
                        m.scale[:] = 0.5 ** min_fbits
                    if -math.log(m.scale) > max_fbits:
                        m.scale[:] = 0.5 ** max_fbits


def fx_adjust_bits_IP1(model, fake_input):
    assert type(fake_input) == torch.Tensor
    QMOD_LIST = list(DEFAULT_QAT_MODULE_MAPPING.values())
    global_bit_adjust(model, QMOD_LIST)

    def adjust_bits(model, fake_input):
        bit_dict = get_layer_bits(model, fake_input)
        for name, mod in model.named_modules():
            if isinstance(mod, qm.QConv2d):
                i_bit = bit_dict[name]['inp'][0]
                conv_bit_adjust_IP1(mod, i_bit)
        #         if changed:
        #             break
        # if changed:
        #     adjust_bits(model, fake_input)


def fx_adjust_bits_IP2(model, fake_input):
    assert type(fake_input) == torch.Tensor
    QMOD_LIST = list(DEFAULT_QAT_MODULE_MAPPING.values())
    global_bit_adjust(model, QMOD_LIST)

    def adjust_bits(model, fake_input):
        bit_dict = get_layer_bits(model, fake_input)
        for name, mod in model.named_modules():
            if isinstance(mod, qm.QStub):
                mod.act_quant.scale.fill_(0.5**7)
            if isinstance(mod, qm.QConv2d):
                i_bit = bit_dict[name]['inp'][0]
                conv_bit_adjust_IP2(mod, i_bit)
            if isinstance(mod, qm.QBatchNorm2d):
                i_bit = bit_dict[name]['inp'][0]
                bn_bit_adjust_IP2(mod, i_bit)
    
    DEVICE = fake_input.device
    fake_input = QTensor(fake_input.cpu()).to(DEVICE)
    adjust_bits(model, fake_input)