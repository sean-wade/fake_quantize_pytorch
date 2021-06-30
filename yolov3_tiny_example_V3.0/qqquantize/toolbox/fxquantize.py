'''
这是 风兴 的量化代码里的 调整模型位宽关系的几个函数实现，需要注意的事项有：
    1、功能：
        目前有如下三种调整，经测试，前两个好像不能用，在 quantize_train.py 中加了也没什么用
            a. fx_adjust_bits_IP1 （陈昱贤）
            b. fx_adjust_bits_IP2 （陈昱贤）
            c. conv_bit_adjust_bias
        第三个是我刚实现的，经测试可以调整，还没经过充分测试

    2、conv_bit_adjust_bias 为例：
        在 quantize_train.py 每个 epoch 开始或者结束时，调用一下，会进行如下操作：
            a. 通过get_layer_bits，获取到各层的 input weight and act float bits，存入字典
            b. 逐层遍历，找到 QConv2d 层，对包含 bias 的卷积层进行判断
            c. 判断条件可根据硬件要求自行更改，目前是 
                max(w_bit + i_bit) < bias_bit
               在不满足时，强制更改 bias 的observer的scale值
            d. 调整完毕，继续训练
'''
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
#                 print(name, type(module.scale), module.scale)
                b = -math.log2(module.scale)
                self.bits[name] = b

                
"""
Get each layer's input weight and act float bits
"""
class GetBitsHookTorch:
    def __init__(self, mod_name):
        self.mod_name = mod_name
        self.bits = {}
        
    def __call__(self, module, inputs, outputs):
#         print("==== ==== Forward Hook: ", self.mod_name)
#         print(outputs.shape, type(outputs))
        
#         if isinstance(outputs, torch.Tensor):
        if isinstance(outputs, QTensor):
            outputs = (outputs, )
            for i, out in enumerate(outputs):
#                 if isinstance(out, torch.Tensor) and hasattr(out, 'scale'):
#                     b = -math.log2(out.scale)
                if isinstance(out, QTensor) and hasattr(out, 'scale'):
                    b = -torch.log2(out.scale)
                    self.bits.setdefault('out', []).append(b)
        for i, inp in enumerate(inputs):
#             if isinstance(inp, torch.Tensor) and hasattr(inp, 'scale') and inp.scale is not None:
            if isinstance(inp, QTensor) and hasattr(inp, 'scale') and inp.scale is not None:
#                 b = -math.log2(inp.scale)
                b = -torch.log2(inp.scale)
                self.bits.setdefault('inp', []).append(b)
        for name, module in module.named_children():
            if isinstance(module, FakeQuantize):
#                 print(name, type(module.scale), module.scale)
#                 b = -math.log2(module.scale)
                b = -torch.log2(module.scale)
                self.bits[name] = b  
#         print("==== ==== Forward Hook Done ...... ", self.mod_name)
#         print(self.bits)
                
                
def get_layer_bits(model, fake_input, qmodules=None):
    bits_dict = {}
    if qmodules is None:
        qmodules = list(DEFAULT_QAT_MODULE_MAPPING.values())

    for mod_name, mod in model.named_modules():
        if type(mod) in qmodules:
#             mod.register_forward_hook(GetBitsHook(mod_name))    # zhanghao modify
            mod.register_forward_hook(GetBitsHookTorch(mod_name))
    
    _ = model(fake_input)

    for mod_name, mod in model.named_modules():
        for h_name, h in list(mod._forward_hooks.items()):
#             if isinstance(h, GetBitsHook):    # zhanghao modify
            if isinstance(h, GetBitsHookTorch):
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
    if i_bit + w_bit - a_bit < 14:    # 2
        new_a_bit = a_bit - i_bit + 14
        print("####    <14 !!!   w_bit = %d, i_bit = %d, a_bit = %d, new_bit = %d"%(w_bit, i_bit, a_bit, new_a_bit))
    if i_bit + w_bit - a_bit > 17:    # 7
        new_a_bit = a_bit - i_bit + 17
        print("####    >17 !!!   w_bit = %d, i_bit = %d, a_bit = %d, new_bit = %d"%(w_bit, i_bit, a_bit, new_a_bit))
    module.weight_quant.scale.fill_(0.5**new_a_bit)
    
    if module.bias is not None:
        b_bit = -math.log2(module.bias_quant.scale)
        new_a_bit = b_bit
        if i_bit + w_bit - b_bit <= 0:
            new_a_bit = i_bit + w_bit
            module.weight_quant.scale.fill_(0.5**new_a_bit)
            print("####    <=0 !!!   w_bit = %d, i_bit = %d, a_bit = %d, new_bit = %d"%(w_bit, i_bit, a_bit, new_a_bit))
    if new_a_bit != a_bit:
        return True
    else:
        return False

    
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
#                     print("m.scale: ", m.scale)
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
#         for name, mod in model.named_modules():
#             if isinstance(mod, qm.QStub):
#                 mod.act_quant.scale.fill_(0.5**7)
#             if isinstance(mod, qm.QConv2d):
#                 i_bit = bit_dict[name]['inp'][0]
#                 res = conv_bit_adjust_IP2(mod, i_bit)
#                 if res:
#                     print("####    [%s] adjust"%name)
#             if isinstance(mod, qm.QBatchNorm2d):
#                 i_bit = bit_dict[name]['inp'][0]
#                 bn_bit_adjust_IP2(mod, i_bit)
    
    DEVICE = fake_input.device
    fake_input = QTensor(fake_input.cpu()).to(DEVICE)
    adjust_bits(model, fake_input)
    


    
########################################################################################
####  zhanghao add below  ####
########################################################################################

def conv_bit_adjust_bias(module, i_bit):
    # 调整 bias 到满足条件 max(w_bit + i_bit) < bias_bit
    if module.bias is None:
        return False
    
    assert isinstance(module, qm.QConv2d)
    device = module.weight_quant.scale.device
    w_bit = -torch.log2(module.weight_quant.scale)    # 这里 per_channel, w_bit是个1*75维tensor
#     a_bit = -math.log2(module.act_quant.scale)    # act_quant scale 没用到
    b_bit = -torch.log2(module.bias_quant.scale)
    new_b_bit = b_bit
          
#     if b_bit > 10:
# #         # 测试能不能改成功
#         new_b_bit = 10
#         module.bias_quant.scale.fill_(0.5 ** new_b_bit)
#         print(module.bias_quant.observer.max_bits)
#         module.bias_quant.observer.max_bits = torch.Tensor([new_b_bit]).to(module.bias.device)
#         print(module.bias_quant.observer.max_bits)
#         print("    bias_bit = %d, modify to %d...... and modify observer"%(b_bit, new_b_bit))
        
    w_i_max_bit = torch.max(w_bit + i_bit)    # 这里的判断条件可以根据实际情况修改
    if w_i_max_bit > new_b_bit:
        new_b_bit = w_i_max_bit
        module.bias_quant.scale.fill_(0.5 ** new_b_bit)
          # bias 需要指定为 FixMaxBiasObserver，才具有 max_bits 属性
        module.bias_quant.observer.max_bits = torch.Tensor([new_b_bit]).to(module.bias.device)
        print("    bias_bit = %d, modify to %d...... and modify observer"%(b_bit, new_b_bit))
    
    if new_b_bit != b_bit:
        return True
    else:
        return False    

    
def fx_adjust_bits_Bias(model, fake_input):
    assert type(fake_input) == torch.Tensor

    def adjust_bits(model, fake_input):
        bit_dict = get_layer_bits(model, fake_input)
#         print(bit_dict)
        for name, mod in model.named_modules():
            if isinstance(mod, qm.QConv2d):
                i_bit = bit_dict[name]['inp'][0]
#                 print(name, i_bit, type(i_bit))
                res = conv_bit_adjust_bias(mod, i_bit)
                if res:
                    print("####    [%s] adjust"%name)

    DEVICE = fake_input.device
    fake_input = QTensor(fake_input.cpu()).to(DEVICE)
    adjust_bits(model, fake_input)
