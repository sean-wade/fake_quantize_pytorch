from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd.function import Function
from torch.nn.modules.utils import _pair
from typing import Optional, List
import math
import sys
# from .dfx import quant, quant_grad, quant_fb, num2fixed


class Quantization(Function):
    '''
    Forward Quantization
    '''
    @staticmethod
    def forward(ctx, input, int_bits=3, dec_bits=4, is_floor=False):

        with torch.no_grad():
            #!!!!!!
            max_pos = 2 ** (int_bits + dec_bits) - 1
            max_neg = - 2 ** (int_bits + dec_bits) + 1
            if is_floor :
                # noise = torch.rand_like(input)
                y = input.mul(2**dec_bits).floor().clamp(max_neg, max_pos).div(2**dec_bits)
            else :
                y = input.mul(2**dec_bits).round().clamp(max_neg, max_pos).div(2**dec_bits)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output
        return grad_input, None, None, None


def quant(input, int_bits=3, dect_bits=4, is_floor=False):
    output = Quantization().apply(input, int_bits, dect_bits, is_floor)

    return output

### ==============================================================================###
###             quant for different data types                                    ###
### ==============================================================================###
act_quant = lambda x : quant(x, 7, 8, True)
weight_quant = lambda x : quant(x, 4, 11, False)
bias_quant = lambda x : quant(x, 7, 8, False)
act_quant_8 = lambda x : quant(x, 3, 4, True)
w_quant_8 = lambda x : quant(x, 0, 7, False)
b_quant_8 = lambda x : quant(x, 3, 4, False)
bn_w_quant_8 = lambda x : quant(x, 6, 1, False)


class IdentityBN(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(IdentityBN, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):

        return input

    def extra_repr(self):
        s = 'IdentityBN,'
        return s


class QReLu(nn.ReLU):
    __constants__ = ['inplace']

    def __init__(self, inplace=False):
        super(nn.ReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        # qinput = act_quant(input)
        out = F.relu(input, self.inplace)
        # out = act_quant(out)

        return out


class QLeakyReLU(nn.LeakyReLU):
    __constants__ = ['inplace', 'negative_slope']
    inplace: bool
    negative_slope: float

    def __init__(self, negative_slope: float = 1e-2, inplace: bool = False) -> None:
        super(nn.LeakyReLU, self).__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, input):
        output = F.leaky_relu(input, self.negative_slope, self.inplace)
        # output = act_quant(output)

        return output


class _ConvBnNd(nn.modules.conv._ConvNd):

    _version = 2

    def __init__(self,
                 # ConvNd args
                 in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups,
                 bias,
                 padding_mode,
                 # BatchNormNd args
                 # num_features: out_channels
                 eps=1e-05, momentum=0.1,
                 # affine: True
                 # track_running_stats: True
                 # Args for this module
                 freeze_bn=True,
                 ):

        nn.modules.conv._ConvNd.__init__(self, in_channels, out_channels, kernel_size,
                                         stride, padding, dilation, transposed,
                                         output_padding, groups, False, padding_mode)

        self.freeze_bn = freeze_bn if self.training else True

        # if self.training : 
        norm_layer = nn.BatchNorm2d
        # else : 
        # norm_layer = IdentityBN
        self.bn = norm_layer(out_channels, eps, momentum, True, True)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_bn_parameters()

        # this needs to be called after reset_bn_parameters,
        # as they modify the same state
        if self.training:
            if freeze_bn:
                self.freeze_bn_stats()
            else:
                self.update_bn_stats()
        else:
            self.freeze_bn_stats()

    def reset_running_stats(self):
        self.bn.reset_running_stats()

    def reset_bn_parameters(self):
        self.bn.reset_running_stats()
        nn.init.uniform_(self.bn.weight)
        nn.init.zeros_(self.bn.bias)
        # note: below is actully for conv, not BN
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)


    def reset_parameters(self):
        super(_ConvBnNd, self).reset_parameters()

    def update_bn_stats(self):
        self.freeze_bn = False
        self.bn.training = True
        return self

    def freeze_bn_stats(self):
        self.freeze_bn = True
        self.bn.training = False
        return self

    def _conv_forward(self, input, weight, bias):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)

        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def _forward(self, input):
        running_std = torch.sqrt(self.bn.running_var + self.bn.eps)
        scale_factor = self.bn.weight / running_std
        # scaled_weight = self.weight * scale_factor.reshape([-1, 1, 1, 1])
        scaled_weight = weight_quant(self.weight * scale_factor.reshape([-1, 1, 1, 1]))

        # scaled bias :
        # with torch.no_grad():
        if self.bias is not None:
            scaled_bias = scale_factor * (self.bias - self.bn.running_mean) + self.bn.bias
        else:
            scaled_bias = - scale_factor * self.bn.running_mean + self.bn.bias
        # scaled_bias_q = scaled_bias
        scaled_bias_q = bias_quant(scaled_bias)

        # this does not include the conv bias
        conv = self._conv_forward(input, scaled_weight.data, scaled_bias_q.data)
        conv_bias = conv
        # conv_bias = conv + scaled_bias_q.reshape([1, -1, 1, 1])

        if self.training:
            conv_bias_orig = conv_bias - scaled_bias.reshape([1, -1, 1, 1])
            conv_orig = conv_bias_orig / scale_factor.reshape([1, -1, 1, 1])

            # conv_orig = conv / scale_factor.reshape([1, -1, 1, 1])
            if self.bias is not None:
                conv_orig = conv_orig + self.bias.reshape([1, -1, 1, 1])
            conv = self.bn(conv_orig)
            return conv
        else:
            return conv_bias
       
    def extra_repr(self):
        # TODO(jerryzh): extend
        return super(_ConvBnNd, self).extra_repr()

    def forward(self, input):
        return self._forward(input)

    def train(self, mode=True):
        """
        Batchnorm's training behavior is using the self.training flag. Prevent
        changing it if BN is frozen. This makes sure that calling `model.train()`
        on a model with a frozen BN will behave properly.
        """
        self.training = mode
        if not self.freeze_bn:
            for module in self.children():
                module.train(mode)
        return self

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)
        if version is None or version == 1:
            # BN related parameters and buffers were moved into the BN module for v2
            v2_to_v1_names = {
                'bn.weight': 'gamma',
                'bn.bias': 'beta',
                'bn.running_mean': 'running_mean',
                'bn.running_var': 'running_var',
                'bn.num_batches_tracked': 'num_batches_tracked',
            }
            for v2_name, v1_name in v2_to_v1_names.items():
                if prefix + v1_name in state_dict:
                    state_dict[prefix + v2_name] = state_dict[prefix + v1_name]
                    state_dict.pop(prefix + v1_name)
                elif strict:
                    missing_keys.append(prefix + v2_name)

        super(_ConvBnNd, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class QConv2d_8(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 ):
        super(QConv2d_8, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, input):
        # input = act_quant(input)

        weight = w_quant_8(self.weight)
        
        if self.bias is not None: 
            bias = b_quant_8(self.bias)
            output = F.conv2d(input, weight.data, bias.data, self.stride,
                              self.padding, self.dilation, self.groups)
        else:
            output = F.conv2d(input, weight.data, None, self.stride,
                            self.padding, self.dilation, self.groups)
            
        # output = act_quant_8(output)

        return output


class QConv2d_16(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 ):
        super(QConv2d_16, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, input):
        # input = act_quant(input)

        weight = weight_quant(self.weight)
        
        if self.bias is not None: 
            bias = bias_quant(self.bias)
            output = F.conv2d(input, weight.data, bias.data, self.stride,
                              self.padding, self.dilation, self.groups)
        else:
            output = F.conv2d(input, weight.data, None, self.stride,
                            self.padding, self.dilation, self.groups)
            
        # output = act_quant(output)

        return output


class QLinear_8(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(QLinear_8, self).__init__(in_features, out_features, bias=bias)

    def forward(self, input):
        weight = w_quant_8(self.weight)

        if self.bias is not None:
            bias = b_quant_8(self.bias)
            output = F.linear(input, weight.data, bias.data)
        else:
            bias = None
            output = F.linear(input, weight.data, None)

        
        # output = act_quant_8(output)

        return output


class QLinear_16(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(QLinear_16, self).__init__(in_features, out_features, bias=bias)

    def forward(self, input):
        weight = weight_quant(self.weight)

        if self.bias is not None:
            bias = bias_quant(self.bias)
            output = F.linear(input, weight.data, bias.data)
        else:
            bias = None
            output = F.linear(input, weight.data, None)

        
        # output = act_quant(output)  # post bias

        return output


class QBatchNorm2d_8(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(QBatchNorm2d_8, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        # input = act_quant(input)

        mean = self.running_mean
        var_sqrt = torch.sqrt(self.running_var + self.eps)
        beta = self.weight
        gamma = self.bias
        zero = mean.new_zeros(mean.shape)
        one = self.running_var.new_ones(self.running_var.shape)

        weight = beta / var_sqrt
        bias = - mean / var_sqrt * beta + gamma
        
        weight = bn_w_quant_8(weight)
        bias = b_quant_8(bias)

        output = F.batch_norm(input, weight=weight.data, bias=bias.data, 
                              running_mean=zero, running_var=one, eps=0)
        
        # output = act_quant_8(output0)

        return output


class QBatchNorm2d_16(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(QBatchNorm2d_16, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        # input = act_quant(input)

        mean = self.running_mean
        var_sqrt = torch.sqrt(self.running_var + self.eps)
        beta = self.weight
        gamma = self.bias
        zero = mean.new_zeros(mean.shape)
        one = self.running_var.new_ones(self.running_var.shape)

        weight = beta / var_sqrt
        bias = - mean / var_sqrt * beta + gamma
        
        weight = weight_quant(weight)
        bias = bias_quant(bias)

        output = F.batch_norm(input, weight=weight.data, bias=bias.data, 
                              running_mean=zero, running_var=one, eps=0)
        
        # output = act_quant_8(output0)

        return output


class QAdaptiveAvgPooling2d(nn.AdaptiveAvgPool2d):
    def __init__(self, output_size):
        super(QAdaptiveAvgPooling2d, self).__init__(output_size)
    
    def forward(self, input):
        output = F.adaptive_avg_pool2d(input, self.output_size)
        # output = act_quant(output)

        return output


class QConvTranspose2d_8(nn.modules.conv._ConvTransposeNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride = 1,
        padding = 0,
        output_padding = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1,
        padding_mode: str = 'zeros'
    ):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        output_padding = _pair(output_padding)
        super(QConvTranspose2d_8, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                                 True, output_padding, groups, bias, padding_mode)

    def forward(self, input: Tensor, output_size: Optional[List[int]] = None) -> Tensor:
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose2d')

        output_padding = self._output_padding(
            input, output_size, self.stride, self.padding, self.kernel_size, self.dilation)
        
        weight = w_quant_8(self.weight)

        if self.bias is not None: 
            bias = b_quant_8(self.bias)
            output = F.conv_transpose2d(input, weight.data, bias.data, self.stride, self.padding,
                                        output_padding, self.groups, self.dilation)
        else:
            output = F.conv_transpose2d(input, weight.data, None, self.stride, self.padding,
                                        output_padding, self.groups, self.dilation)

        return output


class QConvTranspose2d_16(nn.modules.conv._ConvTransposeNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride = 1,
        padding = 0,
        output_padding = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1,
        padding_mode: str = 'zeros'
    ):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        output_padding = _pair(output_padding)
        super(QConvTranspose2d_16, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                                  True, output_padding, groups, bias, padding_mode)

    def forward(self, input: Tensor, output_size: Optional[List[int]] = None) -> Tensor:
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose2d')

        output_padding = self._output_padding(
            input, output_size, self.stride, self.padding, self.kernel_size, self.dilation)
        
        weight = weight_quant(self.weight)

        if self.bias is not None: 
            bias = bias_quant(self.bias)
            output = F.conv_transpose2d(input, weight.data, bias.data, self.stride, self.padding,
                                        output_padding, self.groups, self.dilation)
        else:
            output = F.conv_transpose2d(input, weight.data, None, self.stride, self.padding,
                                        output_padding, self.groups, self.dilation)

        return output


class ConvBn2d(_ConvBnNd, nn.Conv2d):
    r"""
    A ConvBn2d module is a module fused from Conv2d and BatchNorm2d,
    attached with FakeQuantize modules for both output activation and weight,
    used in quantization aware training.

    We combined the interface of :class:`torch.nn.Conv2d` and
    :class:`torch.nn.BatchNorm2d`.

    Implementation details: https://arxiv.org/pdf/1806.08342.pdf section 3.2.2

    Similar to :class:`torch.nn.Conv2d`, with FakeQuantize modules initialized
    to default.

    Attributes:
        freeze_bn:
        activation_post_process: fake quant module for output activation
        weight_fake_quant: fake quant module for weight

    """

    def __init__(self,
                 # ConvNd args
                 in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=None,
                 padding_mode='zeros',
                 # BatchNorm2d args
                 # num_features: out_channels
                 eps=1e-05, momentum=0.1,
                 # affine: True
                 # track_running_stats: True
                 # Args for this module
                 freeze_bn=False):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        _ConvBnNd.__init__(self, in_channels, out_channels, kernel_size, stride,
                           padding, dilation, False, _pair(0), groups, bias, padding_mode,
                           eps, momentum, freeze_bn)
    
    def forward(self, input):
        output = ConvBn2d._forward(self, input)
        # output = act_quant(output)
        return output


class ConvBnReLU2d(ConvBn2d):
    r"""
    A ConvBnReLU2d module is a module fused from Conv2d, BatchNorm2d and ReLU,
    attached with FakeQuantize modules for both output activation and weight,
    used in quantization aware training.

    We combined the interface of :class:`torch.nn.Conv2d` and
    :class:`torch.nn.BatchNorm2d` and :class:`torch.nn.ReLU`.

    Implementation details: https://arxiv.org/pdf/1806.08342.pdf

    Similar to `torch.nn.Conv2d`, with FakeQuantize modules initialized to
    default.

    Attributes:
        observer: fake quant module for output activation, it's called observer
            to align with post training flow
        weight_fake_quant: fake quant module for weight

    """

    def __init__(self,
                 # Conv2d args
                 in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=None,
                 padding_mode='zeros',
                 # BatchNorm2d args
                 # num_features: out_channels
                 eps=1e-05, momentum=0.1,
                 # affine: True
                 # track_running_stats: True
                 # Args for this module
                 freeze_bn=False):
        super(ConvBnReLU2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                           padding, dilation, groups, bias,
                                           padding_mode, eps, momentum,
                                           freeze_bn)

    def forward(self, input):
        output = F.leaky_relu(ConvBn2d._forward(self, input), 0.125)
        # output = act_quant(output)
        return output


if __name__ == "__main__":

    def func(x, int_bit = 1):
        y = x + int_bit
        return y 

    func2 = lambda x : func(x, 2)

    z = func2(3)
    print(z)