import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QBatchNorm2d_d(nn.BatchNorm2d):
    _FLOAT_MODULE = nn.BatchNorm2d
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, qconfig=None):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        assert qconfig, 'qconfig must be provided for QAT module'
        self.qconfig = qconfig
        self.act_quant = qconfig.activation()
        self.weight_quant = qconfig.weight()
        self.bias_quant = qconfig.bias()
        # raise Exception("this module should not be used")

    def forward(self, input):
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean.type(torch.DoubleTensor) if not self.training or self.track_running_stats else None,
            self.running_var.type(torch.DoubleTensor) if not self.training or self.track_running_stats else None,
            self.weight.type(torch.DoubleTensor),
            self.bias.type(torch.DoubleTensor),
            bn_training, exponential_average_factor, self.eps
        )

        if not self.training or self.track_running_stats:
            mean = self.running_mean.float()
            var_sqrt = torch.sqrt(self.running_var + self.eps).float()
        else:
            mean = self.running_mean.new_zeros(self.running_mean.shape).float()
            var_sqrt = torch.sqrt(self.eps).float()
        gamma = self.weight.float()
        beta = self.bias.float()

        weight = gamma / var_sqrt
        bias = - mean / var_sqrt * gamma + beta
        
        shape = [1, -1, 1, 1]
        return self.act_quant(
            input * self.weight_quant(weight.view(*shape).float()).double()
            + self.bias_quant(bias.view(*shape).float()).double()
        )

        # zero = mean.new_zeros(mean.shape)
        # one = self.running_var.new_ones(self.running_var.shape)

        # return self.act_quant(
        #     F.batch_norm(
        #         input,
        #         zero,
        #         one,
        #         self.weight_quant(weight),
        #         self.bias_quant(bias),
        #         bn_training, exponential_average_factor, 0
        #     )
        # )
    
    @classmethod
    def from_float(cls, mod):
        assert type(mod) == cls._FLOAT_MODULE, 'qat.' + cls.__name__ + '.from_float only works for ' + \
            cls._FLOAT_MODULE.__name__
        qconfig = mod.qconfig
        qat_bn = cls(mod.num_features, mod.eps, mod.momentum, mod.affine, mod.track_running_stats, qconfig)
        qat_bn.weight.data.copy_(mod.weight)
        qat_bn.bias.data.copy_(mod.bias)
        qat_bn.running_mean.data.copy_(mod.running_mean)
        qat_bn.running_var.data.copy_(mod.running_var)
        qat_bn.num_batches_tracked.data.copy_(mod.num_batches_tracked)
        
        return qat_bn
