from torch import nn

__all__ = ['InputStub', 'QStub', 'QStubWrapper']

class InputStub(nn.Module):
    r"""only used for convert to QStub"""
    def forward(self, x):
        return x

class QStub(nn.Module):
    _FLOAT_MODULE = InputStub
    def __init__(self, qconfig):
        super().__init__()
        self.qconfig = qconfig
        self.qconfig = qconfig
        self.act_quant = qconfig.activation()
    
    def forward(self, x):
        return self.act_quant(x)

    @classmethod
    def from_float(cls, mod):
        assert type(mod) == cls._FLOAT_MODULE, 'qat.' + cls.__name__ + '.from_float only works for ' + \
            cls._FLOAT_MODULE.__name__
        qconfig = mod.qconfig
        qstub = cls(qconfig)
        return qstub

class QStubWrapper(nn.Module):
    def __init__(self, module, qconfig=None):
        super().__init__()
        self.add_module('inputStub', InputStub())
        self.add_module('module', module)
        self.train(module.training)

    def forward(self, X, *args, **kwargs):
        X = self.inputStub(X)
        return self.module(X, *args, **kwargs)
