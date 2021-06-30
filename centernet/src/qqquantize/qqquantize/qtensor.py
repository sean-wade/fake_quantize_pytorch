import torch

"""
Not neccessary
In convenience of tracking tensor's scale for adjust bits, especially for fx_quantize.
"""
class QTensor(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        scale = args[0].scale if hasattr(args[0], 'scale') else None
        zero_point = args[0].zero_point if hasattr(args[0], 'zero_point') else None
        out = super().__torch_function__(func, types, args, kwargs)
        if isinstance(out, torch.Tensor):
            out.scale = scale
            out.zero_point = zero_point
        return out
    
    def __repr__(self):
        s = super().__repr__()
        scale = self.scale if hasattr(self, 'scale') else None
        zero_point = self.zero_point if hasattr(self, 'zero_point') else None
        s += f' scale={scale}, zero_point={zero_point}'
        return s


if __name__ == '__main__':
    x = QTensor([1,2,3,4])
    x.scale = 1.0
    y = QTensor([1,2,3,4])
    print(y)
    z = x * y
    print(z)
    print(x)