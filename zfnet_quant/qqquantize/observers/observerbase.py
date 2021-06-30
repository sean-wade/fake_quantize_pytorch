from __future__ import absolute_import, division, print_function, unicode_literals

import math
import warnings
from abc import ABCMeta, abstractmethod
from functools import partial
from typing import List, Tuple, Optional

import torch
import torch.nn as nn

def _with_args(cls_or_self, **kwargs):
    r"""Wrapper that allows creation of class factories.

    This can be useful when there is a need to create classes with the same
    constructor arguments, but different instances.

    Example::

        >>> Foo.with_args = classmethod(_with_args)
        >>> foo_builder = Foo.with_args(a=3, b=4).with_args(answer=42)
        >>> foo_instance1 = foo_builder()
        >>> foo_instance2 = foo_builder()
        >>> id(foo_instance1) == id(foo_instance2)
        False
    """
    class _PartialWrapper(object):
        def __init__(self, p):
            self.p = p

        def __call__(self, *args, **keywords):
            return self.p(*args, **keywords)

        def __repr__(self):
            return self.p.__repr__()

        with_args = _with_args
    r = _PartialWrapper(partial(cls_or_self, **kwargs))
    return r


ABC = ABCMeta(str("ABC"), (object,), {})  # compatible with Python 2 *and* 3:


class ObserverBase(ABC, nn.Module):
    r"""Base observer Module.
    Any observer implementation should derive from this class.

    Concrete observers should follow the same API. In forward, they will update
    the statistics of the observed Tensor. And they should provide a
    `calculate_qparams` function that computes the quantization parameters given
    the collected statistics.

    Args:
        dtype: Quantized data type
    """
    def __init__(self):
        super(ObserverBase, self).__init__()

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def calculate_qparams(self, **kwargs):
        pass

    with_args = classmethod(_with_args)

def calculate_qparams_symmetric(min_val, max_val, qmin, qmax):
    """quant_x = int(float_x / scale)
    scale is the unit size
    zero_point is always 0"""
    if min_val == 0 or max_val == 0:
        raise ValueError(
            "must run observer before calling calculate_qparams.\
                                Returning default scale and zero point "
        )
    assert qmin < 0 and qmax > 0, "stupid assertion"
    assert min_val < 0 and max_val > 0, "stupid assertion too"
    max_val = max(abs(min_val), abs(max_val))
    scale = max_val / max(abs(qmin), abs(qmax))
    scale = max(scale, 1e-8)
    scale = 0.5 ** math.floor(math.log(scale, 0.5))
    zero_point = 0
    if scale == 0:
        raise ValueError('scale is 0')
    return scale, zero_point