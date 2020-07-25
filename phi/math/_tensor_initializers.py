import numpy as np

from ._shape import define_shape, EMPTY_SHAPE
from ._tensors import NativeTensor, CollapsedTensor
from .. import math


def zeros(channels=(), batch=None, dtype=None, **spatial):
    """

    :param channels: int or (int,)
    :param batch: int or {name: int} or (Dimension,)
    :param dtype:
    :param spatial:
    :return:
    """
    shape = define_shape(channels, batch, infer_types_if_not_given=True, **spatial)
    zero = NativeTensor(np.zeros([], dtype=dtype), EMPTY_SHAPE)
    return CollapsedTensor(zero, shape)


def random_normal(channels=(), batch=None, dtype=None, **spatial):
    shape = define_shape(channels, batch, infer_types_if_not_given=True, **spatial)
    native = np.random.randn(*shape.sizes)
    native = math.to_float(native) if dtype is None else native.astype(dtype)
    return NativeTensor(native, shape)
