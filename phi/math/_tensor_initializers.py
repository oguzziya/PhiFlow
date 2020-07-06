import numbers

import numpy as np

from .. import math

from ._shape import define_shape, infer_shape, CHANNEL_DIM
from ._tensors import NativeTensor, Shape, CollapsedTensor, TensorStack, AbstractTensor


def zero(channels=(), batch=(), dtype=None, **spatial):
    """

    :param channels: int or (int,)
    :param batch: int or {name: int} or (Dimension,)
    :param dtype:
    :param spatial:
    :return:
    """
    zero = NativeTensor(math.zeros([], dtype=dtype), Shape((), (), ()))
    return CollapsedTensor(zero, define_shape(channels=channels, batch=batch, **spatial))


