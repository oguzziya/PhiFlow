from .. import math

from ._shape import define_shape
from ._tensors import NativeTensor, Shape, CollapsedTensor, TensorStack


def zero(components=(), batch=(), dtype=None, **spatial):
    """

    :param components: int or (int,)
    :param batch: int or {name: int} or (Dimension,)
    :param dtype:
    :param spatial:
    :return:
    """
    zero = NativeTensor(math.zeros([], dtype=dtype), Shape(()))
    return CollapsedTensor(zero, define_shape(components, batch, **spatial))


def as_tensor(obj):
    if isinstance(obj, (tuple, list)):
        return TensorStack([as_tensor(item) for item in obj], )