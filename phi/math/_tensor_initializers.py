import numpy as np

from ._shape import define_shape, EMPTY_SHAPE, spatial_shape, CHANNEL_DIM
from ._tensors import NativeTensor, CollapsedTensor, TensorStack
from phi.math.backend.scipy_backend import SCIPY_BACKEND


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
    native = SCIPY_BACKEND.random_normal(shape.sizes)
    native = native if dtype is None else native.astype(dtype)
    return NativeTensor(native, shape)


def fftfreq(resolution, dtype=None):
    """
    Returns the discrete Fourier transform sample frequencies.
    These are the frequencies corresponding to the components of the result of `math.fft` on a tensor of shape `resolution`.

    :param resolution: grid resolution measured in cells
    :param dtype: data type of the returned tensor
    :return: tensor holding the frequencies of the corresponding values computed by math.fft
    """
    resolution = spatial_shape(resolution)
    k = np.meshgrid(*[np.fft.fftfreq(int(n)) for n in resolution.sizes], indexing='ij')
    k = [SCIPY_BACKEND.to_float(channel) if dtype is None else channel.astype(dtype) for channel in k]
    channel_shape = spatial_shape(k[0].shape)
    k = [NativeTensor(channel, channel_shape) for channel in k]
    return TensorStack(k, 0, CHANNEL_DIM)
