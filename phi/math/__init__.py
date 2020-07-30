from phi.backend import Backend, DYNAMIC_BACKEND, set_precision, extrapolation
from phi.backend.scipy_backend import SCIPY_BACKEND

from phi.struct.struct_backend import StructBroadcastBackend

from ._shape import Shape, define_shape, spatial_shape, infer_shape
from ._tensors import tensor, AbstractTensor, combined_shape
from ._tensor_initializers import zeros, fftfreq, random_normal
from ._tensor_math import (
    is_tensor, as_tensor,
    copy,
    transpose,
    batch_stack, spatial_stack, channel_stack,
    concat,
    pad,
    resample,
    reshape,
    prod,
    divide_no_nan,
    where,
    sum, mean, std,
    py_func,
    zeros_like, ones_like,
    dot,
    matmul,
    einsum,
    while_loop,
    abs,
    sign,
    round, ceil, floor,
    max, min, maximum, minimum,
    clip,
    with_custom_gradient,
    sqrt,
    exp,
    conv,
    shape, staticshape, ndims,
    to_float, to_int, to_complex,
    gather,
    unstack,
    boolean_mask,
    isfinite,
    scatter,
    any, all,
    fft, ifft,
    imag, real,
    cast,
    sin, cos,
    dtype,
    tile,
    sparse_tensor,
    close, assert_close,
)
from ._nd import (
    indices_tensor,
    normalize_to,
    l1_loss, l2_loss, l_n_loss, frequency_loss,
    gradient, laplace,
    fourier_laplace, fourier_poisson, abs_square,
    downsample2x, upsample2x, interpolate_linear,
    spatial_sum, vec_abs
)
from . import _optim as optim

# Setup Backend
DYNAMIC_BACKEND.add_backend(SCIPY_BACKEND)
DYNAMIC_BACKEND.add_backend(StructBroadcastBackend(DYNAMIC_BACKEND))
