from functools import partial

import numpy as np

from ._shape import BATCH_DIM, CHANNEL_DIM, SPATIAL_DIM, Shape, EMPTY_SHAPE
from ..backend import extrapolation, math
from ._tensors import AbstractTensor, tensor, broadcastable_native_tensors, NativeTensor, CollapsedTensor, TensorStack
from ._tensor_initializers import zeros
from ..backend.scipy_backend import SCIPY_BACKEND


def is_tensor(x):
    return isinstance(x, AbstractTensor)


def as_tensor(x, convert_external=True):
    if convert_external:
        return tensor(x)
    else:
        return x


def copy(tensor, only_mutable=False):
    raise NotImplementedError()


def transpose(tensor, axes):
    if isinstance(tensor, AbstractTensor):
        return CollapsedTensor(tensor, tensor.shape[axes])
    else:
        return math.transpose(tensor, axes)


def channel_stack(values, axis=0):
    return _stack(values, axis, CHANNEL_DIM)


def batch_stack(values, axis='batch'):
    return _stack(values, axis, BATCH_DIM)


def spatial_stack(values, axis='x'):
    return _stack(values, axis, SPATIAL_DIM)


def _stack(values, dim, dim_type):
    def inner_stack(*values):
        return TensorStack(values, dim, dim_type, keep_separate='auto')

    result = broadcast_op(inner_stack, values)
    return result



def concat(values, axis):
    tensors = broadcastable_native_tensors(values)
    concatenated = math.concat(tensors, axis)
    return NativeTensor(concatenated, values[0].shape)


def pad(value, pad_width, mode=extrapolation.ZERO):
    assert isinstance(value, AbstractTensor)
    if isinstance(pad_width, dict):  # name -> (lower, upper) or both
        if isinstance(value, NativeTensor):
            native = value.tensor
            ordered_pad_widths = value.shape.order(pad_width, default=0)
            ordered_mode = value.shape.order(mode, default=extrapolation.ZERO)
            result_tensor = math.pad(native, ordered_pad_widths, ordered_mode)
            new_shape = value.shape.with_sizes(math.staticshape(result_tensor))
            return NativeTensor(result_tensor, new_shape)
    raise NotImplementedError()


def resample(inputs, sample_coords, interpolation='linear', boundary=extrapolation.ZERO):
    if isinstance(boundary, (tuple, list)):
        boundary = [extrapolation.ZERO, *boundary, extrapolation.ZERO]

    def atomic_resample(inputs, sample_coords):
        inputs_, _ = _invertible_standard_form(inputs)
        sample_coords_, _ = _invertible_standard_form(sample_coords)  # TODO iterate if keep_separate
        resampled = math.resample(inputs_, sample_coords_, interpolation, boundary)

        batch_shape = inputs.shape.batch & sample_coords.shape.batch
        result_shape = batch_shape & sample_coords.shape.spatial & inputs.shape.channel

        un_reshaped = math.reshape(resampled, result_shape.sizes)
        return NativeTensor(un_reshaped, result_shape)

    result = broadcast_op(atomic_resample, [inputs, sample_coords])
    return result


def broadcast_op(operation, tensors):
    non_atomic_dims = set()
    for tensor in tensors:
        if isinstance(tensor, TensorStack) and tensor.keep_separate:
            non_atomic_dims.add(tensor.stack_dim_name)
    if len(non_atomic_dims) == 0:
        return operation(*tensors)
    elif len(non_atomic_dims) == 1:
        dim = next(iter(non_atomic_dims))
        dim_type = None
        size = None
        unstacked = []
        for tensor in tensors:
            if dim in tensor.shape:
                unstacked_tensor = tensor.unstack(dim)
                unstacked.append(unstacked_tensor)
                if size is None:
                    size = len(unstacked_tensor)
                    dim_type = tensor.shape.get_type(dim)
                else:
                    assert size == len(unstacked_tensor)
                    assert dim_type == tensor.shape.get_type(dim)
            else:
                unstacked.append(tensor)
        result_unstacked = []
        for i in range(size):
            gathered = [t[i] if isinstance(t, tuple) else t for t in unstacked]
            result_unstacked.append(operation(*gathered))
        return TensorStack(result_unstacked, dim, dim_type, keep_separate=True)
    else:
        raise NotImplementedError()


def reshape(value, shape):
    raise NotImplementedError()


def prod(value, axis=None):
    if SCIPY_BACKEND.is_applicable([value]) and axis is None:
        return SCIPY_BACKEND.prod(value)
    raise NotImplementedError()


def divide_no_nan(x, y):
    raise NotImplementedError()


def where(condition, x=None, y=None):
    raise NotImplementedError()


def sum(value: AbstractTensor, axis=None):
    if isinstance(value, AbstractTensor):
        if isinstance(axis, int) and axis < 0:
            raise ValueError('axis argument must be the name of the dimension, not its position')
        result = math.sum(value.native(), axis=value.shape.index(axis), keepdims=False)
        return NativeTensor(result, value.shape.without(axis))
    else:
        if axis is None:
            values = [sum(v).native() for v in value]
            result_scalar = math.sum(values)
            return NativeTensor(result_scalar, EMPTY_SHAPE)
        elif axis == 0:
            shape, tensors = broadcastable_native_tensors(*value)
            result_tensor = math.sum(tensors, axis=0, keepdims=False)
            return NativeTensor(result_tensor, shape)
        else:
            raise NotImplementedError()


def mean(value, axis=None):
    result = math.mean(value.native(), axis=value.shape.index(axis), keepdims=False)
    return NativeTensor(result, value.shape.without(axis))


def std(x: AbstractTensor, axis=None, keepdims=False):
    result = math.std(x.native(), axis=x.shape.index(axis), keepdims=False)
    return NativeTensor(result, x.shape.without(axis))


def py_func(func, inputs, Tout, shape_out, stateful=True, name=None, grad=None):
    raise NotImplementedError()


def zeros_like(tensor):
    return zeros(tensor.shape, dtype=tensor.dtype)


def ones_like(tensor):
    return zeros(tensor.shape, dtype=tensor.dtype) + 1


def dot(a, b, axes):
    raise NotImplementedError()


def matmul(A, b):
    raise NotImplementedError()


def einsum(equation, *tensors):
    raise NotImplementedError()


def abs(x: AbstractTensor):
    return x._op1(math.abs)


def sign(x: AbstractTensor):
    return x._op1(math.sign)


def round(x: AbstractTensor):
    return x._op1(math.round)


def ceil(x: AbstractTensor):
    return x._op1(math.ceil)


def floor(x: AbstractTensor):
    return x._op1(math.floor)


def max(x, axis=None):
    x = tensor(x)
    result = math.max(x.native(), x.shape.index(axis), keepdims=False)
    return NativeTensor(result, x.shape.without(axis))


def min(x, axis=None):
    x = tensor(x)
    result = math.min(x.native(), x.shape.index(axis), keepdims=False)
    return NativeTensor(result, x.shape.without(axis))


def maximum(a, b):
    a_, b_ = tensor(a, b)
    return a_._op2(b_, math.maximum)


def minimum(a, b):
    a_, b_ = tensor(a, b)
    return a_._op2(b_, math.minimum)


def clip(x, minimum, maximum):
    new_shape, (x_, min_, max_) = broadcastable_native_tensors(*tensor(x, minimum, maximum))
    assert new_shape == x.shape, 'Not implemented'
    result_tensor = math.clip(x_, min_, max_)
    return NativeTensor(result_tensor, new_shape)


def with_custom_gradient(function, inputs, gradient, input_index=0, output_index=None, name_base='custom_gradient_func'):
    raise NotImplementedError()


def sqrt(x):
    return tensor(x)._op1(math.sqrt)


def exp(x):
    return tensor(x)._op1(math.exp)


def conv(tensor, kernel, padding='same'):
    raise NotImplementedError()


def shape(tensor):
    return tensor.shape.sizes if isinstance(tensor, AbstractTensor) else math.shape(tensor)


def ndims(tensor):
    return tensor.rank if isinstance(tensor, AbstractTensor) else math.ndims(tensor)


def staticshape(tensor):
    if isinstance(tensor, AbstractTensor):
        return tensor.shape.sizes
    else:
        return math.staticshape(tensor)


def to_float(x, float64=False):
    return tensor(x)._op1(partial(math.to_float, float64=float64))


def to_int(x, int64=False):
    return tensor(x)._op1(partial(math.to_int, int64=int64))


def to_complex(x):
    return tensor(x)._op1(math.to_complex)


def gather(values, indices, batch_dims=0):
    raise NotImplementedError()


def unstack(tensor, axis=0):
    assert isinstance(tensor, AbstractTensor)
    return tensor.unstack(tensor.shape.names[axis])


def boolean_mask(x, mask):
    raise NotImplementedError()


def isfinite(x):
    return tensor(x)._op1(lambda t: math.isfinite(t))


def scatter(points, indices, values, shape, duplicates_handling='undefined'):
    raise NotImplementedError()


def any(boolean_tensor, axis=None, keepdims=False):
    raise NotImplementedError()


def all(boolean_tensor, axis=None, keepdims=False):
    if axis is None:
        if isinstance(boolean_tensor, NativeTensor):
            return math.all(boolean_tensor.tensor)
        elif isinstance(boolean_tensor, CollapsedTensor):
            return all(boolean_tensor.tensor, axis=axis)
        elif isinstance(boolean_tensor, TensorStack):
            return all([all(t, axis=None) for t in boolean_tensor.tensors])
    raise NotImplementedError()


def fft(x):
    raise NotImplementedError()


def ifft(k):
    native, assemble = _invertible_standard_form(k)
    result = math.ifft(native)
    return assemble(result)


def imag(complex):
    raise NotImplementedError()


def real(complex: AbstractTensor):
    return complex._op1(lambda t: math.real(t))


def cast(x, dtype):
    raise NotImplementedError()


def sin(x):
    return tensor(x)._op1(math.sin)


def cos(x):
    return tensor(x)._op1(math.cos)


def dtype(x):
    if isinstance(x, AbstractTensor):
        return x.dtype
    else:
        return math.dtype(x)


def tile(value, multiples):
    raise NotImplementedError()


def expand_channel(x, dim_size, dim_name):
    x = tensor(x)



def sparse_tensor(indices, values, shape):
    raise NotImplementedError()


def _invertible_standard_form(tensor: AbstractTensor):
    normal_order = tensor.shape.normal_order()
    native = tensor.native(normal_order.names)
    standard_form = (tensor.shape.batch.volume,) + tensor.shape.spatial.sizes + (tensor.shape.channel.volume,)
    reshaped = math.reshape(native, standard_form)

    def assemble(reshaped):
            un_reshaped = math.reshape(reshaped, math.shape(native))
            return NativeTensor(un_reshaped, normal_order)

    return reshaped, assemble


def close(*tensors, rel_tolerance=1e-5, abs_tolerance=0):
    """
    Checks whether all tensors have equal values within the specified tolerance.

    Does not check that the shapes exactly match.
    Tensors with different shapes are reshaped before comparing.

    :param tensors: tensor or tensor-like (constant) each
    :param rel_tolerance: relative tolerance
    :param abs_tolerance: absolute tolerance
    """
    tensors = [tensor(t) for t in tensors]
    for other in tensors[1:]:
        if not _close(tensors[0], other, rel_tolerance=rel_tolerance, abs_tolerance=abs_tolerance):
            return False
    return True


def _close(tensor1, tensor2, rel_tolerance=1e-5, abs_tolerance=0):
    if tensor2 is tensor1:
        return True
    new_shape, (native1, native2) = broadcastable_native_tensors(tensor1, tensor2)
    np1 = math.numpy(native1)
    np2 = math.numpy(native2)
    return np.allclose(np1, np2, rel_tolerance, abs_tolerance)


def assert_close(*tensors, rel_tolerance=1e-5, abs_tolerance=0):
    """
    Checks that all tensors have equal values within the specified tolerance.
    Raises an AssertionError if the values of this tensor are not within tolerance of any of the other tensors.

    Does not check that the shapes exactly match.
    Tensors with different shapes are reshaped before comparing.

    :param tensors: tensor or tensor-like (constant) each
    :param rel_tolerance: relative tolerance
    :param abs_tolerance: absolute tolerance
    """
    tensors = [tensor(t) for t in tensors]
    for other in tensors[1:]:
        _assert_close(tensors[0], other, rel_tolerance=rel_tolerance, abs_tolerance=abs_tolerance)


def _assert_close(tensor1, tensor2, rel_tolerance=1e-5, abs_tolerance=0):
    if tensor2 is tensor1:
        return
    if isinstance(tensor2, (int, float, bool)):
        np.testing.assert_allclose(tensor1.numpy(), tensor2, rel_tolerance, abs_tolerance)
    new_shape, (native1, native2) = broadcastable_native_tensors(tensor1, tensor2)
    np1 = math.numpy(native1)
    np2 = math.numpy(native2)
    if not np.allclose(np1, np2, rel_tolerance, abs_tolerance):
        np.testing.assert_allclose(np1, np2, rel_tolerance, abs_tolerance)
