import numbers
import warnings

import numpy as np

from .. import math
from ._shape import Shape, infer_shape, CHANNEL_DIM, BATCH_DIM, SPATIAL_DIM


class AbstractTensor:
    """
    Tensors with grouped and named dimensions.

    All tensors are editable.

    The internal data representation of a tensor can change, even without being edited.
    """

    def native(self, order=None):
        """
        Returns a native tensor object with the dimensions ordered according to `order`.

        Transposes the underlying tensor to match the name order and adds singleton dimensions for new dimension names.

        If a dimension of the tensor is not listed in `order`, a `ValueError` is raised.

        :param order: (optional) list of dimension names. If not given, the current order is kept.
        :return: native tensor object
        :raise: ValueError if the tensor cannot be transposed to match target_shape
        """
        raise NotImplementedError()

    def numpy(self, order=None):
        native = self.native(order=order)
        return math.numpy(native)

    @property
    def dtype(self):
        raise NotImplementedError()

    @property
    def shape(self):
        raise NotImplementedError()

    def _with_shape_replaced(self, new_shape):
        raise NotImplementedError()

    @property
    def rank(self):
        return self.shape.rank

    def __len__(self):
        assert self.rank == 1
        return self.shape.volume

    def __repr__(self):
        if self.rank == 0:
            content = self.numpy()
            return str(content)
        if self.shape.volume <= 4:
            content = self.numpy(order=self.shape.names)
            content = list(math.reshape(content, [-1]))
            content = ', '.join([repr(number) for number in content])
            return "[%s, %s: %s]" % (self.dtype, self.shape, content)
        else:
            return "[%s, %s ...]" % (self.dtype, self.shape)

    def __getitem__(self, item):
        if isinstance(item, (slice, int)):  # Channel dimension
            assert self.shape.channel.rank == 1
            item = {0: item}
        if isinstance(item, (tuple, list)):
            if len(item) == self.shape.channel.rank:
                item = {i: selection for i, selection in enumerate(item)}
            elif len(item) == self.shape.rank:  # legacy indexing
                warnings.warn("Slicing with sequence should only be used for channel dimensions.")
                item = {name: selection for name, selection in zip(self.shape.names, item)}
        assert isinstance(item, dict)  # dict mapping name -> slice/int
        return self._getitem(item)

    def _getitem(self, selection_dict):
        raise NotImplementedError()

    # def __setitem__(self, key, value):
    #     """
    #     All tensors are editable.
    #
    #     :param key: list/tuple of slices / indices
    #     :param value:
    #     :return:
    #     """
    #     raise NotImplementedError()

    def unstack(self, dimension=0):
        """
        Splits this tensor along the specified dimension.
        The returned tensors have the same dimensions as this tensor save the unstacked dimension.

        :param dimension: name of dimension or Dimension or None for component dimension
        :type dimension: str or int or _TensorDim
        :return: tuple of tensors
        """
        raise NotImplementedError()

    def dimension(self, name):
        return _TensorDim(self, name)

    @property
    def dimensions(self):
        return [_TensorDim(self, name) for name in self.shape.names]

    def __getattr__(self, name):
        if name in self.shape:
            return _TensorDim(self, name)
        raise AttributeError("%s has no attribute '%s'" % (self, name))

    def __add__(self, other):
        return self._op2(other, lambda t1, t2: t1 + t2)

    def __sub__(self, other):
        return self._op2(other, lambda t1, t2: t1 - t2)

    def __radd__(self, other):
        return self._op2(other, lambda t2, t1: t1 + t2)  # inverse order

    def __and__(self, other):
        return self._op2(other, lambda t1, t2: t1 & t2)

    def __or__(self, other):
        return self._op2(other, lambda t1, t2: t1 | t2)

    def __xor__(self, other):
        return self._op2(other, lambda t1, t2: t1 ^ t2)

    def __mul__(self, other):
        return self._op2(other, lambda t1, t2: t1 * t2)

    def __rmul__(self, other):
        return self._op2(other, lambda t2, t1: t1 * t2)  # inverse order

    def __truediv__(self, other):
        return self._op2(other, lambda t1, t2: t1 / t2)

    def __rtruediv__(self, other):
        return self._op2(other, lambda t2, t1: t1 / t2)  # inverse order

    def __divmod__(self, other):
        return self._op2(other, lambda t1, t2: divmod(t1, t2))

    def __rdivmod__(self, other):
        return self._op2(other, lambda t2, t1: divmod(t1, t2))  # inverse order

    def __pow__(self, power, modulo=None):
        assert modulo is None
        return self._op2(power, lambda t1, t2: t1 ** t2)

    def __rpow__(self, other):
        return self._op2(other, lambda t2, t1: t1 ** t2)  # inverse order

    def __mod__(self, other):
        return self._op2(other, lambda t1, t2: t1 % t2)

    def __lshift__(self, other):
        return self._op2(other, lambda t1, t2: t1 << t2)

    def __rshift__(self, other):
        return self._op2(other, lambda t1, t2: t1 >> t2)

    def __eq__(self, other):
        return self._op2(other, lambda t1, t2: math.equal(t1, t2))

    def __ne__(self, other):
        return self._op2(other, lambda t1, t2: ~math.equal(t1, t2))

    def __lt__(self, other):
        return self._op2(other, lambda t1, t2: t1 < t2)

    def __le__(self, other):
        return self._op2(other, lambda t1, t2: t1 <= t2)

    def __gt__(self, other):
        return self._op2(other, lambda t1, t2: t1 > t2)

    def __ge__(self, other):
        return self._op2(other, lambda t1, t2: t1 >= t2)

    def __abs__(self):
        return self._op1(lambda t: math.abs(t))

    def as_complex(self):
        return self._op1(lambda t: math.to_complex(t))

    def as_float(self):
        return self._op1(lambda t: math.to_float(t))

    def as_int(self, int64=False):
        return self._op1(lambda t: math.to_int(t, int64=int64))

    def __copy__(self):
        return self._op1(lambda t: math.copy(t, only_mutable=True))

    def __deepcopy__(self, memodict={}):
        return self._op1(lambda t: math.copy(t, only_mutable=False))

    def __neg__(self):
        return self._op1(lambda t: -t)

    def __reversed__(self):
        assert self.shape.channel.rank == 1
        return self[::-1]

    def _op2(self, other, native_function):
        if isinstance(other, AbstractTensor):
            new_shape, (native1, native2) = broadcastable_native_tensors(self, other)
            result_tensor = native_function(native1, native2)
            return NativeTensor(result_tensor, new_shape)
        elif isinstance(other, Shape):
            assert self.shape.channel.rank == 1
            if self.shape.channel.sizes[0] == self.shape.spatial.rank:
                sizes = other.select(*self.shape.spatial.names).sizes
                sizes = math.as_tensor(sizes)
                return self._op2(sizes, native_function)
            else:
                assert other.spatial.rank == self.shape.channel.volume
                return self._op2(other.spatial.sizes, native_function)
        else:
            other_tensor = tensor(other, infer_dimension_types=False)
            if other_tensor.rank in (0, self.rank):
                result_tensor = native_function(self.native(), other)
            elif other_tensor.rank == self.shape.channel.rank:
                return self._op2(other_tensor, native_function)
            else:
                raise ValueError("Cannot broadcast object of rank %d to tensor with shape %s" % (math.rank(other), self.shape))
            return NativeTensor(result_tensor, self.shape.with_sizes(math.staticshape(result_tensor)))

    def _op1(self, native_function):
        return NativeTensor(native_function(self.native()), self.shape)

    def assert_close(self, *others, rel_tolerance=1e-5, abs_tolerance=0):
        """
        Checks that this tensor and all other tensors have the same values.
        Raises an AssertionError if the values of this tensor are not within tolerance of any of the other tensors.

        Does not check that the shapes exactly match.
        Tensors with different shapes are reshaped before comparing.

        :param others: tensor or tensor-like (constant) each
        :param rel_tolerance: relative tolerance
        :param abs_tolerance: absolute tolerance
        """
        for other in others:
            self._assert_close(other, rel_tolerance=rel_tolerance, abs_tolerance=abs_tolerance)

    def _assert_close(self, other, rel_tolerance=1e-5, abs_tolerance=0):
        if other is self:
            return
        if isinstance(other, (int, float, bool)):
            np.testing.assert_allclose(self.numpy(), other, rel_tolerance, abs_tolerance)
        if not isinstance(other, AbstractTensor):
            other = tensor(other)
        new_shape, (native1, native2) = broadcastable_native_tensors(self, other)
        np1 = math.numpy(native1)
        np2 = math.numpy(native2)
        if not np.allclose(np1, np2, rel_tolerance, abs_tolerance):
            np.testing.assert_allclose(np1, np2, rel_tolerance, abs_tolerance)


class _TensorDim:

    def __init__(self, tensor, name):
        self.tensor = tensor
        self.name = name

    def __str__(self):
        return self.name

    def unstack(self):
        return self.tensor.unstack(self.name)

    @property
    def index(self):
        return self.tensor.shape.index(self.name)

    def __int__(self):
        return self.index

    @property
    def size(self):
        return self.tensor.shape.sizes[self.index]

    def as_batch(self):
        shape = self.tensor.shape
        new_types = list(shape.types)
        new_types[self.index] = BATCH_DIM
        new_shape = Shape(shape.sizes, shape.names, new_types)
        return self.tensor._with_shape_replaced(new_shape)

    def as_spatial(self):
        shape = self.tensor.shape
        new_types = list(shape.types)
        new_types[self.index] = SPATIAL_DIM
        new_shape = Shape(shape.sizes, shape.names, new_types)
        return self.tensor._with_shape_replaced(new_shape)

    @property
    def dim_type(self):
        return self.tensor.shape.types[self.index]

    @property
    def is_spatial(self):
        return self.tensor.shape.types[self.index] == SPATIAL_DIM

    @property
    def is_batch(self):
        return self.tensor.shape.types[self.index] == BATCH_DIM

    @property
    def is_channel(self):
        return self.tensor.shape.types[self.index] == CHANNEL_DIM

    def __getitem__(self, item):
        return self.tensor[{self.name: item}]

    def __setitem__(self, key, value):
        self.tensor[{self.name: key}] = value


class NativeTensor(AbstractTensor):

    def __init__(self, native_tensor, shape):
        assert isinstance(shape, Shape)
        assert len(math.staticshape(native_tensor)) == shape.rank
        self.tensor = native_tensor
        self._shape = shape.with_linear_indices()

    def native(self, order=None):
        if order is None or tuple(order) == self.shape.names:
            return self.tensor
        # --- Insert missing dimensions ---
        tensor = self.tensor
        shape = self.shape
        for name in order:
            if name not in self.shape:
                tensor = math.expand_dims(tensor, axis=-1)
                shape = shape.plus(1, name, CHANNEL_DIM, pos=-1)
        # --- Transpose ---
        perm = shape.perm(order)
        tensor = math.transpose(tensor, perm)
        return tensor

    @property
    def dtype(self):
        return math.dtype(self.tensor)

    @property
    def shape(self):
        return self._shape

    def _with_shape_replaced(self, new_shape):
        return NativeTensor(self.tensor, new_shape)

    def _getitem(self, selection_dict):
        new_shape = self.shape
        selections = [slice(None)] * self.rank
        for name, selection in selection_dict.items():
            selections[self.shape.index(name)] = selection
            if isinstance(selection, int):
                new_shape = new_shape.without(name)
        gathered = self.tensor[tuple(selections)]
        new_shape = new_shape.with_sizes(math.staticshape(gathered))
        return NativeTensor(gathered, new_shape)

    def unstack(self, dimension=0):
        dim_index = self.shape.index(dimension)
        new_shape = self.shape.without(dimension)
        tensors = math.unstack(self.tensor, axis=dim_index)
        return tuple([NativeTensor(t, new_shape) for t in tensors])


class CollapsedTensor(AbstractTensor):
    """
    Tiled / Repeated tensor along additional axes.
    """

    def __init__(self, tensor, shape):
        assert isinstance(tensor, AbstractTensor)
        shape = shape.with_linear_indices()
        for name in tensor.shape.names:
            assert name in shape
        for size, name, dim_type in tensor.shape.dimensions:
            assert shape.get_size(name) == size
            assert shape.get_type(name) == dim_type
        self.tensor = tensor
        self._shape = shape
        self._cached = None

    def _cache(self):
        if self._cached is None:
            native = self.tensor.native(order=self.shape.names)
            multiples = [1 if name in self.tensor.shape else size for size, name, _ in self.shape.dimensions]
            tiled = math.tile(native, multiples)
            self._cached = NativeTensor(tiled, self.shape)
        return self._cached

    def native(self, order=None):
        if order is None or tuple(order) == self.shape.names:
            return self._cache().native()
        else:
            native = self.tensor.native(order=order)
            multiples = [1 if name in self.tensor.shape else size for size, name, _ in self.shape.dimensions]
            tiled = math.tile(native, multiples)
            return tiled

    @property
    def dtype(self):
        return self.tensor.dtype

    @property
    def shape(self):
        return self._shape

    def unstack(self, dimension=0):
        unstacked_shape = self.shape.without(dimension)
        if dimension in self.tensor.shape:
            unstacked = self.tensor.unstack(dimension)
            return tuple(CollapsedTensor(t, unstacked_shape) for t in unstacked)
        else:
            return (CollapsedTensor(self.tensor, unstacked_shape),) * self.shape.get_size(dimension)

    def _with_shape_replaced(self, new_shape):
        return CollapsedTensor(self.tensor, new_shape)

    def _getitem(self, selection_dict):
        inner_dict = {name: selection for name, selection in selection_dict.items() if name in self.tensor.shape}
        inner = self.tensor._getitem(inner_dict)
        new_shape = self.shape.after_gather(selection_dict)
        inner.shape.combined(new_shape)  # check that sizes match
        return CollapsedTensor(inner, new_shape)


class TensorStack(AbstractTensor):
    """
    Implicit stack of multiple tensors.
    List of tensors, does not store stacked tensor in memory.
    """

    def __init__(self, tensors, dim_name, dim_type):
        for tensor in tensors:
            assert isinstance(tensor, AbstractTensor)
            assert tensor.dtype == tensors[0].dtype
            assert tensor.shape == tensors[0].shape
        self.tensors = tuple(tensors)
        self.stack_dim_name = dim_name
        self._shape = tensors[0].shape.plus(len(tensors), dim_name, dim_type, pos=None)
        self._cached = None

    def _cache(self):
        if self._cached is None:
            native = math.stack([t.native() for t in self.tensors], axis=self.shape.index(self.stack_dim_name))
            self._cached = NativeTensor(native, self._shape)
        return self._cached

    @property
    def dtype(self):
        return self.tensors[0].dtype

    @property
    def shape(self):
        return self._shape

    def native(self, order=None):
        if self._cached is not None:
            return self._cached.native(order=order)
        # Is only the stack dimension shifted?
        if order is not None and self._shape.without(self.stack_dim_name).names == tuple(filter(lambda name: name != self.stack_dim_name, order)):
            native = math.stack([t.native() for t in self.tensors], axis=tuple(order).index(self.stack_dim_name))
            return native
        return self._cache().native(order=order)

    def _with_shape_replaced(self, new_shape):
        assert isinstance(new_shape, Shape)
        inner_shape = new_shape.without(self.stack_dim_name)
        tensors = [t._with_shape_replaced(inner_shape) for t in self.tensors]
        return TensorStack(tensors, self.stack_dim_name, new_shape.get_type(self.stack_dim_name))

    def _getitem(self, selection_dict):
        if self.stack_dim_name in selection_dict and len(selection_dict) == 1:
            selection = selection_dict[self.stack_dim_name]
            if isinstance(selection, int):
                return self.tensors[selection]
            elif isinstance(selection, slice):
                return TensorStack(self.tensors[selection], self.stack_dim_name, self.shape.get_type(self.stack_dim_name))
            else:
                raise NotImplementedError()
        else:
            return self._cache()._getitem(selection_dict)

    def unstack(self, dimension=0):
        if dimension == self.stack_dim_name:
            return self.tensors
        else:
            return self._cache().unstack(dimension=dimension)


def tensor(*objects, infer_dimension_types=True, batch_dims=None, spatial_dims=None, channel_dims=None):
    if len(objects) == 1:
        return _tensor(objects[0], infer_dimension_types, batch_dims, spatial_dims, channel_dims)
    else:
        return [_tensor(obj, infer_dimension_types, batch_dims, spatial_dims, channel_dims) for obj in objects]


def _tensor(obj, infer_dimension_types=True, batch_dims=None, spatial_dims=None, channel_dims=None):
    if isinstance(obj, AbstractTensor):
        if batch_dims is not None:
            assert obj.shape.batch.rank == batch_dims
        if spatial_dims is not None:
            assert obj.shape.spatial.rank == spatial_dims
        if channel_dims is not None:
            assert obj.shape.channel.rank == channel_dims
        return obj
    if isinstance(obj, np.ndarray) and obj.dtype != np.object:
        if infer_dimension_types and len(obj.shape) > 1:
            shape = infer_shape(obj.shape, batch_dims, spatial_dims, channel_dims)
            tensor = NativeTensor(obj, shape)
            tensor = _remove_singleton_dimensions(tensor)
            return tensor
        else:
            shape = Shape(obj.shape, names=range(len(obj.shape)), types=[CHANNEL_DIM] * len(obj.shape))
            return NativeTensor(obj, shape)
    if isinstance(obj, (tuple, list)):
        array = np.array(obj)
        if array.dtype != np.object:
            return _tensor(array, infer_dimension_types=False)
        else:
            raise NotImplementedError()
            return TensorStack(tensor(obj), dim_name=None, dim_type=CHANNEL_DIM)
    if isinstance(obj, numbers.Number):
        array = np.array(obj)
        return NativeTensor(array, Shape((), (), ()))
    if isinstance(obj, Shape):
        return _tensor(obj.sizes)
    raise ValueError(obj)


def broadcastable_native_tensors(*tensors):
    """
    Expands and transposes the dimensions of the given tensors so that they all have the same dimension order.

    :param tensors: sequence of AbstractTensors
    :return: (shape, native tensors)
    """
    broadcast_shape = tensors[0].shape
    for tensor in tensors[1:]:
        broadcast_shape = broadcast_shape.combined(tensor.shape)
    natives = [tensor.native(order=broadcast_shape.names) for tensor in tensors]
    return broadcast_shape, natives


def _remove_singleton_dimensions(tensor):
    """
    Remove singleton batch and channel dimensions
    :type tensor: AbstractTensor
    :rtype: AbstractTensor
    """
    for dim in tensor.dimensions:
        if dim.size == 1 and (dim.is_batch or dim.is_channel):
            return _remove_singleton_dimensions(dim[0])  # iter over old dimensions is no longer correct
    return tensor


def shapeof(tensor):
    if isinstance(tensor, AbstractTensor):
        return tensor.shape
    else:
        shape = math.staticshape(tensor)
        return infer_shape(shape)
