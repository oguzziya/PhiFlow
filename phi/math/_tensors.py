import numbers
import warnings

import numpy as np

from .. import math
from ._shape import Shape, UNKNOWN_DIM, infer_shape, CHANNEL_DIM


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
        return math.numpy(self.native(order=order))

    @property
    def dtype(self):
        raise NotImplementedError()

    @property
    def shape(self):
        raise NotImplementedError()

    @property
    def rank(self):
        return self.shape.rank

    def __repr__(self):
        if self.shape.volume <= 4:
            content = self.native(order=self.shape.sorted.names)  # TODO self.numpy
            content = list(math.reshape(content, [-1]))
            content = ', '.join([repr(number) for number in content])
            return "[%s, %s: %s]" % (self.dtype, self.shape, content)
        else:
            return "[%s, %s ...]" % (self.dtype, self.shape)

    def __getitem__(self, item):
        raise NotImplementedError()

    def __setitem__(self, key, value):
        """
        All tensors are editable.

        :param key: list/tuple of slices / indices
        :param value:
        :return:
        """
        raise NotImplementedError()

    def unstack(self, dimension=None):
        """
        Splits this tensor along the specified dimension.
        The returned tensors have the same dimensions as this tensor save the unstacked dimension.

        :param dimension: name of dimension or Dimension or None for component dimension
        :type dimension: str or Dimension or _TensorDim
        :return: tuple of tensors
        """
        raise NotImplementedError()

    def dimension(self, name):
        return _TensorDim(self, name)

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


class _TensorDim:

    def __init__(self, tensor, name):
        self.tensor = tensor
        self.name = name

    def __getitem__(self, item):
        return self.tensor[{self.name: item}]

    def __setitem__(self, key, value):
        self.tensor[{self.name: key}] = value

    def unstack(self):
        return self.tensor.unstack(self.name)


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
                shape = shape.plus(1, name, UNKNOWN_DIM)
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
        new_shape = self.shape
        selections = [slice(None)] * self.rank
        for name, selection in item.items():
            selections[self.shape.index(name)] = selection
            if isinstance(selection, int):
                new_shape -= name
        gathered = self.tensor[tuple(selections)]
        new_shape = new_shape.with_sizes(math.staticshape(gathered))
        return NativeTensor(gathered, new_shape)

    def __setitem__(self, key, value):
        pass

    def unstack(self, name=0):
        dim_index = self.shape.index(name)
        new_shape = self.shape - name
        tensors = math.unstack(self.tensor, axis=dim_index)
        return [NativeTensor(t, new_shape) for t in tensors]


class CollapsedTensor(AbstractTensor):

    def __init__(self, tensor, shape):
        assert isinstance(tensor, AbstractTensor)
        for dim in tensor.shape:
            assert dim in shape
        self.tensor = tensor
        self._shape = shape
        self._native = None

    def native(self, order=None):
        reordered_dimensions = [dim for dim in self.shape if dim in self.tensor.shape]
        tensor = transpose(self.tensor, reordered_dimensions)



        # TODO add missing dimensions and tile tensor
        pass

    @property
    def dtype(self):
        return self.tensor.dtype

    @property
    def shape(self):
        return self._shape

    def __getitem__(self, item):
        pass

    def __setitem__(self, key, value):
        pass

    def unstack(self, dimension=None):
        dimension = self.shape[dimension]
        if dimension in self.tensor.shape:
            raise NotImplementedError()
        else:
            unstacked_shape = self.shape - dimension
            return (CollapsedTensor(self.tensor, unstacked_shape),) * dimension.size


class TensorStack(AbstractTensor):

    def __init__(self, tensors, dim_name, dim_type):
        assert isinstance(tensors, (tuple, list, np.ndarray))
        assert len(tensors) > 0
        for tensor in tensors:
            assert isinstance(tensor, AbstractTensor)
        common_shape = tensors[0].shape
        for tensor in tensors[1:]:
            assert tensor.dtype == tensors[0].dtype
            common_shape = common_shape.combined(tensor.shape)
        tensors = [tensor.transpose(common_shape) for tensor in tensors]
        self.tensors = tensors
        self._shape = common_shape.dim_inserted(stack_position, len(tensors), dim_name, dim_type)

    def native(self, order=None):
        np_tensors = [tensor.numpy() for tensor in self.tensors]
        return np.stack(np_tensors, axis=0)

    @property
    def dtype(self):
        return self.tensors[0].dtype

    @property
    def shape(self):
        return self._shape

    def __getitem__(self, item):
        pass

    def __setitem__(self, key, value):
        pass

    def unstack(self, dimension=None):
        pass

    def requires_transpose(self, target_shape):
        pass

    def transpose(self, target_shape):
        pass


def tensor(*objects, infer_dimension_types=True):
    if len(objects) == 1:
        return _tensor(objects[0], infer_dimension_types=infer_dimension_types)
    else:
        return [_tensor(obj, infer_dimension_types=infer_dimension_types) for obj in objects]


def _tensor(obj, infer_dimension_types=True):
    if isinstance(obj, AbstractTensor):
        return obj
    if isinstance(obj, np.ndarray) and obj.dtype != np.object:
        if infer_dimension_types:
            shape = infer_shape(obj.shape)
            tensor = NativeTensor(obj, shape)
            tensor = _remove_singleton_batch_dimensions(tensor)
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
    raise ValueError(obj)


def broadcastable_native_tensors(*tensors):
    broadcast_shape = tensors[0].shape
    for tensor in tensors[1:]:
        broadcast_shape = broadcast_shape.combined(tensor.shape)
    natives = [tensor.native(order=broadcast_shape.names) for tensor in tensors]
    return broadcast_shape, natives


def _remove_singleton_batch_dimensions(tensor):
    for i, size, name, _ in tensor.shape.batch.indexed_dimensions:  # remove singleton batch dimensions
        if size == 1:
            tensor = tensor.dimension(name)[0]
    return tensor
