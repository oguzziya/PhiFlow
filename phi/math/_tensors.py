from collections import namedtuple
import numpy as np

from .. import math
from ._shape import Shape


class AbstractTensor:
    """
    Tensors with grouped and named dimensions.

    All tensors are editable.

    The internal data representation of a tensor can change, even without being edited.
    """

    def native(self):
        raise NotImplementedError()

    def numpy(self):
        return math.numpy(self.native())

    @property
    def dtype(self):
        raise NotImplementedError()

    @property
    def shape(self):
        raise NotImplementedError()

    def __repr__(self):
        return "%s, %s" % (self.dtype, self.shape)

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

    def requires_transpose(self, target_shape):
        """
        Used to decide which shape dominates in tensor-tensor operations.
        :param target_shape:
        :return:
        """
        raise NotImplementedError()


class NativeTensor(AbstractTensor):

    def __init__(self, native_tensor, shape):
        assert isinstance(shape, Shape)
        assert len(math.staticshape(native_tensor)) == shape.rank
        self.tensor = native_tensor
        self._shape = shape

    def native(self):
        return self.tensor

    @property
    def dtype(self):
        return math.dtype(self.tensor)

    @property
    def shape(self):
        return self._shape


class CollapsedTensor(AbstractTensor):

    def __init__(self, tensor, shape):
        assert isinstance(tensor, AbstractTensor)
        for dim in tensor.shape:
            assert dim in shape
        self.tensor = tensor
        self._shape = shape
        self._native = None

    def native(self):
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


class TensorStack(AbstractTensor):

    def __init__(self, tensors, dimension):
        assert isinstance(tensors, (tuple, list, np.ndarray))
        assert len(tensors) > 0
        for tensor in tensors:
            assert isinstance(tensor, AbstractTensor)
            assert tensor.dimensions_match(tensors[0])
        self.tensors = tensors
        self.dimension = dimension

    @property
    def dimensions(self):
        return (self.dimension,) + self.tensors[0].dimensions

    @property
    def numpy(self):
        np_tensors = [tensor.numpy() for tensor in self.tensors]
        return np.stack(np_tensors, axis=0)
