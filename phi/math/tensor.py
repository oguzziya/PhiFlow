from collections import namedtuple
import numpy as np
from .. import math, struct


Dimension = namedtuple('Dimension', ['name', 'size', 'type'])
"""
batch dimensions can be auto-completed

physical dimensions must match or be completely absent

component dimensions can be auto-completed (but only all at the same time)
"""
DIM_TYPES = ('batch', 'physical', 'component')


class Tensor:
    """
    Advantages over regular tensors:
    - Named dimensions
    """

    def numpy(self):
        raise NotImplementedError()

    @property
    def dimensions(self):
        """
        :rtype: tuple
        :return: the ordered dimensions of this tensor
        """
        raise NotImplementedError()

    def shape(self):
        return tuple([dimension.size for dimension in self.dimensions])

    def dimensions_match(self, other):
        return set(self.dimensions) == set(other.dimensions)


class NativeTensor(Tensor):

    def __init__(self, data, dimensions):
        assert len(math.staticshape(data)) == len(dimensions)
        for dim in dimensions:
            assert isinstance(dim, Dimension)
        assert len(set(dimensions)) == len(dimensions)  # no duplicates
        self._data = data
        self._dimensions = tuple(dimensions)

    @property
    def dimensions(self):
        return self._dimensions

    @property
    def numpy(self):
        return math.numpy(self._data)


class Batch(Tensor):

    def __init__(self, tensors, dimension):
        assert isinstance(tensors, (tuple, list, np.ndarray))
        assert len(tensors) > 0
        for tensor in tensors:
            assert isinstance(tensor, Tensor)
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
