import warnings

import numpy as np

from phi import math


CHANNEL_DIM = 0
SPATIAL_DIM = 1
BATCH_DIM = 2
UNKNOWN_DIM = -1


class Shape:

    def __init__(self, sizes, names, types, indices=None):
        """

        :param sizes: list of dimension sizes
        :param names: list of dimension names, either strings (spatial, batch) or integers (channel)
        :param types: list of types, all values must be one of (CHANNEL_DIM, SPATIAL_DIM, BATCH_DIM)
        """
        assert len(sizes) == len(names) == len(types) == len(set(names))  # no duplicates
        self._sizes = np.array(sizes, dtype=np.object)
        self._names = np.array(names, dtype=np.object)
        self._types = np.array(types, dtype=np.int8)
        indices = indices if indices is not None else range(len(sizes))
        self._indices = np.array(indices, dtype=np.int8)
        for i, size in enumerate(sizes):
            if isinstance(size, int) and size == 1:
                warnings.warn("Dimension '%s' at index %d of shape %s has size 1. Is this intentional? Singleton dimensions are not supported." % (names[i], i, sizes))

    @property
    def sizes(self):
        return tuple(self._sizes)

    @property
    def names(self):
        return tuple(self._names)

    @property
    def types(self):
        return tuple(self._types)

    @property
    def indices(self):
        return tuple(self._indices)

    def with_linear_indices(self):
        return Shape(self._sizes, self._names, self._types)

    @property
    def named_sizes(self):
        return {name: size for name, size in zip(self._names, self._sizes)}.items()

    @property
    def dimensions(self):
        return zip(self._sizes, self._names, self._types)

    @property
    def indexed_dimensions(self):
        return zip(self._indices, self._sizes, self._names, self._types)

    def __len__(self):
        return len(self.sizes)

    def __contains__(self, item):
        return item in self.names

    def index(self, name):
        for dim_name, idx in zip(self._names, self._indices):
            if dim_name == name:
                return idx
        raise ValueError("Shape %s does not contain dimension with name '%s'" % (self, name))

    def get_size(self, name):
        return self._sizes[self.names.index(name)]

    def get_type(self, name):
        return self._types[self.names.index(name)]

    def __getitem__(self, selection):
        return Shape(self._sizes[selection], self._names[selection], self._types[selection], indices=self._indices[selection])

    def filtered(self, boolean_mask):
        return self[np.argwhere(boolean_mask)[:, 0]]

    @property
    def channel(self):
        return self.filtered(self._types == CHANNEL_DIM)

    @property
    def spatial(self):
        return self.filtered(self._types == SPATIAL_DIM)

    @property
    def batch(self):
        return self.filtered(self._types == BATCH_DIM)

    def __repr__(self):
        strings = ['%s=%d' % (name, size) if isinstance(name, str) else '%d' % size for size, name, type in self.dimensions]
        return '(' + ', '.join(strings) + ')'

    def __eq__(self, other):
        if not isinstance(other, Shape):
            return False
        return self.names == other.names and self.types == other.types and self.sizes == other.sizes

    def combined(self, other):
        """
        Returns a Shape object that both `self` and `other` can be broadcast to.
        If `self` and `other` are incompatible, raises a ValueError.
        :param other: Shape
        :return:
        :raise: ValueError if shapes don't match
        """
        assert isinstance(other, Shape)
        sizes = list(self.batch._sizes)
        names = list(self.batch._names)
        types = list(self.batch._types)
        for size, name, type in other.batch.dimensions:
            if name not in names:
                names.insert(0, name)
                sizes.insert(0, size)
                types.insert(0, type)
            else:
                self_size = self.get_size(name)
                assert size == self_size, 'Incompatible batch dimensions: %s and %s' % (self, other)
        # --- spatial ---
        # spatial dimensions must match exactly or one shape has none
        if self.spatial.rank == 0:
            sizes.extend(other.spatial._sizes)
            names.extend(other.spatial._names)
            types.extend(other.spatial._types)
        elif other.spatial.rank == 0:
            sizes.extend(self.spatial._sizes)
            names.extend(self.spatial._names)
            types.extend(self.spatial._types)
        else:
            assert self.spatial == other.spatial, 'Incompatible spatial dimensions: %s and %s' % (self, other)
            sizes.extend(self.spatial._sizes)
            names.extend(self.spatial._names)
            types.extend(self.spatial._types)
        # --- channel ---
        # channel dimensions must match exactly or one shape has none
        if self.channel.rank == 0:
            sizes.extend(other.channel._sizes)
            names.extend(other.channel._names)
            types.extend(other.channel._types)
        elif other.channel.rank == 0:
            sizes.extend(self.channel._sizes)
            names.extend(self.channel._names)
            types.extend(self.channel._types)
        else:
            assert self.channel == other.channel, 'Incompatible channel dimensions: %s and %s' % (self, other)
            sizes.extend(self.channel._sizes)
            names.extend(self.channel._names)
            types.extend(self.channel._types)
        return Shape(sizes, names, types)

    def plus(self, size, name, dim_type, pos=None):
        """

        The resulting shape has linear indices.

        :param size:
        :param name:
        :param dim_type:
        :param pos:
        :return:
        """
        if pos is None:
            same_type_dims = self.filtered(self._types == dim_type)
            if len(same_type_dims) > 0:
                pos = same_type_dims._indices[0]
            else:
                pos = {BATCH_DIM: 0, SPATIAL_DIM: self.batch.rank+1, CHANNEL_DIM: self.rank + 1}[dim_type]
        elif pos < 0:
            pos += self.rank + 1
        sizes = list(self._sizes)
        names = list(self._names)
        types = list(self._types)
        sizes.insert(pos, size)
        names.insert(pos, name)
        types.insert(pos, dim_type)
        return Shape(sizes, names, types)

    def __sub__(self, other):
        if isinstance(other, (str, int)):
            return self[np.argwhere(self._names != other)[:, 0]]
        elif isinstance(other, Shape):
            return self[np.argwhere([name not in other._names for name in self._names])[:, 0]]
        else:
            raise ValueError(other)

    @property
    def rank(self):
        return len(self.sizes)

    def with_sizes(self, sizes):
        return Shape(sizes, self._names, self._types, self._indices)

    def perm(self, names):
        assert set(names) == set(self._names), 'names must match existing dimensions %s but got %s' % (self.names, names)
        perm = [self.names.index(name) for name in names]
        return perm

    @property
    def volume(self):
        """
        Returns the total number of values contained in a tensor of this shape.
        This is the product of all dimension sizes.
        """
        return math.prod(self._sizes)

    def order(self, sequence, default=None):
        if isinstance(sequence, dict):
            result = [sequence.get(name, default) for name in self._names]
            return result
        if isinstance(sequence, (tuple, list)):
            assert len(sequence) == self.rank
            return sequence
        else:  # just a constant
            return sequence

    def sequence_get(self, sequence, name):
        if isinstance(sequence, dict):
            return sequence[name]
        if isinstance(sequence, (tuple, list)):
            assert len(sequence) == self.rank
            return sequence[self.names.index(name)]
        else:  # just a constant
            return sequence



def define_shape(channels=(), batch=None, **spatial):
    """

    :param channels: int or (int,)
    :param batch: int or {name: int} or (Dimension,)
    :param dtype:
    :param spatial:
    :return:
    """
    sizes = []
    names = []
    types = []
    # --- Batch dimensions ---
    if isinstance(batch, int):
        sizes.append(batch)
        names.append('batch')
        types.append(BATCH_DIM)
    elif isinstance(batch, dict):
        for name, size in batch.items():
            sizes.append(size)
            names.append(name)
            types.append(BATCH_DIM)
    elif batch is None:
        pass
    else:
        raise ValueError(batch)
    # --- Spatial dimensions ---
    for name, size in spatial.items():
        sizes.append(size)
        names.append(name)
        types.append(SPATIAL_DIM)
    # --- Channel dimensions ---
    if isinstance(channels, int):
        sizes.append(channels)
        names.append(0)
        types.append(CHANNEL_DIM)
    else:
        for i, channel in enumerate(channels):
            sizes.append(channel)
            names.append(i)
            types.append(CHANNEL_DIM)
    return Shape(sizes, names, types)


def infer_shape(shape):
    if isinstance(shape, Shape):
        return shape
    shape = tuple(shape)
    assert len(shape) >= 2
    from phi import geom
    names = ['batch'] + [geom.GLOBAL_AXIS_ORDER.axis_name(i, len(shape) - 2) for i in range(len(shape) - 2)] + [0]
    types = [BATCH_DIM] + [SPATIAL_DIM] * (len(shape) - 2) + [CHANNEL_DIM]
    return Shape(sizes=shape, names=names, types=types)
