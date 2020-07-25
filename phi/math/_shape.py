import warnings

import numpy as np

from phi import math


BATCH_DIM = 0
SPATIAL_DIM = 1
CHANNEL_DIM = 2


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
        if name is None:
            return None
        if isinstance(name, (list, tuple)):
            return tuple(self.index(n) for n in name)
        if isinstance(name, Shape):
            return tuple(self.index(n) for n in name._names)
        for dim_name, idx in zip(self._names, self._indices):
            if dim_name == name:
                return idx
        raise ValueError("Shape %s does not contain dimension with name '%s'" % (self, name))

    def get_size(self, name):
        return self._sizes[self.names.index(name)]

    def get_type(self, name):
        return self._types[self.names.index(name)]

    def __getitem__(self, selection):
        if isinstance(selection, int):
            return self._sizes[selection]
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

    @property
    def non_channel(self):
        return self.filtered(self._types != CHANNEL_DIM)

    @property
    def non_spatial(self):
        return self.filtered(self._types != SPATIAL_DIM)

    @property
    def non_batch(self):
        return self.filtered(self._types != BATCH_DIM)

    @property
    def singleton(self):
        return self.filtered(self._sizes == 1)

    @property
    def non_singleton(self):
        return self.filtered(self._sizes != 1)

    def mask(self, names):
        if isinstance(names, (str, int)):
            names = [names]
        mask = [1 if name in names else 0 for name in self._names]
        return np.array(mask)

    def select(self, *names):
        indices = [self.index(name) for name in names]
        return self[indices]

    def __repr__(self):
        strings = ['%s=%s' % (name, size) if isinstance(name, str) else '%d' % size for size, name, type in self.dimensions]
        return '(' + ', '.join(strings) + ')'

    def __str__(self):
        return repr(self)

    def __eq__(self, other):
        if not isinstance(other, Shape):
            return False
        return self.names == other.names and self.types == other.types and self.sizes == other.sizes

    def __ne__(self, other):
        return not self == other

    def normal_order(self):
        sizes = self.batch.sizes + self.spatial.sizes + self.channel.sizes
        names = self.batch.names + self.spatial.names + self.channel.names
        types = self.batch.types + self.spatial.types + self.channel.types
        indices = [self.index(name) for name in names]
        return Shape(sizes, names, types, indices)

    def combined(self, other, allow_inconsistencies=False):
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

        def _check(size, name):
            self_size = self.get_size(name)
            if size != self_size:
                if not allow_inconsistencies:
                    raise IncompatibleShapes(self, other)
                else:
                    sizes[names.index(name)] = None

        for size, name, type in other.batch.dimensions:
            if name not in names:
                names.insert(0, name)
                sizes.insert(0, size)
                types.insert(0, type)
            else:
                _check(size, name)
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
            sizes.extend(self.spatial._sizes)
            names.extend(self.spatial._names)
            types.extend(self.spatial._types)
            if set(self.spatial.names) != set(other.spatial.names):
                raise IncompatibleShapes(self, other)
            for size, name, type in other.spatial.dimensions:
                _check(size, name)
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
            sizes.extend(self.channel._sizes)
            names.extend(self.channel._names)
            types.extend(self.channel._types)
            if set(self.channel.names) != set(other.channel.names):
                raise IncompatibleShapes(self, other)
            for size, name, type in other.channel.dimensions:
                _check(size, name)
        return Shape(sizes, names, types)

    def __and__(self, other):
        return self.combined(other)

    def plus(self, size, name, dim_type, pos=None):
        """
        Add a dimension to the shape.

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

    def without(self, other):
        if isinstance(other, (str, int)):
            return self[np.argwhere(self._names != other)[:, 0]]
        elif isinstance(other, Shape):
            return self[np.argwhere([name not in other._names for name in self._names])[:, 0]]
        elif other is None:
            return EMPTY_SHAPE
        else:
            raise ValueError(other)

    @property
    def rank(self):
        return len(self.sizes)

    def with_sizes(self, sizes):
        return Shape(sizes, self._names, self._types, self._indices)

    def with_size(self, name, size):
        new_sizes = list(self._sizes)
        new_sizes[self.index(name)] = size
        return self.with_sizes(new_sizes)

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
        """
        If sequence is a dict with dimension names as keys, orders its values according to this shape.

        Otherwise, the sequence is returned unchanged.

        :param sequence: sequence or dict to be ordered
        :type sequence: dict or list or tuple
        :param default: default value used for dimensions not contained in sequence
        :return: ordered sequence of values
        """
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
        if math.is_tensor(sequence):
            assert math.staticshape(sequence) == (self.rank,)
            return sequence[self.names.index(name)]
        else:  # just a constant
            return sequence

    def after_gather(self, selection_dict):
        result = self
        for name, selection in selection_dict.items():
            if isinstance(selection, int):
                result = result.without(name)
            elif isinstance(selection, slice):
                assert selection.step is None
                start = selection.start or 0
                stop = selection.stop or self.get_size(name)
                if stop < 0:
                    stop += self.get_size(name)
                result = result.with_size(name, stop - start)
            else:
                raise NotImplementedError()
        return result


EMPTY_SHAPE = Shape((), (), ())


class IncompatibleShapes(ValueError):
    def __init__(self, shape1, shape2):
        ValueError.__init__(self, shape1, shape2)


def define_shape(channels=(), batch=None, infer_types_if_not_given=False, **spatial):
    """

    :param channels: int or (int,)
    :param batch: int or {name: int} or (Dimension,)
    :param infer_types_if_not_given: if True, detects legacy-style shapes, infers the corresponding types and removes singleton dimensions
    :param spatial:
    :return:
    """
    if isinstance(channels, Shape):
        assert batch is None
        assert len(spatial) == 0
        return channels
    if infer_types_if_not_given and batch is None and len(spatial) == 0 and len(channels) >= 3:
        shape = infer_shape(channels)
        return shape.without(shape.non_spatial.singleton)
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


def infer_shape(shape, batch_dims=None, spatial_dims=None, channel_dims=None):
    if isinstance(shape, Shape):
        return shape
    shape = tuple(shape)
    if len(shape) == 0:
        return EMPTY_SHAPE
    # --- Infer dim types ---
    dims = _infer_dim_group_counts(len(shape), constraints=[batch_dims, spatial_dims, channel_dims])
    if dims is None:  # could not infer
        channel_dims = 1
        dims = _infer_dim_group_counts(len(shape), constraints=[batch_dims, spatial_dims, channel_dims])
        if dims is None:
            batch_dims = 1
            dims = _infer_dim_group_counts(len(shape), constraints=[batch_dims, spatial_dims, channel_dims])
    assert dims is not None, "Could not infer shape from '%s' given constraints batch_dims=%s, spatial_dims=%s, channel_dims=%s" % (shape, batch_dims, spatial_dims, channel_dims)
    batch_dims, spatial_dims, channel_dims = dims
    # --- Construct shape ---
    from phi import geom
    types = [BATCH_DIM] * batch_dims + [SPATIAL_DIM] * spatial_dims + [CHANNEL_DIM] * channel_dims
    names = []
    if batch_dims >= 1:
        names.append('batch')
    for i in range(batch_dims - 1):
        warnings.warn("infer_shape() detected more than 1 batch dimension. They will be named 'batch', 'batch2', ...")
        names.append('batch %d' % (i+2,))
    for i in range(spatial_dims):
        names.append(geom.GLOBAL_AXIS_ORDER.axis_name(i, spatial_dims))
    for i in range(channel_dims):
        names.append(i)
    return Shape(sizes=shape, names=names, types=types)


def _infer_dim_group_counts(rank, constraints: list):
    known_sum = sum([dim or 0 for dim in constraints])
    unknown_count = sum([1 if dim is None else 0 for dim in constraints])
    if known_sum == rank:
        return [dim or 0 for dim in constraints]
    if unknown_count == 1:
        return [rank - known_sum if dim is None else dim for dim in constraints]
    return None


def spatial_shape(sizes):
    """
    If `sizes` is a `Shape`, returns the spatial part of it.

    Otherwise, creates a Shape with the given sizes as spatial dimensions.
    The sizes are assumed to be ordered according to the GLOBAL_AXIS_ORDER and the dimensions are named accordingly.

    :param sizes: list of integers or Shape
    :return: Shape containing only spatial dimensions
    """
    if isinstance(sizes, Shape):
        return sizes.spatial
    else:
        return infer_shape(sizes, batch_dims=0, channel_dims=0)


def channel_shape(sizes):
    if isinstance(sizes, Shape):
        return sizes.channel
    else:
        return infer_shape(sizes, batch_dims=0, spatial_dims=0)
