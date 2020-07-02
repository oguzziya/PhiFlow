
class Dimension:

    def __init__(self, name, size, dim_type):
        assert isinstance(name, str) or name is None
        assert dim_type in ('batch', 'spatial', 'component')
        self.name = name
        self.size = size
        self.dim_type = dim_type

    def __repr__(self):
        if self.name is not None:
            return '%s=%d' % (self.name, self.size)
        else:
            return '%d' % self.size

    def __eq__(self, other):
        raise TypeError('Dimension does not support == operator')


class Shape:

    def __init__(self, dimensions):
        for dim in dimensions:
            assert isinstance(dim, Dimension)
        named_dimensions = [dim for dim in dimensions if dim.name is not None]
        assert len(set(dim.name for dim in named_dimensions)) == len(named_dimensions)  # no duplicates
        self._dimensions = tuple(dimensions)
        self._dim_dict = {dim.name: dim for dim in dimensions}

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._dimensions[item]
        elif isinstance(item, str):
            return self._dim_dict[item]
        raise ValueError(item)

    def __len__(self):
        return len(self._dimensions)

    def __contains__(self, item):
        return item in self._dimensions

    def __iter__(self):
        return iter(self._dimensions)

    def __repr__(self):
        return repr(self._dimensions)

    def __and__(self, other):
        """
        Returns a Shape object that both `self` and `other` can be broadcast to.
        If `self` and `other` are incompatible, raises a ValueError.
        :param other: Shape
        :return:
        """
        assert isinstance(other, Shape)
        # TODO check that groups match
        # TODO check that dimension sizes match

    @property
    def rank(self):
        return len(self._dimensions)

    @property
    def spatial_rank(self):
        return sum([1 if dim.dim_type == 'spatial' else 0 for dim in self._dimensions])

    @property
    def spatial(self):
        return Shape([dim for dim in self._dimensions if dim.dim_type == 'spatial'])

    @property
    def batch(self):
        return Shape([dim for dim in self._dimensions if dim.dim_type == 'batch'])

    @property
    def component(self):
        return Shape([dim for dim in self._dimensions if dim.dim_type == 'component'])


def define_shape(components=(), batch=(), **spatial):
    """

    :param components: int or (int,)
    :param batch: int or {name: int} or (Dimension,)
    :param dtype:
    :param spatial:
    :return:
    """
    dimensions = []
    if isinstance(batch, (tuple, list)):
        for dim in batch:
            assert isinstance(dim, Dimension)
            assert dim.dim_type == 'batch'
            dimensions.append(dim)
    elif isinstance(batch, int):
        dimensions.append(Dimension(name='batch', size=batch, dim_type='batch'))
    elif isinstance(batch, dict):
        for name, size in batch.items():
            dimensions.append(Dimension(name=name, size=size, dim_type='batch'))
    else:
        raise ValueError(batch)
    for name, size in spatial.items():
        dimensions.append(Dimension(name=name, size=size, dim_type='spatial'))
    if isinstance(components, int):
        dimensions.append(Dimension(name=None, size=components, dim_type='component'))
    else:
        for component in components:
            dimensions.append(Dimension(name=None, size=component, dim_type='component'))
    return Shape(dimensions)
