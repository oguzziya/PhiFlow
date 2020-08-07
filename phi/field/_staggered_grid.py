# coding=utf-8
import numpy as np

from phi import math
from phi.geom import AABox, GLOBAL_AXIS_ORDER, Geometry
from phi.geom import assert_same_rank
from ._field import Field, IncompatibleFieldTypes
from ._grid import CenteredGrid, Grid
from ..geom._box import AbstractBox, GridCell
from ..geom._stack import GeometryStack
from ..math import tensor, Shape
from ..math._shape import CHANNEL_DIM
from ..math._tensors import TensorStack


class StaggeredGrid(Grid):

    def __init__(self, data, box=None, extrapolation=math.extrapolation.ZERO):
        assert isinstance(data, TensorStack)
        self._data = data
        x = self._data[0 if GLOBAL_AXIS_ORDER.is_x_first else -1]
        self._shape = x.shape.with_size('x', x.shape.get_size('x') - 1).expand(x.rank, 0, CHANNEL_DIM)
        Grid.__init__(self, self.resolution, box, extrapolation)
        assert_same_rank(self._data.shape.get_size(0), self.box, 'StaggeredGrid.data does not match box.')

    @staticmethod
    def from_staggered_tensor(staggered_tensor, box, extrapolation=math.extrapolation.ZERO):
        components = unstack_staggered_tensor(staggered_tensor)
        return StaggeredGrid(components, box, extrapolation)

    @staticmethod
    def sample(value, resolution, box, extrapolation=math.extrapolation.ZERO):
        """
        Sampmles the value to a staggered grid.

        :param value: Either constant, staggered tensor, or Field
        :return: Sampled values in staggered grid form matching domain resolution
        :rtype: StaggeredGrid
        """
        if isinstance(value, Field):
            assert_same_rank(value.rank, box.rank, 'rank of value (%s) does not match domain (%s)' % (value.rank, box.rank))
            if isinstance(value, StaggeredGrid) and value.box == box and np.all(value.resolution == resolution):
                return value
            else:
                components = value.unstack(0) if 0 in value.shape else [value] * box.rank
                tensors = []
                for dim, comp in zip(resolution.spatial.names, components):
                    comp_res, comp_box = extend_symmetric(resolution, box, dim)
                    comp_grid = CenteredGrid.sample(comp, comp_res, comp_box, extrapolation)
                    tensors.append(comp_grid.data)
                return StaggeredGrid(math.channel_stack(tensors), box, extrapolation)
        elif callable(value):
            x = CenteredGrid.getpoints(domain.box, domain.resolution).copied_with(extrapolation=Material.extrapolation_mode(domain.boundaries), name=name)
            value = value(x)
            return value
        else:  # value is constant
            tensors = []
            for dim in resolution.spatial.names:
                comp_res, comp_box = extend_symmetric(resolution, box, dim)
                tensors.append(math.zeros(comp_res) + value)
            return StaggeredGrid(math.channel_stack(tensors), box, extrapolation)

    @property
    def data(self):
        return self._data

    def with_data(self, data):
        if isinstance(data, StaggeredGrid):
            return StaggeredGrid(data._data, self._box, self._extrapolation)
        else:
            assert math.is_tensor(data)
            return StaggeredGrid(data, self._box, self._extrapolation)

    @property
    def shape(self):
        return self._shape

    def sample_at(self, points, reduce_channels=()):
        if isinstance(points, Geometry):
            points = points.center
        if len(reduce_channels) == 0:
            channels = [component.sample_at(points) for component in self.unstack()]
        else:
            assert len(reduce_channels) == 1
            points = points.unstack(reduce_channels[0])
            channels = [component.sample_at(p) for p, component in zip(points, self.unstack())]
        return math.channel_stack(channels)

    # def _resample_to(self, representation: Field) -> Field:
    #     if isinstance(representation, StaggeredGrid) and representation.box == self.box and np.allclose(representation.resolution, self.resolution):
    #         return self
    #     points = representation.points
    #     resampled = [centeredgrid.at(representation) for centeredgrid in self.data]
    #     data = math.concat([field.data for field in resampled], -1)
    #     return representation.copied_with(data=data, flags=propagate_flags_resample(self, representation.flags, representation.rank))

    def at_centers(self):
        centered = []
        for grid in self.unstack():
            centered.append(CenteredGrid.sample(grid, self.resolution, self._box, self._extrapolation).data)
        tensor = math.channel_stack(centered)
        return CenteredGrid(tensor, self._box, self._extrapolation)

    def unstack(self, dimension=0):
        if dimension == 0:
            result = []
            for dim, data in zip(self.resolution.spatial.names, self._data.unstack()):
                result.append(CenteredGrid(data, extend_symmetric(self.resolution, self.box, dim)[1], self.extrapolation))
            return tuple(result)
        else:
            raise NotImplementedError()

    @property
    def x(self):
        return self.unstack()[self.resolution.index('x')]

    @property
    def y(self):
        return self.unstack()[self.resolution.index('y')]

    @property
    def z(self):
        return self.unstack()[self.resolution.index('z')]

    @property
    def elements(self):
        grids = [grid.elements for grid in self.unstack()]
        return GeometryStack(grids, 'staggered')

    def __repr__(self):
        return 'StaggeredGrid[%s, size=%s]' % (self.shape, self.box.size.numpy())

    def compatible(self, other_field):
        if not other_field.has_points:
            return True
        if isinstance(other_field, StaggeredGrid):
            return self.box == other_field.box and np.all(self.resolution == other_field.resolution)
        else:
            return False

    def _op1(self, operator):
        data = operator(self._data)
        extrapolation_ = operator(self._extrapolation)
        return StaggeredGrid(data, self._box, extrapolation_)

    def _op2(self, other, operator):
        if isinstance(other, Field):
            if self.resolution == other.resolution and self.box == other.box:
                return self.with_data(operator(self._data, other._data))
            else:
                raise IncompatibleFieldTypes(self, other)
        else:
            return self.with_data(operator(self._data, other))

    def staggered_tensor(self):
        return stack_staggered_components(self._data)
    #
    # def padded(self, widths):
    #     new_grids = [grid.padded(widths) for grid in self.unstack()]
    #     if isinstance(widths, int):
    #         widths = [[widths, widths]] * self.rank
    #     w_lower, w_upper = np.transpose(widths)
    #     box = AABox(self.box.lower - w_lower * self.dx, self.box.upper + w_upper * self.dx)
    #     return self.copied_with(data=new_grids, box=box)
    #
    # def downsample2x(self):
    #     data = []
    #     for axis in range(self.rank):
    #         grid = self.unstack()[axis].data
    #         grid = grid[tuple([slice(None, None, 2) if d - 1 == axis else slice(None) for d in range(self.rank + 2)])]  # Discard odd indices along axis
    #         grid = math.downsample2x(grid, axes=tuple(filter(lambda ax2: ax2 != axis, range(self.rank))))  # Interpolate values along other axes
    #         data.append(grid)
    #     return self.with_data(data)


def unstack_staggered_tensor(tensor):
    tensors = math.unstack(tensor, -1)
    result = []
    for i, dim in enumerate(math.spatial_dimensions(tensor)):
        slices = [slice(None, -1) if d != dim else slice(None) for d in math.spatial_dimensions(tensor)]
        result.append(math.expand_dims(tensors[i][tuple([slice(None)]+slices)], -1))
    return result


def stack_staggered_components(tensors):
    for i, tensor in enumerate(tensors):
        paddings = [[0, 1] if d != i else [0, 0] for d in range(len(tensors))]
        tensors[i] = math.pad(tensor, [[0, 0]] + paddings + [[0, 0]])
    return math.concat(tensors, -1)


def extend_symmetric(resolution: Shape, box: AbstractBox, axis, cells=1):
    axis_mask = np.array(resolution.mask(axis)) * cells
    unit = box.size / resolution * axis_mask
    delta_size = unit / 2
    box = AABox(box.lower - delta_size, box.upper + delta_size)
    ext_res = resolution.sizes + axis_mask
    return resolution.with_sizes(ext_res), box
