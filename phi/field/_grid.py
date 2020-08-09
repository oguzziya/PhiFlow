from abc import ABC

import numpy as np

from phi import math
from phi.geom import AABox, GridCell, Geometry
from phi.geom import assert_same_rank
from phi.struct.functions import mappable
from ._field import Field, SampledField
from phi.math.backend.tensorop import collapse
from phi.math.backend import Extrapolation, general_grid_sample_nd
from ..math import Shape, tensor, AbstractTensor


class Grid(SampledField, ABC):

    def __init__(self, resolution, box, extrapolation=math.extrapolation.ZERO):
        assert isinstance(extrapolation, (Extrapolation, tuple, list)), extrapolation
        self._extrapolation = collapse(extrapolation)
        self._box = AABox.to_box(box, resolution_hint=resolution)

    @property
    def box(self) -> AABox:
        return self._box

    @property
    def resolution(self) -> Shape:
        return self.shape.spatial

    @property
    def dx(self) -> AbstractTensor:
        return self.box.size / self.resolution

    @property
    def extrapolation(self) -> Extrapolation:
        return self._extrapolation

    def __repr__(self):
        return '%s[%s, size=%s, extrapolation=%s]' % (self.__class__.__name__, self.shape, self.box.size, self._extrapolation)


class CenteredGrid(Grid):

    def __init__(self, data, box=None, extrapolation=math.extrapolation.ZERO):
        """
        Create CenteredGrid given its data and dimensions.

        :param data: numerical values to be set as values of CenteredGrid (immutable)
        :type data: array-like
        :param box: numerical values describing the surrounding area of the CenteredGrid, defaults to None
        :type box: domain.box, optional
        :param extrapolation: set conditions for boundaries, defaults to 'boundary'
        :type extrapolation: str, optional
        """
        self._data = tensor(data)
        Grid.__init__(self, self.resolution, box, extrapolation)
        assert_same_rank(self._data.shape, self._box, 'data dimensions %s do not match box %s' % (self._data.shape, self._box))

    @staticmethod
    def sample(value, resolution, box, extrapolation=math.extrapolation.ZERO):
        if isinstance(value, Field):
            elements = GridCell(resolution, box)
            data = value.sample_at(elements)
        else:
            if callable(value):
                x = GridCell(resolution, box).center
                value = value(x)
            value = tensor(value, infer_dimension_types=False)
            data = math.zeros(resolution) + value
        return CenteredGrid(data, box, extrapolation)

    @property
    def data(self):
        return self._data

    def with_data(self, data):
        if isinstance(data, (tuple, list)):
            assert len(data) == 1
            data = data[0]
        return CenteredGrid(data, self._box, self._extrapolation)

    @property
    def shape(self):
        return self._data.shape

    @property
    def elements(self):
        return GridCell(self.resolution, self._box)

    def sample_at(self, points, reduce_channels=()):
        if isinstance(points, (tuple, list)):
            return tuple(self.sample_at(p) for p in points)
        elif isinstance(points, GridCell) and points.bounds == self.box and points.resolution == self.resolution:
            return self._data
        elif isinstance(points, GridCell) and math.close(self.dx, points.size):
            fast_resampled = self._shift_resample(points.resolution, points.bounds)
            if fast_resampled is not NotImplemented:
                return fast_resampled
        elif isinstance(points, Geometry):
            points = points.center
        local_points = self.box.global_to_local(points)
        local_points = local_points * self.resolution - 0.5
        if len(reduce_channels) == 0:
            return math.resample(self.data, local_points, 'linear', self.extrapolation)
        else:
            assert self.shape.channel.sizes == points.shape.get_size(reduce_channels)
            if len(reduce_channels) > 1:
                raise NotImplementedError()
            channels = []
            for i, channel in enumerate(self._data.vector.unstack()):
                channels.append(math.resample(channel, local_points[{reduce_channels[0]: i}], 'linear', self.extrapolation))
            return math.channel_stack(channels, 'vector')

    def _shift_resample(self, resolution, box):
        paddings = _required_paddings_transposed(self.box, self.dx, box)
        if math.sum(paddings) == 0:
            origin_in_local = self.box.global_to_local(box.lower) * self.resolution
            data = math.interpolate_linear(self._data, origin_in_local, resolution.sizes)
            return data
        elif math.sum(paddings) < 16:
            padded = self.padded(np.transpose(paddings).tolist())
            return padded.at(representation)

    def general_sample_at(self, points, reduce):
        local_points = self.box.global_to_local(points)
        local_points = math.mul(local_points, math.to_float(self.resolution)) - 0.5
        result = general_grid_sample_nd(self.data, local_points, boundary=_pad_mode(self.extrapolation), constant_values=_pad_value(self.extrapolation_value), math=math.choose_backend([self.data, points]), reduce=reduce)
        return result

    def compatible(self, other):
        return isinstance(other, CenteredGrid) and other.box == self.box and other.resolution == self.resolution

    def unstack(self, dimension=0):
        components = self.data.unstack(dimension=dimension)
        return [CenteredGrid(component, box=self.box) for i, component in enumerate(components)]

    def __getitem__(self, item):
        return CenteredGrid(self._data[item], self._box, self._extrapolation)

    def _op1(self, operator):
        data = operator(self._data)
        extrapolation_ = operator(self._extrapolation)
        return CenteredGrid(data, self._box, extrapolation_)

    def _op2(self, other, operator):
        if self.compatible(other):
            data = operator(self._data, other.data)
            extrapolation = operator(self.extrapolation, other.extrapolation)
            return CenteredGrid(data, self.box, extrapolation)
        else:
            data = operator(self.data, other)
            return CenteredGrid(data, self.box, self.extrapolation)
        # if isinstance(other, CenteredGrid):
        #     assert self.compatible(other), 'Fields are not compatible: %s and %s' % (self, other)
        #     self_data = self.data if self.has_points else self.at(other).data
        #     other_data = other.data if other.has_points else other.at(self).data
        #     data = data_operator(self_data, other_data)
        # else:
        #     data = data_operator(self.data, other)
        # return self.copied_with(data=data)

    def __neg__(self):
        return CenteredGrid(-self.data, self.box, self.extrapolation)


def _required_paddings_transposed(box, dx, target, threshold=1e-5):
    lower = math.to_int(math.ceil(math.maximum(0, box.lower - target.lower) / dx - threshold))
    upper = math.to_int(math.ceil(math.maximum(0, target.upper - box.upper) / dx - threshold))
    return [lower, upper]


@mappable()
def _gradient_extrapolation(field_extrapolation):
    return field_extrapolation.gradient()



