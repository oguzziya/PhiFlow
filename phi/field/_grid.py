import numpy as np
import six

from phi import math
from phi.backend.backend_helper import general_grid_sample_nd
from phi.geom import AABox, box
from phi.geom.geometry import assert_same_rank
from phi.physics.domain import Domain
from phi.physics.material import Material
from phi.struct.functions import mappable
from phi.struct.tensorop import collapse

from ._field import Field
from ..math import Shape
from ..math._shape import CHANNEL_DIM
from ..math._tensors import tensor, NativeTensor, TensorStack


def _crop_for_interpolation(data, offset_float, window_resolution):
    offset = math.to_int(offset_float)
    slices = [slice(o, o + res + 1) for o, res in zip(offset, window_resolution)]
    data = data[tuple([slice(None)] + slices + [slice(None)])]
    return data


class CenteredGrid(Field):

    def __init__(self, data, box=None, extrapolation='boundary', extrapolation_value=0):
        """Create new CenteredGrid from array like data

        :param data: numerical values to be set as values of CenteredGrid (immutable)
        :type data: array-like
        :param box: numerical values describing the surrounding area of the CenteredGrid, defaults to None
        :type box: domain.box, optional
        :param extrapolation: set conditions for boundaries, defaults to 'boundary'
        :type extrapolation: str, optional
        """
        assert extrapolation in ('periodic', 'constant', 'boundary') or isinstance(extrapolation, (tuple, list)), extrapolation
        self._data = tensor(data)
        self._extrapolation = collapse(extrapolation)
        self._extrapolation_value = collapse(extrapolation_value)
        self._box = AABox.to_box(box, resolution_hint=self.resolution)

    @staticmethod
    def sample(value, domain, batch_size=None, name=None):
        assert isinstance(domain, Domain)
        if isinstance(value, Field):
            assert_same_rank(value.rank, domain.rank, 'rank of value (%s) does not match domain (%s)' % (value.rank, domain.rank))
            if isinstance(value, CenteredGrid) and value.box == domain.box and np.all(value.resolution == domain.resolution):
                data = value.data
            else:
                point_field = CenteredGrid.getpoints(domain.box, domain.resolution)
                point_field._batch_size = batch_size
                data = value.at(point_field).data
        else:  # value is constant
            if callable(value):
                x = CenteredGrid.getpoints(domain.box, domain.resolution).copied_with(extrapolation=Material.extrapolation_mode(domain.boundaries), name=name)
                value = value(x)
                return value
            components = math.staticshape(value)[-1] if math.ndims(value) > 0 else 1
            data = math.add(math.zeros((batch_size,) + tuple(domain.resolution) + (components,)), value)
        return CenteredGrid(data, box=domain.box, extrapolation=Material.extrapolation_mode(domain.boundaries))

    @property
    def data(self):
        return self._data

    @property
    def box(self):
        return self._box

    @property
    def resolution(self):
        return self.data.shape.spatial

    @property
    def dx(self):
        return self.box.size / self.resolution

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def extrapolation(self):
        return self._extrapolation

    @property
    def extrapolation_value(self):
        return self._extrapolation_value

    def sample_at(self, points):
        local_points = self.box.global_to_local(points)
        local_points = math.mul(local_points, math.to_float(self.resolution)) - 0.5
        resampled = math.resample(self.data, local_points, boundary=_pad_mode(self.extrapolation), interpolation=self.interpolation, constant_values=_pad_value(self.extrapolation_value))
        return resampled

    def general_sample_at(self, points, reduce):
        local_points = self.box.global_to_local(points)
        local_points = math.mul(local_points, math.to_float(self.resolution)) - 0.5
        result = general_grid_sample_nd(self.data, local_points, boundary=_pad_mode(self.extrapolation), constant_values=_pad_value(self.extrapolation_value), math=math.choose_backend([self.data, points]), reduce=reduce)
        return result

    def at(self, other_field):
        if self.compatible(other_field):
            return self
        if isinstance(other_field, CenteredGrid) and np.allclose(self.dx, other_field.dx):
            paddings = _required_paddings_transposed(self.box, self.dx, other_field.box)
            if math.sum(paddings) == 0:
                origin_in_local = self.box.global_to_local(other_field.box.lower) * self.resolution
                data = _crop_for_interpolation(self.data, origin_in_local, other_field.resolution)
                dimensions = self.resolution != other_field.resolution
                dimensions = [d for d in math.spatial_dimensions(data) if dimensions[d - 1]]
                data = math.interpolate_linear(data, origin_in_local % 1.0, dimensions)
                return CenteredGrid(data, other_field.box)
            elif math.sum(paddings) < 16:
                padded = self.padded(np.transpose(paddings).tolist())
                return padded.at(other_field)
        return Field.at(self, other_field)

    def unstack(self, dimension=0):
        components = self.data.unstack(dimension=dimension)
        return [CenteredGrid(component, box=self.box) for i, component in enumerate(components)]

    @property
    def elements(self):
        idx_zyx = np.meshgrid(*[np.linspace(0.5 / dim, 1 - 0.5 / dim, dim) for dim in self.resolution.sizes], indexing="ij")
        idx = [NativeTensor(t, self.resolution) for t in idx_zyx]
        local_coords = TensorStack(idx, 0, CHANNEL_DIM)
        points = self.box.local_to_global(local_coords)
        return box(center=points, size=self.dx)

    def __repr__(self):
        return 'Grid[%s(%d), size=%s, %s]' % ('x'.join([str(r) for r in self.resolution]), self.component_count, self.box.size, self.dtype.data)

    # def padded(self, widths):
    #     if isinstance(widths, int):
    #         widths = [[widths, widths]] * self.rank
    #     data = math.pad(self.data, [[0, 0]] + widths + [[0, 0]], _pad_mode(self.extrapolation), constant_values=_pad_value(self.extrapolation_value))
    #     w_lower, w_upper = np.transpose(widths)
    #     box = AABox(self.box.lower - w_lower * self.dx, self.box.upper + w_upper * self.dx)
    #     return self.copied_with(data=data, box=box)
    #
    # def axis_padded(self, axis, lower, upper):
    #     widths = [[lower, upper] if ax == axis else [0,0] for ax in range(self.rank)]
    #     return self.padded(widths)

    # def laplace(self, physical_units=True, axes=None):
    #     if not physical_units:
    #         data = math.laplace(self.data, padding=_pad_mode(self.extrapolation), axes=axes)
    #     else:
    #         if not self.has_cubic_cells:
    #             raise NotImplementedError('Only cubic cells supported.')
    #         laplace = math.laplace(self.data, padding=_pad_mode(self.extrapolation), axes=axes)
    #         data = laplace / self.dx[0] ** 2
    #     extrapolation = map_for_axes(_gradient_extrapolation, self.extrapolation, axes, self.rank)
    #     return self.copied_with(data=data, extrapolation=extrapolation, flags=())
    #
    # def gradient(self, physical_units=True, difference='central'):
    #     if not physical_units or self.has_cubic_cells:
    #         data = math.gradient(self.data, dx=np.mean(self.dx), difference=difference, padding=_pad_mode(self.extrapolation))
    #         return self.copied_with(data=data, extrapolation=_gradient_extrapolation(self.extrapolation), flags=())
    #     else:
    #         raise NotImplementedError('Only cubic cells supported.')


def _required_paddings_transposed(box, dx, target, threshold=1e-5):
    lower = math.to_int(math.ceil(math.maximum(0, box.lower - target.lower) / dx - threshold))
    upper = math.to_int(math.ceil(math.maximum(0, target.upper - box.upper) / dx - threshold))
    return [lower, upper]


def _pad_mode(extrapolation):
    """ Inserts 'constant' padding for batch dimension and channel dimension. """
    if isinstance(extrapolation, six.string_types):
        return _pad_mode_str(extrapolation)
    else:
        return _pad_mode_str(['constant'] + list(extrapolation) + ['constant'])


def _pad_value(value):
    if math.is_tensor(value):
        return value
    else:
        return [0] + list(value) + [0]


@mappable()
def _pad_mode_str(extrapolation):
    """
Converts an extrapolation string (or struct of strings) to a string that can be passed to math functions like math.pad or math.resample.
    :param extrapolation: field extrapolation
    :return: padding mode, same type as extrapolation
    """
    return {'periodic': 'circular',
            'boundary': 'replicate',
            'constant': 'constant'}[extrapolation]


@mappable()
def _gradient_extrapolation(extrapolation):
    """
Given the extrapolation of a field, returns the extrapolation mode of the corresponding gradient field.
    :param extrapolation: string or struct of strings
    :return: same type as extrapolation
    """
    return {'periodic': 'periodic',
            'boundary': 'constant',
            'constant': 'constant'}[extrapolation]
