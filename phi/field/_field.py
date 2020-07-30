from __future__ import annotations

from phi.geom import Geometry
from phi.math import Shape


class Field:

    @property
    def elements(self) -> Geometry:
        """
        Returns a geometrical representation of the discretized volume elements.
        The result is a tuple of Geometry objects, each of which can have additional spatial (but not batch) dimensions.

        For grids, the geometries are boxes while particle fields may be represented as spheres.

        If this Field has no discrete points, this method returns an empty geometry.

        :return: Geometry with all batch/spatial dimensions of this Field. Staggered sample points are modelled using extra batch dimensions.
        """
        raise NotImplementedError(self)

    @property
    def points(self):
        return self.elements.center

    @property
    def shape(self) -> Shape:
        """
        Returns a shape with the following properties

        * The spatial dimension names match the dimensions of this Field
        * The batch dimensions match the batch dimensions of this Field
        * The channel dimensions match the channels of this Field
        """
        raise NotImplementedError()

    @property
    def rank(self) -> int:
        """
        Spatial rank of the field (1 for 1D, 2 for 2D, 3 for 3D).
        This is equal to the spatial rank of the `data`.
        """
        return self.shape.spatial.rank

    def sample_at(self, points, reduce_channels=()):
        """
        Sample this field at the world-space locations (in physical units) given by points.

        Points must be one of the following:

        * **Tensor** with exactly one channel dimension.
          The channel dimension holds the vectors that reference the locations in world-space.
          Batch dimensions are matched with the batch dimensions of this Field.
          Spatial dimensions can be used to sample a grid of locations.

        * **Geometry**. Approximates the mean field value inside the volume.
          For small volumes, the value at the volume's center may be sampled.
          The batch dimensions of the geometry are matched with this Field.
          Spatial dimensions can be used to sample a grid of geometries.

        * **List** or **tuple** of any of these. This broadcasts the sampling for all entries in the list.
          The result will have the same (nested) structure.

        :param points: world-space locations
        :param reduce_channels: batch dimensions to be reduced against channel dimensions. Indicates that the different channels of this field should be sampled at different locations.
        :return: object of same kind as points
        """
        # * **Field**. The values of that field are interpreted as the sample locations. Analytic fields cannot be used.
        raise NotImplementedError(self)

    def _resample_to(self, representation: Field) -> Field:
        return NotImplemented

    def _resample_from(self, data: Field) -> Field:
        return NotImplemented

    # def resample(self, other: Field) -> Field:
    #     """
    #     Changes the underlying data structure of this Field to one that matches the other.
    #     This typically involves interpolation.
    #
    #     Note that the values of other are ignored by this method, only its sample points are of concern.
    #
    #     This method differs from sample_at(other.elements) in that not all components may be sampled at all points.
    #
    #     :param other: Field with discrete data structure (other.elements must not be None)
    #     :return: Field of same type as other
    #     """
    #     raise NotImplementedError(self)

    def unstack(self, dimension=0) -> tuple:
        """
        Unstack the field along one of its dimensions.
        The dimension can be batch, spatial or channel.

        :param dimension: name of the dimension to unstack, must be part of `self.shape`
        :return: tuple of Fields
        """
        raise NotImplementedError()

    def __mul__(self, other):
        return self._op2(other, lambda d1, d2: d1 * d2)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self._op2(other, lambda d1, d2: d1 / d2)

    def __rtruediv__(self, other):
        return self._op2(other, lambda d1, d2: d2 / d1)

    def __sub__(self, other):
        return self._op2(other, lambda d1, d2: d1 - d2)

    def __rsub__(self, other):
        return self._op2(other, lambda d1, d2: d2 - d1)

    def __add__(self, other):
        return self._op2(other, lambda d1, d2: d1 + d2)

    __radd__ = __add__

    def __pow__(self, power, modulo=None):
        return self._op2(power, lambda f, p: f ** p)

    def __neg__(self):
        return self._op1(lambda x: -x)

    def __gt__(self, other):
        return self._op2(other, lambda x, y: x > y)

    def __ge__(self, other):
        return self._op2(other, lambda x, y: x >= y)

    def __lt__(self, other):
        return self._op2(other, lambda x, y: x < y)

    def __le__(self, other):
        return self._op2(other, lambda x, y: x <= y)

    def _op1(self, operator):
        """
        Perform an operation on the data of this field.

        :param operator: function that accepts tensors and extrapolations and returns objects of the same type and dimensions
        :return: Field of same type
        """
        raise NotImplementedError()

    def _op2(self, other, operator):
        raise NotImplementedError()


class IncompatibleFieldTypes(Exception):
    def __init__(self, *args):
        Exception.__init__(self, *args)


def resample(data: Field, representation: Field):
    result = data._resample_to(representation)
    if result != NotImplemented:
        return result
    result = representation._resample_from(data)
    if result != NotImplemented:
        return result
    elements = representation.elements
    resampled = data.sample_at(elements, reduce_channels=elements.shape.non_channel.without(representation.shape).names)
    return representation.with_data(resampled)
