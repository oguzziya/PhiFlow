from phi import math
from phi.math import Shape
from phi.geom import Geometry


class Field:

    @property
    def elements(self):
        """
        Returns the geometry of all cells/particles/elements at which this Field is sampled.
        If the components of this field are sampled at different locations, this method raises StaggeredSamplePoints.
        If this field has no sample points, this method returns None.
        :rtype: Geometry
        :return: all sample elements
        """
        raise NotImplementedError(self)

    @property
    def shape(self):
        """
        Returns a shape with the following properties
        * The spatial dimensions match the dimensions of this Field
        * The batch dimensions match the batch dimensions of this Field
        * The channel dimensions match the channels of this Field
        :rtype: Shape
        """
        raise NotImplementedError()

    def sample_at(self, points):
        """
        Samples this field at the given points.
        :param points: tensor or rank >= 2 containing world-space vectors
        :return: tensor of shape (*location.shape[:-1], field.component_count)
        """
        raise NotImplementedError(self)

    def approximate_mean_value_in(self, geometry):
        """
        Computes the (approximate) mean field value inside the region specified by `geometry`.
        The geometry is assumed to be small compared to the structure of this field and may be approximated as a simpler shape.

        Let V be the volume of `geometry`. This method approximates `1/V * integral_V field.sample_at(x) dx`.
        :param geometry: (batched) Geometry
        :type geometry: Geometry
        :return: float tensor of shape (*geometry.batch_dimensions, field.component_count)
        """
        assert isinstance(geometry, Geometry)
        return self.sample_at(geometry.center)

    def at(self, other_field):
        """
        Resample this field at the same points as other_field.
        The returned Field is compatible with other_field.
        :param other_field: Field
        :return: a new Field which samples all components of this field at the points of other_field
        """
        try:
            values = self.approximate_mean_value_in(other_field.elements)
            result = other_field.copied_with(data=values)
            return result
        except StaggeredSamplePoints:  # other_field is staggered
            return broadcast_at(self, other_field)

    @property
    def rank(self):
        """
        Spatial rank of the field (1 for 1D, 2 for 2D, 3 for 3D).
        This is equal to the spatial rank of the `data`.
        :return: int
        """
        return self.shape.spatial.rank

    def unstack(self, dimension=0):
        """
        Split the Field by components.
        If the field only has one component, returns a list containing itself.
        :return: tuple of Fields
        :rtype: tuple
        """
        raise NotImplementedError()

    @property
    def has_points(self):
        try:
            return self.points is not None
        except StaggeredSamplePoints:
            return True

    def __mul__(self, other):
        return self.__dataop__(other, True, lambda d1, d2: math.mul(d1, d2))

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self.__dataop__(other, True, lambda d1, d2: math.div(d1, d2))

    def __rtruediv__(self, other):
        return self.__dataop__(other, False, lambda d1, d2: math.div(d2, d1))

    def __sub__(self, other):
        return self.__dataop__(other, False, lambda d1, d2: math.sub(d1, d2))

    def __rsub__(self, other):
        return self.__dataop__(other, False, lambda d1, d2: math.sub(d2, d1))

    def __add__(self, other):
        return self.__dataop__(other, False, lambda d1, d2: math.add(d1, d2))

    __radd__ = __add__

    def __pow__(self, power, modulo=None):
        return self.__dataop__(power, False, lambda f, p: math.pow(f, p))

    def __neg__(self):
        return self * -1

    def __gt__(self, other):
        return self.__dataop__(other, False, lambda x, y: x > y)

    def __ge__(self, other):
        return self.__dataop__(other, False, lambda x, y: x >= y)

    def __lt__(self, other):
        return self.__dataop__(other, False, lambda x, y: x < y)

    def __le__(self, other):
        return self.__dataop__(other, False, lambda x, y: x <= y)

    def __dataop__(self, other, linear_if_scalar, data_operator):
        if isinstance(other, Field):
            assert self.compatible(other), 'Fields are not compatible: %s and %s' % (self, other)
            self_data = self.data if self.has_points else self.at(other).data
            other_data = other.data if other.has_points else other.at(self).data
            data = data_operator(self_data, other_data)
        else:
            data = data_operator(self.data, other)
        return self.copied_with(data=data)

    def default_physics(self):
        from phi.physics.effect import FieldPhysics
        return FieldPhysics(self.name)


class StaggeredSamplePoints(Exception):

    def __init__(self, *args):
        Exception.__init__(self, *args)


class IncompatibleFieldTypes(Exception):
    def __init__(self, *args):
        Exception.__init__(self, *args)


def broadcast_at(field1, field2):
    if field1.component_count != field2.component_count and field1.component_count != 1:
        raise IncompatibleFieldTypes('Can only resample to staggered fields with same number of components.\n%s\n%s' % (field1, field2))
    if field1.component_count == 1:
        new_components = [field1.at(f2) for f2 in field2.unstack()]
    else:
        new_components = [f1.at(f2) for f1, f2 in zip(field1.unstack(), field2.unstack())]
    return field2.copied_with(data=tuple(new_components))
