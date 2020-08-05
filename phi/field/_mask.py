from phi import math
from phi.geom import Geometry
from ._analytic import AnalyticField


class GeometryMask(AnalyticField):
    """

    Antialias:
        If False, field values are either 0 (outside) or 1 (inside) and the field is not differentiable w.r.t. the geometry.
        If True, field values smoothly go from 0 to 1 at the surface and the field is differentiable w.r.t. the geometry.

    """

    def __init__(self, geometry: Geometry):
        self.geometry = geometry

    @property
    def shape(self):
        return self.geometry.shape.non_channel

    def sample_at(self, points, reduce_channels=()):
        if isinstance(points, Geometry):
            return self.geometry.approximate_fraction_inside(points)
        else:
            return math.to_float(self.geometry.lies_inside(points))


mask = GeometryMask
