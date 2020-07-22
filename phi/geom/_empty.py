import numpy as np

from ._geom import Geometry
from ..math import Shape


class _NoGeometry(Geometry):

    @property
    def shape(self):
        return Shape((), (), ())

    @property
    def center(self):
        return 0

    def bounding_radius(self):
        return 0

    def bounding_half_extent(self):
        return 0

    def rank(self):
        return None

    def approximate_signed_distance(self, location):
        return np.inf

    def lies_inside(self, location):
        return False

    def approximate_fraction_inside(self, other_geometry):
        return 0

    def shifted(self, delta):
        return self

    def rotated(self, angle):
        return self


NO_GEOMETRY = _NoGeometry()
