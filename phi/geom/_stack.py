from ._geom import Geometry
from ..math._shape import BATCH_DIM, EMPTY_SHAPE
from ..math._tensors import TensorStack


class GeometryStack(Geometry):

    def __init__(self, geometries, dim_name):
        self._shape = EMPTY_SHAPE
        for geometry in geometries:
            assert isinstance(geometry, Geometry)
            self._shape = self._shape.combined(geometry.shape, allow_inconsistencies=True)
        self._shape = self._shape.expand(len(geometries), dim_name, BATCH_DIM, pos=0)
        self.geometries = tuple(geometries)
        self.stack_dim_name = dim_name

    @property
    def center(self):
        centers = [g.center for g in self.geometries]
        return TensorStack(centers, self.stack_dim_name, BATCH_DIM, keep_separate=True)

    @property
    def shape(self):
        return self._shape

    def lies_inside(self, location):
        raise NotImplementedError()

    def approximate_signed_distance(self, location):
        raise NotImplementedError()

    def bounding_radius(self):
        raise NotImplementedError()

    def bounding_half_extent(self):
        raise NotImplementedError()

    def shifted(self, delta):
        raise NotImplementedError()

    def rotated(self, angle):
        raise NotImplementedError()

