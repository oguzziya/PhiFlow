import numpy as np

from phi import math, struct
from phi.geom import AABox, GridCell
from phi.struct.tensorop import collapse, collapsed_gather_nd
from phi.field import CenteredGrid, StaggeredGrid
from . import State
from .material import OPEN, Material
from ..math._shape import spatial_shape


class Domain:

    def __init__(self, resolution, boundaries=OPEN, box=None):
        """
        Simulation domain that specifies size and boundary conditions.

        If all boundary surfaces should have the same behaviour, pass a single Material instance.

        To specify the boundary constants_dict per dimension or surface, pass a tuple or list with as many elements as there are spatial dimensions (highest dimension first).
        Each element can either be a Material, specifying the faces perpendicular to that axis, or a pair
        of Material holding (lower_face_material, upper_face_material).

        Examples:

        Domain(grid, OPEN) - all surfaces are open

        DomainBoundary(grid, boundaries=[(SLIPPY, OPEN), SLIPPY]) - creates a 2D domain with an open top and otherwise solid boundaries

        :param resolution: 1D tensor specifying the grid dimensions
        :param boundaries: Material or list of Material/Pair of Material
        :param box: physical size of the domain, box-like
        """
        self.resolution = spatial_shape(resolution)
        self.box = AABox.to_box(box, resolution_hint=self.resolution)
        assert isinstance(boundaries, (Material, list, tuple))
        if isinstance(boundaries, (tuple, list)):
            assert len(boundaries) == self.rank
        self.boundaries = collapse(boundaries)

    @staticmethod
    def as_domain(domain_like):
        assert domain_like is not None
        if isinstance(domain_like, Domain):
            return domain_like
        if isinstance(domain_like, int):
            return Domain([domain_like])
        if isinstance(domain_like, (tuple, list)):
            return Domain(domain_like)
        raise ValueError('Not a valid domain: %s' % domain_like)

    @property
    def dx(self):
        return self.box.size / self.resolution

    def cells(self):
        return GridCell(self.resolution, self.box)

    @property
    def rank(self):
        return len(self.resolution)

    def __repr__(self):
        return '(%s, size=%s)' % (self.resolution, self.box.size)

    def center_points(self):
        return self.cells().center

    def staggered_points(self, dimension):
        idx_zyx = np.meshgrid(*[np.arange(0.5, dim + 1.5, 1) if dim != dimension else np.arange(0, dim + 1, 1) for dim in self.resolution], indexing="ij")
        return math.expand_dims(math.stack(idx_zyx, axis=-1), 0)

    def indices(self):
        """
        Constructs a grid containing the index-location as components.
        Each index denotes the location within the tensor starting from zero.
        Indices are encoded as vectors in the index tensor.

        :param dtype: a numpy data type (default float32)
        :return: an index tensor of shape (1, spatial dimensions..., spatial rank)
        """
        idx_zyx = np.meshgrid(*[range(dim) for dim in self.resolution], indexing="ij")
        return math.expand_dims(np.stack(idx_zyx, axis=-1))

    @staticmethod
    def equal(domain1, domain2):
        assert isinstance(domain1, Domain), 'Not a Domain: %s' % type(domain1)
        assert isinstance(domain2, Domain), 'Not a Domain: %s' % type(domain2)
        return np.all(domain1.resolution == domain2.resolution) and domain1.box == domain2.box

    def centered_grid(self, value):
        return CenteredGrid.sample(value, self.resolution, self.box, Material.extrapolation_mode(self.boundaries))

    def staggered_grid(self, value):
        return StaggeredGrid.sample(value, self.resolution, self.box, Material.vector_extrapolation_mode(self.boundaries))

    def grid(self, value, staggered=False):
        if staggered:
            return self.staggered_grid(value)
        else:
            return self.centered_grid(value)

    def surface_material(self, axis=0, upper_boundary=False):
        return collapsed_gather_nd(self.boundaries, axis, upper_boundary)


def _friction_mask(masks_and_multipliers):
    for mask, multiplier in masks_and_multipliers:
        return mask


def tensor_shape(batch_size, resolution, components):
    return np.concatenate([[batch_size], resolution, [components]])


def _extend1(shape, axis):
    shape = list(shape)
    shape[axis + 1] += 1
    return shape


@struct.definition()
class DomainState(State):

    @struct.constant()
    def domain(self, domain):
        return Domain.as_domain(domain)

    @property
    def resolution(self):
        return self.domain.resolution

    @property
    def rank(self):
        return self.domain.rank

    def centered_grid(self, name, value, components=1, dtype=None):
        extrapolation = Material.extrapolation_mode(self.domain.boundaries)
        return self.domain.centered_grid(value, dtype=dtype, name=name, components=components, batch_size=self._batch_size, extrapolation=extrapolation)

    def staggered_grid(self, name, value, dtype=None):
        extrapolation = Material.vector_extrapolation_mode(self.domain.boundaries)
        return self.domain.staggered_grid(value, dtype=dtype, name=name, batch_size=self._batch_size, extrapolation=extrapolation)
