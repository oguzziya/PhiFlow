import warnings

import numpy as np

from phi import math, struct
from phi.math.backend.tensorop import collapse, collapsed_gather_nd
from phi.math import spatial_shape
from phi.geom import AABox, GridCell
from phi.field import CenteredGrid, StaggeredGrid
from . import State
from .material import OPEN, Material


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

    @property
    def cells(self):
        return GridCell(self.resolution, self.box)

    @property
    def rank(self):
        return len(self.resolution)

    def __repr__(self):
        return '(%s, size=%s)' % (self.resolution, self.box.size)

    def center_points(self):
        return self.cells.center

    @staticmethod
    def equal(domain1, domain2):
        assert isinstance(domain1, Domain), 'Not a Domain: %s' % type(domain1)
        assert isinstance(domain2, Domain), 'Not a Domain: %s' % type(domain2)
        return np.all(domain1.resolution == domain2.resolution) and domain1.box == domain2.box

    def grid(self, value, type:type = CenteredGrid, extrapolation=Material.extrapolation_mode):
        """
        Creates a grid matching the domain by sampling the given value.

        This method uses Material.extrapolation_mode of the domain's boundaries.

        :param value: Field or tensor or tensor function
        :param type: class of Grid to create, must be either CenteredGrid or StaggeredGrid
        :param extrapolation: function: Material -> Extrapolation
        :return: Grid of specified type
        """
        if type is CenteredGrid:
            return CenteredGrid.sample(value, self.resolution, self.box, extrapolation(self.boundaries))
        elif type is StaggeredGrid:
            return StaggeredGrid.sample(value, self.resolution, self.box, extrapolation(self.boundaries))
        else:
            raise ValueError('Unknown grid type: %s' % type)

    def vec_grid(self, value, type:type = CenteredGrid, extrapolation=Material.vector_extrapolation_mode):
        """
        Creates a vector grid matching the domain by sampling the given value.

        This method uses Material.vector_extrapolation_mode of the domain's boundaries.

        :param value: Field or tensor or tensor function
        :param type: class of Grid to create, must be either CenteredGrid or StaggeredGrid
        :param extrapolation: function: Material -> Extrapolation
        :return: Grid of specified type
        """
        if type is CenteredGrid:
            grid = CenteredGrid.sample(value, self.resolution, self.box, extrapolation(self.boundaries))
            if grid.shape.channel.rank == 0:
                grid = grid.with_data(math.expand_channel(grid.data, self.rank, 0))
            else:
                assert grid.shape.channel.sizes[0] == self.rank
            return grid
        elif type is StaggeredGrid:
            return StaggeredGrid.sample(value, self.resolution, self.box, extrapolation(self.boundaries))
        else:
            raise ValueError('Unknown grid type: %s' % type)

    def surface_material(self, axis=0, upper_boundary=False):
        return collapsed_gather_nd(self.boundaries, axis, upper_boundary)


@struct.definition()
class DomainState(State):

    @struct.constant()
    def domain(self, domain: Domain) -> Domain:
        return Domain.as_domain(domain)

    @property
    def resolution(self):
        return self.domain.resolution

    @property
    def rank(self):
        return self.domain.rank

    def centered_grid(self, name, value, components=1, dtype=None):
        warnings.warn("DomainState.centered_grid() is deprecated. The arguments 'name, components, dtype' were ignored.", DeprecationWarning)
        return self.domain.grid(value, CenteredGrid)

    def staggered_grid(self, name, value, dtype=None):
        warnings.warn("DomainState.staggered_grid() is deprecated. The arguments 'name, components, dtype' were ignored.", DeprecationWarning)
        return self.domain.vec_grid(value, StaggeredGrid)
