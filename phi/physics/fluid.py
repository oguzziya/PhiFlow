"""
Definition of Fluid, IncompressibleFlow as well as fluid-related functions.
"""
import warnings
from functools import partial
from numbers import Number

import numpy as np

from phi import math, struct, field
from phi.geom import union, GridCell
from phi.field import Field, mask, AngularVelocity, CenteredGrid, StaggeredGrid, resample, Grid

from .domain import Domain, DomainState
from .effect import Gravity, effect_applied, gravity_tensor, FieldEffect, FieldPhysics
from .material import OPEN, Material
from .physics import Physics, StateDependency
from .pressuresolver.solver_api import poisson_solve
from . import advect
from ..math._helper import _multi_roll


@struct.definition()
class Fluid(DomainState):
    """
    A Fluid state consists of a density field (centered grid) and a velocity field (staggered grid).
    """

    def __init__(self, domain, density=0.0, velocity=0.0, buoyancy_factor=0.0, tags=('fluid', 'velocityfield', 'velocity'), name='fluid', **kwargs):
        DomainState.__init__(self, **struct.kwargs(locals()))

    def default_physics(self): return INCOMPRESSIBLE_FLOW

    @struct.variable(default=0, dependencies=DomainState.domain)
    def density(self, density):
        """
The marker density is stored in a CenteredGrid with dimensions matching the domain.
It describes the number of particles per physical volume.
        """
        return self.centered_grid('density', density)

    @struct.variable(default=0, dependencies=DomainState.domain)
    def velocity(self, velocity):
        """
The velocity is stored in a StaggeredGrid with dimensions matching the domain.
        """
        return self.staggered_grid('velocity', velocity)

    @struct.constant(default=0.0)
    def buoyancy_factor(self, fac):
        """
The default fluid physics can apply Boussinesq buoyancy as an upward force, proportional to the density.
This force is scaled with the buoyancy_factor (float).
        """
        return fac

    @struct.variable(default={}, holds_data=False)
    def solve_info(self, solve_info):
        return dict(solve_info)

    def __repr__(self):
        return "Fluid[density: %s, velocity: %s]" % (self.density, self.velocity)


def create_smoke(domain, density=0.0, velocity=0.0, buoyancy_factor=0.1):
    velocity_field = StaggeredGrid.sample(velocity, domain, name='velocity')
    velocity_physics = IncompressibleVFlow(domain.boundaries)
    density_field = CenteredGrid.sample(density, domain, name='density')
    density_physics = [Drift(), FieldPhysics('density')]
    buoyancy_state = FieldEffect(None, targets='velocity')
    buoyancy_physics = ProportionalGForce('density', -buoyancy_factor)
    return (velocity_field, density_field, buoyancy_state), (velocity_physics, density_physics, buoyancy_physics)


class IncompressibleFlow(Physics):
    """
Physics modelling the incompressible Navier-Stokes equations.
Supports buoyancy proportional to the marker density.
Supports obstacles, density effects, velocity effects, global gravity.
    """

    def __init__(self, pressure_solver=None, make_input_divfree=False, make_output_divfree=True, conserve_density=True):
        Physics.__init__(self, [StateDependency('obstacles', 'obstacle', blocking=True),
                                StateDependency('gravity', 'gravity', single_state=True),
                                StateDependency('density_effects', 'density_effect', blocking=True),
                                StateDependency('velocity_effects', 'velocity_effect', blocking=True)])
        self.pressure_solver = pressure_solver
        self.make_input_divfree = make_input_divfree
        self.make_output_divfree = make_output_divfree
        self.conserve_density = conserve_density

    def step(self, fluid, dt=1.0, obstacles=(), gravity=Gravity(), density_effects=(), velocity_effects=()):
        # pylint: disable-msg = arguments-differ
        gravity = gravity_tensor(gravity, fluid.rank)
        velocity = fluid.velocity
        density = fluid.density
        if self.make_input_divfree:
            velocity, solve_info = divergence_free(velocity, fluid.domain, obstacles, pressure_solver=self.pressure_solver, return_info=True)
        # --- Advection ---
        density = advect.semi_lagrangian(density, velocity, dt=dt)
        velocity = advected_velocity = advect.semi_lagrangian(velocity, velocity, dt=dt)
        if self.conserve_density and np.all(Material.solid(fluid.domain.boundaries)):
            density = density.normalized(fluid.density)
        # --- Effects ---
        for effect in density_effects:
            density = effect_applied(effect, density, dt)
        for effect in velocity_effects:
            velocity = effect_applied(effect, velocity, dt)
        velocity += (density * -gravity * fluid.buoyancy_factor * dt).at(velocity)
        divergent_velocity = velocity
        # --- Pressure solve ---
        if self.make_output_divfree:
            velocity, solve_info = divergence_free(velocity, fluid.domain, obstacles, pressure_solver=self.pressure_solver, return_info=True)
        solve_info['advected_velocity'] = advected_velocity
        solve_info['divergent_velocity'] = divergent_velocity
        return fluid.copied_with(density=density, velocity=velocity, age=fluid.age + dt, solve_info=solve_info)


class IncompressibleVFlow(Physics):

    def __init__(self, boundaries, pressure_solver=None):
        Physics.__init__(self, dependencies=[
            StateDependency('obstacles', 'obstacle'),
            StateDependency('velocity_effects', 'velocity_effect', blocking=True),
        ])
        self.boundaries =  boundaries
        self.pressure_solver = pressure_solver

    def step(self, velocity, dt=1.0, obstacles=(), velocity_effects=()):
        velocity = advect.semi_lagrangian(velocity, velocity, dt=dt)
        for effect in velocity_effects:  # this is where buoyancy is applied
            velocity = effect_applied(effect, velocity, dt)
        velocity, _solve_info = divergence_free(velocity, Domain(velocity.resolution, self.boundaries, velocity.box), obstacles, pressure_solver=self.pressure_solver, return_info=True)
        return velocity.copied_with(age=velocity.age + dt)


INCOMPRESSIBLE_FLOW = IncompressibleFlow()


class Drift(Physics):
    """
Passive advection with external velocity field.
This Physics requires the world to contain a single velocity field or velocity-carrying state such as Fluid.

This Physics can be applied to all built-in Fields.
The fields will then be advected with the velocity field each time step.
    """

    def __init__(self, use_updated_velocity=False, conserve=True, velocity_field_name='velocity'):
        Physics.__init__(self, dependencies=[StateDependency('velocity', velocity_field_name, single_state=True, blocking=use_updated_velocity)])
        self.conserve = conserve

    def step(self, field, dt=1.0, velocity=None):
        if not isinstance(velocity, Field):
            velocity = velocity.velocity
        advected = advect.advect(field, velocity, dt=dt).copied_with(age=field.age + dt)
        if self.conserve and isinstance(field, (CenteredGrid, StaggeredGrid)) and np.all(~np.char.equal(struct.flatten(field.extrapolation), 'constant')):  # If field has zero extrapolation, it cannot be conserved
            advected = advected.normalized(field)
        return advected


def buoyancy(density, gravity, buoyancy_factor):
    """
Computes the buoyancy force proportional to the density.
    :param density: CenteredGrid
    :param gravity: vector or float
    :param buoyancy_factor: float
    :return: StaggeredGrid for the domain of the density
    """
    warnings.warn('buoyancy() is deprecated. Use (density * -gravity * buoyancy_factor).at(target_grid) instead.', DeprecationWarning)
    if isinstance(gravity, (int, float)):
        gravity = math.to_float(math.as_tensor([gravity] + ([0] * (density.rank - 1))))
    result = StaggeredGrid.from_scalar(density, -gravity * buoyancy_factor)
    return result


class ProportionalGForce(Physics):
    """
Computes a force field proportional to the scalar `source` field that points in the direction of gravity.
A ProportionalGForce object must be accompanied by a FieldEffect state object.
    """
    def __init__(self, source, factor):
        Physics.__init__(self, dependencies=[
            StateDependency('source_field', source, single_state=True, blocking=True),
            StateDependency('gravity', 'gravity', single_state=True)
        ])
        self.factor = factor

    def step(self, effect, dt=1.0, source_field=None, gravity=Gravity()):
        gravity = gravity_tensor(gravity, source_field.rank)
        return effect.copied_with(field=source_field * gravity * self.factor)


def _is_div_free(velocity, is_div_free):
    assert is_div_free in (True, False, None)
    if isinstance(is_div_free, bool):
        return is_div_free
    if isinstance(velocity, Number):
        return True
    return False


def solve_pressure(divergence, fluiddomain, pressure_solver=None, guess=None):
    """
Computes the pressure from the given velocity divergence using the specified solver.
    :param divergence: CenteredGrid
    :param fluiddomain: FluidDomain instance
    :param pressure_solver: PressureSolver to use, None for default
    :param guess: CenteredGrid with same size and resolution as divergence
    :return: pressure field, iteration count
    :rtype: CenteredGrid, int
    """
    return poisson_solve(divergence, fluiddomain, solver=pressure_solver, guess=guess)


def divergence_free(velocity: Grid, domain: Domain, obstacles=(), relative_tolerance: float = 1e-5, absolute_tolerance: float = 0.0, max_iterations: int = 1000, return_info=False, gradient='implicit'):
    """
Projects the given velocity field by solving for and subtracting the pressure.
    :param return_info: if True, returns a dict holding information about the solve as a second object
    :param velocity: StaggeredGrid
    :param domain: Domain matching the velocity field, used for boundary conditions
    :param obstacles: list of Obstacles
    :return: divergence-free velocity as StaggeredGrid
    """
    obstacle_mask = mask(union([obstacle.geometry for obstacle in obstacles]))
    active_mask = 1 - obstacle_mask.sample_at(GridCell(velocity.resolution, velocity.box).center)
    active_mask = CenteredGrid(active_mask, velocity.box, math.extrapolation.ZERO)
    accessible_mask = CenteredGrid(active_mask.data, active_mask.box, Material.accessible_extrapolation_mode(domain.boundaries))
    # --- Boundary Conditions---
    hard_bcs = field.stagger(accessible_mask, math.minimum)
    velocity *= hard_bcs
    for obstacle in obstacles:
        if not obstacle.is_stationary:
            obs_mask = mask(obstacle.geometry)
            angular_velocity = AngularVelocity(location=obstacle.geometry.center, strength=obstacle.angular_velocity, falloff=None)
            velocity = ((1 - obs_mask) * velocity + obs_mask * (angular_velocity + obstacle.velocity)).at(velocity)
    # --- Pressure solve ---
    divergence_field = field.divergence(velocity)
    pressure_guess = domain.grid(0)
    laplace_fun = partial(laplace_pressure, active=active_mask, accessible=accessible_mask)
    converged, pressure, iterations = field.conjugate_gradient(laplace_fun, divergence_field, pressure_guess, relative_tolerance, absolute_tolerance, max_iterations, gradient)
    if not math.all(converged):
        raise ValueError('pressure solve did not converge')
    gradp = field.staggered_gradient(pressure)
    gradp *= hard_bcs
    velocity -= gradp
    return velocity if not return_info else (velocity, {'pressure': pressure, 'iterations': iterations, 'divergence': divergence_field})


def laplace_pressure(pressure: CenteredGrid, active: CenteredGrid, accessible: CenteredGrid):
    extended_active_mask = field.pad(active, 1).data
    extended_fluid_mask = field.pad(accessible, 1).data
    extended_pressure = field.pad(pressure, 1).data
    active_pressure = extended_active_mask * extended_pressure
    by_dim = []
    for dim in pressure.shape.spatial.names:
        lower_active_pressure, upper_active_pressure = _multi_roll(active_pressure, dim, (-1, 1), diminish_others=(1, 1), names=pressure.shape.spatial.names)
        lower_accessible, upper_accessible = _multi_roll(extended_fluid_mask, dim, (-1, 1), diminish_others=(1, 1), names=pressure.shape.spatial.names)
        upper = upper_active_pressure * active.data
        lower = lower_active_pressure * active.data
        center = (- lower_accessible - upper_accessible) * pressure.data
        by_dim.append(center + upper + lower)
    data = math.sum(by_dim, axis=0)
    return CenteredGrid(data, pressure.box, pressure.extrapolation.gradient())
