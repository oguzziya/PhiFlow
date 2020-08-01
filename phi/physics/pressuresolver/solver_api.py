# coding=utf-8
from phi import math
from phi import field
from phi.backend import ConstantExtrapolation
from phi.field import CenteredGrid, Field
from phi.struct.functions import mappable


class PoissonSolver(object):
    """
    Base class for Poisson solvers
    """

    def __init__(self, name, supported_devices, supports_guess, supports_loop_counter, supports_continuous_masks):
        """Assign details such as name, supported device (CPU/GPU), etc."""
        self.name = name
        self.supported_devices = supported_devices
        self.supports_guess = supports_guess
        self.supports_loop_counter = supports_loop_counter
        self.supports_continuous_masks = supports_continuous_masks

    def solve(self, y: Field, active, accessible, x0, enable_backprop) -> tuple:
        """
        Solves the Poisson equation Δp = d for p for all active fluid cells where active cells are given by the active_mask.
        p is expected to fulfill (Δp-d) ≤ accuracy for every active cell.

        :param enable_backprop: Whether automatic differentiation should be enabled. If False, the gradient of this operation is undefined.
        :param y: scalar input field to the solve, e.g. the divergence of the velocity channel, ∇·v
        :param active: Scalar field encoding active cells as ones and inactive (open/obstacle) as zero.
        Active cells are those for which physical constants_dict such as pressure or velocity are calculated.
        :param accessible: Scalar field encoding cells that are accessible, i.e. not solid, as ones and obstacles as zero.
        :param x0: (Optional) Pressure channel which can be used as an initial state for the solver
        :return: 1. pressure tensor (same shape as divergence tensor),
          2. number of iterations (integer, 1D integer tensor or None if unknown)
        """
        raise NotImplementedError(self.__class__)

    def __repr__(self):
        """representation = name"""
        return self.name

    def __and__(self, other):
        if isinstance(self, _PoissonSolverChain):
            return _PoissonSolverChain(self.solvers + (other,))
        if isinstance(other, _PoissonSolverChain):
            return _PoissonSolverChain((self,) + other.solvers)
        else:
            return _PoissonSolverChain([self, other])


PressureSolver = PoissonSolver


@mappable()
def _active_extrapolation(boundaries):
    return math.extrapolation.PERIODIC if boundaries == math.extrapolation.PERIODIC else math.extrapolation.ZERO


def poisson_solve(input_field: CenteredGrid, active_mask: CenteredGrid, accessible_mask: CenteredGrid,
                  guess: CenteredGrid, solver: PoissonSolver = None, gradient: str = 'implicit'):
    """
    Solves the Poisson equation Δp = input_field for p.

    :param gradient: one of ('implicit', 'autodiff', 'inverse')
        If 'autodiff', use the built-in autodiff for backpropagation.
        The intermediate results of each loop iteration will be permanently stored if backpropagation is used.
        If 'implicit', computes a forward pressure solve in reverse accumulation backpropagation.
        This requires less memory but is only accurate if the solution is fully converged.
    :param input_field: CenteredGrid
    :param active_mask:
    :param accessible_mask:
    :param solver: PoissonSolver to use, None for default
    :param guess: CenteredGrid with same size and resolution as input_field
    :return: p as CenteredGrid, iteration count as int or None if not available
    :rtype: CenteredGrid, int
    """
    if solver is None:
        from .geom import GeometricCG
        solver = GeometricCG(accuracy=1e-3)
        # solver = _choose_solver(input_field.resolution, math.choose_backend([input_field.data, active_mask.data, accessible_mask.data]))
    if not isinstance(input_field.extrapolation, ConstantExtrapolation):  # has no open boundary TODO
        input_field = input_field - field.mean(input_field)

    pressure, iteration = solver.solve(input_field.data, active_mask, accessible_mask, guess, enable_backprop=True)
    # assert gradient in ('autodiff', 'implicit', 'inverse')
    # if gradient == 'autodiff':
    #     pressure, iteration = solver.solve(input_field.data, active_mask, accessible_mask, guess, enable_backprop=True)
    # else:
    #     if gradient == 'implicit':
    #         def poisson_gradient(_op, grad):
    #             return poisson_solve(CenteredGrid.sample(grad, poisson_domain.domain), poisson_domain, solver, None, gradient=gradient)[0].data
    #     else:  # gradient = 'inverse'
    #         def poisson_gradient(_op, grad):
    #             return CenteredGrid.sample(grad, poisson_domain.domain).laplace(physical_units=False).data
    #     pressure, iteration = math.with_custom_gradient(solver.solve, [input_field.data, poisson_domain, guess, False], poisson_gradient, input_index=0, output_index=0, name_base='poisson_solve')

    return pressure, iteration


class _PoissonSolverChain(PoissonSolver):

    def __init__(self, solvers):
        PoissonSolver.__init__(self, 'chain%s' % solvers, supported_devices=(), supports_guess=solvers[0].supports_guess, supports_loop_counter=solvers[-1].supports_loop_counter, supports_continuous_masks=False)
        self.solvers = tuple(solvers)
        for solver in solvers[1:]:
            assert solver.supports_guess

    def solve(self, field, active, accessible, guess, enable_backprop):
        iterations = None
        for solver in self.solvers:
            guess, iterations = solver.solve(field, active, accessible, guess, enable_backprop)
        return guess, iterations


def _choose_solver(resolution, backend):
    use_fourier = math.max(resolution) > 64
    if backend.precision == 64:
        from .fourier import FourierSolver
        from .geom import GeometricCG
        return FourierSolver() & GeometricCG(accuracy=1e-8) if use_fourier else GeometricCG(accuracy=1e-8)
    elif backend.precision == 32 and backend.matches_name('SciPy'):
        from .sparse import SparseSciPy
        return SparseSciPy()
    elif backend.precision == 32:
        from .fourier import FourierSolver
        from .sparse import SparseCG
        return FourierSolver() & SparseCG(accuracy=1e-5) if use_fourier else SparseCG(accuracy=1e-5)
    else:  # lower precision
        from .geom import GeometricCG
        return GeometricCG(accuracy=1e-2)
