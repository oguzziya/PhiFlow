from numbers import Number

from phi import math, field
from phi.math._helper import _dim_shifted
from phi.field import CenteredGrid
from .solver_api import PoissonSolver
from phi.physics.material import Material


class GeometricCG(PoissonSolver):

    def __init__(self, accuracy=1e-5, max_iterations=2000):
        """
Conjugate gradient solver that geometrically calculates laplace pressure in each iteration.
Unlike most other solvers, this algorithm is TPU compatible but usually performs worse than SparseCG.

Obstacles are allowed to vary between examples but the same number of iterations is performed for each example in one batch.

        :param accuracy: the maximally allowed error on the divergence channel for each cell
        :param max_iterations: integer specifying maximum conjugent gradient loop iterations or None for no limit
        """
        PoissonSolver.__init__(self, 'Single-Phase Conjugate Gradient', supported_devices=('CPU', 'GPU', 'TPU'), supports_guess=True, supports_loop_counter=True, supports_continuous_masks=True)
        self.accuracy = accuracy
        self.max_iterations = max_iterations

    def solve(self, y, active, accessible, x0: CenteredGrid, enable_backprop):

        def masked_laplace(x):
            x = x0.with_data(x)
            return field.laplace(x).data

        result, iterations = math.optim.conjugate_gradient(masked_laplace, y, x0, self.accuracy, self.max_iterations, back_prop=enable_backprop)
        return x0.with_data(result), iterations


def _weighted_sliced_laplace_nd(tensor, weights):
    if tensor.shape[-1] != 1:
        raise ValueError('Laplace operator requires a scalar channel as input')
    dims = range(math.spatial_rank(tensor))
    components = []
    for dimension in dims:
        lower_weights, center_weights, upper_weights = _dim_shifted(weights, dimension, (-1, 0, 1), diminish_others=(1, 1))
        lower_values, center_values, upper_values = _dim_shifted(tensor, dimension, (-1, 0, 1), diminish_others=(1, 1))
        diff = math.mul(upper_values, upper_weights * center_weights) + math.mul(lower_values, lower_weights * center_weights) + math.mul(center_values, - lower_weights - upper_weights)
        components.append(diff)
    return math.sum(components, 0)
