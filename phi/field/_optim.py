from phi import math

from ._grid import Grid
from ._util import expose_tensors
from ..math._optim import SolveResult


def conjugate_gradient(function, y: Grid, x0: Grid, accuracy=1e-5, max_iterations=1000, back_prop=False):
    data_function = expose_tensors(function, y)
    solve = math.optim.conjugate_gradient(data_function, y.data, x0.data, accuracy, max_iterations, back_prop)
    return SolveResult(solve.iterations, x0.with_data(solve.x), x0.with_data(solve.residual))
