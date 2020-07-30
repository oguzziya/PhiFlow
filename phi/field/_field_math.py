from phi import math
from phi.geom import AABox
from ._field import Field
from ._grid import CenteredGrid, Grid, _pad_value, _pad_mode, _gradient_extrapolation


def laplace(field: Grid, axes=None):
    return field._op1(lambda tensor: math.laplace(tensor, dx=field.dx, padding=field.extrapolation, axes=axes))


def gradient(field: Grid, axes=None, difference='central'):
    if not physical_units or self.has_cubic_cells:
        data = math.gradient(self.data, dx=np.mean(self.dx), difference=difference, padding=_pad_mode(self.extrapolation))
        return self.copied_with(data=data, extrapolation=_gradient_extrapolation(self.extrapolation), flags=())
    else:
        raise NotImplementedError('Only cubic cells supported.')


def pad(grid: Grid, widths):
    if isinstance(widths, int):
        widths = [[widths, widths]] * grid.rank
    if isinstance(grid, CenteredGrid):
        data = math.pad(self.data, [[0, 0]] + widths + [[0, 0]], _pad_mode(self.extrapolation), constant_values=_pad_value(self.extrapolation_value))
        w_lower, w_upper = np.transpose(widths)
        box = AABox(self.box.lower - w_lower * self.dx, self.box.upper + w_upper * self.dx)
        return CenteredGrid(data, box, grid.extrapolation)


def squared(field: Field):
    raise NotImplementedError()


def real(field: Field):
    raise NotImplementedError()


def imag(field: Field):
    raise NotImplementedError()


def fftfreq(grid: Grid):
    raise NotImplementedError()


def fft(grid: Grid):
    raise NotImplementedError()


def ifft(grid: Grid):
    raise NotImplementedError()
