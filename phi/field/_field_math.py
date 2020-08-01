from phi import math
from phi.geom import AABox
from . import StaggeredGrid
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


def staggered_gradient(field: CenteredGrid):
    return stagger(field, lambda lower, upper: (upper - lower) / field.dx)


def stagger(field: CenteredGrid, face_function=math.minimum):
    all_lower = []
    all_upper = []
    for dim in field.shape.spatial.names:
        all_upper.append(math.pad(field.data, {dim: (0, 1)}, field.extrapolation))
        all_lower.append(math.pad(field.data, {dim: (1, 0)}, field.extrapolation))
    all_upper = math.channel_stack(all_upper)
    all_lower = math.channel_stack(all_lower)
    channels = face_function(all_lower, all_upper)
    return math.channel_stack(channels)


def divergence(field: StaggeredGrid):
    components = []
    for i, dim in field.shape.spatial.indexed_names:
        div_dim = math.gradient(field.data[i], dx=field.dx[i], difference='forward', padding=None, axes=[dim])[0]
        components.append(div_dim)
    data = math.sum(components, 0)
    return CenteredGrid(data, field.box, field.extrapolation.gradient())


def mean(field: Grid):
    return math.mean(field.data, field.shape.spatial.names)


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


def staggered_curl_2d(grid, pad_width=(1, 2)):
    assert isinstance(grid, CenteredGrid)
    kernel = math.zeros((3, 3, 1, 2))
    kernel[1, :, 0, 0] = [0, 1, -1]  # y-component: - dz/dx
    kernel[:, 1, 0, 1] = [0, -1, 1]  # x-component: dz/dy
    scalar_potential = grid.padded([pad_width, pad_width]).data
    vector_field = math.conv(scalar_potential, kernel, padding='valid')
    return StaggeredGrid(vector_field, box=grid.box)
