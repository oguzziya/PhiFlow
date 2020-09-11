import numpy as np
from numba import njit, types, cfunc
from phi.flow import *
import time

@njit([types.void(types.float32[:,:,:,:,:], types.int64)], cache=True)
def initialize_data(data, res):
    data[..., (res // 4 * 2):(res // 4 * 3), (res // 4 * 1):(res // 4 * 3), (res // 4):(res // 4 * 3), 0] = \
        np.ones_like(data[..., (res // 4 * 2):(res // 4 * 3), (res // 4 * 1):(res // 4 * 3), (res // 4):(res // 4 * 3), 0])

@njit([types.float32[:,:,:,:,:](types.float32[:,:,:,:,:], types.float32[:,:,:,:,:], types.float32)], cache=True)
def semi_lagrangian_numba(x0, v, dt):
    x = x0 - v * dt
    return x

def sample_at(data, points, box_size, box_lower, resolution):

    # Batch allignment
    box_list = [box_size, box_lower]
    box_array = np.array(box_list)
    points_array = np.array(points)

    ndims = len(box_array.shape)
    target_ndims = len(points_array.shape)

    if ndims <= 1 or target_ndims == ndims:
        size, lower = box_array
    else:
        for _i in range(target_ndims - ndims):
            points_array = np.expand_dims(points_array, - 2)
        size, lower = points_array

    # Global to local
    local_points = (points_array - lower) / size
    local_points = np.array(local_points) * np.float(resolution) - 0.5

    # Resample
    resampled = math.resample(data, local_points, boundary=_pad_mode('boundary'),
                             interpolation="linear", constant_values=_pad_value(self.extrapolation_value))
    return resampled

def semi_lagrangian(field, velocity_field, dt):
    x = semi_lagrangian_numba(field.points.data,
                              velocity_field.at(field.points).data,
                              dt)
    data = field.sample_at(x)
    return_val = field.with_data(data)
    return return_val