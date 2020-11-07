import numpy as np
from numba import cuda

@cuda.jit
def initialize_data_2d(data, res):
    i, j = cuda.grid(2)

    if i < 128 and j < 128:
        if i >= (res // 10 * 1) and i < (res // 10 * 3):
            if j >= (res // 6 * 2) and j < (res // 6 * 3):
                data[0, i, j, 0] = -1.0

@cuda.jit
def initialize_data_3d(data, res):
    i, j, k = cuda.grid(3)
    if i < 128 and j < 128 and k < 128:
        if i >= (res // 4 * 2) and i < (res // 4 * 3):
            if j >= (res // 4 * 1) and j < (res // 4 * 3):
                if k >= (res // 4) and k < (res // 4 * 3):
                    data[0, i, j, k, 0] = -1.0

def semi_lagrangian_update(x, v, dt):
    x -= v*dt
    return x

@cuda.jit
def semi_lagrangian_update2d(x, v, dt):
    i, j = cuda.grid(2)
    if i < 128 and j < 128:
        x[0, i, j, 0] -= v[0, i, j, 0] * dt

@cuda.jit
def semi_lagrangian_update3d(x, v, dt):
    i, j, k = cuda.grid(3)
    if i < 128 and j < 128 and k < 128:
        x[0, i, j, k, 0] -= v[0, i, j, k, 0] * dt

@cuda.jit
def patch_inflow2d(inflow_tensor, x, dt):
    i, j = cuda.grid(2)
    if i < 128 and j < 128:
        x[0, i, j, 0] += inflow_tensor[0, i, j, 0] * dt

@cuda.jit
def patch_inflow3d(inflow_tensor, x, dt):
    i, j, k = cuda.grid(3)
    if i < 128 and j < 128 and k < 128:
        x[0, i, j, k, 0] += inflow_tensor[0, i, j, k, 0] * dt

from PIL import Image

def save_img(array, scale, name, idx=0):
    if len(array.shape) <= 4:
        ima = np.reshape(array[idx], [array.shape[1], array.shape[2]])  # remove channel dimension , 2d
    else:
        ima = array[idx, :, array.shape[1] // 2, :, 0]  # 3d , middle z slice
    ima = np.reshape(ima, [array.shape[1], array.shape[2]])  # remove channel dimension
    # ima = ima[::-1, :]  # flip along y
    image = Image.fromarray(np.asarray(ima * scale, dtype='i'))
    print("    Writing image '" + name + "'")
    image.save(name)