import numpy as np
from numba import cuda

@cuda.jit
def initialize_data_2d(data, res):
    i, j = cuda.grid(2)

    if i < 128 and j < 128:
        if i >= (res // 10 * 1) and i < (res // 10 * 3):
            if j >= (res // 6 * 2) and j < (res // 6 * 3):
                data[0, i, j, 0] = -0.5

def initialize_data_3d(data, res):
    data[0, (res // 4 * 2):(res // 4 * 3), (res // 4 * 1):(res // 4 * 3), (res // 4):(res // 4 * 3), 0] = -1.0
    return data

def semi_lagrangian_update(x, v, dt):
    x_new = x - v*dt
    return x_new

@cuda.jit
def patch_inflow(inflow_tensor, x, dt):
    i, j = cuda.grid(2)

    if i < 128 and j < 128:
        x[0, i, j, 0] += inflow_tensor[0, i, j, 0] * dt

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