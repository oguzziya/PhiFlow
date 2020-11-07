import numpy as np
from numba import cuda

def initialize_data_2d(data, res):
    data[0, (res // 10 * 1):(res // 10 * 3), (res // 6 * 2):(res // 6 * 3), 0] = -0.5
    return data

def initialize_data_3d(data, res):
    data[0, (res // 4 * 2):(res // 4 * 3), (res // 4 * 1):(res // 4 * 3), (res // 4):(res // 4 * 3), 0] = -1.0
    return data

def semi_lagrangian_update(x, v, dt):
    x_new = x - v*dt
    return x_new

def patch_inflow(inflow_tensor, x, dt):
    x += inflow_tensor*dt
    return x

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