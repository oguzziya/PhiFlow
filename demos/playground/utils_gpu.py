import numpy as np
from numba import cuda
import resample_torch_cuda
from phi.backend.tensorop import collapsed_gather_nd
import torch
from copy import copy

@cuda.jit
def initialize_data2d(data, res):
    i, j = cuda.grid(2)
    if i < res and j < res:
        if i >= (res // 10 * 1) and i < (res // 10 * 3):
            if j >= (res // 6 * 2) and j < (res // 6 * 3):
                data[0, i, j, 0] = 0.1
@cuda.jit
def initialize_data3d(data, res):
    i, j, k = cuda.grid(3)
    if i >= (res // 4 * 2) and i < (res // 4 * 3):
        if j >= (res // 4 * 1) and j < (res // 4 * 3):
            if k >= (res // 4) and k < (res // 4 * 3):
                data[0, i, j, k, 0] = -1.0

def semi_lagrangian_update(x, v, dt):
    x -= v*dt
    return x

@cuda.jit
def semi_lagrangian_update2d(x, vx, vy, dt, res):
    i, j = cuda.grid(2)
    if i < res and j < res:
        x[0, i, j, 0] = x[0, i, j, 0] - vx[0, i, j, 0] * dt
        x[0, i, j, 1] = x[0, i, j, 1] - vy[0, i, j, 0] * dt

@cuda.jit
def semi_lagrangian_update3d(x, vx, vy, vz, dt, res):
    i, j, k = cuda.grid(3)
    if i < res and j < res and k < res:
        x[0, i, j, k, 0] -= vx[0, i, j, k, 0] * dt
        x[0, i, j, k, 1] -= vy[0, i, j, k, 0] * dt
        x[0, i, j, k, 2] -= vz[0, i, j, k, 0] * dt

@cuda.jit
def patch_inflow2d(inflow_tensor, data, dt, res):
    i, j = cuda.grid(2)
    if i < res and j < res:
        data[0, i, j, 0] += inflow_tensor[0, i, j, 0] * dt

@cuda.jit
def patch_inflow3d(inflow_tensor, data, dt, res):
    i, j, k = cuda.grid(3)
    if i < res and j < res and k < res:
        data[0, i, j, k, 0] += inflow_tensor[0, i, j, k, 0] * dt

@cuda.jit
def global_to_local2d(points, size, lower, res, offset):
    i,j = cuda.grid(2)
    if i < res and j < res:
        points[0, i, j, 0] = (points[0, i, j, 0] - lower[0])/(size[0]) * float(res) - offset[0]
        points[0, i, j, 1] = (points[0, i, j, 1] - lower[1])/(size[1]) * float(res) - offset[1]

@cuda.jit
def global_to_local3d(points, size, lower, res):
    i,j,k = cuda.grid(3)
    if i < res and j < res and k < res:
        points[0, i, j, k, 0] = (points[0, i, j, k, 0] - lower[0])/(size[0]) * float(res) - 0.5
        points[0, i, j, k, 1] = (points[0, i, j, k, 1] - lower[1])/(size[1]) * float(res) - 0.5
        points[0, i, j, k, 2] = (points[0, i, j, k, 2] - lower[2])/(size[2]) * float(res) - 0.5

@cuda.jit
def buoyancy2d(vel, rho, gravity, buoyancy_factor, res):
    i, j = cuda.grid(2)
    if i < res and j < res:
        vel[0, i, j, 0] = vel[0, i, j, 0] + (rho[0, i, j, 0] * gravity * buoyancy_factor)

@cuda.jit
def buoyancy3d(vel, rho, factor, res):
    i, j, k = cuda.grid(3)
    if i < res and j < res and k < res:
        vel[0, i, j, k, 0] -= (rho[0, i, j, k, 0] * factor)

def resample2d(inputs, sample_coords, boundary_array, box_sizes, box_lower, blocks, threads, res, output):
    global_to_local2d[blocks, threads](sample_coords, box_sizes, box_lower, res)
    return copy(resample_torch_cuda.resample_op(inputs, sample_coords, boundary_array, output))

def resample3d(inputs, sample_coords, boundary_array, box_sizes, box_lower, blocks, threads, res, output):
    global_to_local3d[blocks, threads](sample_coords, box_sizes, box_lower, res)
    return copy(resample_torch_cuda.resample_op(inputs, sample_coords, boundary_array, output))

def get_boundary_array(shape):
    ZERO = 0
    REPLICATE = 1
    CIRCULAR = 2
    SYMMETRIC = 3
    REFLECT = 4
    dims = len(shape) - 2
    boundary_array = np.zeros((dims, 2), np.int32)
    for i in range(dims):
        for j in range(2):
            current_boundary = collapsed_gather_nd('zero', [i, j]).lower()
            if current_boundary == 'zero' or current_boundary == 'constant':
                boundary_array[i, j] = ZERO
            elif current_boundary == 'replicate':
                boundary_array[i, j] = REPLICATE
            elif current_boundary == 'circular' or current_boundary == 'wrap':
                boundary_array[i, j] = CIRCULAR
            elif current_boundary == 'symmetric':
                boundary_array[i, j] = SYMMETRIC
            elif current_boundary == 'reflect':
                boundary_array[i, j] = REFLECT
    return boundary_array

from PIL import Image

def save_img(array, scale, name, idx=0):
    if len(array.shape) <= 4:
        ima = np.reshape(array[0], [array.shape[1], array.shape[2]])  # remove channel dimension , 2d
    else:
        ima = array[0, :, array.shape[1] // 2, :, 0]  # 3d , middle z slice
        ima = np.reshape(ima, [array.shape[1], array.shape[2]])  # remove channel dimension
    ima = np.reshape(ima, [array.shape[1], array.shape[2]])  # remove channel dimension
    image = Image.fromarray(np.asarray(ima * scale, dtype='i'))
    print("    Writing image '" + name + "'")
    image.save(name)