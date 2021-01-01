import torch
from numba import cuda
import resample_torch_cuda
import utils_gpu as utils
from copy import copy

class ManualGPUData:
    def __init__(self, field, device, dim, res, blocks, threads, boundary_array, offset):
        self.dim = dim
        self.res = res
        self.blocks = blocks
        self.threads = threads
        self.boundary_array = boundary_array
        self.offset = cuda.as_cuda_array(torch.as_tensor(offset).to(device))

        self.data = torch.as_tensor(field.data).to(device)
        self.points = torch.as_tensor(field.points.data).to(device)
        self.box_sizes = torch.as_tensor(field.box.size).to(device)
        self.box_lower = torch.as_tensor(field.box.lower).to(device)

        self.data_numba = cuda.as_cuda_array(self.data)
        self.points_numba = cuda.as_cuda_array(self.points)
        self.box_sizes_numba = cuda.as_cuda_array(self.box_sizes)
        self.box_lower_numba = cuda.as_cuda_array(self.box_lower)

    def update(self):
        self.data_numba = cuda.as_cuda_array(self.data)
        self.points_numba = cuda.as_cuda_array(self.points)

    def resample(self, sample_coords, offset):
        if self.dim == 2:
            utils.global_to_local2d[self.blocks, self.threads](sample_coords, self.box_sizes_numba, self.box_lower_numba, self.res, offset)
            return copy(resample_torch_cuda.resample_op(self.data, sample_coords, self.boundary_array))
        elif self.dim == 3:
            utils.global_to_local3d[self.blocks, self.threads](sample_coords, self.box_sizes_numba, self.box_lower_numba, self.res, offset)
            return copy(resample_torch_cuda.resample_op(self.data, sample_coords, self.boundary_array))

def advection_step2d(data, vx, vy, dt):
    vx.data = vx.resample(data.points, data.offset)
    vy.data = vy.resample(data.points, data.offset)
    vx.update()
    vy.update()
    utils.semi_lagrangian_update2d[data.blocks, data.threads](data.points_numba, vx.data_numba, vy.data_numba, dt, data.res)
    data.data = data.resample(data.points, data.offset)
    data.update()
    return data

def advection_step3d(data, vx, vy, vz, dt):
    vx.data = vx.resample(data.points, data.offset)
    vy.data = vy.resample(data.points, data.offset)
    vz.data = vz.resample(data.points, data.offset)
    vx.update()
    vy.update()
    vz.update()
    utils.semi_lagrangian_update3d[data.blocks, data.threads](data.points_numba, vx.data_numba, vy.data_numba, vz.data_numba, dt, data.res)
    data.data = data.resample(data.points, data.offset)
    data.update()
    return data