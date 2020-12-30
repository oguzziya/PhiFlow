import torch
from numba import cuda
import resample_torch_cuda
import utils_gpu as utils
from copy import copy

class ManualGPUData:
    def __init__(self, field, device, dim, res, blocks, threads):
        self.dim = dim
        self.res = res
        self.blocks = blocks
        self.threads = threads

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

    def resample(self, sample_coords, boundary_array, output):
        if self.dim == 2:
            utils.global_to_local2d[self.blocks, self.threads](sample_coords, self.box_sizes_numba, self.box_lower_numba, self.res)
            return copy(resample_torch_cuda.resample_op(self.data, sample_coords, boundary_array, output))
        elif self.dim == 3:
            utils.global_to_local3d[self.blocks, self.threads](sample_coords, self.box_sizes_numba, self.box_lower_numba, self.res)
            return copy(resample_torch_cuda.resample_op(self.data, sample_coords, boundary_array, output))