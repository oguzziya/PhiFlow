import torch
from numba import cuda
import resample_torch_cuda
from resample_torch_gradient_cuda import resample_gradient_op
import utils_gpu as utils
from copy import copy
import time
import torch.nn.functional as torchf

class ManualGPUData:
    def __init__(self, field, device, dim, res, blocks, threads, boundaries, offset):
        self.dim = dim
        self.res = res
        self.blocks = blocks
        self.threads = threads
        self.boundaries = boundaries
        self.offset = cuda.as_cuda_array(torch.as_tensor(offset).to(device))
        self.device = device

        self.data = torch.as_tensor(field.data).to(device)
        self.points = torch.as_tensor(field.points.data).to(device)
        self.box_sizes = torch.as_tensor(field.box.size).to(device)
        self.box_lower = torch.as_tensor(field.box.lower).to(device)

        self.vx_buffer = torch.zeros([res, res+1]).to(device)
        self.vy_buffer = torch.zeros([res+1, res]).to(device)

        if self.box_sizes.shape == torch.Size([1]):
            self.box_sizes = torch.zeros([2]).to(device)
            self.box_sizes[0] = res
            self.box_sizes[1] = res

        if self.box_lower.shape == torch.Size([1]):
            self.box_lower = torch.zeros([2]).to(device)

        self.data_grad = torch.zeros_like(field.data).to(device)
        self.points_grad = torch.zeros_like(field.data).to(device)

        self.data_numba = cuda.as_cuda_array(self.data)
        self.points_numba = cuda.as_cuda_array(self.points)
        self.data_grad_numba = cuda.as_cuda_array(self.data_grad)
        self.points_grad_numba = cuda.as_cuda_array(self.points_grad)

        self.box_sizes_numba = cuda.as_cuda_array(self.box_sizes)
        self.box_lower_numba = cuda.as_cuda_array(self.box_lower)

    def update(self):
        self.data_numba = cuda.as_cuda_array(self.data)
        self.points_numba = cuda.as_cuda_array(self.points)
        self.data_grad_numba = cuda.as_cuda_array(self.data_grad)
        self.points_grad_numba = cuda.as_cuda_array(self.points_grad)

    def global_to_local(self, coords):
      coords[0,:,:,0] = (coords[0,:,:,0] - self.box_lower[0]) / self.box_sizes[0]
      coords[0,:,:,1] = (coords[0,:,:,1] - self.box_lower[1]) / self.box_sizes[1]
      return coords

    def resample(self, sample_coords):
        if self.dim == 2:
            local_coords = (sample_coords - self.box_lower) / self.box_sizes * float(self.res) - 0.5
            local_coords = 2. * local_coords / (self.res - 1.) - 1.
            local_coords = torch.flip(local_coords, dims=[-1])
            result = torchf.grid_sample(self.data.permute(*((0, -1) + tuple(range(1, len(self.data.shape) - 1)))), local_coords, mode='bilinear', padding_mode='zeros', align_corners=True)
            return result.permute((0,) + tuple(range(2, len(result.shape))) + (1,))

    def advect(self, vx, vy, dt):

        self.points = self.points - torch.cat((vy.resample(self.points), vx.resample(self.points)), 3) * dt
        result = self.resample(self.points)
        self.data = result
        return result

def advection_step2d(data, vx, vy, dt):
    profiling_dict = {"Resampling": 0.0 , "Advection": 0.0, "Step": 0}

    sample_start = time.time()
    data.vx_buffer = vx.resample(data.points)
    data.vy_buffer = vy.resample(data.points)
    profiling_dict["Resampling"] += time.time() - sample_start

    advection_start = time.time()
    data.points[0, :, :, 0] -= data.vy_buffer[0, :, :, 0] * dt
    data.points[0, :, :, 1] -= data.vx_buffer[0, :, :, 0] * dt
    profiling_dict["Advection"] += time.time() - advection_start

    sample_start = time.time()
    data.data = data.resample(data.points)
    profiling_dict["Resampling"] += time.time() - sample_start

    return data, profiling_dict

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

def advection_backward2d(rho, vx, vy, dt):
    # RHO ADVECTION
    rho.data_grad = resample_gradient_op(torch.ones_like(rho.data, dtype=torch.float32).to(rho.device), rho.data, rho.points, rho.boundaries, 1)
    dpoints = resample_gradient_op(torch.ones_like(rho.points).to(rho.device), rho.data, rho.points, rho.boundaries, 2)

    rho.points_grad = utils.semi_lagrangian_backward(dpoints, dt, 1)
    vx_grad = utils.semi_lagrangian_backward(dpoints, dt, 2)
    vy_grad = utils.semi_lagrangian_backward(dpoints, dt, 3)

    vx.data_grad = resample_gradient_op(vx_grad, vx.data, rho.points, vx.boundaries, 1)
    vy.data_grad = resample_gradient_op(vy_grad, vy.data, rho.points, vy.boundaries, 1)
    rho.points_grad += (resample_gradient_op(vx_grad, vx.data, rho.points, vx.boundaries, 2) + resample_gradient_op(vy_grad, vy.data, rho.points, vy.boundaries, 2))

    # VX ADVECTION
    vx.data_grad += resample_gradient_op(torch.ones_like(vx.data).to(vx.device), vx.data, vx.points, vx.boundaries, 1)
    dpoints = resample_gradient_op(torch.ones_like(vx.points).to(vx.device), vx.data, vx.points, vx.boundaries, 2)

    vx.points_grad = utils.semi_lagrangian_backward(dpoints, dt, 1)
    vx_grad = utils.semi_lagrangian_backward(dpoints, dt, 2)
    vy_grad = utils.semi_lagrangian_backward(dpoints, dt, 3)

    vx.data_grad += resample_gradient_op(vx_grad, vx.data, vx.points, vx.boundaries, 1)
    vy.data_grad += resample_gradient_op(vy_grad, vy.data, vx.points, vy.boundaries, 1)
    vx.points_grad += (resample_gradient_op(vx_grad, vx.data, vx.points, vx.boundaries, 2) + resample_gradient_op(vy_grad, vy.data, vx.points, vy.boundaries, 2))

    # VY ADVECTION
    vy.data_grad = resample_gradient_op(torch.ones_like(vy.data).to(vy.device), vy.data, vy.points, vy.boundaries, 1)
    dpoints = resample_gradient_op(torch.ones_like(vy.points).to(vy.device), vy.data, vy.points, vy.boundaries, 2)

    vy.points_grad = utils.semi_lagrangian_backward(dpoints, dt, 1)
    vx_grad = utils.semi_lagrangian_backward(dpoints, dt, 2)
    vy_grad = utils.semi_lagrangian_backward(dpoints, dt, 3)

    vx.data_grad += resample_gradient_op(vx_grad, vx.data, vy.points, vx.boundaries, 1)
    vy.data_grad += resample_gradient_op(vy_grad, vy.data, vy.points, vy.boundaries, 1)
    vy.points_grad += (resample_gradient_op(vx_grad, vx.data, vy.points, vx.boundaries, 2) + resample_gradient_op(vy_grad, vy.data, vy.points, vy.boundaries, 2))

