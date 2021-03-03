import torch
from copy import copy
import time
import torch.nn.functional as torchf

class CustomField:
    def __init__(self, field, device, dim, res):
        self.dim = dim
        self.res = res
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

