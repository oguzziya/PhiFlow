import torch
from copy import copy
import time
import torch.nn.functional as torchf

class CustomField:
    def __init__(self, field, device, dim, res):  
        self.dim = dim
        self.res = res
        self.device = device

        self.data = torch.as_tensor(field.data, dtype=torch.float64).to(device)
        self.points = torch.as_tensor(field.points.data, dtype=torch.float64).to(device)
        self.box_sizes = torch.as_tensor(field.box.size, dtype=torch.float64).to(device)
        self.box_lower = torch.as_tensor(field.box.lower, dtype=torch.float64).to(device)

        if self.box_sizes.shape == torch.Size([1]):
            self.box_sizes = torch.zeros([2]).to(device)
            self.box_sizes[0] = res
            self.box_sizes[1] = res

        if self.box_lower.shape == torch.Size([1]):
            self.box_lower = torch.zeros([2]).to(device)

    def resample(self, sample_coords):
        local_coords = torch.div((sample_coords - self.box_lower), self.box_sizes) * self.res - 0.5
        local_coords = 2. * local_coords / (self.res - 1.) - 1.
        local_coords = torch.flip(local_coords, dims=[-1])
        result = torchf.grid_sample(self.data.permute(*((0, -1) + tuple(range(1, len(self.data.shape) - 1)))), local_coords, mode='bilinear', padding_mode='zeros', align_corners=True)
        return result.permute((0,) + tuple(range(2, len(result.shape))) + (1,))

    def advect(self, vx, vy, dt):
        x = self.points - torch.cat((vy.resample(self.points), vx.resample(self.points)), 3) * dt
        self.data = self.resample(x)

