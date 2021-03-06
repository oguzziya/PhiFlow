from custom_field import CustomField
import matplotlib.pyplot as plt
import torch
import time
from copy import copy
import sys
import os
import subprocess

def numpy_manual(resolutions, steps, dimension):
    SAVE_IMAGE = False

    DIM = dimension
    BATCH_SIZE = 1
    DT = 0.6
    GRAVITY = 9.81
    BUOYANCY = 0.2

    IMG_PATH = os.path.expanduser("~/Repos/Simulations/phi/torch/gpu")

    if SAVE_IMAGE:
        if not os.path.exists(IMG_PATH):
            os.makedirs(IMG_PATH)

    print("resolution,rho_means,vx_means,vy_means")
    for RES in resolutions:
      
      FLOW = Fluid(Domain([RES] * DIM, boundaries=OPEN), batch_size=BATCH_SIZE, buoyancy_factor=BUOYANCY)
      DENSITY = FLOW.density
      VELOCITY = FLOW.velocity
      
      INFLOW_DENSITY = math.zeros_like(FLOW.density)
      INFLOW_DENSITY.data[..., (RES // 10 * 1):(RES // 10 * 3), (RES // 6 * 2):(RES // 6 * 3), 0] = -0.5

      for i in range(steps):
          DENSITY = advect.semi_lagrangian(DENSITY, VELOCITY, DT) + DT * INFLOW_DENSITY
          VELOCITY = advect.semi_lagrangian(VELOCITY, VELOCITY, DT) + buoyancy(DENSITY, GRAVITY, FLOW.buoyancy_factor) * DT
          print("{},{},{},{}".format(RES, np.mean(DENSITY.data), np.mean(VELOCITY.unstack()[1].data), np.mean(VELOCITY.unstack()[0].data)))

def torch_manual(resolutions, steps, dimension):
    device = "cuda:0"
    SAVE_IMAGE = False

    DIM = dimension
    BATCH_SIZE = 1
    STEPS = steps
    DT = 0.6
    GRAVITY = -9.81
    BUOYANCY = 0.2

    torch.set_default_dtype(torch.float64)

    IMG_PATH = os.path.expanduser("~/Repos/Simulations/phi/torch/gpu")

    if SAVE_IMAGE:
        if not os.path.exists(IMG_PATH):
            os.makedirs(IMG_PATH)

    print("resolution,rho_means,vx_means,vy_means")
    for RES in resolutions:        
        FLOW = Fluid(Domain([RES] * DIM, boundaries=OPEN), batch_size=BATCH_SIZE, buoyancy_factor=0.2)

        DENSITY = FLOW.density
        VELOCITY = FLOW.velocity

        density_shape = [1, RES, RES, 1]
        inflow_tensor = torch.zeros(size=density_shape).to(device)
        inflow_tensor[0, (RES // 10 * 1):(RES // 10 * 3), (RES // 6 * 2):(RES // 6 * 3), 0] = -0.5

        VEL_X = VELOCITY.unstack()[1]
        VEL_Y = VELOCITY.unstack()[0]

        RHO = CustomField(DENSITY, device, DIM, RES)
        VX = CustomField(VEL_X, device, DIM, RES)
        VY = CustomField(VEL_Y, device, DIM, RES)

        for i in range(STEPS):          
            RHO.advect(VX, VY, DT)    
            VX.advect(VX, VY, DT)
            VY.advect(VX, VY, DT)

            RHO.data += inflow_tensor * DT
            VY.data += RHO.resample(VY.points) * DT * BUOYANCY * GRAVITY
                
            print("{},{},{},{}".format(RES, torch.mean(RHO.data.cpu()), torch.mean(VX.data.cpu()), torch.mean(VY.data.cpu())))

            if SAVE_IMAGE:
                array = RHO.data.cpu().data.numpy()
                if len(array.shape) <= 4:
                    ima = np.reshape(array[0], [array.shape[1], array.shape[2]])  # remove channel dimension , 2d
                else:
                    ima = array[0, :, array.shape[1] // 2, :, 0]  # 3d , middle z slice
                    ima = np.reshape(ima, [array.shape[1], array.shape[2]])  # remove channel dimension
                cmap = plt.cm.get_cmap("inferno")
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                cs = ax.contourf(ima, cmap=cmap)
                fig.colorbar(cs, ax=ax)
                plt.savefig(os.path.join(IMG_PATH, "torchGPU_" + str(i) + ".png"))
                plt.close()

if __name__ == '__main__':
    resolutions = [int(res) for res in sys.argv[1:-1]]
    steps = 100
    case = sys.argv[-1]

    if case == "cpu":
      from phi.flow import *
      numpy_manual(resolutions, steps, 2)
    elif case == "gpu":
      from phi.torch.flow import *
      torch_manual(resolutions, steps, 2)
    else:
      print("Spcify: cpu or gpu")