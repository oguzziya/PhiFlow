from custom_field import CustomField
from phi.torch.flow import *
import matplotlib.pyplot as plt
import torch
import time
from copy import copy
import sys
import os
import subprocess

def torch_gpu_manual(resolutions, steps, dimension):
    device = "cpu"
    SAVE_IMAGE = False
    ANIMATE = False
    IS_PRESSURE = False

    DIM = dimension
    BATCH_SIZE = 1
    STEPS = steps
    DT = 0.6

    IMG_PATH = os.path.expanduser("~/Repos/Simulations/phi/torch/gpu")

    if SAVE_IMAGE:
        if not os.path.exists(IMG_PATH):
            os.makedirs(IMG_PATH)

    mean_dict = {"rho": [], "vx": [], "vy": []}

    for RES in resolutions:
        FLOW = Fluid(Domain([RES] * DIM, boundaries=OPEN), batch_size=BATCH_SIZE, buoyancy_factor=0.2)
        FLOW_TORCH = torch_from_numpy(FLOW)

        DENSITY = FLOW_TORCH.density
        VELOCITY = FLOW_TORCH.velocity

        # Set the GPU threads
        if DIM == 2:
            density_shape = [1, RES, RES, 1]
            inflow_tensor = torch.zeros(size=density_shape).to(device)
            inflow_tensor[0, (RES // 10 * 1):(RES // 10 * 3), (RES // 6 * 2):(RES // 6 * 3), 0] = -0.5
        elif DIM == 3:
            density_shape = [1, RES, RES, RES, 1]
            inflow_tensor = torch.zeros(size=density_shape)
            inflow_tensor[0, (res // 4 * 2):(res // 4 * 3), (res // 4 * 1):(res // 4 * 3), (res // 4):(res // 4 * 3), 0] = -1.0

        VEL_X = VELOCITY.unstack()[1]
        VEL_Y = VELOCITY.unstack()[0]
        if DIM == 3:
            VEL_Z = VELOCITY.unstack()[2]

        if DIM == 2:
            RHO = CustomField(DENSITY, device, DIM, RES)
            RHO_BUO = copy(RHO)
            VX = CustomField(VEL_X, device, DIM, RES)
            VY = CustomField(VEL_Y, device, DIM, RES)

        if DIM == 3:
            RHO = CustomField(DENSITY, device, DIM, RES)
            RHO_BUO = copy(RHO)
            VX = CustomField(VEL_X, device, DIM, RES)
            VY = CustomField(VEL_Y, device, DIM, RES)
            VZ = CustomField(VEL_Z, device, DIM, RES)

        print("step,resample-time,advection-time")
        for i in range(STEPS):

            if DIM == 2:            
                RHO.data = RHO.advect(VX, VY, DT) + inflow_tensor * DT
                VX.data = VX.advect(VX, VY, DT)
                VY.data = VY.advect(VX, VY, DT) + RHO.resample(VY.points) * DT * 0.2 * (-9.81)
                
                print("RHO Mean: {:5f} - VX Mean: {:5f} - VY Mean: {:5f}".format(torch.mean(RHO.data.cpu()), torch.mean(VX.data.cpu()), torch.mean(VY.data.cpu())))
                
                mean_dict["rho"].append(torch.mean(RHO.data.cpu()))
                mean_dict["vx"].append(torch.mean(VX.data.cpu()))
                mean_dict["vy"].append(torch.mean(VY.data.cpu()))

            if ANIMATE or SAVE_IMAGE:
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
                if ANIMATE:
                    plt.draw()
                    plt.pause(0.5)
                    plt.close()
                else:
                    plt.savefig(os.path.join(IMG_PATH, "torchGPU_" + str(i) + ".png"))
                    plt.close()
    return mean_dict

def run():
    resolutions = [int(res) for res in sys.argv[1:]]
    
    steps = 20
 
    #cpu_means = torch_cpu_manual(resolutions, steps, 2)
    #print("-------------------------------------------")
    gpu_means = torch_gpu_manual(resolutions, steps, 2)

    # rho_diff = np.square(np.asarray(gpu_means["rho"]) - np.asarray(cpu_means["rho"]))
    # vx_diff  = np.square(np.asarray(gpu_means["vx"]) - np.asarray(cpu_means["vx"]))
    # vy_diff  = np.square(np.asarray(gpu_means["vy"]) - np.asarray(cpu_means["vy"]))

    # plt.figure()
    # plt.plot(np.linspace(1, steps, steps), rho_diff)
    # plt.plot(np.linspace(1, steps, steps), vx_diff)
    # plt.plot(np.linspace(1, steps, steps), vy_diff)
    # plt.legend(["RHO Error", "VX Error", "VY Error"])
    # plt.savefig("/home/oguzziya/Repos/Simulations/torch_cpu_gpu_error.png")


if __name__ == '__main__':
    run()