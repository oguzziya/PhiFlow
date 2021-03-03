from phi.torch.flow import *
import matplotlib.pyplot as plt
import torch
import time
from resample_torch_cuda import resample_op
from manual_gpu_data import *
from copy import copy
import sys
import os
import subprocess

def torch_cpu_manual(resolutions, steps, dimension):
    import utils as utils

    device = "cpu"
    SAVE_IMAGE = False
    ANIMATE = False
    IS_PRESSURE = False

    DIM = dimension
    BATCH_SIZE = 1
    STEPS = steps
    DT = 0.6

    IMG_PATH = os.path.expanduser("~/Repos/Simulations/phi/torch/cpu")

    mean_dict = {"rho": [], "vx": [], "vy": []}

    if SAVE_IMAGE:
        IMG_PATH = os.path.expanduser(IMG_PATH)
        if not os.path.exists(IMG_PATH):
            os.makedirs(IMG_PATH)

    for RES in resolutions:
        
        FLOW = Fluid(Domain([RES] * DIM, boundaries=OPEN), batch_size=BATCH_SIZE, buoyancy_factor=0.2)
        FLOW_TORCH = torch_from_numpy(FLOW)

        DENSITY = FLOW_TORCH.density
        VELOCITY = FLOW_TORCH.velocity

        if DIM == 2:
            density_shape = [1, RES, RES, 1]
            inflow_tensor = torch.zeros(size=density_shape)
            inflow_tensor = utils.initialize_data_2d(inflow_tensor, RES)
        elif DIM == 3:
            density_shape = [1, RES, RES, RES, 1]
            inflow_tensor = torch.zeros(size=density_shape)
            inflow_tensor = utils.initialize_data_3d(inflow_tensor, RES)

        for i in range(STEPS):

            x_rho = torch_from_numpy(DENSITY.points.data)
            v_rho = VELOCITY.sample_at(x_rho)

            x_rho = utils.semi_lagrangian_update(x_rho, v_rho, DT)

            x_rho = DENSITY.sample_at(x_rho)

            x_rho = utils.patch_inflow(inflow_tensor, x_rho, DT)

            x_vel_list = []

            for component in VELOCITY.unstack():
                x_vel = torch_from_numpy(component.points.data)
                v_vel = VELOCITY.sample_at(x_vel)

                x_vel = utils.semi_lagrangian_update(x_vel, v_vel, DT)

                x_vel = component.sample_at(x_vel)

                x_vel_list.append(x_vel)

            # Update the object data
            DENSITY = DENSITY.with_data(x_rho)
            VELOCITY = VELOCITY.with_data(x_vel_list)

            VELOCITY += buoyancy(DENSITY, 9.81, FLOW.buoyancy_factor)

            print("RHO Mean: {:5f} - VX Mean: {:5f} - VY Mean: {:5f}".format(torch.mean(DENSITY.data), torch.mean(VELOCITY.unstack()[1].data.cpu()), torch.mean(VELOCITY.unstack()[0].data.cpu())))

            mean_dict["rho"].append(torch.mean(DENSITY.data))
            mean_dict["vx"].append(torch.mean(VELOCITY.unstack()[1].data.cpu()))
            mean_dict["vy"].append(torch.mean(VELOCITY.unstack()[0].data.cpu()))

            if ANIMATE or SAVE_IMAGE:
                array = DENSITY.data.numpy()
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
                    plt.savefig(os.path.join(IMG_PATH, "torchCPU_" + str(i) + ".png"))
                    plt.close()

    return mean_dict

def torch_gpu_manual(resolutions, steps, dimension):
    import utils_gpu as utils
    from numba import cuda

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
            threads_per_block = (min(RES, 32), min(RES, 32))
            blocks_per_grid = (int(np.ceil(RES / threads_per_block[0])), int(np.ceil(RES / threads_per_block[1])))

            density_shape = [1, RES, RES, 1]

            inflow_tensor = torch.zeros(size=density_shape).to(device)
            inflow_tensor[0, (RES // 10 * 1):(RES // 10 * 3), (RES // 6 * 2):(RES // 6 * 3), 0] = -0.5

        elif DIM == 3:
            threads_per_block = (8, 8, 8)
            blocks_per_grid = (int(RES / threads_per_block[0]), int(RES / threads_per_block[1]), int(RES / threads_per_block[2]))

            density_shape = [1, RES, RES, RES, 1]
            inflow_tensor = torch.zeros(size=density_shape)
            inflow_tensor_numba = cuda.as_cuda_array(inflow_tensor.to(device))
            utils.initialize_data3d[blocks_per_grid, threads_per_block](inflow_tensor_numba, RES)

        VEL_X = VELOCITY.unstack()[1]
        VEL_Y = VELOCITY.unstack()[0]
        if DIM == 3:
            VEL_Z = VELOCITY.unstack()[2]

        print("Rho Box Lower: {} - Size: {}".format(DENSITY.box.lower, DENSITY.box.size))
        print("Velx Box Lower: {} - Size: {}".format(VEL_X.box.lower, VEL_X.box.size))
        print("Vely Box Lower: {} - Size: {}".format(VEL_Y.box.lower, VEL_Y.box.size))

        rho_boundary_array = torch.as_tensor(utils.get_boundary_array(DENSITY.data.shape)).to(device)
        vx_boundary_array = torch.as_tensor(utils.get_boundary_array(VEL_X.data.shape)).to(device)
        vy_boundary_array = torch.as_tensor(utils.get_boundary_array(VEL_Y.data.shape)).to(device)
        if DIM == 3:
            vz_boundary_array = torch.as_tensor(utils.get_boundary_array(VEL_Z.data.shape)).to(device)

        # Create Torch data on GPU and Numba pointers
        if DIM == 2:
            RHO = ManualGPUData(DENSITY, device, DIM, RES, blocks_per_grid, threads_per_block, rho_boundary_array, [0.0, 0.0])
            RHO_BUO = copy(RHO)
            VX = ManualGPUData(VEL_X, device, DIM, RES, blocks_per_grid, threads_per_block, vx_boundary_array, [0.0, 0.5])
            VY = ManualGPUData(VEL_Y, device, DIM, RES, blocks_per_grid, threads_per_block, vy_boundary_array, [0.5, 0.0])

        if DIM == 3:
            RHO = ManualGPUData(DENSITY, device, DIM, RES, blocks_per_grid, threads_per_block, rho_boundary_array, [0.0, 0.0, 0.0])
            VX = ManualGPUData(VEL_X, device, DIM, RES, blocks_per_grid, threads_per_block, vx_boundary_array, [0.5, 0.0, 0.0])
            VY = ManualGPUData(VEL_Y, device, DIM, RES, blocks_per_grid, threads_per_block, vy_boundary_array, [0.0, 0.5, 0.0])
            VZ = ManualGPUData(VEL_Z, device, DIM, RES, blocks_per_grid, threads_per_block, vz_boundary_array, [0.0, 0.0, 0.5])

        print("step,resample-time,advection-time")
        for i in range(STEPS):

            if DIM == 2:            
                RHO.data = RHO.advect(VX, VY, DT) + inflow_tensor * DT
                VX.data = VX.advect(VX, VY, DT)
                #RHO_BUO.data = RHO.data * 0.2 * (-9.81) 
                #VY.data = VY.advect(VX, VY, DT) + RHO_BUO.resample(VY.points) * DT
                VY.data = VY.advect(VX, VY, DT) + RHO.resample(VY.points) * DT * 0.2 * (-9.81)
                
                print("RHO Mean: {:5f} - VX Mean: {:5f} - VY Mean: {:5f}".format(torch.mean(RHO.data.cpu()), torch.mean(VX.data.cpu()), torch.mean(VY.data.cpu())))
                
                mean_dict["rho"].append(torch.mean(RHO.data.cpu()))
                mean_dict["vx"].append(torch.mean(VX.data.cpu()))
                mean_dict["vy"].append(torch.mean(VY.data.cpu()))

            else:
                RHO = advection_step3d(RHO, VX, VY, VZ, DT)

                utils.patch_inflow3d[blocks_per_grid, threads_per_block](inflow_tensor_numba, RHO.data_numba, DT, RES)

                VX = advection_step3d(VX, VX, VY, VZ, DT)
                VY = advection_step3d(VY, VX, VY, VZ, DT)
                VZ = advection_step3d(VZ, VX, VZ, VZ, DT)

                utils.buoyancy3d[blocks_per_grid, threads_per_block](VY.data_numba, RHO.data_numba, 9.81,
                                                                     FLOW.buoyancy_factor, RES)

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
