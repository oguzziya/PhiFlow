from phi.torch.flow import *
import matplotlib.pyplot as plt
import torch
import time
from resample_torch_cuda import resample_op
from manual_gpu_data import *
from copy import copy

IS_PRESSURE = False
device = "cpu"
SAVE_IMAGE = True
ANIMATE = False

DIM = 2
BATCH_SIZE = 1
STEPS = 10
DT = 0.6

DT = np.float32(DT)

IMG_PATH = os.path.expanduser("~/Repos/Simulations/phi/data/manual/torch")

if SAVE_IMAGE:
    IMG_PATH = os.path.expanduser("~/Repos/Simulations/phi/data/manual/torch")
    if not os.path.exists(IMG_PATH):
        os.makedirs(IMG_PATH)

sample_timings = {"GPU": {}, "CPU": {}}
inflow_timings = {"GPU": {}, "CPU": {}}
advect_timings = {"GPU": {}, "CPU": {}}

resolutions_float = np.linspace(100, 2000, 20)
resolutions = [128, ]
for RUN_GPU in [True,]:
    for RES in resolutions:
        FLOW = Fluid(Domain([RES] * DIM, boundaries=OPEN), batch_size=BATCH_SIZE, buoyancy_factor=0.2)
        FLOW_TORCH = torch_from_numpy(FLOW)

        DENSITY = FLOW_TORCH.density
        VELOCITY = FLOW_TORCH.velocity

        if RUN_GPU:
            device = 'cuda:0'
            import utils_gpu as utils
            from numba import cuda

            # Set the GPU threads
            if DIM == 2:
                threads_per_block = (min(RES, 32), min(RES, 32))
                blocks_per_grid = (int(np.ceil(RES / threads_per_block[0])), int(np.ceil(RES / threads_per_block[1])))

                density_shape = [1, RES, RES, 1]

                inflow_tensor = torch.zeros(size=density_shape)
                inflow_tensor_numba = cuda.as_cuda_array(inflow_tensor.to(device))
                utils.initialize_data2d[blocks_per_grid, threads_per_block](inflow_tensor_numba, RES)

            elif DIM == 3:
                threads_per_block = (8, 8, 8)
                blocks_per_grid = (int(RES / threads_per_block[0]), int(RES / threads_per_block[1]), int(RES / threads_per_block[2]))

                density_shape = [1, RES, RES, RES, 1]
                inflow_tensor = torch.zeros(size=density_shape)
                inflow_tensor_numba = cuda.as_cuda_array(inflow_tensor.to(device))
                utils.initialize_data3d[blocks_per_grid, threads_per_block](inflow_tensor_numba, RES)

            VEL_X = VELOCITY.unstack()[0]
            VEL_Y = VELOCITY.unstack()[1]
            if DIM == 3:
                VEL_Z = VELOCITY.unstack()[2]

            rho_boundary_array = torch.as_tensor(utils.get_boundary_array(DENSITY.data.shape)).to(device)
            vx_boundary_array = torch.as_tensor(utils.get_boundary_array(VEL_X.data.shape)).to(device)
            vy_boundary_array = torch.as_tensor(utils.get_boundary_array(VEL_Y.data.shape)).to(device)
            if DIM == 3:
                vz_boundary_array = torch.as_tensor(utils.get_boundary_array(VEL_Z.data.shape)).to(device)

            # Create Torch data on GPU and Numba pointers
            if DIM == 2:
                RHO = ManualGPUData(DENSITY, device, DIM, RES, blocks_per_grid, threads_per_block, rho_boundary_array, [0.5, 0.5])
                VX = ManualGPUData(VEL_X, device, DIM, RES, blocks_per_grid, threads_per_block, vx_boundary_array, [0.5, 0.0])
                VY = ManualGPUData(VEL_Y, device, DIM, RES, blocks_per_grid, threads_per_block, vy_boundary_array, [0.0, 0.5])

            if DIM == 3:
                RHO = ManualGPUData(DENSITY, device, DIM, RES, blocks_per_grid, threads_per_block, rho_boundary_array, [0.0, 0.0, 0.0])
                VX = ManualGPUData(VEL_X, device, DIM, RES, blocks_per_grid, threads_per_block, vx_boundary_array, [0.5, 0.0, 0.0])
                VY = ManualGPUData(VEL_Y, device, DIM, RES, blocks_per_grid, threads_per_block, vy_boundary_array, [0.0, 0.5, 0.0])
                VZ = ManualGPUData(VEL_Z, device, DIM, RES, blocks_per_grid, threads_per_block, vz_boundary_array, [0.0, 0.0, 0.5])

            for i in range(STEPS):

                sample_time = 0.0
                advect_time = 0.0
                inflow_time = 0.0

                if DIM == 2:
                    RHO = advection_step2d(RHO, VX, VY, DT)

                    utils.patch_inflow2d[blocks_per_grid, threads_per_block](inflow_tensor_numba, RHO.data_numba, DT, RES)

                    VX = advection_step2d(VX, VX, VY, DT)
                    VY = advection_step2d(VY, VX, VY, DT)

                    utils.buoyancy2d[blocks_per_grid, threads_per_block](VY.data_numba, RHO.data_numba, 9.81, FLOW.buoyancy_factor, RES)

                    print("Step {}".format(i))
                    print(" - Mean value of rho: {}".format(np.mean(RHO.data.cpu().data.numpy())))
                    print(" - Mean value of vx : {}".format(np.mean(VX.data.cpu().data.numpy())))
                    print(" - Mean value of vy : {}".format(np.mean(VY.data.cpu().data.numpy())))

                else:
                    RHO = advection_step3d(RHO, VX, VY, VZ, DT)

                    utils.patch_inflow3d[blocks_per_grid, threads_per_block](inflow_tensor_numba, RHO.data_numba, DT, RES)

                    VX = advection_step3d(VX, VX, VY, VZ, DT)
                    VY = advection_step3d(VY, VX, VY, VZ, DT)
                    VZ = advection_step3d(VZ, VX, VZ, VZ, DT)

                    utils.buoyancy3d[blocks_per_grid, threads_per_block](VY.data_numba, RHO.data_numba, 9.81,
                                                                         FLOW.buoyancy_factor, RES)

                    print("Step {}".format(i))
                    print(" - Mean value of rho: {}".format(np.mean(RHO.data.cpu().data.numpy())))
                    print(" - Mean value of vx : {}".format(np.mean(VX.data.cpu().data.numpy())))
                    print(" - Mean value of vy : {}".format(np.mean(VY.data.cpu().data.numpy())))

                if IS_PRESSURE:
                    pressure_solve_start = time.time()
                    VELOCITY = divergence_free(VELOCITY, FLOW.domain, obstacles=(), pressure_solver=SparseCG(max_iterations=100))
                    pressure_solve_end = time.time()

                    pressure_time = pressure_solve_end - pressure_solve_start

                if ANIMATE or SAVE_IMAGE:
                    array = RHO.data.cpu().data.numpy()
                    if len(array.shape) <= 4:
                        ima = np.reshape(array[0], [array.shape[1], array.shape[2]])  # remove channel dimension , 2d
                    else:
                        ima = array[0, :, array.shape[1] // 2, :, 0]  # 3d , middle z slice
                        ima = np.reshape(ima, [array.shape[1], array.shape[2]])  # remove channel dimension
                    fig = plt.figure()
                    ax = fig.add_subplot(1, 1, 1)
                    ax.contourf(ima, cmap='inferno')
                    if ANIMATE:
                        plt.draw()
                        plt.pause(0.5)
                        plt.close()
                    else:
                        plt.savefig(os.path.join(IMG_PATH, "torchGPU_" + str(i) + ".png"))

        else:
            device = 'cpu'
            import utils as utils

            sample_time = 0.0
            advect_time = 0.0
            inflow_time = 0.0

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
                v_rho = VELOCITY.sample_at(x_rho, device=device)

                x_rho = utils.semi_lagrangian_update(x_rho, v_rho, DT)

                x_rho = DENSITY.sample_at(x_rho, device)

                x_rho = utils.patch_inflow(inflow_tensor, x_rho, DT)

                x_vel_list = []

                for component in VELOCITY.unstack():
                    x_vel = torch_from_numpy(component.points.data)
                    v_vel = VELOCITY.sample_at(x_vel, device)

                    x_vel = utils.semi_lagrangian_update(x_vel, v_vel, DT)

                    x_vel = component.sample_at(x_vel, device)

                    x_vel_list.append(x_vel)

                # Update the object data
                DENSITY = DENSITY.with_data(x_rho)
                VELOCITY = VELOCITY.with_data(x_vel_list)

                VELOCITY += buoyancy(DENSITY, 9.81, FLOW.buoyancy_factor)

                if ANIMATE:
                    array = DENSITY.data.numpy()
                    if len(array.shape) <= 4:
                        ima = np.reshape(array[0], [array.shape[1], array.shape[2]])  # remove channel dimension , 2d
                    else:
                        ima = array[0, :, array.shape[1] // 2, :, 0]  # 3d , middle z slice
                        ima = np.reshape(ima, [array.shape[1], array.shape[2]])  # remove channel dimension
                    fig = plt.figure()
                    ax = fig.add_subplot(1, 1, 1)
                    ax.contourf(ima, cmap='inferno')
                    plt.draw()
                    plt.pause(0.5)
                    plt.close()