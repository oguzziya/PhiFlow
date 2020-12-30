from phi.torch.flow import *
import matplotlib.pyplot as plt
import torch
import time
from resample_torch_cuda import resample_op
from manual_gpu_data import ManualGPUData
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
resolutions = [32,]
for RUN_GPU in [True, False]:
    for RES in resolutions:
        FLOW = Fluid(Domain([RES] * DIM, boundaries=OPEN), batch_size=BATCH_SIZE, buoyancy_factor=0.2)
        FLOW_TORCH = torch_from_numpy(FLOW)

        DENSITY = FLOW_TORCH.density
        VELOCITY = FLOW_TORCH.velocity

        if RUN_GPU:
            sample_timings["GPU"][RES] = []
            advect_timings["GPU"][RES] = []
            inflow_timings["GPU"][RES] = []
        else:
            sample_timings["CPU"][RES] = []
            advect_timings["CPU"][RES] = []
            inflow_timings["CPU"][RES] = []

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

            # Create Torch data on GPU and Numba pointers
            if DIM == 2:
                RHO = ManualGPUData(DENSITY, device, DIM, RES, blocks_per_grid, threads_per_block)
                VX = ManualGPUData(VEL_X, device, DIM, RES, blocks_per_grid, threads_per_block)
                VY = ManualGPUData(VEL_Y, device, DIM, RES, blocks_per_grid, threads_per_block)

            if DIM == 3:
                RHO = ManualGPUData(DENSITY, device, DIM, RES, blocks_per_grid, threads_per_block)
                VX = ManualGPUData(VEL_X, device, DIM, RES, blocks_per_grid, threads_per_block)
                VY = ManualGPUData(VEL_Y, device, DIM, RES, blocks_per_grid, threads_per_block)
                VZ = ManualGPUData(VEL_Z, device, DIM, RES, blocks_per_grid, threads_per_block)

            rho_data_resample = torch.as_tensor(DENSITY.data).to(device)
            vx_data_resample = copy(torch.as_tensor(VEL_X.data).to(device))
            vy_data_resample = copy(torch.as_tensor(VEL_Y.data).to(device))
            if(DIM == 3):
                vz_data_resample = torch.as_tensor(VEL_Z.data).to(device)

            rho_boundary_array = torch.as_tensor(utils.get_boundary_array(DENSITY.data.shape)).to(device)
            vx_boundary_array = torch.as_tensor(utils.get_boundary_array(VEL_X.data.shape)).to(device)
            vy_boundary_array = torch.as_tensor(utils.get_boundary_array(VEL_Y.data.shape)).to(device)
            if DIM == 3:
                vz_boundary_array = torch.as_tensor(utils.get_boundary_array(VEL_Z.data.shape)).to(device)

            for i in range(STEPS):

                sample_time = 0.0
                advect_time = 0.0
                inflow_time = 0.0

                if DIM == 2:
                    sample_start = time.time()
                    VX.data = VX.resample(RHO.points, vx_boundary_array, vx_data_resample)
                    VY.data = VY.resample(RHO.points, vy_boundary_array, vy_data_resample)
                    VX.update()
                    VY.update()
                    sample_time += time.time() - sample_start

                    advect_start = time.time()
                    utils.semi_lagrangian_update2d[blocks_per_grid, threads_per_block](RHO.points_numba, VX.data_numba, VY.data_numba, DT, RES)
                    advect_time += time.time() - advect_start

                    sample_start = time.time()
                    RHO.data = RHO.resample(RHO.points, rho_boundary_array, rho_data_resample)
                    RHO.update()
                    sample_time += time.time() - sample_start

                    inflow_start = time.time()
                    utils.patch_inflow2d[blocks_per_grid, threads_per_block](inflow_tensor_numba, RHO.data_numba, DT, RES)
                    inflow_time += time.time() - inflow_start

                    sample_start = time.time()
                    vx_y_data = VX.resample(VY.points, vx_boundary_array, vx_data_resample)
                    vy_x_data = VY.resample(VX.points, vy_boundary_array, vy_data_resample)
                    vx_y_data_numba = cuda.as_cuda_array(vx_y_data)
                    vy_x_data_numba = cuda.as_cuda_array(vy_x_data)
                    sample_time += time.time() - sample_start

                    advect_start = time.time()
                    utils.semi_lagrangian_update2d[blocks_per_grid, threads_per_block](VX.points_numba, VX.data_numba, vy_x_data_numba, DT, RES)
                    utils.semi_lagrangian_update2d[blocks_per_grid, threads_per_block](VY.points_numba, vx_y_data_numba, VY.data_numba, DT, RES)
                    advect_time += time.time() - advect_start

                    sample_start = time.time()
                    VX.data = VX.resample(VX.points, vx_boundary_array, vx_data_resample)
                    VY.data = VY.resample(VY.points, vy_boundary_array, vy_data_resample)
                    VX.update()
                    VY.update()
                    sample_time = time.time() - sample_start

                    utils.buoyancy2d[blocks_per_grid, threads_per_block](VY.data_numba, RHO.data_numba, 9.81, FLOW.buoyancy_factor, RES)

                    print("Step {}".format(i))
                    print(" - Mean value of rho: {}".format(np.mean(RHO.data.cpu().data.numpy())))
                    print(" - Mean value of vx : {}".format(np.mean(VX.data.cpu().data.numpy())))
                    print(" - Mean value of vy : {}".format(np.mean(VY.data.cpu().data.numpy())))

                    sample_timings["GPU"][RES].append(sample_time)
                    inflow_timings["GPU"][RES].append(inflow_time)
                    advect_timings["GPU"][RES].append(advect_time)

                else:
                    sample_start = time.time()
                    VX.data = VX.resample(RHO.points, vx_boundary_array, vx_data_resample)
                    VY.data = VY.resample(RHO.points, vy_boundary_array, vy_data_resample)
                    VZ.data = VZ.resample(RHO.points, vz_boundary_array, vz_data_resample)
                    VX.update()
                    VY.update()
                    VZ.update()
                    sample_time += time.time() - sample_start

                    advect_start = time.time()
                    utils.semi_lagrangian_update3d[blocks_per_grid, threads_per_block](RHO.points_numba, VX.data_numba, VY.data_numba, VZ.data_numba, DT, RES)
                    advect_time += time.time() - advect_start

                    sample_start = time.time()
                    RHO.data = RHO.resample(RHO.points, rho_boundary_array, rho_data_resample)
                    RHO.update()
                    sample_time += time.time() - sample_start

                    inflow_start = time.time()
                    utils.patch_inflow3d[blocks_per_grid, threads_per_block](inflow_tensor_numba, RHO.data_numba, DT, RES)
                    inflow_time += time.time() - inflow_start

                    sample_start = time.time()
                    vx_y_data = VX.resample(VY.points, vx_boundary_array, vx_data_resample)
                    vx_z_data = VX.resample(VZ.points, vx_boundary_array, vx_data_resample)

                    vy_x_data = VY.resample(VX.points, vy_boundary_array, vy_data_resample)
                    vy_z_data = VY.resample(VZ.points, vy_boundary_array, vy_data_resample)

                    vz_x_data = VZ.resample(VX.points, vz_boundary_array, vz_data_resample)
                    vz_y_data = VZ.resample(VY.points, vz_boundary_array, vz_data_resample)

                    vx_y_data_numba = cuda.as_cuda_array(vx_y_data)
                    vx_z_data_numba = cuda.as_cuda_array(vx_z_data)
                    vy_x_data_numba = cuda.as_cuda_array(vy_x_data)
                    vy_z_data_numba = cuda.as_cuda_array(vy_z_data)
                    vz_x_data_numba = cuda.as_cuda_array(vz_x_data)
                    vz_y_data_numba = cuda.as_cuda_array(vz_y_data)
                    sample_time += time.time() - sample_start

                    advect_start = time.time()
                    utils.semi_lagrangian_update3d[blocks_per_grid, threads_per_block](VX.points_numba, VX.data_numba, vy_x_data_numba, vz_x_data_numba, DT, RES)
                    utils.semi_lagrangian_update3d[blocks_per_grid, threads_per_block](VY.points_numba, vx_y_data_numba, VY.data_numba, vz_y_data_numba, DT, RES)
                    utils.semi_lagrangian_update3d[blocks_per_grid, threads_per_block](VZ.points_numba, vx_z_data_numba, vy_z_data_numba, VZ.data_numba, DT, RES)
                    advect_time += time.time() - advect_start

                    sample_start = time.time()
                    VX.data = VX.resample(VX.points, vx_boundary_array, vx_data_resample)
                    VY.data = VY.resample(VY.points, vy_boundary_array, vy_data_resample)
                    VZ.data = VZ.resample(VZ.points, vz_boundary_array, vz_data_resample)
                    VX.update()
                    VY.update()
                    VZ.update()
                    sample_time = time.time() - sample_start

                    utils.buoyancy2d[blocks_per_grid, threads_per_block](VY.data_numba, RHO.data_numba, 9.81, FLOW.buoyancy_factor, RES)

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

                sample_start = time.time()
                x_rho = torch_from_numpy(DENSITY.points.data)
                v_rho = VELOCITY.sample_at(x_rho, device=device)
                sample_time += time.time() - sample_start

                advect_start = time.time()
                x_rho = utils.semi_lagrangian_update(x_rho, v_rho, DT)
                advect_time += time.time() - advect_start

                sample_start = time.time()
                x_rho = DENSITY.sample_at(x_rho, device)
                sample_time += time.time() - sample_start

                inflow_start = 0.0
                x_rho = utils.patch_inflow(inflow_tensor, x_rho, DT)
                inflow_time += time.time() - inflow_start

                x_vel_list = []

                for component in VELOCITY.unstack():
                    sample_start = time.time()
                    x_vel = torch_from_numpy(component.points.data)
                    v_vel = VELOCITY.sample_at(x_vel, device)
                    sample_time += time.time() - sample_start

                    advect_start = time.time()
                    x_vel = utils.semi_lagrangian_update(x_vel, v_vel, DT)
                    advect_time += time.time() - advect_start

                    sample_start = time.time()
                    x_vel = component.sample_at(x_vel, device)
                    sample_time += time.time() - sample_start

                    x_vel_list.append(x_vel)

                # Update the object data
                DENSITY = DENSITY.with_data(x_rho)
                VELOCITY = VELOCITY.with_data(x_vel_list)

                VELOCITY += buoyancy(DENSITY, 9.81, FLOW.buoyancy_factor)

                sample_timings["CPU"][RES].append(sample_time)
                inflow_timings["CPU"][RES].append(inflow_time)
                advect_timings["CPU"][RES].append(advect_time)

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


sample_gpu_means = []
advect_gpu_means = []
inflow_gpu_means = []

sample_cpu_means = []
advect_cpu_means = []
inflow_cpu_means = []

for res, timings in sample_timings["GPU"].items():
    sample_gpu_means.append(np.mean(np.asarray(timings)))

for res, timings in advect_timings["GPU"].items():
    advect_gpu_means.append(np.mean(np.asarray(timings)))

for res, timings in sample_timings["GPU"].items():
    inflow_gpu_means.append(np.mean(np.asarray(timings)))

for res, timings in sample_timings["CPU"].items():
    sample_cpu_means.append(np.mean(np.asarray(timings)))

for res, timings in advect_timings["CPU"].items():
    advect_cpu_means.append(np.mean(np.asarray(timings)))

for res, timings in sample_timings["CPU"].items():
    inflow_cpu_means.append(np.mean(np.asarray(timings)))


plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'text.usetex': True
})

fig_sample = plt.figure(figsize=(4,3))
fig_advect = plt.figure(figsize=(4,3))
fig_inflow = plt.figure(figsize=(4,3))

ax_sample = fig_sample.add_subplot(1,1,1)
ax_advect = fig_advect.add_subplot(1,1,1)
ax_inflow = fig_inflow.add_subplot(1,1,1)

ax_sample.plot(resolutions, sample_gpu_means, linestyle="-", color="black", marker="o")
ax_sample.plot(resolutions, sample_cpu_means, linestyle=":", color="black", marker="o")
ax_sample.legend(["GPU", "CPU"])
ax_sample.set_xlabel("Resolution")
ax_sample.set_ylabel("Time (s)")
ax_sample.set_xticks(resolutions)
ax_sample.set_title("Resampling")

ax_advect.plot(resolutions, advect_gpu_means, linestyle="-", color="black", marker="o")
ax_advect.plot(resolutions, advect_cpu_means, linestyle=":", color="black", marker="o")
ax_advect.legend(["GPU", "CPU"])
ax_advect.set_xlabel("Resolution")
ax_advect.set_ylabel("Time (s)")
ax_advect.set_xticks(resolutions)
ax_advect.set_title("Advection")

ax_inflow.plot(resolutions, inflow_gpu_means, linestyle="-", color="black", marker="o")
ax_inflow.plot(resolutions, inflow_cpu_means, linestyle=":", color="black", marker="o")
ax_inflow.legend(["GPU", "CPU"])
ax_inflow.set_xlabel("Resolution")
ax_inflow.set_ylabel("Time (s)")
ax_inflow.set_xticks(resolutions)
ax_inflow.set_title("Inflow Patching")

fig_sample.savefig(os.path.join(IMG_PATH, "sample_performance.png"))
fig_advect.savefig(os.path.join(IMG_PATH, "advect_performance.png"))
fig_inflow.savefig(os.path.join(IMG_PATH, "inflow_performance.png"))