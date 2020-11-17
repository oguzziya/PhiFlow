from phi.torch.flow import *
import matplotlib.pyplot as plt
import torch
import time

IS_PRESSURE = False
device = "cpu"
SAVE_IMAGE = False
ANIMATE = False

DIM = 2
BATCH_SIZE = 1
STEPS = 20
DT = 0.6

DT = np.float32(DT)

if SAVE_IMAGE:
    IMG_PATH = os.path.expanduser("~/Simulations/phi/data/manual/torch")
    if not os.path.exists(IMG_PATH):
        os.makedirs(IMG_PATH)

advection_percentage_cpu = {}
sampling_percentage_cpu = {}
inflow_patch_percentage_cpu = {}
total_time_cpu = {}

advection_percentage_gpu = {}
sampling_percentage_gpu = {}
inflow_patch_percentage_gpu = {}
total_time_gpu = {}

resolutions = [32, 64, 128, 256, 512, 1024, 2048]

for RUN_GPU in [False, True]:
    for RES in resolutions:
        if RUN_GPU:
            device = "cuda:0"
            import utils_gpu as utils
            from numba import cuda
        else:
            import utils as utils

        if RUN_GPU:
            advection_percentage_gpu[RES] = []
            sampling_percentage_gpu[RES] = []
            inflow_patch_percentage_gpu[RES] = []
            total_time_gpu[RES] = []
        else:
            advection_percentage_cpu[RES] = []
            sampling_percentage_cpu[RES] = []
            inflow_patch_percentage_cpu[RES] = []
            total_time_cpu[RES] = []

        FLOW = Fluid(Domain([RES] * DIM, boundaries=OPEN), batch_size=BATCH_SIZE, buoyancy_factor=0.2)
        FLOW_TORCH = torch_from_numpy(FLOW)

        DENSITY = FLOW_TORCH.density
        VELOCITY = FLOW_TORCH.velocity

        threads_per_block = None
        blocks_per_grid = None

        if DIM == 2:
            threads_per_block = (min(RES, 32), min(RES, 32))
            blocks_per_grid = (int(np.ceil(RES / threads_per_block[0])), int(np.ceil(RES / threads_per_block[1])))
        else:
            threads_per_block = (8, 8, 8)
            blocks_per_grid = (int(RES / threads_per_block[0]), int(RES / threads_per_block[1]), int(RES / threads_per_block[2]))

        inflow_tensor_numba = None
        inflow_tensor = None

        if DIM == 2:
            density_shape = [1, RES, RES, 1]
            if RUN_GPU:
                inflow_tensor = torch.zeros(size=density_shape)
                inflow_tensor_numba = cuda.as_cuda_array(inflow_tensor.to(device))
                utils.initialize_data_2d[blocks_per_grid, threads_per_block](inflow_tensor_numba, RES)
            else:
                inflow_tensor = torch.zeros(size=density_shape)
                inflow_tensor = utils.initialize_data_2d(inflow_tensor, RES)
        else:
            density_shape = [1, RES, RES, RES, 1]
            if RUN_GPU:
                inflow_tensor = torch.zeros(size=density_shape)
                inflow_tensor_numba = cuda.as_cuda_array(inflow_tensor.to(device))
                utils.initialize_data_3d[blocks_per_grid, threads_per_block](inflow_tensor_numba, RES)
            else:
                inflow_tensor = torch.zeros(size=density_shape)
                inflow_tensor = utils.initialize_data_3d(inflow_tensor, RES)


        for i in range(STEPS):
            start = time.time()
            sample_counter = 0.0
            advection_counter = 0.0

            x_rho = torch_from_numpy(DENSITY.points.data)

            sample_start = time.time()
            v_rho = VELOCITY.sample_at(x_rho)
            sample_end = time.time()
            sample_counter += sample_end - sample_start

            advection_start = time.time()
            if RUN_GPU:
                x_rho_gpu = x_rho.to(device)
                x_rho_numba = cuda.as_cuda_array(x_rho_gpu)
                v_rho_gpu = v_rho.to(device)
                v_rho_numba = cuda.as_cuda_array(v_rho_gpu)
                if DIM == 2:
                    utils.semi_lagrangian_update2d[blocks_per_grid, threads_per_block](x_rho_numba, v_rho_numba, DT, RES)
                else:
                    utils.semi_lagrangian_update3d[blocks_per_grid, threads_per_block](x_rho_numba, v_rho_numba, DT)
                x_rho = x_rho_gpu.to("cpu")
            else:
                x_rho = utils.semi_lagrangian_update(x_rho, v_rho, DT)
            advection_end = time.time()
            advection_counter += advection_end - advection_start

            sample_start = time.time()
            x_rho = DENSITY.sample_at(x_rho)
            sample_end = time.time()
            sample_counter += sample_end - sample_start

            inflow_patch_start = time.time()
            if RUN_GPU:
                x_rho_gpu = x_rho.to(device)
                x_rho_numba = cuda.as_cuda_array(x_rho_gpu)
                if DIM == 2:
                    utils.patch_inflow2d[blocks_per_grid, threads_per_block](inflow_tensor_numba, x_rho_numba, DT, RES)
                else:
                    utils.patch_inflow3d[blocks_per_grid, threads_per_block](inflow_tensor_numba, x_rho_numba, DT)
                x_rho = x_rho_gpu.to("cpu")
            else:
                x_rho = utils.patch_inflow(inflow_tensor, x_rho, DT)
            inflow_patch_end = time.time()
            inflow_patch_time = inflow_patch_end - inflow_patch_start

            x_vel_list = []

            for component in VELOCITY.unstack():
                x_vel = torch_from_numpy(component.points.data)

                sample_start = time.time()
                v_vel = VELOCITY.sample_at(x_vel)
                sample_end = time.time()
                sample_counter += sample_end - sample_start

                advection_start = time.time()
                if RUN_GPU:
                    x_vel_gpu = x_vel.to(device)
                    x_vel_numba = cuda.as_cuda_array(x_vel_gpu)
                    v_vel_gpu = v_vel.to(device)
                    v_vel_numba = cuda.as_cuda_array(v_vel_gpu)
                    if DIM == 2:
                        utils.semi_lagrangian_update2d[blocks_per_grid, threads_per_block](x_vel_numba, v_vel_numba, DT, RES)
                    else:
                        utils.semi_lagrangian_update3d[blocks_per_grid, threads_per_block](x_vel_numba, v_vel_numba, DT)
                    x_vel = x_vel_gpu.to("cpu")
                else:
                    x_vel = utils.semi_lagrangian_update(x_vel, v_vel, DT)
                advection_end = time.time()
                advection_counter += advection_end - advection_start

                sample_start = time.time()
                x_vel = component.sample_at(x_vel)
                sample_end = time.time()
                sample_counter += sample_end - sample_start

                x_vel_list.append(x_vel)

            # Update the object data
            DENSITY = DENSITY.with_data(x_rho)
            VELOCITY = VELOCITY.with_data(x_vel_list)

            VELOCITY += buoyancy(DENSITY, 9.81, FLOW.buoyancy_factor)

            if IS_PRESSURE:
                pressure_solve_start = time.time()
                VELOCITY = divergence_free(VELOCITY, FLOW.domain, obstacles=(), pressure_solver=SparseCG(max_iterations=100))
                pressure_solve_end = time.time()

                pressure_time = pressure_solve_end - pressure_solve_start

            if SAVE_IMAGE:
                utils.save_img(DENSITY.data, 10000., IMG_PATH + "/torch_%04d.png" % i)

            if ANIMATE:
                array = DENSITY.data
                if len(array.shape) <= 4:
                    ima = np.reshape(array[0], [array.shape[1], array.shape[2]])  # remove channel dimension , 2d
                else:
                    ima = array[0, :, array.shape[1] // 2, :, 0]  # 3d , middle z slice
                ima = np.reshape(ima, [array.shape[1], array.shape[2]])  # remove channel dimension

                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                ax.contourf(ima, cmap='inferno')
                plt.pause(0.5)
                plt.close(fig)

            end = time.time()

            total_time = end-start

            if RUN_GPU:
                advection_percentage_gpu[RES].append(advection_counter)
                sampling_percentage_gpu[RES].append(sample_counter)
                inflow_patch_percentage_gpu[RES].append(inflow_patch_time)
                total_time_gpu[RES].append(total_time)
            else:
                advection_percentage_cpu[RES].append(advection_counter)
                sampling_percentage_cpu[RES].append(sample_counter)
                inflow_patch_percentage_cpu[RES].append(inflow_patch_time)
                total_time_cpu[RES].append(total_time)

            print("-------------------------------------------------------------------------------------")
            print("Step", i, "executed in", total_time, "seconds with GPU", RUN_GPU)
            if IS_PRESSURE:
                print("- Pressure Solver: ", pressure_time/total_time * 100, "%")
            print("- Advection: ", advection_counter / total_time * 100, "%")
            print("- Sampling: ", sample_counter / total_time * 100, "%")
            print("- Inflow patch: ", inflow_patch_time / total_time * 100, "%")

advection_percentage_means_gpu = []
sampling_percentage_means_gpu = []
inflow_patch_percentage_means_gpu = []
total_time_means_gpu = []

advection_percentage_means_cpu = []
sampling_percentage_means_cpu = []
inflow_patch_percentage_means_cpu = []
total_time_means_cpu = []

for res in resolutions:
    advection_percentage_means_gpu.append(np.mean(advection_percentage_gpu[res]))
    sampling_percentage_means_gpu.append(np.mean(sampling_percentage_gpu[res]))
    inflow_patch_percentage_means_gpu.append(np.mean(inflow_patch_percentage_gpu[res]))
    total_time_means_gpu.append(np.mean(total_time_gpu[res]))

    advection_percentage_means_cpu.append(np.mean(advection_percentage_cpu[res]))
    sampling_percentage_means_cpu.append(np.mean(sampling_percentage_cpu[res]))
    inflow_patch_percentage_means_cpu.append(np.mean(inflow_patch_percentage_cpu[res]))
    total_time_means_cpu.append(np.mean(total_time_cpu[res]))

fig, (ax1, ax2) = plt.subplots(2,1)

advection_plot_gpu = ax1.plot(resolutions, advection_percentage_means_gpu, label="Advection Patching - GPU", marker="o", linestyle="-")
inflow_plot_gpu = ax1.plot(resolutions, inflow_patch_percentage_means_gpu, label="Inflow Patching - GPU", marker="x", linestyle="-")
advection_plot_cpu = ax1.plot(resolutions, advection_percentage_means_cpu, label="Advection Patching - CPU", marker="*", linestyle="-")
inflow_plot_cpu = ax1.plot(resolutions, inflow_patch_percentage_means_cpu, label="Inflow Patching - CPU", marker="h", linestyle="-")
ax1.grid()
ax1.set_xticks(resolutions)
ax1.set_xlabel("Resolution", fontsize=18)
ax1.set_ylabel("Time (s)", fontsize=18)
ax1.set_ylim([0.0, 0.1])
ax1.set_title("Execution Time per Time Step - 2D - PyTorch-Numba", fontsize=12)
ax1.legend()

total_time_cpu = ax2.plot(resolutions, total_time_means_cpu, label="Total Time - CPU", marker="o", linestyle="-")
total_time_gpu = ax2.plot(resolutions, total_time_means_gpu, label="Total Time - GPU", marker="x", linestyle="-")
ax2.grid()
ax2.set_xticks(resolutions)
ax2.set_xlabel("Resolution", fontsize=18)
ax2.set_ylabel("Time (s)", fontsize=18)
ax2.set_ylim([0.0, 7])
ax2.set_title("Total Execution Time per Time Step - 2D - PyTorch-Numba", fontsize=12)
ax2.legend()

plt.show()