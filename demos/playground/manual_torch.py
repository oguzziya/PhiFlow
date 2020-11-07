from phi.torch.flow import *
import matplotlib.pyplot as plt
import torch
import time

RUN_GPU = False
device = "cpu"

if RUN_GPU:
    device = "cuda:0"
    import utils_gpu as utils
    from numba import cuda
else:
    import utils as utils

DIM = 2
BATCH_SIZE = 1
STEPS = 30
RES = 128
DT = 0.6

DT = np.float32(DT)

SAVE_IMAGE = False
ANIMATE = False

if SAVE_IMAGE:
    IMG_PATH = os.path.expanduser("~/Simulations/phi/data/manual/torch")
    if not os.path.exists(IMG_PATH):
        os.makedirs(IMG_PATH)

FLOW = Fluid(Domain([RES] * DIM, boundaries=OPEN), batch_size=BATCH_SIZE, buoyancy_factor=0.2)
FLOW_TORCH = torch_from_numpy(FLOW)

DENSITY = FLOW_TORCH.density
VELOCITY = FLOW_TORCH.velocity

threads_per_block = None
blocks_per_grid = None

if DIM == 2:
    threads_per_block = (32, 32)
    blocks_per_grid = (int(RES / threads_per_block[0]), int(RES / threads_per_block[1]))
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

    x_rho = torch_from_numpy(DENSITY.points.data)

    sample_start = time.time()
    v_rho = VELOCITY.sample_at(x_rho)
    sample_end = time.time()

    sample_counter += sample_end - sample_start

    x_rho = utils.semi_lagrangian_update(x_rho, v_rho, DT)

    sample_start = time.time()
    x_rho = DENSITY.sample_at(x_rho)
    sample_end = time.time()

    sample_counter += sample_end - sample_start

    inflow_patch_start = time.time()
    if RUN_GPU:
        x_rho_gpu = x_rho.to(device)
        x_rho_numba = cuda.as_cuda_array(x_rho_gpu)
        if DIM == 2:
            utils.patch_inflow2d[blocks_per_grid, threads_per_block](inflow_tensor_numba, x_rho_numba, DT)
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

        x_vel = utils.semi_lagrangian_update(x_vel, v_vel, DT)

        sample_start = time.time()
        x_vel = component.sample_at(x_vel)
        sample_end = time.time()
        sample_counter += sample_end - sample_start

        x_vel_list.append(x_vel)

    # Update the object data
    DENSITY = DENSITY.with_data(x_rho)
    VELOCITY = VELOCITY.with_data(x_vel_list)

    VELOCITY += buoyancy(DENSITY, 9.81, FLOW.buoyancy_factor)

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

    #print("Step Done: ", i, " Means: ", DENSITY.data.mean(), ", ", VELOCITY.staggered_tensor().mean())
    total_time = end-start

    print("---------------------------------------------")
    print("Step", i, "executed in", total_time, "seconds")
    print("- Pressure Solver: ", pressure_time/total_time * 100, "%")
    print("- Sampling: ", sample_counter / total_time * 100, "%")
    print("- Inflow patch: ", inflow_patch_time / total_time * 100, "%")