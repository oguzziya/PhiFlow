from phi.torch.flow import *
import matplotlib.pyplot as plt
import torch
import time
from resample_torch_cuda import resample_op
from copy import copy
IS_PRESSURE = False
device = "cpu"
SAVE_IMAGE = False
ANIMATE = True

DIM = 3
BATCH_SIZE = 1
STEPS = 10
DT = 0.6

DT = np.float32(DT)

if SAVE_IMAGE:
    IMG_PATH = os.path.expanduser("~/Simulations/phi/data/manual/torch")
    if not os.path.exists(IMG_PATH):
        os.makedirs(IMG_PATH)

resolutions = [32,]
for RUN_GPU in [True]:
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

            # Obtain the domain boxes and move them to the device
            box_sizes_rho = torch.as_tensor(DENSITY.box.size).to(device)
            box_lower_rho = torch.as_tensor(DENSITY.box.lower).to(device)
            box_sizes_vx = torch.as_tensor(VEL_X.box.size).to(device)
            box_lower_vx = torch.as_tensor(VEL_X.box.lower).to(device)
            box_lower_vy = torch.as_tensor(VEL_Y.box.lower).to(device)
            box_sizes_vy = torch.as_tensor(VEL_Y.box.size).to(device)
            if DIM == 3:
                box_lower_vz = torch.as_tensor(VEL_Z.box.lower).to(device)
                box_sizes_vz = torch.as_tensor(VEL_Z.box.size).to(device)

            box_sizes_rho_numba = cuda.as_cuda_array(box_sizes_rho)
            box_lower_rho_numba = cuda.as_cuda_array(box_lower_rho)
            box_sizes_vx_numba = cuda.as_cuda_array(box_sizes_vx)
            box_lower_vx_numba = cuda.as_cuda_array(box_lower_vx)
            box_sizes_vy_numba = cuda.as_cuda_array(box_sizes_vy)
            box_lower_vy_numba = cuda.as_cuda_array(box_lower_vy)
            if DIM == 3:
                box_sizes_vz_numba = cuda.as_cuda_array(box_sizes_vz)
                box_lower_vz_numba = cuda.as_cuda_array(box_lower_vz)

            rho_data = torch.as_tensor(DENSITY.data).to(device)
            vx_data = torch.as_tensor(VEL_X.data).to(device)
            vy_data = torch.as_tensor(VEL_Y.data).to(device)
            if DIM == 3:
                vz_data = torch.as_tensor(VEL_Z.data).to(device)

            rho_points = torch.as_tensor(DENSITY.points.data).to(device)
            rho_points_x = torch.as_tensor(DENSITY.points.data).to(device)
            rho_points_y = torch.as_tensor(DENSITY.points.data).to(device)
            if DIM == 3:
                rho_points_z = torch.as_tensor(DENSITY.points.data).to(device)

            vx_points = torch.as_tensor(VEL_X.points.data).to(device)
            vy_points = torch.as_tensor(VEL_Y.points.data).to(device)
            if DIM == 3:
                vz_points = torch.as_tensor(VEL_Z.points.data).to(device)

            rho_data_numba = cuda.as_cuda_array(rho_data)
            vx_data_numba = cuda.as_cuda_array(vx_data)
            vy_data_numba = cuda.as_cuda_array(vy_data)
            if DIM == 3:
                vz_data_numba = cuda.as_cuda_array(vz_data)

            rho_points_numba = cuda.as_cuda_array(rho_points)
            rho_points_x_numba = cuda.as_cuda_array(rho_points_x)
            rho_points_y_numba = cuda.as_cuda_array(rho_points_y)
            if DIM == 3:
                rho_points_z_numba = cuda.as_cuda_array(rho_points_z)

            vx_points_numba = cuda.as_cuda_array(vx_points)
            vy_points_numba = cuda.as_cuda_array(vy_points)
            if DIM == 3:
                vz_points_numba = cuda.as_cuda_array(vz_points)

            rho_boundary_array = torch.as_tensor(utils.get_boundary_array(density_shape)).to(device)
            vx_boundary_array = torch.as_tensor(utils.get_boundary_array(vx_data.shape)).to(device)
            vy_boundary_array = torch.as_tensor(utils.get_boundary_array(vy_data.shape)).to(device)
            if DIM == 3:
                vz_boundary_array = torch.as_tensor(utils.get_boundary_array(vz_data.shape)).to(device)

            for i in range(STEPS):
                if DIM == 2:
                    rho_points_x_numba = copy(rho_points_numba)
                    rho_points_y_numba = copy(rho_points_numba)

                    utils.global_to_local2d[blocks_per_grid, threads_per_block](rho_points_x_numba, box_sizes_vx_numba, box_lower_vx_numba, RES)
                    vx_data = resample_op(vx_data, rho_points_x, vx_boundary_array, vx_data)

                    utils.global_to_local2d[blocks_per_grid, threads_per_block](rho_points_y_numba, box_sizes_vy_numba, box_lower_vy_numba, RES)
                    vy_data = resample_op(vy_data, rho_points_y, vy_boundary_array, vy_data)

                    utils.semi_lagrangian_update2d[blocks_per_grid, threads_per_block](rho_points_numba, vx_data_numba, vy_data_numba, DT, RES)

                    utils.global_to_local2d[blocks_per_grid, threads_per_block](rho_points_numba, box_sizes_rho_numba, box_lower_rho_numba, RES)
                    rho_data = resample_op(rho_data, rho_points, rho_boundary_array, rho_data)
                    utils.patch_inflow2d[blocks_per_grid, threads_per_block](inflow_tensor_numba, rho_data_numba, DT, RES)

                    utils.global_to_local2d[blocks_per_grid, threads_per_block](vx_points_numba, box_sizes_vx_numba, box_lower_vx_numba, RES)
                    utils.global_to_local2d[blocks_per_grid, threads_per_block](vy_points_numba, box_sizes_vy_numba, box_lower_vy_numba, RES)

                    vx_x_data = resample_op(vx_data, vx_points, vx_boundary_array, vx_data)
                    vx_y_data = resample_op(vx_data, vy_points, vx_boundary_array, vx_data)

                    vy_x_data = resample_op(vy_data, vx_points, vy_boundary_array, vy_data)
                    vy_y_data = resample_op(vy_data, vy_points, vy_boundary_array, vy_data)

                    vx_x_data_numba = cuda.as_cuda_array(vx_x_data)
                    vx_y_data_numba = cuda.as_cuda_array(vx_y_data)
                    vy_x_data_numba = cuda.as_cuda_array(vy_x_data)
                    vy_y_data_numba = cuda.as_cuda_array(vy_y_data)

                    utils.semi_lagrangian_update2d[blocks_per_grid, threads_per_block](vx_points_numba, vx_x_data_numba, vy_x_data_numba, DT, RES)
                    utils.semi_lagrangian_update2d[blocks_per_grid, threads_per_block](vy_points_numba, vx_y_data_numba, vy_y_data_numba, DT, RES)

                    utils.global_to_local2d[blocks_per_grid, threads_per_block](vx_points_numba, box_sizes_vx_numba, box_lower_vx_numba, RES)
                    utils.global_to_local2d[blocks_per_grid, threads_per_block](vy_points_numba, box_sizes_vy_numba, box_lower_vy_numba, RES)

                    vx_data = resample_op(vx_data, vx_points, vx_boundary_array, vx_data)
                    vy_data = resample_op(vy_data, vy_points, vy_boundary_array, vy_data)

                    utils.buoyancy2d[blocks_per_grid, threads_per_block](vy_data_numba, rho_data_numba, 9.81, RES)
                else:
                    rho_points_x_numba = copy(rho_points_numba)
                    rho_points_y_numba = copy(rho_points_numba)
                    rho_points_z_numba = copy(rho_points_numba)

                    utils.global_to_local3d[blocks_per_grid, threads_per_block](rho_points_x_numba, box_sizes_vx_numba, box_lower_vx_numba, RES)
                    vx_data = resample_op(vx_data, rho_points_x, vx_boundary_array, vx_data)

                    utils.global_to_local3d[blocks_per_grid, threads_per_block](rho_points_y_numba, box_sizes_vy_numba, box_lower_vy_numba, RES)
                    vy_data = resample_op(vy_data, rho_points_y, vy_boundary_array, vy_data)

                    utils.global_to_local3d[blocks_per_grid, threads_per_block](rho_points_z_numba, box_sizes_vz_numba, box_lower_vz_numba, RES)
                    vz_data = resample_op(vz_data, rho_points_z, vz_boundary_array, vz_data)

                    utils.semi_lagrangian_update3d[blocks_per_grid, threads_per_block](rho_points_numba, vx_data_numba, vy_data_numba, vz_data_numba, DT, RES)

                    utils.global_to_local3d[blocks_per_grid, threads_per_block](rho_points_numba, box_sizes_rho_numba, box_lower_rho_numba, RES)
                    rho_data = resample_op(rho_data, rho_points, rho_boundary_array, rho_data)
                    utils.patch_inflow3d[blocks_per_grid, threads_per_block](inflow_tensor_numba, rho_data_numba, DT, RES)

                    utils.global_to_local3d[blocks_per_grid, threads_per_block](vx_points_numba, box_sizes_vx_numba, box_lower_vx_numba, RES)
                    utils.global_to_local3d[blocks_per_grid, threads_per_block](vy_points_numba, box_sizes_vy_numba, box_lower_vy_numba, RES)
                    utils.global_to_local3d[blocks_per_grid, threads_per_block](vz_points_numba, box_sizes_vz_numba, box_lower_vz_numba, RES)

                    vx_x_data = resample_op(vx_data, vx_points, vx_boundary_array, vx_data)
                    vx_y_data = resample_op(vx_data, vy_points, vx_boundary_array, vx_data)
                    vx_z_data = resample_op(vx_data, vz_points, vy_boundary_array, vx_data)

                    vy_x_data = resample_op(vy_data, vx_points, vy_boundary_array, vy_data)
                    vy_y_data = resample_op(vy_data, vy_points, vy_boundary_array, vy_data)
                    vy_z_data = resample_op(vy_data, vz_points, vy_boundary_array, vy_data)

                    vz_x_data = resample_op(vz_data, vx_points, vy_boundary_array, vz_data)
                    vz_y_data = resample_op(vz_data, vy_points, vy_boundary_array, vz_data)
                    vz_z_data = resample_op(vz_data, vz_points, vz_boundary_array, vz_data)

                    vx_x_data_numba = cuda.as_cuda_array(vx_x_data)
                    vx_y_data_numba = cuda.as_cuda_array(vx_y_data)
                    vx_z_data_numba = cuda.as_cuda_array(vx_z_data)

                    vy_x_data_numba = cuda.as_cuda_array(vy_x_data)
                    vy_y_data_numba = cuda.as_cuda_array(vy_y_data)
                    vy_z_data_numba = cuda.as_cuda_array(vy_z_data)

                    vz_x_data_numba = cuda.as_cuda_array(vz_x_data)
                    vz_y_data_numba = cuda.as_cuda_array(vz_y_data)
                    vz_z_data_numba = cuda.as_cuda_array(vz_z_data)

                    utils.semi_lagrangian_update3d[blocks_per_grid, threads_per_block](vx_points_numba, vx_x_data_numba, vy_x_data_numba, vz_x_data_numba, DT, RES)
                    utils.semi_lagrangian_update3d[blocks_per_grid, threads_per_block](vy_points_numba, vx_y_data_numba, vy_y_data_numba, vz_y_data_numba, DT, RES)
                    utils.semi_lagrangian_update3d[blocks_per_grid, threads_per_block](vz_points_numba, vx_z_data_numba, vy_z_data_numba, vz_z_data_numba, DT, RES)

                    utils.global_to_local3d[blocks_per_grid, threads_per_block](vx_points_numba, box_sizes_vx_numba, box_lower_vx_numba, RES)
                    utils.global_to_local3d[blocks_per_grid, threads_per_block](vy_points_numba, box_sizes_vy_numba, box_lower_vy_numba, RES)
                    utils.global_to_local3d[blocks_per_grid, threads_per_block](vz_points_numba, box_sizes_vz_numba, box_lower_vz_numba, RES)

                    vx_data = resample_op(vx_data, vx_points, vx_boundary_array, vx_data)
                    vy_data = resample_op(vy_data, vy_points, vy_boundary_array, vy_data)
                    vz_data = resample_op(vz_data, vz_points, vz_boundary_array, vz_data)

                    utils.buoyancy3d[blocks_per_grid, threads_per_block](vy_data_numba, rho_data_numba, 9.81, RES)

                if IS_PRESSURE:
                    pressure_solve_start = time.time()
                    VELOCITY = divergence_free(VELOCITY, FLOW.domain, obstacles=(), pressure_solver=SparseCG(max_iterations=100))
                    pressure_solve_end = time.time()

                    pressure_time = pressure_solve_end - pressure_solve_start

                if ANIMATE:
                    array = rho_data.cpu().data.numpy()
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

        else:
            device = 'cpu'
            import utils as utils

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