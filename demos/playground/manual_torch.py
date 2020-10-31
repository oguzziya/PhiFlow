from phi.torch.flow import *
import utils
import matplotlib.pyplot as plt

DIM = 2
BATCH_SIZE = 1
STEPS = 10
RES = 128
DT = 0.6

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

if DIM == 2:
    density_shape = [1, RES, RES, 1]
    inflow_tensor = torch.zeros(size=density_shape)
    inflow_tensor = utils.initialize_data_2d(inflow_tensor, RES)
else:
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
    VELOCITY = divergence_free(VELOCITY, FLOW.domain, obstacles=(), pressure_solver=SparseCG(max_iterations=100))

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
        plt.pause(0.00001)
        plt.close(fig)

    print("Step Done: ", i, " Means: ", x_rho.mean(), ", ", VELOCITY.staggered_tensor().mean())
