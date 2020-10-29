from phi.flow import *
import numpy as np

import jax_utils
import jax.numpy as jnp
from jax import device_put

from PIL import Image

def save_img(array, scale, name, idx=0):
    if len(array.shape) <= 4:
        ima = np.reshape(array[idx], [array.shape[1], array.shape[2]])  # remove channel dimension , 2d
    else:
        ima = array[idx, :, array.shape[1] // 2, :, 0]  # 3d , middle z slice
    ima = np.reshape(ima, [array.shape[1], array.shape[2]])  # remove channel dimension
    # ima = ima[::-1, :]  # flip along y
    image = Image.fromarray(np.asarray(ima * scale, dtype='i'))
    print("    Writing image '" + name + "'")
    image.save(name)

DIM = 3  # 2d / 3d
BATCH_SIZE = 1  # process multiple independent simulations at once
STEPS = 10  # number of simulation STEPS
GRAPH_STEPS = 10  # how many STEPS to unroll in TF graph

RES = 32
DT = 0.6

# by default, creates a numpy state, i.e. "FLOW.density.data" is a numpy array
FLOW = Fluid(Domain([RES] * DIM, boundaries=OPEN), batch_size=BATCH_SIZE, buoyancy_factor=0.2)

IMG_PATH = os.path.expanduser("~/Simulations/phi/data/manual/numpy")
if not os.path.exists(IMG_PATH):
    os.makedirs(IMG_PATH)

if DIM == 2:
    density_shape = [1, RES, RES, 1]
else:
    density_shape = [1, RES, RES, RES, 1]

DENSITY = FLOW.density
VELOCITY = FLOW.velocity

density_tensor = DENSITY.data
velocity_tensor = [component.data for component in VELOCITY.unstack()]
inflow_tensor = jnp.zeros(shape=density_shape)

if DIM == 2:
    inflow_tensor = jax_utils.initialize_data_2d(inflow_tensor, RES)
    inflow_tensor = device_put(inflow_tensor)
else:
    inflow_tensor = jax_utils.initialize_data_3d(inflow_tensor, RES)
    inflow_tensor = device_put(inflow_tensor)

for i in range(GRAPH_STEPS):

    # Extract data and move to GPU
    x_rho_np = device_put(DENSITY.points.data)
    v_rho_np = device_put(VELOCITY.sample_at(x_rho_np))

    # Advect the density
    x_rho_np = jax_utils.semi_lagrangian_update(x_rho_np, v_rho_np, DT)
    x_rho_np = DENSITY.sample_at(x_rho_np)
    x_rho_np = jax_utils.patch_inflow(inflow_tensor, x_rho_np, DT)

    x_vel_list = []

    for component in VELOCITY.unstack():
        # Extract data and move to GPU
        x_vel = device_put(component.points.data)
        v_vel = device_put(VELOCITY.sample_at(x_vel))

        # Advect the velocity, move to GPU again necessary because sample_at results in numpy array again
        x_vel = jax_utils.semi_lagrangian_update(x_vel, v_vel, DT)
        x_vel = device_put(component.sample_at(x_vel))
        print(type(x_vel))
        x_vel_list.append(x_vel)

    # Update the object data
    DENSITY = DENSITY.with_data(x_rho_np)
    VELOCITY = VELOCITY.with_data(x_vel_list)

    VELOCITY += buoyancy(DENSITY, 9.81, FLOW.buoyancy_factor)
    VELOCITY = divergence_free(VELOCITY, FLOW.domain, obstacles=())

    save_img(DENSITY.data, 10000., IMG_PATH + "/numpy_%04d.png" % i)
    print("Numpy step %d done, means %s %s" % (i, np.mean(DENSITY.data), np.mean(VELOCITY.staggered_tensor())))
