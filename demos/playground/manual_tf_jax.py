# example that runs a "manual" simple incompressible fluid sim either in numpy or TF
# note, this example does not use the dash GUI, instead it creates PNG images via PIL

from phi.tf.flow import *  # Use TensorFlow
import tensorflow as tf
import jax_utils

import jax.numpy as jnp

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
GRAPH_STEPS = 1  # how many STEPS to unroll in TF graph

RES = 64
DT = 0.6

# by default, creates a numpy state, i.e. "FLOW.density.data" is a numpy array
FLOW = Fluid(Domain([RES] * DIM, boundaries=OPEN), batch_size=BATCH_SIZE, buoyancy_factor=0.2)

SCENE = Scene.create("~/Simulations/phi/data/manual")
SESSION = Session(SCENE)
IMG_PATH = SCENE.path

if DIM == 2:
    density_shape = [1, RES, RES, 1]
else:
    density_shape = [1, RES, RES, RES, 1]

# create TF placeholders with the correct shapes
#
FLOW_IN = FLOW.copied_with(density=placeholder, velocity=placeholder)
DENSITY = FLOW_IN.density
VELOCITY = FLOW_IN.velocity

density_tensor = DENSITY.data
velocity_tensor = [component.data for component in VELOCITY.unstack()]
inflow_tensor = jnp.zeros(shape=density_shape)

# Set the inflow density value
if DIM == 2:
    inflow_tensor = jax_utils.initialize_data_2d(inflow_tensor, RES)
else:
    inflow_tensor = jax_utils.initialize_data_3d(inflow_tensor, RES)


for i in range(GRAPH_STEPS):

    # Advect density
    x = DENSITY.points.data
    v = VELOCITY.sample_at(x)
    x = jax_utils.semi_lagrangian_update(x, v, DT)
    x = DENSITY.sample_at(x)
    x = jax_utils.patch_inflow(inflow_tensor, x, DT)

    x_vel_list = []

    for component in VELOCITY.unstack():
        x_vel = component.points.data
        v_vel = VELOCITY.sample_at(x_vel)
        x_vel = jax_utils.semi_lagrangian_update(x_vel, v_vel, DT)
        x_vel = component.sample_at(x_vel)
        x_vel_list.append(x_vel)

    # Update the object data
    DENSITY = DENSITY.with_data(x)
    VELOCITY = VELOCITY.with_data(x_vel_list)

    # Buoyancy effects and incompressible flow
    VELOCITY += buoyancy(DENSITY, 9.81, FLOW.buoyancy_factor)
    VELOCITY = divergence_free(VELOCITY, FLOW.domain, obstacles=())

# main , step 2: do actual sim run (TF only)

# for TF, all the work still needs to be done, feed empty state and start simulation
FLOW_OUT = FLOW.copied_with(density=DENSITY, velocity=VELOCITY, age=FLOW.age + DT)

# run session
for i in range(STEPS // GRAPH_STEPS):
    FLOW = SESSION.run(FLOW_OUT, feed_dict={FLOW_IN: FLOW})  # Passes DENSITY and VELOCITY tensors

    save_img(FLOW.density.data, 10000., IMG_PATH + "/tf_%04d.png" % (GRAPH_STEPS * (i + 1) - 1))

    # for TF, we only have results now after each GRAPH_STEPS iterations
    print("Step SESSION.run %04d done, DENSITY shape %s, means %s %s" %
          (i, FLOW.density.data.shape, np.mean(FLOW.density.data), np.mean(FLOW.velocity.staggered_tensor())))