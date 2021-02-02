# example that runs a "manual" simple incompressible fluid sim either in numpy or TF
# note, this example does not use the dash GUI, instead it creates PNG images via PIL

import sys
from phi.tf.flow import *

def run_tensorflow(resolution):
    DIM = 2  # 2d / 3d
    BATCH_SIZE = 1  # process multiple independent simulations at once
    STEPS = 11  # number of simulation STEPS
    GRAPH_STEPS = 3  # how many STEPS to unroll in TF graph

    RES = resolution
    DT = 0.6

    # by default, creates a numpy state, i.e. "FLOW.density.data" is a numpy array
    FLOW = Fluid(Domain([RES] * DIM, boundaries=OPEN), batch_size=BATCH_SIZE, buoyancy_factor=0.2)

    SCENE = Scene.create("~/phi/data/manual")
    SESSION = Session(SCENE)
    IMG_PATH = SCENE.path
    # create TF placeholders with the correct shapes
    FLOW_IN = FLOW.copied_with(density=placeholder, velocity=placeholder)
    DENSITY = FLOW_IN.density
    VELOCITY = FLOW_IN.velocity

    # optional , write images
    SAVE_IMAGES = False

    # main , step 1: run FLOW sim (numpy), or only set up graph for TF

    for i in range(GRAPH_STEPS):
        # simulation step; note that the core is only 3 lines for the actual simulation
        # the RESt is setting up the inflow, and debug info afterwards

        INFLOW_DENSITY = math.zeros_like(FLOW.density)
        if DIM == 2:
            # (batch, y, x, components)
            INFLOW_DENSITY.data[..., (RES // 4 * 2):(RES // 4 * 3), (RES // 4):(RES // 4 * 3), 0] = 1.
        else:
            # (batch, z, y, x, components)
            INFLOW_DENSITY.data[..., (RES // 4 * 2):(RES // 4 * 3), (RES // 4 * 1):(RES // 4 * 3), (RES // 4):(RES // 4 * 3), 0] = 1.

        DENSITY = advect.semi_lagrangian(DENSITY, VELOCITY, DT) + DT * INFLOW_DENSITY
        VELOCITY = advect.semi_lagrangian(VELOCITY, VELOCITY, DT) + buoyancy(DENSITY, 9.81, FLOW.buoyancy_factor) * DT
        VELOCITY = divergence_free(VELOCITY, FLOW.domain, obstacles=())

    # main , step 2: do actual sim run (TF only)
    # for TF, all the work still needs to be done, feed empty state and start simulation
    FLOW_OUT = FLOW.copied_with(density=DENSITY, velocity=VELOCITY, age=FLOW.age + DT)

    # run session
    for i in range(STEPS // GRAPH_STEPS):
        FLOW = SESSION.run(FLOW_OUT, feed_dict={FLOW_IN: FLOW})  # Passes DENSITY and VELOCITY tensors

def run():
    run_tensorflow(int(sys.argv[1]))

if __name__ == '__main__':
    run()
