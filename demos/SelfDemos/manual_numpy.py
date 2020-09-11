# This example runs NumPy backend which is tried to be optimized by Numba

# Use NumPy backend
from phi.flow import *
import time
import numba_utils

DIM = 3  # 2d / 3d
BATCH_SIZE = 1  # process multiple independent simulations at once
STEPS = 5  # number of simulation STEPS
GRAPH_STEPS = 2  # how many STEPS to unroll in TF graph

RES = 32
DT = 0.6

# by default, creates a numpy state, i.e. "FLOW.density.data" is a numpy array
FLOW = Fluid(Domain([RES] * DIM, boundaries=OPEN), batch_size=BATCH_SIZE, buoyancy_factor=0.2)

DENSITY = FLOW.density
VELOCITY = FLOW.velocity
INFLOW_DENSITY = math.zeros_like(FLOW.density)

density_data = DENSITY.data
velocity_data = VELOCITY.data
inflow_data = INFLOW_DENSITY.data

## Remove class structres and work directly on the underlying data
# - What to do for the sample_at?: In the semi-lagrangian calculation we get the updated x values
# - then the "sample_at" function calculates the updated velocity field.

for i in range(STEPS):
    start = time.time()

    density_data = numba_utils.semi_lagrangian(DENSITY, VELOCITY, DT) +  DT * inflow_data
    advected_data = [numba_utils.semi_lagrangian(component, VELOCITY, DT) for component in VELOCITY.unstack()]

    VELOCITY = VELOCITY.with_data(velocity_data)
    VELOCITY = divergence_free(VELOCITY, FLOW.domain, obstacles=())

    end = time.time()

    print("Step %i is calculated in %f seconds" % (i + 1, end - start))

print("Simulation has finished.")