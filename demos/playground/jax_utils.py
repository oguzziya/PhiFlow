from phi.tf.flow import *
import jax.numpy as jnp
from jax.ops import index, index_update
import eagerpy as ep

# IMPORTANT
# Jax allocates 90 percent of available GPU memory, which causes out of memory errors when used with
# TensorFlow GPU
# To suppress: export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Jax does not support mutation of the arrays. Therefore index_update should be used
def initialize_data_2d(data, res):
    patched = index_update(data, index[..., (res // 4 * 2):(res // 4 * 3), (res // 4):(res // 4 * 3), 0], 1.0)
    return patched

def initialize_data_3d(data, res):
    patched = index_update(data, index[..., (res // 4 * 2):(res // 4 * 3), (res // 4 * 1):(res // 4 * 3), (res // 4):(res // 4 * 3), 0], 1.0)
    return patched

def semi_lagrangian_update(x, v, dt):
    x_new = x - v*dt
    return x_new

def patch_inflow(inflow_tensor, x, dt):
    return x + inflow_tensor * dt