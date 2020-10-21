from phi.tf.flow import *
import jax.numpy as jnp
from jax.ops import index, index_update

# IMPORTANT
# Jax allocates 90 percent of available memory, which causes out of memory errors when used with
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
    x -= v * dt
    return x

def semi_lagrangian_tf(field, velocity_field, dt, field_shape):
    x = field.points.data
    v = velocity_field.at(field.points)

    x = semi_lagrangian_update(x, v.data, dt)

    data = field.sample_at(x)

    return data