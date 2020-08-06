from phi.tf.flow import *

domain = Domain([64, 80], boundaries=CLOSED, box=box[0:100, 0:100])
dt = 1.0
buoyancy_factor = 0.1

velocity = domain.grid(0, StaggeredGrid)
density = domain.grid(0)
inflow = mask(Sphere(center=(50, 10), radius=5)).sample_at(density.elements) * 0.2
divergence = domain.grid(0)


def step():
    global velocity, density, divergence
    density = advect.semi_lagrangian(density, velocity, dt) + inflow
    velocity = advect.semi_lagrangian(velocity, velocity, dt) + resample(density * (0, buoyancy_factor), velocity)
    velocity = divergence_free(velocity, domain)
    divergence = field.divergence(velocity)


step()


app = App('Simple Plume', framerate=10)
app.add_field('Velocity', lambda: velocity)
app.add_field('Density', lambda: density)
app.add_field('Divergence', lambda: divergence)
app.step = step
show(app)

# while True:

    # app.update({'velocity': velocity, 'density': density})
