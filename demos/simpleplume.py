from phi.flow import *
# physics_config.x_first()

domain = Domain([80, 64], boundaries=CLOSED)
dt = 1.0
buoyancy_factor = 0.1
velocity = domain.staggered_grid(0)
density = domain.grid(0)
inflow = field.resample(mask(Sphere(center=(10, 32), radius=5)), density) * 0.2


def step():
    global velocity, density
    density = advect.semi_lagrangian(density, velocity, dt) + inflow
    velocity = advect.semi_lagrangian(velocity, velocity, dt) + resample(density * (buoyancy_factor, 0), velocity)
    velocity = divergence_free(velocity, domain)


step()


app = show(App('Simple Plume', framerate=10))

# while True:

    # app.update({'velocity': velocity, 'density': density})
