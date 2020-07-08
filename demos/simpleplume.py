from phi.flow import *
# physics_config.x_first()

domain = Domain([80, 64], boundaries=CLOSED)
dt = 1.0
buoyancy_factor = 0.1
inflow = mask(Sphere(center=(10, 32), radius=5)) * 0.2
velocity = StaggeredGrid.sample(0, domain)
density = CenteredGrid.sample(0, domain)

app = show(App('Simple Plume', framerate=10, fields={'velocity': velocity, 'density': density}))
while True:
    density = advect.semi_lagrangian(density, velocity, dt) + inflow
    velocity = advect.semi_lagrangian(velocity, velocity, dt) + density * (buoyancy_factor, 0)
    velocity = divergence_free(velocity, domain)
    app.update({'velocity': velocity, 'density': density})
