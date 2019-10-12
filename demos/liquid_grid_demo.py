import sys
if 'tf' in sys.argv:
    from phi.tf.flow import *  # Use TensorFlow
    mode = 'TensorFlow'
else:
    from phi.flow import *  # Use NumPy
    mode = 'NumPy'


class GridDemo(FieldSequenceModel):

    def __init__(self):
        FieldSequenceModel.__init__(self, 'Grid-based simulation', "Grid-based liquid simulation using %s backend." % mode, stride=1)

        size = [64, 80]
        domain = Domain(size, SLIPPERY)
        self.dt = 0.1

        self.initial_density = zeros(domain.grid.shape())
        self.initial_velocity = zeros(domain.grid.staggered_shape())
        self.initial_density[:, size[-2] * 6 // 8 : size[-2] * 8 // 8 - 1, size[-1] * 2 // 8 : size[-1] * 6 // 8, :] = 1
        self.initial_density[:, size[-2] * 0 // 8 : size[-2] * 2 // 8, size[-1] * 0 // 8 : size[-1] * 8 // 8, :] = 1
        self.initial_velocity.staggered[:, size[-2] * 5 // 8 : size[-2] * 7 // 8 + 0, size[-1] * 2 // 8 : size[-1] * 6 // 8 + 1, :] = [0, -0.0]

        self.liquid = world.GridLiquid(domain=domain, density=self.initial_density, velocity=self.initial_velocity, gravity=-9.81)
        #world.Inflow(Sphere((10,32), 5), rate=0.2)

        self.add_field("Density", lambda: self.liquid.density)
        self.add_field("Velocity", lambda: self.liquid.velocity.staggered)


    def step(self):
        world.step(dt=self.dt)

    def action_reset(self):
        self.liquid.density = self.initial_density
        self.liquid.velocity = self.initial_velocity
        self.time = 0


app = GridDemo().show(production=__name__ != "__main__", framerate=3, display=("Density", "Velocity"))