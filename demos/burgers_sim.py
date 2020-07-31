from phi.flow import *

domain = Domain([64, 64], boundaries=PERIODIC, box=box[0:100, 0:100])
# velocity = domain.grid(Noise((2,))) * 2
# velocity = domain.grid(Noise((2,)), type=StaggeredGrid) * 2

velocity = domain.vec_grid(mask(Sphere([50, 50], radius=15))) #  * [2, 0] + [1, 0]
assert velocity.shape.channel.volume == 2


def step():
    global velocity
    velocity = diffuse(velocity, 0.1, 1)
    velocity = advect.semi_lagrangian(velocity, velocity, 1.0)


step()

app = App('Burgers Equation in %dD' % len(domain.resolution), framerate=5)
app.add_field('Velocity', lambda: velocity)
app.step = step
show(app)
