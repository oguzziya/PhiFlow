# pylint: disable-msg = unused-import
"""
Use this module as your main PhiFlow import.

from phi.flow import *

Contains:

* App class, show()
* phi.math as math
* Geometry objects
* phi.field and common Field classes
* Common physics functions such as diffuse, divergence_free, advect family
* I/O functions such as write_sim_frame
"""

import numpy
import numpy as np

from phi import math, struct

from phi.geom import Geometry, Sphere, box, GLOBAL_AXIS_ORDER, union

from phi import field
from phi.field import Grid, CenteredGrid, StaggeredGrid, GeometryMask

from phi.physics.domain import Domain
from phi.physics.material import Material, OPEN, CLOSED, PERIODIC, NO_SLIP, NO_STICK, STICKY, SLIPPERY
from phi.physics.common_physics import diffuse
from phi.physics import advect
from phi.physics.fluid import divergence_free, masked_laplace

from phi.data.fluidformat import write_sim_frame

from phi.app.app import App
from phi.viz.display import show

physics_config = GLOBAL_AXIS_ORDER
