from ._field import Field, StaggeredSamplePoints, IncompatibleFieldTypes
from ._constant import ConstantField
from ._grid import CenteredGrid
from ._staggered_grid import StaggeredGrid, unstack_staggered_tensor, stack_staggered_components, staggered_component_box
from ._sampled import SampledField
from ._analytic import AnalyticField, SymbolicFieldBackend
from ._mask import GeometryMask, mask
from ._noise import Noise
from ._util import diffuse, data_bounds, staggered_curl_2d
from ._angular_velocity import AngularVelocity
from . import _advect as advect
from . import _manta


from phi import math

math.DYNAMIC_BACKEND.add_backend(SymbolicFieldBackend(math.DYNAMIC_BACKEND), priority=True)
