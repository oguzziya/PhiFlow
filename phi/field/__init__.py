from ._field import Field, IncompatibleFieldTypes, resample
from ._constant import ConstantField
from ._grid import Grid, CenteredGrid
from ._staggered_grid import StaggeredGrid, unstack_staggered_tensor, stack_staggered_components
from ._sampled import SampledField
from ._analytic import AnalyticField, SymbolicFieldBackend
from ._mask import GeometryMask, mask
from ._noise import Noise
from ._util import diffuse, data_bounds, staggered_curl_2d
from ._angular_velocity import AngularVelocity
from . import _advect as advect
from . import _manta
from . import _field_math as math


from phi import math as _math
_math.DYNAMIC_BACKEND.add_backend(SymbolicFieldBackend(_math.DYNAMIC_BACKEND), priority=True)
