from . import _extrapolation as extrapolation
from ._extrapolation import Extrapolation, ConstantExtrapolation
from ._backend import Backend
from ._dynamic_backend import DYNAMIC_BACKEND, set_precision, NoBackendFound
from ._backend_helper import split_multi_mode_pad, pad_constant_boundaries, apply_boundary, PadSettings, general_grid_sample_nd, circular_pad, replicate_pad
from ._scipy_backend import SCIPY_BACKEND, SciPyBackend

DYNAMIC_BACKEND.add_backend(SCIPY_BACKEND)
DYNAMIC_BACKEND.default_backend = SCIPY_BACKEND
math = DYNAMIC_BACKEND
