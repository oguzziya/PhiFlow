from phi import math
from .tf_backend import TF_BACKEND

math.DYNAMIC_BACKEND.add_backend(TF_BACKEND, priority=True)
if math.DYNAMIC_BACKEND.default_backend is None:
    math.DYNAMIC_BACKEND.default_backend = TF_BACKEND
