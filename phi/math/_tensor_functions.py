import numpy as np
from phi.backend.dynamic_backend import DYNAMIC_BACKEND as math
from ._tensors import AbstractTensor, tensor, broadcastable_native_tensors


def close(*tensors, rel_tolerance=1e-5, abs_tolerance=0):
    """
    Checks whether all tensors have equal values within the specified tolerance.

    Does not check that the shapes exactly match.
    Tensors with different shapes are reshaped before comparing.

    :param tensors: tensor or tensor-like (constant) each
    :param rel_tolerance: relative tolerance
    :param abs_tolerance: absolute tolerance
    """
    tensors = [tensor(t) for t in tensors]
    for other in tensors[1:]:
        if not _close(tensors[0], other, rel_tolerance=rel_tolerance, abs_tolerance=abs_tolerance):
            return False
    return True


def _close(tensor1, tensor2, rel_tolerance=1e-5, abs_tolerance=0):
    if tensor2 is tensor1:
        return True
    new_shape, (native1, native2) = broadcastable_native_tensors(tensor1, tensor2)
    np1 = math.numpy(native1)
    np2 = math.numpy(native2)
    return np.allclose(np1, np2, rel_tolerance, abs_tolerance)


def assert_close(*tensors, rel_tolerance=1e-5, abs_tolerance=0):
    """
    Checks that all tensors have equal values within the specified tolerance.
    Raises an AssertionError if the values of this tensor are not within tolerance of any of the other tensors.

    Does not check that the shapes exactly match.
    Tensors with different shapes are reshaped before comparing.

    :param tensors: tensor or tensor-like (constant) each
    :param rel_tolerance: relative tolerance
    :param abs_tolerance: absolute tolerance
    """
    tensors = [tensor(t) for t in tensors]
    for other in tensors[1:]:
        _assert_close(tensors[0], other, rel_tolerance=rel_tolerance, abs_tolerance=abs_tolerance)


def _assert_close(tensor1, tensor2, rel_tolerance=1e-5, abs_tolerance=0):
    if tensor2 is tensor1:
        return
    if isinstance(tensor2, (int, float, bool)):
        np.testing.assert_allclose(tensor1.numpy(), tensor2, rel_tolerance, abs_tolerance)
    new_shape, (native1, native2) = broadcastable_native_tensors(tensor1, tensor2)
    np1 = math.numpy(native1)
    np2 = math.numpy(native2)
    if not np.allclose(np1, np2, rel_tolerance, abs_tolerance):
        np.testing.assert_allclose(np1, np2, rel_tolerance, abs_tolerance)
