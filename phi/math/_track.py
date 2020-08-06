import numpy as np

from .backend import DYNAMIC_BACKEND as math, Extrapolation, extrapolation
from ._tensors import AbstractTensor, NativeTensor, combined_shape


class SparseLinearOperation(AbstractTensor):

    def __init__(self, source: AbstractTensor, dependency_matrix, shape):
        self.source = source
        self.dependency_matrix = dependency_matrix
        self._shape = shape

    def native(self, order=None):
        # raise ValueError()
        native_source = math.reshape(self.source.native(), (self.source.shape.batch.volume, self.source.shape.non_batch.volume))
        native = math.matmul(self.dependency_matrix, native_source)
        new_shape = self.source.shape.batch.combined(self._shape)
        native = math.reshape(native, new_shape.sizes)
        return NativeTensor(native, new_shape).native(order)

    @property
    def dtype(self):
        return self.source.dtype

    @property
    def shape(self):
        return self._shape

    def _with_shape_replaced(self, new_shape):
        raise NotImplementedError()

    def _getitem(self, selection_dict):
        indices = NativeTensor(math.reshape(math.range(self.shape.volume), self.shape.sizes), self.shape)
        selected_indices = indices[selection_dict]
        selected_indices_native = math.flatten(selected_indices.native())
        selected_deps = math.gather(self.dependency_matrix, (selected_indices_native, slice(None)))
        return SparseLinearOperation(self.source, selected_deps, selected_indices.shape)

    def unstack(self, dimension=0):
        raise NotImplementedError()

    def _op1(self, native_function):
        deps = native_function(self.dependency_matrix)
        return SparseLinearOperation(self.source, deps, self._shape)

    def _op2(self, other, native_function):
        if isinstance(other, SparseLinearOperation):
            assert self.source is other.source
            assert self._shape == other._shape
            new_matrix = native_function(self.dependency_matrix, other.dependency_matrix)
            return SparseLinearOperation(self.source, new_matrix, self._shape)
        else:
            other = self._tensor(other)
            broadcast_shape = combined_shape(self, other)
            if other.shape.volume > 1:
                flat = math.flatten(other.native(broadcast_shape.names))
                vertical = math.expand_dims(flat, -1)
                new_matrix = native_function(self.dependency_matrix, vertical)  # this can cause matrix to become dense...
            else:
                scalar = other[{dim: 0 for dim in other.shape.names}].native()
                new_matrix = native_function(self.dependency_matrix, scalar)
            return SparseLinearOperation(self.source, new_matrix, broadcast_shape)


def as_sparse_linear_operation(tensor: AbstractTensor):
    tracking_shape = tensor.shape.non_batch
    idx = math.range(tracking_shape.volume)
    ones = math.ones_like(idx)
    sparse_diag = math.choose_backend(tensor.native()).sparse_tensor([idx, idx], ones, shape=(tracking_shape.volume,) * 2)
    return SparseLinearOperation(tensor, sparse_diag, tracking_shape)


def pad_operator(tensor: SparseLinearOperation, pad_widths: dict, mode: Extrapolation):
    # TODO create large index array reshape(range(...)), then pad?
    (row, col), data = math.coordinates(tensor.dependency_matrix, unstack_coordinates=True)
    if mode == extrapolation.ZERO:
        assert len(tensor.shape) == 2  # TODO nd
        y = row // tensor.shape[1]
        dy0, dy1 = pad_widths[tensor.shape.names[0]]
        dx0, dx1 = pad_widths[tensor.shape.names[1]]
        padded_row = row + dy0 * (tensor.shape[1] + dx0 + dx1) + dx0 * (y + 1) + dx1 * y
        new_sizes = list(tensor.shape.sizes)
        for i, dim in tensor.shape.enumerated_names:
            new_sizes[i] += sum(pad_widths[dim])
        new_shape = tensor.shape.with_sizes(new_sizes)
        padded_matrix = math.sparse_tensor((padded_row, col), data, shape=(new_shape.volume, tensor.dependency_matrix.shape[1]))
        return SparseLinearOperation(tensor.source, padded_matrix, new_shape)
    elif mode == extrapolation.PERIODIC:
        raise NotImplementedError()
    elif mode == extrapolation.BOUNDARY:
        raise NotImplementedError()
    else:
        raise NotImplementedError(mode)


def sum_operators(operators):
    for o in operators[1:]:
        assert isinstance(o, SparseLinearOperation)
        assert o.source is operators[0].source
    new_matrix = math.sum([o.dependency_matrix for o in operators], axis=0)
    return SparseLinearOperation(operators[0].source, new_matrix, operators[0].shape)
