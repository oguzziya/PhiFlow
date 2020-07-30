from abc import ABC

from phi.backend import Backend
from phi import math, struct
from phi.geom import assert_same_rank

from ._field import Field


class AnalyticField(Field, ABC):

    @property
    def elements(self):
        return None

    def unstack(self, dimension=0):
        components = []
        size = self.shape.get_size(dimension)
        for i in range(size):
            def _context(index=i):
                return lambda x: x.unstack()[index]
            components.append(_SymbolicOpField(_context(i), [self]))
        return components

    def _op2(self, other, operator):
        pass

    def _op1(self, operator):
        pass


class SymbolicFieldBackend(Backend):
    # Abstract mehtods are overridden generically.
    # pylint: disable-msg = abstract-method

    def __init__(self, backend):
        Backend.__init__(self, 'Symbolic Field Backend')
        self.backend = backend
        for fname in dir(self):
            if fname not in ('__init__', 'is_tensor', 'is_applicable', 'symbolic_call') and not fname.startswith('__'):
                function = getattr(self, fname)
                if callable(function):
                    def context(fname=fname):
                        def proxy(*args, **kwargs):
                            return self.symbolic_call(fname, args, kwargs)
                        return proxy
                    setattr(self, fname, context())

    def symbolic_call(self, func, args, kwargs):
        assert len(kwargs) == 0, 'kwargs not supported'
        backend_func = getattr(self.backend, func)
        return _SymbolicOpField(backend_func, args)

    def is_tensor(self, x, only_native=False):
        return isinstance(x, AnalyticField)


class _SymbolicOpField(AnalyticField):

    def __init__(self, function, function_args, **kwargs):
        fields = filter(lambda arg: isinstance(arg, Field), function_args)
        AnalyticField.__init__(self, _determine_rank(fields), name=function.__name__, **struct.kwargs(locals(), ignore='fields'))
        self.fields = tuple(fields)
        self.channels = _determine_component_count(function_args)

    @struct.constant()
    def function_args(self, args):
        return args

    @struct.constant()
    def function(self, function):
        return function

    def at(self, other_field):
        args = []
        for arg in self.function_args:
            if isinstance(arg, Field):
                arg = arg.at(other_field)
            args.append(arg)
        applied = self.function(*args)
        return applied

    def sample_at(self, points):
        raise NotImplementedError()

    @property
    def component_count(self):
        return self.channels

    def unstack(self):
        unstacked = {}
        for arg in self.function_args:
            if isinstance(arg, Field):
                unstacked[arg] = arg.unstack()
            elif math.is_tensor(arg) and math.ndims(arg) > 0:
                unstacked[arg] = math.unstack(arg, axis=-1, keepdims=True)
            else:
                unstacked[arg] = [arg] * self.component_count
            assert len(unstacked[arg]) == self.component_count
        result = [_SymbolicOpField(self.function, [unstacked[arg][i] for arg in self.function_args]) for i in range(self.component_count)]
        return result

    @property
    def points(self):
        if len(self.fields) == 0:
            return None
        else:
            return self.fields[0].points

    def compatible(self, other_field):
        if len(self.fields) == 0:
            return True
        else:
            return self.fields[0].compatible(other_field)


def _determine_rank(fields):
    rank = None
    for field in fields:
        if rank is None:
            rank = field.rank
        else:
            assert_same_rank(rank, field.rank, 'All fields must have the same rank')
    return rank


def _determine_component_count(args):
    result = None
    for arg in args:
        arg_channels = None
        if isinstance(arg, Field):
            arg_channels = arg.component_count
        elif math.is_tensor(arg) and math.ndims(arg) > 0:
            arg_channels = arg.shape[-1]
        if result is None:
            result = arg_channels
        elif result == 1 and arg_channels is not None:
            result = arg_channels
        else:
            assert result == arg_channels or arg_channels is None
    return result

