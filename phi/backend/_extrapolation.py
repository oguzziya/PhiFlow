
class IncompatibleExtrapolations(ValueError):
    def __init__(self, extrapolation1, extrapolation2):
        ValueError.__init__(self, extrapolation1, extrapolation2)


class Extrapolation:

    def gradient(self):
        """
        Returns the extrapolation for the spatial gradient of a tensor/field with this extrapolation.

        :rtype: _Extrapolation
        """
        raise NotImplementedError()


class ConstantExtrapolation(Extrapolation):

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return repr(self.value)

    def gradient(self):
        return ZERO

    def __eq__(self, other):
        from . import math
        return isinstance(other, ConstantExtrapolation) and math.all(math.equal(other.value, self.value))

    def is_zero(self):
        return self == ZERO

    def is_one(self):
        return self == ONE

    def __add__(self, other):
        if isinstance(other, ConstantExtrapolation):
            return ConstantExtrapolation(self.value + other.value)
        elif self.is_zero():
            return other
        else:
            raise IncompatibleExtrapolations(self, other)

    def __sub__(self, other):
        if isinstance(other, ConstantExtrapolation):
            return ConstantExtrapolation(self.value - other.value)
        else:
            raise IncompatibleExtrapolations(self, other)

    def __rsub__(self, other):
        if isinstance(other, ConstantExtrapolation):
            return ConstantExtrapolation(other.value - self.value)
        elif self.is_zero():
            return other
        else:
            raise IncompatibleExtrapolations(self, other)

    def __mul__(self, other):
        if isinstance(other, ConstantExtrapolation):
            return ConstantExtrapolation(self.value * other.value)
        elif self.is_one():
            return other
        elif self.is_zero():
            return self
        else:
            raise IncompatibleExtrapolations(self, other)

    def __truediv__(self, other):
        if isinstance(other, ConstantExtrapolation):
            return ConstantExtrapolation(self.value / other.value)
        elif self.is_zero():
            return self
        else:
            raise IncompatibleExtrapolations(self, other)

    def __rtruediv__(self, other):
        if isinstance(other, ConstantExtrapolation):
            return ConstantExtrapolation(other.value / self.value)
        elif self.is_one():
            return other
        else:
            raise IncompatibleExtrapolations(self, other)

    def __lt__(self, other):
        if isinstance(other, ConstantExtrapolation):
            return ConstantExtrapolation(self.value < other.value)
        else:
            raise IncompatibleExtrapolations(self, other)

    def __gt__(self, other):
        if isinstance(other, ConstantExtrapolation):
            return ConstantExtrapolation(self.value > other.value)
        else:
            raise IncompatibleExtrapolations(self, other)


class _StatelessExtrapolation(Extrapolation):

    def gradient(self):
        raise NotImplementedError()

    def __eq__(self, other):
        return type(other) == type(self)

    def _op(self, other, op):
        if type(other) == type(self):
            return self
        elif isinstance(other, Extrapolation) and not isinstance(other, _StatelessExtrapolation):
            op = getattr(other, op.__name__)
            return op(self)
        else:
            raise IncompatibleExtrapolations(self, other)

    def __add__(self, other):
        return self._op(other, ConstantExtrapolation.__add__)

    def __mul__(self, other):
        return self._op(other, ConstantExtrapolation.__mul__)

    def __sub__(self, other):
        return self._op(other, ConstantExtrapolation.__rsub__)

    def __truediv__(self, other):
        return self._op(other, ConstantExtrapolation.__rtruediv__)

    def __lt__(self, other):
        return self._op(other, ConstantExtrapolation.__gt__)

    def __gt__(self, other):
        return self._op(other, ConstantExtrapolation.__lt__)


class _BoundaryExtrapolation(_StatelessExtrapolation):
    """
    Uses the closest defined value for points lying outside the defined region.
    """
    def __repr__(self):
        return 'boundary'

    def gradient(self):
        return ZERO


class _PeriodicExtrapolation(_StatelessExtrapolation):
    def __repr__(self):
        return 'periodic'

    def gradient(self):
        return self


class _SymmetricExtrapolation(_StatelessExtrapolation):
    """
    Mirror with the boundary value occurring twice.
    """
    def __repr__(self):
        return 'symmetric'

    def gradient(self):
        return -self


class _ReflectExtrapolation(_StatelessExtrapolation):
    """
    Mirror of inner elements. The boundary value is not duplicated.
    """
    def __repr__(self):
        return 'reflect'

    def gradient(self):
        return -self


ZERO = ConstantExtrapolation(0)
ONE = ConstantExtrapolation(1)
PERIODIC = _PeriodicExtrapolation()
BOUNDARY = _BoundaryExtrapolation()
SYMMETRIC = _SymmetricExtrapolation()
REFLECT = _ReflectExtrapolation()


