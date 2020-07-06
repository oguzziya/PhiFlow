from unittest import TestCase
from phi.flow import *
from phi.math._shape import *
from phi.math._tensor_initializers import zero
from phi.math._tensors import tensor

import numpy as np


class TestTensors(TestCase):

    def test_define_shapes(self):
        self.assertEqual('(2)', repr(define_shape(2)))
        self.assertEqual('(1, 2, 3)', repr(define_shape((1, 2, 3))))
        self.assertEqual('(batch=10)', repr(define_shape(batch=10)))
        self.assertEqual('(batch=10, time=5)', repr(define_shape(batch={'batch': 10, 'time': 5})))
        self.assertEqual('(2, 1 | batch=10)', repr(define_shape((2, 1), 10)))
        physics_config.x_first()
        self.assertEqual('(x=3, y=4, z=5)', repr(define_shape(y=4, z=5, x=3)))
        self.assertEqual('(3 | x=3, y=4, z=5 | batch=10)', repr(define_shape(3, y=4, z=5, x=3, batch=10)))
        self.assertEqual((10, 4, 5, 3, 3), define_shape(3, y=4, z=5, x=3, batch=10).sizes)
        self.assertEqual(('batch', 'y', 'z', 'x', 0), define_shape(3, y=4, z=5, x=3, batch=10).names)

    def test_tensor_creation(self):
        v = tensor(np.ones([1, 4, 3, 2]))
        self.assertEqual((4, 3, 2), v.shape.sizes)
        v = tensor(np.ones([10, 4, 3, 2]))
        self.assertEqual((10, 4, 3, 2), v.shape.sizes)
        a = tensor([1, 2, 3])
        self.assertEqual((3,), a.shape.sizes)

    def test_native_unstack(self):
        physics_config.x_first()
        v = tensor(np.ones([10, 4, 3, 2]))
        vx, vy = v.unstack()
        self.assertEqual('(x=4, y=3 | batch=10)', repr(vx.shape))

    def test_native_add(self):
        physics_config.x_first()
        v = tensor(np.ones([1, 4, 3, 2]))
        v2 = v + 1
        v2 = v + [0, 1]
        d = v.unstack()[0]
        v + d
        d + v

    def test_math_functions(self):
        v = tensor(np.ones([1, 4, 3, 2]))
        v0 = math.minimum(0, v)
