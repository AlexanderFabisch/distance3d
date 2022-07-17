import numpy as np
import pytransform3d.transformations as pt
from distance3d.utils import (
    norm_vector, invert_transform, transform_point, inverse_transform_point,
    scalar_triple_product)
from numpy.testing import assert_array_almost_equal
from pytest import approx


def test_norm_vector():
    assert_array_almost_equal(norm_vector(np.array([0, 1, 0], dtype=float)), [0, 1, 0])
    assert_array_almost_equal(norm_vector(np.array([0, 2, 0], dtype=float)), [0, 1, 0])
    assert_array_almost_equal(norm_vector(np.array([0, 0, 0], dtype=float)), [0, 0, 0])
    assert approx(1.0) == np.linalg.norm(norm_vector(np.array([1, -2.3, 1.3], dtype=float)))


def test_invert_transform():
    random_state = np.random.RandomState(333)
    for _ in range(10):
        A2B = pt.random_transform(random_state)
        B2A = invert_transform(A2B)
        pt.assert_transform(B2A)
        assert_array_almost_equal(pt.concat(A2B, B2A), np.eye(4))


def test_inverse_transform_point():
    random_state = np.random.RandomState(334)
    for _ in range(10):
        A2B = pt.random_transform(random_state)
        point_in_B = random_state.randn(3)
        B2A = invert_transform(A2B)
        assert_array_almost_equal(transform_point(B2A, point_in_B),
                                  inverse_transform_point(A2B, point_in_B))


def test_scalar_triple_product():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    c = np.array([0.0, 0.0, 1.0])

    assert approx(scalar_triple_product(a, b, c)) == 1

    assert approx(scalar_triple_product(a, b, -c)) == -1

    assert approx(scalar_triple_product(a, b, b)) == 0
