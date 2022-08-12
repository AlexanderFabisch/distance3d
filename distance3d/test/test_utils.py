import numpy as np
import pytransform3d.transformations as pt
from distance3d.utils import (
    norm_vector, invert_transform, transform_point, inverse_transform_point,
    scalar_triple_product, transform_points, transform_directions)
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


def test_transform_points():
    random_state = np.random.RandomState(335)
    A2B = pt.random_transform(random_state)
    points_in_A = random_state.randn(10, 3)
    points_in_B = transform_points(A2B, points_in_A)
    for i in range(len(points_in_A)):
        assert_array_almost_equal(
            points_in_B[i], transform_point(A2B, points_in_A[i]))


def test_transform_directions():
    random_state = np.random.RandomState(336)
    A2B = pt.random_transform(random_state)
    directions_in_A = random_state.randn(10, 3)
    directions_in_B = transform_directions(A2B, directions_in_A)
    for i in range(len(directions_in_A)):
        assert_array_almost_equal(
            directions_in_B[i], A2B[:3, :3].dot(directions_in_A[i]))
