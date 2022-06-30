import numpy as np
from distance3d.geometry import (
    support_function_box, support_function_cone, support_function_disk,
    support_function_ellipse)
from distance3d.random import rand_cone, rand_circle, rand_ellipse
from distance3d.utils import plane_basis_from_normal
from distance3d.containment import ellipse_aabb
from numpy.testing import assert_array_almost_equal
from pytest import approx


def test_box_extreme_along_direction():
    search_direction = np.array([1, 0, 0], dtype=float)
    extreme = support_function_box(
        search_direction, np.eye(4), np.ones(1))
    assert_array_almost_equal(extreme, search_direction)

    search_direction = np.array([0, 1, 0], dtype=float)
    extreme = support_function_box(
        search_direction, np.eye(4), np.ones(1))
    assert_array_almost_equal(extreme, search_direction)

    search_direction = np.array([0, 0, 1], dtype=float)
    extreme = support_function_box(
        search_direction, np.eye(4), np.ones(1))
    assert_array_almost_equal(extreme, search_direction)


def test_cone_support_function():
    random_state = np.random.RandomState(2323)
    cone2origin, radius, height = rand_cone(random_state)

    p = support_function_cone(cone2origin[:3, 2], cone2origin, radius, height)
    assert_array_almost_equal(
        p, cone2origin[:3, 3] + height * cone2origin[:3, 2])

    p = support_function_cone(cone2origin[:3, 1], cone2origin, radius, height)
    assert_array_almost_equal(
        p, cone2origin[:3, 3] + radius * cone2origin[:3, 1])

    p = support_function_cone(cone2origin[:3, 0], cone2origin, radius, height)
    assert_array_almost_equal(
        p, cone2origin[:3, 3] + radius * cone2origin[:3, 0])

    p = support_function_cone(np.zeros(3), cone2origin, radius, height)
    assert_array_almost_equal(p, cone2origin[:3, 3])


def test_disk_support_function():
    random_state = np.random.RandomState(2223)
    center, radius, normal = rand_circle(random_state)
    x, y = plane_basis_from_normal(normal)

    p = support_function_disk(x, center, radius, normal)
    assert_array_almost_equal(p, center + radius * x)

    p = support_function_disk(y, center, radius, normal)
    assert_array_almost_equal(p, center + radius * y)

    p = support_function_disk(normal, center, radius, normal)
    assert approx(np.dot(p - center, normal)) == 0.0
    assert approx(np.linalg.norm(p - center)) == radius

    p = support_function_disk(np.zeros(3), center, radius, normal)
    assert_array_almost_equal(p, center)


def test_ellipse_support_function():
    random_state = np.random.RandomState(873)
    center, axes, radii = rand_ellipse(random_state)
    mins, maxs = ellipse_aabb(center, axes, radii)

    xm = support_function_ellipse(np.array([-1.0, 0.0, 0.0]), center, axes, radii)
    assert approx(mins[0]) == xm[0]
    xp = support_function_ellipse(np.array([1.0, 0.0, 0.0]), center, axes, radii)
    assert approx(maxs[0]) == xp[0]

    ym = support_function_ellipse(np.array([0.0, -1.0, 0.0]), center, axes, radii)
    assert approx(mins[1]) == ym[0]
    yp = support_function_ellipse(np.array([0.0, 1.0, 0.0]), center, axes, radii)
    assert approx(maxs[1]) == yp[0]

    zm = support_function_ellipse(np.array([0.0, 0.0, -1.0]), center, axes, radii)
    assert approx(mins[2]) == zm[0]
    zp = support_function_ellipse(np.array([0.0, 0.0, 1.0]), center, axes, radii)
    assert approx(maxs[2]) == zp[0]
