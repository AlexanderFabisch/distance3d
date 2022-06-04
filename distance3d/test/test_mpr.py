import numpy as np
from distance3d import colliders, mpr, gjk, epa, utils
from numpy.testing import assert_array_almost_equal
from pytest import approx


def test_intersection_mpr_vs_gjk():
    random_state = np.random.RandomState(24)

    for _ in range(50):
        vertices1 = random_state.rand(6, 3) * np.array([[2, 2, 2]])
        convex1 = colliders.Convex(vertices1)

        vertices2 = random_state.rand(6, 3) * np.array([[1, 1, 1]])
        convex2 = colliders.Convex(vertices2)

        dist, closest_point1, closest_point2, _ = gjk.gjk_with_simplex(
            convex1, convex2)
        gjk_intersection = dist < 1e-16
        mpr_intersection = mpr.mpr_intersection(convex1, convex2)
        assert gjk_intersection == mpr_intersection


def test_penetration():
    sphere1 = colliders.Sphere(np.array([0.0, 0.0, 0.0]), 1.0)
    sphere2 = colliders.Sphere(np.array([0.0, 0.0, 0.0]), 1.0)
    intersection, depth, penetration_direction, contact_point = mpr.mpr_penetration(
        sphere1, sphere2)
    assert intersection
    assert depth == 2.0
    assert approx(np.linalg.norm(penetration_direction)) == 1.0
    assert_array_almost_equal(contact_point, [0, 0, 0])

    sphere1 = colliders.Sphere(np.array([0.0, 0.0, 0.0]), 1.0)
    sphere2 = colliders.Sphere(np.array([0.0, 0.0, 1.0]), 0.5)
    intersection, depth, penetration_direction, contact_point = mpr.mpr_penetration(
        sphere1, sphere2)
    assert intersection
    assert depth == 0.5
    assert_array_almost_equal(penetration_direction, [0, 0, 1])
    assert_array_almost_equal(contact_point, [0, 0, 0.75])

    sphere1 = colliders.Sphere(np.array([0.0, 0.0, 0.0]), 1.0)
    sphere2 = colliders.Sphere(np.array([1.0, 0.0, 0.0]), 0.5)
    intersection, depth, penetration_direction, contact_point = mpr.mpr_penetration(
        sphere1, sphere2)
    assert intersection
    assert depth == 0.5
    assert_array_almost_equal(penetration_direction, [1, 0, 0])
    assert_array_almost_equal(contact_point, [0.75, 0, 0])
