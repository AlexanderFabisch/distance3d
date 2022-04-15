import numpy as np
from distance3d.containment import (
    axis_aligned_bounding_box, sphere_aabb, box_aabb, cylinder_aabb,
    capsule_aabb)
from numpy.testing import assert_array_almost_equal


def test_aabb():
    P = np.array([
        [0, 1, 2],
        [2, 1, 0],
        [1, 2, 1]
    ])
    mins, maxs = axis_aligned_bounding_box(P)
    assert_array_almost_equal(mins, [0, 1, 0])
    assert_array_almost_equal(maxs, [2, 2, 2])


def test_sphere_aabb():
    mins, maxs = sphere_aabb(np.array([0, 1, 2]), 3.0)
    assert_array_almost_equal(mins, [-3, -2, -1])
    assert_array_almost_equal(maxs, [3, 4, 5])


def test_box_aabb():
    box2origin = np.eye(4)
    size = np.array([1, 2, 3])
    mins, maxs = box_aabb(box2origin, size)
    assert_array_almost_equal(mins, [-0.5, -1, -1.5])
    assert_array_almost_equal(maxs, [0.5, 1, 1.5])


def test_cylinder_aabb():
    cylinder2origin = np.eye(4)
    radius = 1
    length = 10
    mins, maxs = cylinder_aabb(cylinder2origin, radius, length)
    assert_array_almost_equal(mins, [-1, -1, -5])
    assert_array_almost_equal(maxs, [1, 1, 5])


def test_capsule_aabb():
    capsule2origin = np.eye(4)
    radius = 1
    height = 5
    mins, maxs = capsule_aabb(capsule2origin, radius, height)
    assert_array_almost_equal(mins, [-1, -1, -3.5])
    assert_array_almost_equal(maxs, [1, 1, 3.5])
