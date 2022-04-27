import numpy as np
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
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

    cylinder2origin = pt.transform_from(
        R=pr.active_matrix_from_extrinsic_euler_xyz([0.5, 0.3, 0.2]),
        p=np.array([0.2, 0.3, 0.4])
    )
    radius = 2
    length = 4
    mins, maxs = cylinder_aabb(cylinder2origin, radius, length)
    assert_array_almost_equal(mins, [-2.372774, -2.353267, -2.366925])
    assert_array_almost_equal(maxs, [2.772774, 2.953267, 3.166925])


def test_capsule_aabb():
    capsule2origin = np.eye(4)
    radius = 1
    height = 5
    mins, maxs = capsule_aabb(capsule2origin, radius, height)
    assert_array_almost_equal(mins, [-1, -1, -3.5])
    assert_array_almost_equal(maxs, [1, 1, 3.5])
