import numpy as np
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
from distance3d.containment import (
    axis_aligned_bounding_box, sphere_aabb, box_aabb, cylinder_aabb,
    capsule_aabb, ellipsoid_aabb)
from distance3d.geometry import (
    cylinder_extreme_along_direction, capsule_extreme_along_direction)
from distance3d import random
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
    size = np.array([1, 2, 3], dtype=float)
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

    random_state = np.random.RandomState(3)
    for _ in range(100):
        cylinder2origin, radius, length = random.rand_cylinder(random_state)
        mins1, maxs1 = cylinder_aabb(cylinder2origin, radius, length)
        mins2, maxs2 = cylinder_aabb_slow(cylinder2origin, radius, length)
        assert_array_almost_equal(mins1, mins2)
        assert_array_almost_equal(maxs1, maxs2)


XM = np.array([-1.0, 0.0, 0.0])
YM = np.array([0.0, -1.0, 0.0])
ZM = np.array([0.0, 0.0, -1.0])
XP = np.array([1.0, 0.0, 0.0])
YP = np.array([0.0, 1.0, 0.0])
ZP = np.array([0.0, 0.0, 1.0])


def cylinder_aabb_slow(cylinder2origin, radius, length):
    negative_vertices = np.vstack((
        cylinder_extreme_along_direction(XM, cylinder2origin, radius, length),
        cylinder_extreme_along_direction(YM, cylinder2origin, radius, length),
        cylinder_extreme_along_direction(ZM, cylinder2origin, radius, length),
    ))
    mins = np.min(negative_vertices, axis=0)
    positive_vertices = np.vstack((
        cylinder_extreme_along_direction(XP, cylinder2origin, radius, length),
        cylinder_extreme_along_direction(YP, cylinder2origin, radius, length),
        cylinder_extreme_along_direction(ZP, cylinder2origin, radius, length),
    ))
    maxs = np.max(positive_vertices, axis=0)
    return mins, maxs


def test_capsule_aabb():
    capsule2origin = np.eye(4)
    radius = 1
    height = 5
    mins, maxs = capsule_aabb(capsule2origin, radius, height)
    assert_array_almost_equal(mins, [-1, -1, -3.5])
    assert_array_almost_equal(maxs, [1, 1, 3.5])

    random_state = np.random.RandomState(3)
    for _ in range(100):
        capsule2origin, radius, height = random.rand_capsule(random_state)
        mins1, maxs1 = capsule_aabb(capsule2origin, radius, height)
        mins2, maxs2 = capsule_aabb_slow(capsule2origin, radius, height)
        assert_array_almost_equal(mins1, mins2)
        assert_array_almost_equal(maxs1, maxs2)


def capsule_aabb_slow(capsule2origin, radius, height):
    negative_vertices = np.vstack((
        capsule_extreme_along_direction(XM, capsule2origin, radius, height),
        capsule_extreme_along_direction(YM, capsule2origin, radius, height),
        capsule_extreme_along_direction(ZM, capsule2origin, radius, height),
    ))
    mins = np.min(negative_vertices, axis=0)
    positive_vertices = np.vstack((
        capsule_extreme_along_direction(XP, capsule2origin, radius, height),
        capsule_extreme_along_direction(YP, capsule2origin, radius, height),
        capsule_extreme_along_direction(ZP, capsule2origin, radius, height),
    ))
    maxs = np.max(positive_vertices, axis=0)
    return mins, maxs


def test_ellipsoid_aabb():
    ellipsoid2origin = np.eye(4)
    radii = np.ones(3)
    mins, maxs = ellipsoid_aabb(ellipsoid2origin, radii)
    assert_array_almost_equal(mins, -radii)
    assert_array_almost_equal(maxs, radii)

    radii = np.array([0.2, 1.0, 0.5])
    mins, maxs = ellipsoid_aabb(ellipsoid2origin, radii)
    assert_array_almost_equal(mins, -radii)
    assert_array_almost_equal(maxs, radii)

    ellipsoid2origin = np.array([
        [1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])
    mins, maxs = ellipsoid_aabb(ellipsoid2origin, radii)
    assert_array_almost_equal(mins, [-0.2, -0.5, -1])
    assert_array_almost_equal(maxs, [0.2, 0.5, 1])

    ellipsoid2origin = pt.transform_from(
        R=pr.active_matrix_from_extrinsic_euler_xyz([0.2, 0.3, 0.5]),
        p=np.zeros(3)
    )
    mins, maxs = ellipsoid_aabb(ellipsoid2origin, radii)
    assert_array_almost_equal(mins, [-0.376639, -0.83155, -0.491812])
    assert_array_almost_equal(maxs, [0.376639, 0.83155, 0.491812])

    ellipsoid2origin = pt.transform_from(
        R=pr.active_matrix_from_extrinsic_euler_xyz([0.2, 0.3, 0.5]),
        p=np.array([-0.3, 0.8, 0.4])
    )
    mins, maxs = ellipsoid_aabb(ellipsoid2origin, radii)
    assert_array_almost_equal(mins, [-0.676639, -0.03155, -0.091812])
    assert_array_almost_equal(maxs, [0.076639, 1.63155, 0.891812])
