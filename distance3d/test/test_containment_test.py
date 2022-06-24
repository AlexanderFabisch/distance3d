import numpy as np
from distance3d import containment_test


def test_points_in_sphere():
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 2.0],
        [1.0, 1.0, 0.0]
    ])
    contained = containment_test.points_in_sphere(points, np.zeros(3), 1.0)
    assert all(contained == [True, True, True, False, False])


def test_points_in_capsule():
    points = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.0, -0.5, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.1],
        [0.0, 0.0, -1.1],
        [0.5, 0.5, 0.0]
    ])
    contained = containment_test.points_in_capsule(points, np.eye(4), 0.5, 1.0)
    assert all(contained == [True, True, True, True, True, False, False, False])


def test_points_in_ellipsoid():
    points = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.0, -0.3, 0.0],
        [0.0, 0.0, 1.0],
        [-0.11, 0.0, 0.0],
        [0.0, 0.31, 0.0],
        [0.0, 0.0, -1.01],
        [0.1, 0.3, 0.0]
    ])
    contained = containment_test.points_in_ellipsoid(
        points, np.eye(4), np.array([0.1, 0.3, 1.0]))
    assert all(contained == [True, True, True, True, False, False, False, False])


def test_points_in_disk():
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 0.001],
        [1.0, 1.0, 0.0]
    ])
    contained = containment_test.points_in_disk(
        points, np.zeros(3), 1.0, np.array([0.0, 0.0, 1.0]))
    assert all(contained == [True, True, True, False, False])


def test_points_in_cone():
    points = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.0, -0.5, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -0.01],
        [0.3, 0.0, 0.5],
        [0.01, 0.0, 1.0]
    ])
    contained = containment_test.points_in_cone(
        points, np.eye(4), 0.5, 1.0)
    assert all(contained == [True, True, True, True, False, False, False])


def test_points_in_cylinder():
    points = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.0, -0.5, 0.0],
        [0.0, 0.0, -0.5],
        [0.0, 0.0, 0.5],
        [0.0, 0.0, 0.51],
        [0.0, 0.0, -0.51],
        [0.5, 0.5, 0.0]
    ])
    contained = containment_test.points_in_cylinder(points, np.eye(4), 0.5, 1.0)
    assert all(contained == [True, True, True, True, True, False, False, False])


def test_points_in_box():
    points = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.3, 1.0],
        [-0.1, -0.3, -1.0],
        [0.11, 0.3, 1.0],
        [0.0, -0.31, 0.0],
    ])
    contained = containment_test.points_in_box(
        points, np.eye(4), np.array([0.2, 0.6, 2.0]))
    assert all(contained == [True, True, True, False, False])
