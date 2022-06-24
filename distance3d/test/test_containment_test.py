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
