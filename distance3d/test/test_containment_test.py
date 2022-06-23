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
