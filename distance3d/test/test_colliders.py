import numpy as np

from distance3d import colliders
from numpy.testing import assert_array_almost_equal


def test_convex_collider():
    convex = colliders.Convex(np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]))
    assert_array_almost_equal(
        convex.aabb().limits, np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 1.0]]))
    assert_array_almost_equal(convex.center(), [0, 0.5, 0.5])


def test_box_collider():
    box = colliders.Box(np.eye(4), np.array([1.0, 1.0, 1.0]))
    assert_array_almost_equal(
        box.aabb().limits, np.array([[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]]))
    assert_array_almost_equal(box.center(), [0, 0, 0])
