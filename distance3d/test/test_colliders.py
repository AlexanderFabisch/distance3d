import numpy as np

from distance3d import colliders
from numpy.testing import assert_array_almost_equal


def test_convex_collider():
    convex = colliders.ConvexHullVertices(np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]))
    assert_array_almost_equal(
        convex.aabb().limits, np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 1.0]]))
    assert_array_almost_equal(convex.center(), [0, 0.5, 0.5])


def test_box_collider():
    box = colliders.Box(np.eye(4), np.array([1.0, 1.0, 1.0]))
    assert_array_almost_equal(
        box.aabb().limits, np.array([[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]]))
    assert_array_almost_equal(box.center(), [0, 0, 0])


def test_cone_collider():
    cone = colliders.Cone(np.eye(4), 0.5, 1.0)
    assert_array_almost_equal(
        cone.aabb().limits, np.array([[-0.5, 0.5], [-0.5, 0.5], [0.0, 1.0]]))
    assert_array_almost_equal(cone.center(), [0, 0, 0.5])
    cone.make_artist(c=(1, 0, 0))
    assert cone.artist_ is not None
    cone2origin = np.eye(4)
    cone2origin[:3, 3] = 0.1, 0.2, 0.3
    cone.update_pose(cone2origin)
    assert_array_almost_equal(
        cone.aabb().limits, np.array([[-0.4, 0.6], [-0.3, 0.7], [0.3, 1.3]]))


def test_disk_collider():
    disk = colliders.Disk(np.zeros(3), 0.5, np.array([0.0, 0.0, 1.0]))
    assert_array_almost_equal(
        disk.aabb().limits, np.array([[-0.5, 0.5], [-0.5, 0.5], [0.0, 0.0]]))
    assert_array_almost_equal(disk.center(), [0, 0, 0])
    disk.make_artist(c=(1, 0, 0))
    assert disk.artist_ is not None
    disk2origin = np.eye(4)
    disk2origin[:3, 3] = -0.1, -0.2, -0.3
    disk.update_pose(disk2origin)
    assert_array_almost_equal(
        disk.aabb().limits, np.array([[-0.6, 0.4], [-0.7, 0.3], [-0.3, -0.3]]))
