import numpy as np

from distance3d import colliders, random
from numpy.testing import assert_array_almost_equal


collider_classes = {
    "sphere": (colliders.Sphere, random.rand_sphere),
    "ellipsoid": (colliders.Ellipsoid, random.rand_ellipsoid),
    "capsule": (colliders.Capsule, random.rand_capsule),
    "disk": (colliders.Disk, random.rand_circle),
    "ellipse": (colliders.Ellipse, random.rand_ellipse),
    "cone": (colliders.Cone, random.rand_cone),
    "cylinder": (colliders.Cylinder, random.rand_cylinder),
    "box": (colliders.Box, random.rand_box),
    "mesh": (colliders.MeshGraph, random.randn_convex),
}


def test_all_colliders():
    random_state = np.random.RandomState(4)
    for Collider, rand in collider_classes.values():
        print(Collider)
        c = Collider(*rand(random_state))
        limits = np.array(c.aabb().limits)
        center = c.center()
        assert all(limits[:, 0] <= center)
        assert all(center <= limits[:, 1])
        first_vertex = c.first_vertex()
        assert all(limits[:, 0] <= first_vertex)
        assert all(first_vertex <= limits[:, 1])

        c.make_artist(c=(0, 0, 1))
        assert c.artist_ is not None
        a = c.artist_
        aabb = a.geometries[0].get_axis_aligned_bounding_box()

        c.update_pose(np.eye(4))
        aabb2 = a.geometries[0].get_axis_aligned_bounding_box()
        assert all(np.asarray(aabb.min_bound) != np.asarray(aabb2.min_bound))
        assert all(np.asarray(aabb.max_bound) != np.asarray(aabb2.max_bound))

        c.artist_ = None
        c.update_pose(np.eye(4))


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


def test_margin():
    box = colliders.Box(np.eye(4), np.array([1.0, 1.0, 1.0]))
    box_with_margin = colliders.Margin(box, 0.1)

    assert_array_almost_equal(
        box_with_margin.aabb().limits,
        np.array([[-0.6, 0.6], [-0.6, 0.6], [-0.6, 0.6]]))
    assert_array_almost_equal(box_with_margin.center(), [0, 0, 0])

    new_pose = np.eye(4)
    new_pose[:3, 3] = 0.1, 0.2, 0.3
    box_with_margin.update_pose(new_pose)
    assert_array_almost_equal(box_with_margin.center(), [0.1, 0.2, 0.3])

    box_with_margin.make_artist()
    box_with_margin.update_pose(np.eye(4))
    assert_array_almost_equal(box_with_margin.center(), [0, 0, 0])
