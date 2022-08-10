import numpy as np
from distance3d.geometry import (
    support_function_box, support_function_cone, support_function_disk,
    support_function_ellipse, barycentric_coordinates_tetrahedron)
from distance3d.random import rand_cone, rand_circle, rand_ellipse
from distance3d.utils import plane_basis_from_normal
from distance3d.containment import ellipse_aabb
from numpy.testing import assert_array_almost_equal
from pytest import approx


def test_box_extreme_along_direction():
    search_direction = np.array([1, 0, 0], dtype=float)
    extreme = support_function_box(
        search_direction, np.eye(4), np.ones(1))
    assert_array_almost_equal(extreme, search_direction)

    search_direction = np.array([0, 1, 0], dtype=float)
    extreme = support_function_box(
        search_direction, np.eye(4), np.ones(1))
    assert_array_almost_equal(extreme, search_direction)

    search_direction = np.array([0, 0, 1], dtype=float)
    extreme = support_function_box(
        search_direction, np.eye(4), np.ones(1))
    assert_array_almost_equal(extreme, search_direction)


def test_cone_support_function():
    random_state = np.random.RandomState(2323)
    cone2origin, radius, height = rand_cone(random_state)

    d = np.ascontiguousarray(cone2origin[:3, 2])
    p = support_function_cone(d, cone2origin, radius, height)
    assert_array_almost_equal(
        p, cone2origin[:3, 3] + height * cone2origin[:3, 2])

    d = np.ascontiguousarray(cone2origin[:3, 1])
    p = support_function_cone(d, cone2origin, radius, height)
    assert_array_almost_equal(
        p, cone2origin[:3, 3] + radius * cone2origin[:3, 1])

    d = np.ascontiguousarray(cone2origin[:3, 0])
    p = support_function_cone(d, cone2origin, radius, height)
    assert_array_almost_equal(
        p, cone2origin[:3, 3] + radius * cone2origin[:3, 0])

    p = support_function_cone(np.zeros(3), cone2origin, radius, height)
    assert_array_almost_equal(p, cone2origin[:3, 3])


def test_disk_support_function():
    random_state = np.random.RandomState(2223)
    center, radius, normal = rand_circle(random_state)
    x, y = plane_basis_from_normal(normal)

    p = support_function_disk(x, center, radius, normal)
    assert_array_almost_equal(p, center + radius * x)

    p = support_function_disk(y, center, radius, normal)
    assert_array_almost_equal(p, center + radius * y)

    p = support_function_disk(normal, center, radius, normal)
    assert approx(np.dot(p - center, normal)) == 0.0
    assert approx(np.linalg.norm(p - center)) == radius

    p = support_function_disk(np.zeros(3), center, radius, normal)
    assert_array_almost_equal(p, center)


def test_ellipse_support_function():
    random_state = np.random.RandomState(873)
    center, axes, radii = rand_ellipse(random_state)
    mins, maxs = ellipse_aabb(center, axes, radii)

    negative_vertices = np.vstack((
        support_function_ellipse(np.array([-1.0, 0.0, 0.0]), center, axes, radii),
        support_function_ellipse(np.array([0.0, -1.0, 0.0]), center, axes, radii),
        support_function_ellipse(np.array([0.0, 0.0, -1.0]), center, axes, radii)))
    assert_array_almost_equal(np.min(negative_vertices, axis=0), mins)

    positive_vertices = np.vstack((
        support_function_ellipse(np.array([1.0, 0.0, 0.0]), center, axes, radii),
        support_function_ellipse(np.array([0.0, 1.0, 0.0]), center, axes, radii),
        support_function_ellipse(np.array([0.0, 0.0, 1.0]), center, axes, radii)))
    assert_array_almost_equal(np.max(positive_vertices, axis=0), maxs)


def test_barycentric_coordinates_tetrahedron():
    random_state = np.random.RandomState(874)
    for _ in range(100):
        tetrahedron_points = random_state.randn(4, 3)

        coordinates = random_state.rand(4)
        coordinates[random_state.rand(4) > 0.75] = 0.0
        if np.all(coordinates == 0.0):
            coordinates[0] = 1.0
        coordinates /= np.sum(coordinates)

        p = np.dot(coordinates, tetrahedron_points)

        reconstructed_coordinates = barycentric_coordinates_tetrahedron(
            p, tetrahedron_points)
        assert_array_almost_equal(coordinates, reconstructed_coordinates)
