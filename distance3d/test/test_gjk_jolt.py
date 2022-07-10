import numpy as np
from distance3d import random, distance, utils, colliders
from distance3d.gjk import gjk_distance_original, gjk_distance
from distance3d.gjk._gjk_jolt import (
    get_barycentric_coordinates_line, get_barycentric_coordinates_plane,
    get_barycentric_coordinates_tetrahedron, closest_point_line,
    closest_point_triangle, closest_point_tetrahedron)
from numpy.testing import assert_array_almost_equal


def test_barycentric_coordinates_line():
    random_state = np.random.RandomState(0)
    for _ in range(10):
        a = random.randn_point(random_state)
        b = random.randn_point(random_state)
        u, v = get_barycentric_coordinates_line(a, b)
        cp1 = u * a + v * b
        _, cp2 = distance.point_to_line(np.zeros(3), a, utils.norm_vector(b - a))
        assert_array_almost_equal(cp1, cp2)


def test_barycentric_coordinates_plane():
    random_state = np.random.RandomState(1)
    for _ in range(10):
        a = random.randn_point(random_state)
        b = random.randn_point(random_state)
        c = random.randn_point(random_state)
        u, v, w = get_barycentric_coordinates_plane(a, b, c)
        cp1 = u * a + v * b + w * c
        d1 = utils.norm_vector(b - a)
        d2 = utils.norm_vector(c - a)
        n = utils.norm_vector(np.cross(d1, d2))
        _, cp2 = distance.point_to_plane(np.zeros(3), a, n)
        assert_array_almost_equal(cp1, cp2)


def test_barycentric_coordinates_tetrahedron():
    random_state = np.random.RandomState(2)
    for _ in range(100):
        a = random.randn_point(random_state)
        b = random.randn_point(random_state)
        c = random.randn_point(random_state)
        d = random.randn_point(random_state)
        u, v, w, x = get_barycentric_coordinates_tetrahedron(a, b, c, d)
        cp1 = u * a + v * b + w * c + x * d
        c1 = colliders.ConvexHullVertices(np.row_stack((a, b, c, d)))
        c2 = colliders.ConvexHullVertices(np.zeros((1, 3)))
        dist, cp2 = gjk_distance_original(c1, c2)[:2]
        if dist < utils.EPSILON:
            assert_array_almost_equal(cp1, cp2)


def test_closest_point_line():
    random_state = np.random.RandomState(0)
    for _ in range(10):
        a = random.randn_point(random_state)
        b = random.randn_point(random_state)
        cp1, simplex = closest_point_line(a, b)
        _, cp2 = distance.point_to_line(np.zeros(3), a, utils.norm_vector(b - a))
        assert_array_almost_equal(cp1, cp2)


def test_closest_point_plane():
    random_state = np.random.RandomState(1)
    for _ in range(10):
        a = random.randn_point(random_state)
        b = random.randn_point(random_state)
        c = random.randn_point(random_state)
        cp1, simplex = closest_point_triangle(a, b, c)
        _, cp2 = distance.point_to_triangle(np.zeros(3), np.row_stack((a, b, c)))
        assert_array_almost_equal(cp1, cp2)


def test_closest_point_tetrahedron():
    random_state = np.random.RandomState(2)
    for _ in range(100):
        a = random.randn_point(random_state)
        b = random.randn_point(random_state)
        c = random.randn_point(random_state)
        d = random.randn_point(random_state)
        cp1, simplex = closest_point_tetrahedron(a, b, c, d)
        c1 = colliders.ConvexHullVertices(np.row_stack((a, b, c, d)))
        c2 = colliders.ConvexHullVertices(np.zeros((1, 3)))
        cp2 = gjk_distance_original(c1, c2)[1]
        assert_array_almost_equal(cp1, cp2)


def test_too_far_away():
    c1 = colliders.ConvexHullVertices(np.array([[0.0, 0.0, 0.0]]))
    c2 = colliders.ConvexHullVertices(np.array([[100000000.0, 0.0, 0.0]]))
    dist, p1, p2, simplex = gjk_distance(c1, c2)
    assert dist == np.finfo(float).max
    assert p1 is None
    assert p2 is None
    assert simplex is None
