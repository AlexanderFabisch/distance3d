import numpy as np
import pytransform3d.transformations as pt
from distance3d.random import (
    randn_point, randn_direction, randn_line, randn_line_segment,
    randn_plane, randn_triangle, randn_rectangle, rand_circle, rand_box,
    rand_capsule, rand_ellipsoid, rand_cylinder, rand_sphere, rand_cone,
    randn_convex)
from pytest import approx
from numpy.testing import assert_array_almost_equal


def test_randn_point():
    random_state = np.random.RandomState(101)
    p = randn_point(random_state)
    assert len(p) == 3


def test_randn_direction():
    random_state = np.random.RandomState(102)
    d = randn_direction(random_state)
    assert len(d) == 3
    assert approx(np.linalg.norm(d)) == 1


def test_randn_line():
    random_state = np.random.RandomState(103)
    p, d = randn_line(random_state)
    assert len(p) == 3
    assert len(d) == 3
    assert approx(np.linalg.norm(d)) == 1


def test_randn_line_segment():
    random_state = np.random.RandomState(104)
    s, e = randn_line_segment(random_state)
    assert len(s) == 3
    assert len(e) == 3


def test_randn_plane():
    random_state = np.random.RandomState(1040)
    p, n = randn_plane(random_state)
    assert len(p) == 3
    assert len(n) == 3
    assert approx(np.linalg.norm(n)) == 1


def test_randn_triangle():
    random_state = np.random.RandomState(105)
    triangle_points = randn_triangle(random_state)
    assert_array_almost_equal(triangle_points.shape, (3, 3))


def test_randn_rectangle():
    random_state = np.random.RandomState(106)
    rectangle_center, rectangle_axes, rectangle_lengths = randn_rectangle(
        random_state)
    assert len(rectangle_center) == 3
    assert_array_almost_equal(np.linalg.norm(rectangle_axes, axis=1), [1, 1])
    assert all(rectangle_lengths > 0)


def test_rand_circle():
    random_state = np.random.RandomState(1060)
    center, radius, normal = rand_circle(random_state)
    assert len(center) == 3
    assert radius > 0
    assert len(normal) == 3
    assert approx(np.linalg.norm(normal)) == 1


def test_rand_box():
    random_state = np.random.RandomState(107)
    box2origin, size = rand_box(random_state)
    pt.assert_transform(box2origin)
    assert all(size > 0)


def test_rand_capsule():
    random_state = np.random.RandomState(108)
    capsule2origin, radius, height = rand_capsule(random_state)
    pt.assert_transform(capsule2origin)
    assert 0 < radius <= 1
    assert 0 < height <= 1


def test_rand_ellipsoid():
    random_state = np.random.RandomState(1080)
    ellipsoid2origin, radii = rand_ellipsoid(random_state, min_radius=1.0)
    pt.assert_transform(ellipsoid2origin)
    assert 1.0 < radii[0] <= 2.0
    assert 1.0 < radii[1] <= 2.0
    assert 1.0 < radii[2] <= 2.0


def test_rand_cylinder():
    random_state = np.random.RandomState(109)
    cylinder2origin, radius, length = rand_cylinder(random_state)
    pt.assert_transform(cylinder2origin)
    assert 0 < radius <= 1
    assert 0 < length <= 1


def test_rand_sphere():
    random_state = np.random.RandomState(110)
    center, radius = rand_sphere(random_state)
    assert len(center) == 3
    assert 0 < radius <= 1


def test_rand_cone():
    random_state = np.random.RandomState(1101)
    capsule2origin, radius, height = rand_cone(
        random_state, center_scale=2.0, min_radius=0.5, min_height=0.3,
        radius_scale=2.0, height_scale=3.0)
    pt.assert_transform(capsule2origin)
    assert 0.5 < radius <= 2.5
    assert 0.3 < height <= 3.3


def test_randn_convex():
    random_state = np.random.RandomState(111)
    n_points = 10
    mesh2origin, vertices, triangles = randn_convex(random_state, n_points)
    pt.assert_transform(mesh2origin)
    assert len(vertices) == n_points
    assert np.min(triangles) == 0
    assert np.max(triangles) == len(vertices) - 1
