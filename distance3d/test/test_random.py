import numpy as np
import pytransform3d.transformations as pt
from distance3d.random import (
    randn_point, randn_direction, randn_line, randn_line_segment,
    randn_triangle, randn_rectangle, rand_box, rand_capsule, rand_cylinder,
    rand_sphere)
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
