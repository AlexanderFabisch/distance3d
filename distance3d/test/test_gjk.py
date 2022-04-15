import numpy as np
from distance3d import colliders, gjk, geometry, random, distance
from pytest import approx
from numpy.testing import assert_array_almost_equal


def test_gjk_boxes():
    box2origin = np.array([
        [-0.29265666, -0.76990535, 0.56709596, 0.1867558],
        [0.93923897, -0.12018753, 0.32153556, -0.09772779],
        [-0.17939408, 0.62673815, 0.75829879, 0.09500884],
        [0., 0., 0., 1.]])
    size = np.array([2.89098828, 1.15032456, 2.37517511])
    box_collider = colliders.Box(box2origin, size)

    box2origin2 = np.array([
        [-0.29265666, -0.76990535, 0.56709596, 3.73511598],
        [0.93923897, -0.12018753, 0.32153556, -1.95455576],
        [-0.17939408, 0.62673815, 0.75829879, 1.90017684],
        [0., 0., 0., 1.]])
    size2 = np.array([0.96366276, 0.38344152, 0.79172504])

    box_collider2 = colliders.Box(box2origin2, size2)
    dist, closest_point_box, closest_point_box2, _ = gjk.gjk_with_simplex(
        box_collider, box_collider2)

    assert approx(dist) == 1.7900192730149391


def test_gjk_spheres():
    sphere1 = colliders.Sphere(center=np.array([0, 0, 0]), radius=1.0)
    sphere2 = colliders.Sphere(center=np.array([1, 1, 1]), radius=1.0)
    dist, closest_point1, closest_point2, _ = gjk.gjk_with_simplex(
        sphere1, sphere2)
    assert approx(dist) == 0.0
    assert_array_almost_equal(closest_point1, np.array([0.5, 0.5, 0.633975]))
    assert_array_almost_equal(closest_point1, closest_point2)

    sphere1 = colliders.Sphere(center=np.array([0, 0, 0]), radius=1.0)
    sphere2 = colliders.Sphere(center=np.array([0, 0, 3]), radius=1.0)
    dist, closest_point1, closest_point2, _ = gjk.gjk_with_simplex(
        sphere1, sphere2)
    assert approx(dist) == 1
    assert_array_almost_equal(closest_point1, np.array([0, 0, 1]))
    assert_array_almost_equal(closest_point2, np.array([0, 0, 2]))


def test_gjk_capsules():
    capsule1 = colliders.Capsule(np.eye(4), 1, 1)
    A2B = np.eye(4)
    A2B[:3, 3] = np.array([3, 0, 0])
    capsule2 = colliders.Capsule(A2B, 1, 1)
    dist, closest_point1, closest_point2, _ = gjk.gjk_with_simplex(
        capsule1, capsule2)
    assert approx(dist) == 1
    assert_array_almost_equal(closest_point1, np.array([1, 0, -0.5]))
    assert_array_almost_equal(closest_point2, np.array([2, 0, -0.5]))

    A2B = np.eye(4)
    A2B[:3, 3] = np.array([0, 0, 4])
    capsule2 = colliders.Capsule(A2B, 1, 1)
    dist, closest_point1, closest_point2, _ = gjk.gjk_with_simplex(
        capsule1, capsule2)
    assert approx(dist) == 1
    assert_array_almost_equal(closest_point1, np.array([0, 0, 1.5]))
    assert_array_almost_equal(closest_point2, np.array([0, 0, 2.5]))


def test_gjk_convex():
    random_state = np.random.RandomState(23)

    for _ in range(50):
        vertices1 = random_state.rand(6, 3) * np.array([[2, 5, 1]])
        convex1 = colliders.Convex(vertices1)

        vertices2 = random_state.rand(6, 3) * np.array([[1, 3, 1]])
        convex2 = colliders.Convex(vertices2)

        dist, closest_point1, closest_point2, _ = gjk.gjk_with_simplex(
            convex1, convex2)
        assert 0 <= closest_point1[0] < 2
        assert 0 <= closest_point1[1] < 5
        assert 0 <= closest_point1[2] < 1
        assert 0 <= closest_point2[0] < 1
        assert 0 <= closest_point2[1] < 3
        assert 0 <= closest_point2[2] < 1
        assert approx(dist) == np.linalg.norm(closest_point2 - closest_point1)

    for _ in range(50):
        vertices1 = random_state.rand(6, 3) * np.array([[2, 5, 1]])
        convex1 = colliders.Convex(vertices1)

        vertices2 = random_state.rand(6, 3) * np.array([[-2, -3, 1]])
        convex2 = colliders.Convex(vertices2)

        dist, closest_point1, closest_point2, _ = gjk.gjk_with_simplex(
            convex1, convex2)
        assert 0 <= closest_point1[0] < 2
        assert 0 <= closest_point1[1] < 5
        assert 0 <= closest_point1[2] < 1
        assert -2 < closest_point2[0] <= 2
        assert -3 < closest_point2[1] <= 5
        assert 0 <= closest_point2[2] < 1
        assert approx(dist) == np.linalg.norm(closest_point2 - closest_point1)


def test_gjk_triangle_to_triangle():
    random_state = np.random.RandomState(81)
    for _ in range(10):
        triangle_points = random.randn_triangle(random_state)
        triangle_points2 = random.randn_triangle(random_state)
        dist, closest_point_triangle, closest_point_triangle2 = gjk.gjk(
            triangle_points, triangle_points2)
        dist2, closest_point_triangle_2, closest_point_triangle2_2 = distance.triangle_to_triangle(
            triangle_points, triangle_points2)
        assert approx(dist) == dist2
        assert_array_almost_equal(
            closest_point_triangle, closest_point_triangle_2)
        assert_array_almost_equal(
            closest_point_triangle2, closest_point_triangle2_2)


def test_gjk_triangle_to_rectangle():
    random_state = np.random.RandomState(82)
    for _ in range(10):
        triangle_points = random.randn_triangle(random_state)
        rectangle_center, rectangle_axes, rectangle_lengths = random.randn_rectangle(
            random_state)
        rectangle_points = geometry.convert_rectangle_to_vertices(
            rectangle_center, rectangle_axes, rectangle_lengths)
        dist, closest_point_triangle, closest_point_rectangle = gjk.gjk(
            triangle_points, rectangle_points)
        dist2, closest_point_triangle2, closest_point_rectangle2 = distance.triangle_to_rectangle(
            triangle_points, rectangle_center, rectangle_axes, rectangle_lengths)
        assert approx(dist) == dist2
        assert_array_almost_equal(
            closest_point_triangle, closest_point_triangle2)
        assert_array_almost_equal(
            closest_point_rectangle, closest_point_rectangle2)
