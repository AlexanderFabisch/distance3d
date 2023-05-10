import numpy as np
from distance3d import colliders, gjk, geometry, random, distance
from pytest import approx
from numpy.testing import assert_array_almost_equal


def test_gjk_points():
    p1 = colliders.ConvexHullVertices(np.array([[0.0, 0.0, 0.0]]))
    assert gjk.gjk_intersection_jolt(p1, p1)
    assert gjk.gjk_intersection_libccd(p1, p1)
    assert gjk.gjk_distance_original(p1, p1)[0] == 0.0
    assert gjk.gjk_distance_jolt(p1, p1)[0] == 0.0

    p2 = colliders.ConvexHullVertices(np.array([[1.0, 0.0, 0.0]]))
    assert not gjk.gjk_intersection_jolt(p1, p2)
    assert not gjk.gjk_intersection_libccd(p1, p2)
    assert gjk.gjk_distance_original(p1, p2)[0] == 1.0
    assert gjk.gjk_distance_jolt(p1, p2)[0] == 1.0


def test_gjk_line_segments():
    s1 = colliders.ConvexHullVertices(np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    assert gjk.gjk_intersection_jolt(s1, s1)
    assert gjk.gjk_intersection_libccd(s1, s1)
    assert gjk.gjk_distance_original(s1, s1)[0] == 0.0
    assert gjk.gjk_distance_jolt(s1, s1)[0] == 0.0

    p1 = colliders.ConvexHullVertices(np.array([[0.0, 0.0, 0.0]]))
    assert gjk.gjk_intersection_jolt(s1, p1)
    assert gjk.gjk_intersection_libccd(s1, p1)
    assert gjk.gjk_distance_original(s1, p1)[0] == 0.0
    assert gjk.gjk_distance_jolt(s1, p1)[0] == 0.0

    p2 = colliders.ConvexHullVertices(np.array([[1.0, 0.0, 0.0]]))
    assert gjk.gjk_intersection_jolt(s1, p2)
    assert gjk.gjk_intersection_libccd(s1, p2)
    assert gjk.gjk_distance_original(s1, p2)[0] == 0.0
    assert gjk.gjk_distance_jolt(s1, p2)[0] == 0.0

    s2 = colliders.ConvexHullVertices(np.array([
        [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]))
    assert not gjk.gjk_intersection_jolt(s1, s2)
    assert not gjk.gjk_intersection_libccd(s1, s2)
    assert gjk.gjk_distance_original(s1, s2)[0] == 1.0
    assert gjk.gjk_distance_jolt(s1, s2)[0] == 1.0

    s2 = colliders.ConvexHullVertices(np.array([
        [0.5, -1.0, 0.0], [0.5, 1.0, 0.0]]))
    assert gjk.gjk_intersection_jolt(s1, s2)
    assert gjk.gjk_intersection_libccd(s1, s2)
    assert gjk.gjk_distance_original(s1, s2)[0] == 0.0
    assert gjk.gjk_distance_jolt(s1, s2)[0] == 0.0


def test_gjk_boxes():
    box2origin = np.eye(4)
    size = np.ones(3)
    box_collider = colliders.Box(box2origin, size)

    # complete overlap
    dist, closest_point1, closest_point2, _ = gjk.gjk_distance_original(
        box_collider, box_collider)
    assert approx(dist) == 0.0
    assert_array_almost_equal(closest_point1, np.array([-0.5, -0.5, -0.5]))
    assert_array_almost_equal(closest_point1, closest_point2)

    assert gjk.gjk_nesterov_accelerated_intersection(
        box_collider, box_collider)

    # touching faces, edges, or points
    for dim1 in range(3):
        for dim2 in range(3):
            for dim3 in range(3):
                for sign1 in [-1, 1]:
                    for sign2 in [-1, 1]:
                        for sign3 in [-1, 1]:
                            box2origin2 = np.eye(4)
                            box2origin2[dim1, 3] = sign1
                            box2origin2[dim2, 3] = sign2
                            box2origin2[dim3, 3] = sign3
                            size2 = np.ones(3)
                            box_collider2 = colliders.Box(box2origin2, size2)

                            dist, closest_point1, closest_point2, _ = gjk.gjk_distance_original(
                                box_collider, box_collider2)
                            assert approx(dist) == 0.0
                            expected = -0.5 * np.ones(3)
                            expected[dim1] = 0.5 * sign1
                            expected[dim2] = 0.5 * sign2
                            expected[dim3] = 0.5 * sign3
                            assert_array_almost_equal(closest_point1, expected)
                            assert_array_almost_equal(
                                closest_point1, closest_point2)

                            dist2, _, _, _ = gjk.gjk_distance_jolt(
                                box_collider, box_collider2)
                            assert approx(dist2) == 0.0

                            assert gjk.gjk_nesterov_accelerated_intersection(
                                box_collider, box_collider2)


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

    dist, closest_point1, closest_point2, _ = gjk.gjk_distance_original(
        box_collider, box_collider2)
    assert approx(dist) == 1.7900192730149391

    dist2, _, _, _ = gjk.gjk_distance_jolt(
        box_collider, box_collider2)
    assert approx(dist2) == 1.7900192730149391

    assert not gjk.gjk_nesterov_accelerated_intersection(
        box_collider, box_collider2)


def test_gjk_spheres():
    sphere1 = colliders.Sphere(center=np.array([0, 0, 0], dtype=float), radius=1.0)
    dist, closest_point1, closest_point2, _ = gjk.gjk_distance_original(
        sphere1, sphere1)
    assert approx(dist) == 0.0
    assert_array_almost_equal(closest_point1, np.array([0, 0, 1]))
    assert_array_almost_equal(closest_point1, closest_point2)

    dist, _, _, _ = gjk.gjk_distance_jolt(sphere1, sphere1)
    assert approx(dist) == 0.0

    assert gjk.gjk_nesterov_accelerated_intersection(sphere1, sphere1)
    dist = gjk.gjk_nesterov_accelerated_distance(sphere1, sphere1)
    assert approx(dist) == 0.0

    sphere2 = colliders.Sphere(center=np.array([1, 1, 1], dtype=float), radius=1.0)
    dist, closest_point1, closest_point2, _ = gjk.gjk_distance_original(
        sphere1, sphere2)
    assert approx(dist) == 0.0
    assert_array_almost_equal(closest_point1, np.array([0.5, 0.5, 0.633975]))
    assert_array_almost_equal(closest_point1, closest_point2)

    dist, _, _, _ = gjk.gjk_distance_jolt(sphere1, sphere2)
    assert approx(dist) == 0.0

    assert gjk.gjk_nesterov_accelerated_intersection(sphere1, sphere2)
    dist = gjk.gjk_nesterov_accelerated_distance(sphere1, sphere2)
    assert approx(dist) == 0.0

    sphere1 = colliders.Sphere(center=np.array([0, 0, 0], dtype=float), radius=1.0)
    sphere2 = colliders.Sphere(center=np.array([0, 0, 3], dtype=float), radius=1.0)
    dist, closest_point1, closest_point2, _ = gjk.gjk_distance_original(
        sphere1, sphere2)
    assert approx(dist) == 1
    assert_array_almost_equal(closest_point1, np.array([0, 0, 1]))
    assert_array_almost_equal(closest_point2, np.array([0, 0, 2]))

    dist, _, _, _ = gjk.gjk_distance_jolt(sphere1, sphere2)
    assert approx(dist) == 1.0

    assert not gjk.gjk_nesterov_accelerated_intersection(sphere1, sphere2)
    dist = gjk.gjk_nesterov_accelerated_distance(sphere1, sphere2)
    assert approx(dist) == 1.0


def test_gjk_cylinders():
    cylinder1 = colliders.Cylinder(np.eye(4), 1, 1)
    dist, closest_point1, closest_point2, _ = gjk.gjk_distance_original(
        cylinder1, cylinder1)
    assert approx(dist) == 0
    assert_array_almost_equal(closest_point1, np.array([1, 0, 0.5]))
    assert_array_almost_equal(closest_point2, np.array([1, 0, 0.5]))

    dist, _, _, _ = gjk.gjk_distance_jolt(cylinder1, cylinder1)
    assert approx(dist) == 0

    assert gjk.gjk_nesterov_accelerated_intersection(cylinder1, cylinder1)
    dist = gjk.gjk_nesterov_accelerated_distance(cylinder1, cylinder1)
    assert approx(dist) == 0.0


    A2B = np.eye(4)
    A2B[:3, 3] = np.array([3, 0, 0])
    cylinder2 = colliders.Cylinder(A2B, 1, 1)
    dist, closest_point1, closest_point2, _ = gjk.gjk_distance_original(
        cylinder1, cylinder2)
    assert approx(dist) == 1
    assert_array_almost_equal(closest_point1, np.array([1, 0, 0.5]))
    assert_array_almost_equal(closest_point2, np.array([2, 0, 0.5]))

    dist, _, _, _ = gjk.gjk_distance_jolt(cylinder1, cylinder2)
    assert approx(dist) == 1

    assert not gjk.gjk_nesterov_accelerated_intersection(cylinder1, cylinder2)
    # dist, _, _, _ = gjk.gjk_nesterov_accelerated_distance(cylinder1, cylinder2)
    # assert approx(dist) == 1

    A2B = np.eye(4)
    A2B[:3, 3] = np.array([0, 0, 4])
    cylinder2 = colliders.Cylinder(A2B, 1, 1)
    dist, closest_point1, closest_point2, _ = gjk.gjk_distance_original(
        cylinder1, cylinder2)
    assert approx(dist) == 3
    assert_array_almost_equal(closest_point1, np.array([1, 0, 0.5]))
    assert_array_almost_equal(closest_point2, np.array([1, 0, 3.5]))

    dist, _, _, _ = gjk.gjk_distance_jolt(cylinder1, cylinder2)
    assert approx(dist) == 3

    assert not gjk.gjk_nesterov_accelerated_intersection(cylinder1, cylinder2)
    dist = gjk.gjk_nesterov_accelerated_distance(cylinder1, cylinder2)
    assert approx(dist, rel=1e-3) == 3


def test_gjk_capsules():
    capsule1 = colliders.Capsule(np.eye(4), 1, 1)
    dist, closest_point1, closest_point2, _ = gjk.gjk(
        capsule1, capsule1)
    assert approx(dist) == 0
    assert_array_almost_equal(closest_point1, closest_point2)

    dist, _, _, _ = gjk.gjk_distance_jolt(capsule1, capsule1)
    assert approx(dist) == 0

    assert gjk.gjk_nesterov_accelerated_intersection(capsule1, capsule1)
    dist = gjk.gjk_nesterov_accelerated_distance(capsule1, capsule1)
    assert approx(dist) == 0.0


    A2B = np.eye(4)
    A2B[:3, 3] = np.array([3, 0, 0])
    capsule2 = colliders.Capsule(A2B, 1, 1)
    dist, closest_point1, closest_point2, _ = gjk.gjk(
        capsule1, capsule2)
    assert approx(dist) == 1
    assert_array_almost_equal(closest_point1, np.array([1, 0, -0.5]))
    assert_array_almost_equal(closest_point2, np.array([2, 0, -0.5]))

    dist, _, _, _ = gjk.gjk_distance_jolt(capsule1, capsule2)
    assert approx(dist) == 1

    assert not gjk.gjk_nesterov_accelerated_intersection(capsule1, capsule2)
    dist = gjk.gjk_nesterov_accelerated_distance(capsule1, capsule2)
    assert approx(dist) == 1


    A2B = np.eye(4)
    A2B[:3, 3] = np.array([0, 0, 4])
    capsule2 = colliders.Capsule(A2B, 1, 1)
    dist, closest_point1, closest_point2, _ = gjk.gjk(
        capsule1, capsule2)
    assert approx(dist) == 1
    assert_array_almost_equal(closest_point1, np.array([0, 0, 1.5]))
    assert_array_almost_equal(closest_point2, np.array([0, 0, 2.5]))

    dist, _, _, _ = gjk.gjk_distance_jolt(capsule1, capsule2)
    assert approx(dist) == 1

    assert not gjk.gjk_nesterov_accelerated_intersection(capsule1, capsule2)
    dist = gjk.gjk_nesterov_accelerated_distance(capsule1, capsule2)
    assert approx(dist) == 1


def test_gjk_ellipsoids():
    random_state = np.random.RandomState(83)
    for _ in range(10):
        ellipsoid2origin1, radii1 = random.rand_ellipsoid(
            random_state, center_scale=2.0)
        ellipsoid2origin2, radii2 = random.rand_ellipsoid(
            random_state, center_scale=2.0)
        dist, closest_point1, closest_point2, _ = gjk.gjk_distance_original(
            colliders.Ellipsoid(ellipsoid2origin1, radii1),
            colliders.Ellipsoid(ellipsoid2origin2, radii2))

        dist12, closest_point12 = distance.point_to_ellipsoid(
            closest_point1, ellipsoid2origin2, radii2)
        dist21, closest_point21 = distance.point_to_ellipsoid(
            closest_point2, ellipsoid2origin1, radii1)
        assert approx(dist) == dist12
        assert_array_almost_equal(closest_point2, closest_point12)
        assert approx(dist) == dist21
        assert_array_almost_equal(closest_point1, closest_point21)

        dist, closest_point1, closest_point2, _ = gjk.gjk_distance_original(
            colliders.Ellipsoid(ellipsoid2origin1, radii1),
            colliders.Ellipsoid(ellipsoid2origin2, radii2))
        assert approx(dist) == dist12
        assert_array_almost_equal(closest_point2, closest_point12)
        assert approx(dist) == dist21
        assert_array_almost_equal(closest_point1, closest_point21)


def test_compare_gjk_intersection_flavours_with_random_shapes():
    random_state = np.random.RandomState(84)
    shape_names = list(colliders.COLLIDERS.keys())
    for i in range(100):
        shape1 = shape_names[random_state.randint(len(shape_names))]
        args1 = random.RANDOM_GENERATORS[shape1](random_state)
        shape2 = shape_names[random_state.randint(len(shape_names))]
        args2 = random.RANDOM_GENERATORS[shape2](random_state)
        collider1 = colliders.COLLIDERS[shape1](*args1)
        collider2 = colliders.COLLIDERS[shape2](*args2)

        intersection_jolt = gjk.gjk_intersection_jolt(collider1, collider2)
        intersection_libccd = gjk.gjk_intersection_libccd(collider1, collider2)
        assert intersection_jolt == intersection_libccd

def test_compare_gjk_distance_flavours_with_random_shapes():
    random_state = np.random.RandomState(85)
    shape_names = list(colliders.COLLIDERS.keys())
    for _ in range(100):
        shape1 = shape_names[random_state.randint(len(shape_names))]
        args1 = random.RANDOM_GENERATORS[shape1](random_state)
        shape2 = shape_names[random_state.randint(len(shape_names))]
        args2 = random.RANDOM_GENERATORS[shape2](random_state)
        collider1 = colliders.COLLIDERS[shape1](*args1)
        collider2 = colliders.COLLIDERS[shape2](*args2)

        dist_jolt, cp1_jolt, cp2_jolt, _ = gjk.gjk_distance_jolt(collider1, collider2)
        dist_orig, cp1_orig, cp2_orig, _ = gjk.gjk_distance_original(collider1, collider2)
        assert approx(dist_jolt) == dist_orig
        assert approx(np.linalg.norm(cp1_orig - cp2_orig)) == np.linalg.norm(cp1_jolt - cp2_jolt)


def test_gjk_random_points():
    random_state = np.random.RandomState(23)

    for _ in range(50):
        vertices1 = random_state.rand(6, 3) * np.array([[2, 5, 1]])
        convex1 = colliders.ConvexHullVertices(vertices1)

        vertices2 = random_state.rand(6, 3) * np.array([[1, 3, 1]])
        convex2 = colliders.ConvexHullVertices(vertices2)

        dist, closest_point1, closest_point2, _ = gjk.gjk(
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
        convex1 = colliders.ConvexHullVertices(vertices1)

        vertices2 = random_state.rand(6, 3) * np.array([[-2, -3, 1]])
        convex2 = colliders.ConvexHullVertices(vertices2)

        dist, closest_point1, closest_point2, _ = gjk.gjk(
            convex1, convex2)
        assert 0 <= closest_point1[0] < 2
        assert 0 <= closest_point1[1] < 5
        assert 0 <= closest_point1[2] < 1
        assert -2 < closest_point2[0] <= 2
        assert -3 < closest_point2[1] <= 5
        assert 0 <= closest_point2[2] < 1
        assert approx(dist) == np.linalg.norm(closest_point2 - closest_point1)


def test_gjk_point_subset():
    random_state = np.random.RandomState(333)

    for _ in range(50):
        vertices1 = random_state.rand(15, 3)
        convex1 = colliders.ConvexHullVertices(vertices1)
        vertices2 = vertices1[::2]
        convex2 = colliders.ConvexHullVertices(vertices2)
        dist, closest_point1, closest_point2, _ = gjk.gjk_distance_original(
            convex1, convex2)
        assert approx(dist) == 0.0
        assert_array_almost_equal(closest_point1, closest_point2)
        assert closest_point1 in vertices1
        assert closest_point2 in vertices2


def test_gjk_triangle_to_triangle():
    random_state = np.random.RandomState(81)
    for _ in range(10):
        triangle_points = random.randn_triangle(random_state)
        triangle_points2 = random.randn_triangle(random_state)
        dist, closest_point_triangle, closest_point_triangle2, _ = gjk.gjk_distance_original(
            colliders.ConvexHullVertices(triangle_points), colliders.ConvexHullVertices(triangle_points2))
        dist2, closest_point_triangle_2, closest_point_triangle2_2 = distance.triangle_to_triangle(
            triangle_points, triangle_points2)
        assert approx(dist) == dist2
        assert_array_almost_equal(
            closest_point_triangle, closest_point_triangle_2)
        assert_array_almost_equal(
            closest_point_triangle2, closest_point_triangle2_2)
        dist3, closest_point_triangle_3, closest_point_triangle2_3, _ = gjk.gjk_distance_original(
            colliders.ConvexHullVertices(triangle_points), colliders.ConvexHullVertices(triangle_points2))
        assert approx(dist) == dist3
        assert_array_almost_equal(
            closest_point_triangle, closest_point_triangle_3)
        assert_array_almost_equal(
            closest_point_triangle2, closest_point_triangle2_3)


def test_gjk_triangle_to_rectangle():
    random_state = np.random.RandomState(82)
    for _ in range(10):
        triangle_points = random.randn_triangle(random_state)
        rectangle_center, rectangle_axes, rectangle_lengths = random.randn_rectangle(
            random_state)
        rectangle_points = geometry.convert_rectangle_to_vertices(
            rectangle_center, rectangle_axes, rectangle_lengths)
        dist, closest_point_triangle, closest_point_rectangle, _ = gjk.gjk(
            colliders.ConvexHullVertices(triangle_points), colliders.ConvexHullVertices(rectangle_points))
        dist2, closest_point_triangle2, closest_point_rectangle2 = distance.triangle_to_rectangle(
            triangle_points, rectangle_center, rectangle_axes, rectangle_lengths)
        assert approx(dist) == dist2
        assert_array_almost_equal(
            closest_point_triangle, closest_point_triangle2)
        assert_array_almost_equal(
            closest_point_rectangle, closest_point_rectangle2)
        dist3, closest_point_triangle3, closest_point_rectangle3 = distance.triangle_to_rectangle(
            triangle_points, rectangle_center, rectangle_axes, rectangle_lengths)
        assert approx(dist) == dist3
        assert_array_almost_equal(
            closest_point_triangle, closest_point_triangle3)
        assert_array_almost_equal(
            closest_point_rectangle, closest_point_rectangle3)


def test_gjk_distance_with_margin():
    box = colliders.Box(np.eye(4), np.array([2.0, 2.0, 2.0]))
    box_with_margin = colliders.Margin(box, 0.1)
    sphere = colliders.Sphere(np.array([2.0, 0.0, 0.0]), 0.5)
    dist_without_margin = gjk.gjk(box, sphere)[0]
    dist_with_margin = gjk.gjk(box_with_margin, sphere)[0]
    assert approx(dist_without_margin) == dist_with_margin + 0.1
