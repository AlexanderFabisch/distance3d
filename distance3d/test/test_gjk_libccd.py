import numpy as np
from distance3d import colliders, gjk, random


def test_gjk_no_intersection():
    random_state = np.random.RandomState(25)

    for _ in range(10):
        vertices1 = random_state.rand(6, 3)
        vertices2 = random_state.rand(6, 3) + np.ones(3)

        convex1 = colliders.ConvexHullVertices(vertices1)
        convex2 = colliders.ConvexHullVertices(vertices2)

        assert not gjk.gjk_intersection_libccd(convex1, convex2)


def test_gjk_intersection_libccd_vs_gjk_distance_original():
    random_state = np.random.RandomState(24)

    for _ in range(500):
        vertices1 = random_state.rand(6, 3) * np.array([[2, 2, 2]])
        vertices2 = random_state.rand(6, 3) * np.array([[1, 1, 1]])

        if random_state.rand() > 0.5:
            vertices2[3] = vertices1[0]
        if random_state.rand() > 0.75:
            vertices2[2] = vertices1[1]
        if random_state.rand() > 0.875:
            vertices2[1] = vertices1[2]

        convex1 = colliders.ConvexHullVertices(vertices1)

        convex2 = colliders.ConvexHullVertices(vertices2)

        dist, closest_point1, closest_point2, _ = gjk.gjk_distance_original(convex1, convex2)
        gjk_intersection = dist < 10.0 * np.finfo(float).eps
        gjk_intersection2 = gjk.gjk_intersection_libccd(convex1, convex2)
        assert gjk_intersection == gjk_intersection2


def test_gjk_max_iterations():
    random_state = np.random.RandomState(27)
    c1 = colliders.MeshGraph(*random.randn_convex(
        random_state, n_vertices=10000, center_scale=1.0))
    c2 = colliders.MeshGraph(*random.randn_convex(
        random_state, n_vertices=10000, center_scale=1.0))
    # not enough iterations to find correct solution
    intersection = gjk.gjk_intersection_libccd(c1, c2, max_iterations=2)
    assert not intersection


def test_gjk_touching_contact():
    random_state = np.random.RandomState(26)
    for _ in range(10):
        vertices1_xy = random_state.rand(6, 2)
        vertices2_xy = random_state.rand(6, 2)

        n_common_vertices = random_state.randint(1, 5)
        common_vertices_xy = random_state.rand(n_common_vertices, 2)
        common_vertices = np.hstack((
            common_vertices_xy, np.ones((n_common_vertices, 1))))

        vertices1 = np.vstack((
            np.hstack((vertices1_xy, np.zeros((len(vertices1_xy), 1)))),
            common_vertices
        ))
        vertices2 = np.vstack((
            common_vertices,
            np.hstack((vertices2_xy, 2 * np.ones((len(vertices2_xy), 1))))
        ))

        convex1 = colliders.ConvexHullVertices(vertices1)
        convex2 = colliders.ConvexHullVertices(vertices2)

        assert gjk.gjk_intersection_libccd(convex1, convex2)
