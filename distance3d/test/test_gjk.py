import numpy as np
from distance3d import colliders, gjk, geometry, random, distance
from pytest import approx
from numpy.testing import assert_array_almost_equal


def test_select_line_segment():
    simplex = gjk.SimplexInfo()
    simplex.n_simplex_points = 4
    random_state = np.random.RandomState(23)
    simplex.points[:, :] = random_state.randn(*simplex.points.shape)
    simplex.dot_product_table = simplex.points.dot(simplex.points.T)
    simplex.indices_polytope1 = np.arange(4, dtype=int)
    simplex.indices_polytope2 = np.arange(4, dtype=int)
    for i in range(1, 4):
        for j in range(i + 1, 4):
            simplex.dot_product_table[i, j] = float("nan")

    assert_dot_product_table(simplex)

    simplex_backup = gjk.SimplexInfo()
    simplex_backup.copy_from(simplex)

    for i in range(4):
        for j in range(1, 4):
            if i == j:
                continue
            simplex.copy_from(simplex_backup)
            simplex.select_line_segment(i, j)
            assert_dot_product_table(simplex)


def test_select_face():
    simplex = gjk.SimplexInfo()
    simplex.n_simplex_points = 4
    random_state = np.random.RandomState(24)
    simplex.points[:, :] = random_state.randn(*simplex.points.shape)
    simplex.dot_product_table = simplex.points.dot(simplex.points.T)
    simplex.indices_polytope1 = np.arange(4, dtype=int)
    simplex.indices_polytope2 = np.arange(4, dtype=int)
    for i in range(1, 4):
        for j in range(i + 1, 4):
            simplex.dot_product_table[i, j] = float("nan")

    assert_dot_product_table(simplex)

    simplex_backup = gjk.SimplexInfo()
    simplex_backup.copy_from(simplex)

    for i in range(4):
        for j in range(1, 4):
            for k in range(2, 4):
                if i == j or i == k or j == k:
                    continue
                simplex.copy_from(simplex_backup)
                simplex.select_face(i, j, k)
                assert_dot_product_table(simplex)


def assert_dot_product_table(simplex):
    for i in range(simplex.n_simplex_points):
        for j in range(i + 1):
            assert approx(simplex.dot_product_table[i, j]) == np.dot(
                simplex.points[i], simplex.points[j])


def test_simplex_reorder():
    simplex = gjk.SimplexInfo()
    simplex.n_simplex_points = 4
    random_state = np.random.RandomState(24)
    simplex.points[:, :] = random_state.randn(*simplex.points.shape)
    simplex.dot_product_table = simplex.points.dot(simplex.points.T)
    simplex.indices_polytope1 = np.arange(4, dtype=int)
    simplex.indices_polytope2 = np.arange(4, dtype=int)
    for i in range(1, 4):
        for j in range(i + 1, 4):
            simplex.dot_product_table[i, j] = float("nan")

    simplex_backup = gjk.SimplexInfo()
    simplex_backup.copy_from(simplex)

    simplex.reorder(np.array([3, 2, 1, 0], dtype=int))
    assert_array_almost_equal(simplex.indices_polytope1, [3, 2, 1, 0])
    assert_array_almost_equal(simplex.indices_polytope2, [3, 2, 1, 0])
    assert_dot_product_table(simplex)
    assert len(simplex) == 4

    simplex.reorder(np.array([3, 2, 1, 0], dtype=int))
    assert_array_almost_equal(simplex.indices_polytope1, [0, 1, 2, 3])
    assert_array_almost_equal(simplex.indices_polytope2, [0, 1, 2, 3])
    assert_dot_product_table(simplex)


def test_vertex_optimal_point():
    simplex = gjk.SimplexInfo()
    simplex.set_first_point(0, 0, np.array([0, 0, 1], dtype=float))

    solution, backup = gjk.distance_subalgorithm_with_backup_procedure(
        simplex, gjk.Solution(), False)
    assert not backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:1], [1])


def test_vertex_optimal_point_backup():
    simplex = gjk.SimplexInfo()
    simplex.set_first_point(0, 0, np.array([0, 0, 1], dtype=float))

    solution, backup = gjk.distance_subalgorithm_with_backup_procedure(
        simplex, gjk.Solution(), True)
    assert backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:1], [1])


def test_line_segment_optimal_point():
    simplex = gjk.SimplexInfo()
    simplex.set_first_point(0, 0, np.array([0, 0, 2], dtype=float))
    simplex.add_new_point(1, 1, np.array([0, 0, 1], dtype=float))

    barycentric_coordinates = gjk.BarycentricCoordinates()
    barycentric_coordinates.line_segment_coordinates_0(simplex)
    barycentric_coordinates.line_segment_coordinates_1(simplex)
    assert barycentric_coordinates.vertex_0_of_line_segment_optimal()

    solution, backup = gjk.distance_subalgorithm_with_backup_procedure(
        simplex, gjk.Solution(), False)
    assert not backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:1], [1])

    simplex = gjk.SimplexInfo()
    simplex.set_first_point(0, 0, np.array([0, 0, 1], dtype=float))
    simplex.add_new_point(1, 1, np.array([0, 0, 2], dtype=float))

    barycentric_coordinates = gjk.BarycentricCoordinates()
    barycentric_coordinates.line_segment_coordinates_0(simplex)
    barycentric_coordinates.line_segment_coordinates_1(simplex)
    assert barycentric_coordinates.vertex_1_of_line_segment_optimal()

    solution, backup = gjk.distance_subalgorithm_with_backup_procedure(
        simplex, gjk.Solution(), False)
    assert not backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:1], [1])

    simplex = gjk.SimplexInfo()
    simplex.set_first_point(0, 0, np.array([-1, 0, 1], dtype=float))
    simplex.add_new_point(1, 1, np.array([1, 0, 1], dtype=float))

    barycentric_coordinates = gjk.BarycentricCoordinates()
    barycentric_coordinates.line_segment_coordinates_0(simplex)
    barycentric_coordinates.line_segment_coordinates_1(simplex)
    assert barycentric_coordinates.line_segment_01_of_line_segment_optimal()

    solution, backup = gjk.distance_subalgorithm_with_backup_procedure(
        simplex, gjk.Solution(), False)
    assert not backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:2], [0.5, 0.5])


def test_line_segment_optimal_point_backup():
    simplex = gjk.SimplexInfo()
    simplex.set_first_point(0, 0, np.array([0, 0, 2], dtype=float))
    simplex.add_new_point(1, 1, np.array([0, 0, 1], dtype=float))

    barycentric_coordinates = gjk.BarycentricCoordinates()
    barycentric_coordinates.line_segment_coordinates_0(simplex)
    barycentric_coordinates.line_segment_coordinates_1(simplex)
    assert barycentric_coordinates.vertex_0_of_line_segment_optimal()

    solution, backup = gjk.distance_subalgorithm_with_backup_procedure(
        simplex, gjk.Solution(), True)
    assert backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:1], [1])

    simplex = gjk.SimplexInfo()
    simplex.set_first_point(0, 0, np.array([0, 0, 1], dtype=float))
    simplex.add_new_point(1, 1, np.array([0, 0, 2], dtype=float))

    barycentric_coordinates = gjk.BarycentricCoordinates()
    barycentric_coordinates.line_segment_coordinates_0(simplex)
    barycentric_coordinates.line_segment_coordinates_1(simplex)
    assert barycentric_coordinates.vertex_1_of_line_segment_optimal()

    solution, backup = gjk.distance_subalgorithm_with_backup_procedure(
        simplex, gjk.Solution(), True)
    assert backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:1], [1])

    simplex = gjk.SimplexInfo()
    simplex.set_first_point(0, 0, np.array([-1, 0, 1], dtype=float))
    simplex.add_new_point(1, 1, np.array([1, 0, 1], dtype=float))

    barycentric_coordinates = gjk.BarycentricCoordinates()
    barycentric_coordinates.line_segment_coordinates_0(simplex)
    barycentric_coordinates.line_segment_coordinates_1(simplex)
    assert barycentric_coordinates.line_segment_01_of_line_segment_optimal()

    solution, backup = gjk.distance_subalgorithm_with_backup_procedure(
        simplex, gjk.Solution(), True)
    assert backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:2], [0.5, 0.5])


def test_face_optimal_point():
    simplex = gjk.SimplexInfo()
    simplex.set_first_point(0, 0, np.array([1, 0, 2], dtype=float))
    simplex.add_new_point(1, 1, np.array([0, 0, 2], dtype=float))
    simplex.add_new_point(2, 2, np.array([0, 0, 1], dtype=float))

    barycentric_coordinates = gjk.BarycentricCoordinates()
    barycentric_coordinates.face_coordinates_0(simplex)
    barycentric_coordinates.face_coordinates_1(simplex)
    e123 = barycentric_coordinates.face_coordinates_2(simplex)
    barycentric_coordinates.face_coordinates_3(simplex, e123)
    assert_array_almost_equal(simplex.points[0], [0, 0, 1])
    assert barycentric_coordinates.vertex_0_of_face_optimal()

    solution, backup = gjk.distance_subalgorithm_with_backup_procedure(
        simplex, gjk.Solution(), False)
    assert not backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:1], [1])

    simplex = gjk.SimplexInfo()
    simplex.set_first_point(0, 0, np.array([0, 0, 1], dtype=float))
    simplex.add_new_point(1, 1, np.array([0, 0, 2], dtype=float))
    simplex.add_new_point(2, 2, np.array([1, 0, 2], dtype=float))

    barycentric_coordinates = gjk.BarycentricCoordinates()
    barycentric_coordinates.face_coordinates_0(simplex)
    barycentric_coordinates.face_coordinates_1(simplex)
    e123 = barycentric_coordinates.face_coordinates_2(simplex)
    barycentric_coordinates.face_coordinates_3(simplex, e123)
    assert_array_almost_equal(simplex.points[1], [0, 0, 1])
    assert barycentric_coordinates.vertex_1_of_face_optimal()

    solution, backup = gjk.distance_subalgorithm_with_backup_procedure(
        simplex, gjk.Solution(), False)
    assert not backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:1], [1])

    simplex = gjk.SimplexInfo()
    simplex.set_first_point(0, 0, np.array([0, 0, 2], dtype=float))
    simplex.add_new_point(1, 1, np.array([0, 0, 1], dtype=float))
    simplex.add_new_point(2, 2, np.array([1, 0, 2], dtype=float))

    barycentric_coordinates = gjk.BarycentricCoordinates()
    barycentric_coordinates.face_coordinates_0(simplex)
    barycentric_coordinates.face_coordinates_1(simplex)
    e123 = barycentric_coordinates.face_coordinates_2(simplex)
    barycentric_coordinates.face_coordinates_3(simplex, e123)
    assert_array_almost_equal(simplex.points[2], [0, 0, 1])
    assert barycentric_coordinates.vertex_2_of_face_optimal()

    solution, backup = gjk.distance_subalgorithm_with_backup_procedure(
        simplex, gjk.Solution(), False)
    assert not backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:1], [1])

    simplex = gjk.SimplexInfo()
    simplex.set_first_point(0, 0, np.array([-1, 0, 1], dtype=float))
    simplex.add_new_point(1, 1, np.array([0, 0, 2], dtype=float))
    simplex.add_new_point(2, 2, np.array([1, 0, 1], dtype=float))

    barycentric_coordinates = gjk.BarycentricCoordinates()
    barycentric_coordinates.face_coordinates_0(simplex)
    barycentric_coordinates.face_coordinates_1(simplex)
    e123 = barycentric_coordinates.face_coordinates_2(simplex)
    barycentric_coordinates.face_coordinates_3(simplex, e123)
    assert barycentric_coordinates.line_segment_01_of_face_optimal()

    solution, backup = gjk.distance_subalgorithm_with_backup_procedure(
        simplex, gjk.Solution(), False)
    assert not backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:2], [0.5, 0.5])

    simplex = gjk.SimplexInfo()
    simplex.set_first_point(0, 0, np.array([1, 0, 1], dtype=float))
    simplex.add_new_point(1, 1, np.array([-1, 0, 1], dtype=float))
    simplex.add_new_point(2, 2, np.array([0, 0, 2], dtype=float))

    barycentric_coordinates = gjk.BarycentricCoordinates()
    barycentric_coordinates.face_coordinates_0(simplex)
    barycentric_coordinates.face_coordinates_1(simplex)
    e123 = barycentric_coordinates.face_coordinates_2(simplex)
    barycentric_coordinates.face_coordinates_3(simplex, e123)
    assert barycentric_coordinates.line_segment_12_of_face_optimal()

    solution, backup = gjk.distance_subalgorithm_with_backup_procedure(
        simplex, gjk.Solution(), False)
    assert not backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:2], [0.5, 0.5])

    simplex = gjk.SimplexInfo()
    simplex.set_first_point(0, 0, np.array([0, 0, 2], dtype=float))
    simplex.add_new_point(1, 1, np.array([1, 0, 1], dtype=float))
    simplex.add_new_point(2, 2, np.array([-1, 0, 1], dtype=float))

    barycentric_coordinates = gjk.BarycentricCoordinates()
    barycentric_coordinates.face_coordinates_0(simplex)
    barycentric_coordinates.face_coordinates_1(simplex)
    e123 = barycentric_coordinates.face_coordinates_2(simplex)
    barycentric_coordinates.face_coordinates_3(simplex, e123)
    assert barycentric_coordinates.line_segment_02_of_face_optimal()

    solution, backup = gjk.distance_subalgorithm_with_backup_procedure(
        simplex, gjk.Solution(), False)
    assert not backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:2], [0.5, 0.5])

    simplex = gjk.SimplexInfo()
    simplex.set_first_point(0, 0, np.array([0, 1, 1], dtype=float))
    simplex.add_new_point(1, 1, np.array([1, -1, 1], dtype=float))
    simplex.add_new_point(2, 2, np.array([-1, -1, 1], dtype=float))

    barycentric_coordinates = gjk.BarycentricCoordinates()
    barycentric_coordinates.face_coordinates_0(simplex)
    barycentric_coordinates.face_coordinates_1(simplex)
    e123 = barycentric_coordinates.face_coordinates_2(simplex)
    barycentric_coordinates.face_coordinates_3(simplex, e123)
    assert barycentric_coordinates.face_012_of_face_optimal()

    solution, backup = gjk.distance_subalgorithm_with_backup_procedure(
        simplex, gjk.Solution(), False)
    assert not backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(
        solution.barycentric_coordinates[:3], [0.25, 0.5, 0.25])


def test_face_optimal_point_backup():
    simplex = gjk.SimplexInfo()
    simplex.set_first_point(0, 0, np.array([1, 0, 2], dtype=float))
    simplex.add_new_point(1, 1, np.array([0, 0, 2], dtype=float))
    simplex.add_new_point(2, 2, np.array([0, 0, 1], dtype=float))

    barycentric_coordinates = gjk.BarycentricCoordinates()
    barycentric_coordinates.face_coordinates_0(simplex)
    barycentric_coordinates.face_coordinates_1(simplex)
    e123 = barycentric_coordinates.face_coordinates_2(simplex)
    barycentric_coordinates.face_coordinates_3(simplex, e123)
    assert_array_almost_equal(simplex.points[0], [0, 0, 1])
    assert barycentric_coordinates.vertex_0_of_face_optimal()

    solution, backup = gjk.distance_subalgorithm_with_backup_procedure(
        simplex, gjk.Solution(), True)
    assert backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:1], [1])

    simplex = gjk.SimplexInfo()
    simplex.set_first_point(0, 0, np.array([0, 0, 1], dtype=float))
    simplex.add_new_point(1, 1, np.array([0, 0, 2], dtype=float))
    simplex.add_new_point(2, 2, np.array([1, 0, 2], dtype=float))

    barycentric_coordinates = gjk.BarycentricCoordinates()
    barycentric_coordinates.face_coordinates_0(simplex)
    barycentric_coordinates.face_coordinates_1(simplex)
    e123 = barycentric_coordinates.face_coordinates_2(simplex)
    barycentric_coordinates.face_coordinates_3(simplex, e123)
    assert_array_almost_equal(simplex.points[1], [0, 0, 1])
    assert barycentric_coordinates.vertex_1_of_face_optimal()

    solution, backup = gjk.distance_subalgorithm_with_backup_procedure(
        simplex, gjk.Solution(), True)
    assert backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:1], [1])

    simplex = gjk.SimplexInfo()
    simplex.set_first_point(0, 0, np.array([0, 0, 2], dtype=float))
    simplex.add_new_point(1, 1, np.array([0, 0, 1], dtype=float))
    simplex.add_new_point(2, 2, np.array([1, 0, 2], dtype=float))

    barycentric_coordinates = gjk.BarycentricCoordinates()
    barycentric_coordinates.face_coordinates_0(simplex)
    barycentric_coordinates.face_coordinates_1(simplex)
    e123 = barycentric_coordinates.face_coordinates_2(simplex)
    barycentric_coordinates.face_coordinates_3(simplex, e123)
    assert_array_almost_equal(simplex.points[2], [0, 0, 1])
    assert barycentric_coordinates.vertex_2_of_face_optimal()

    solution, backup = gjk.distance_subalgorithm_with_backup_procedure(
        simplex, gjk.Solution(), True)
    assert backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:1], [1])

    simplex = gjk.SimplexInfo()
    simplex.set_first_point(0, 0, np.array([-1, 0, 1], dtype=float))
    simplex.add_new_point(1, 1, np.array([0, 0, 2], dtype=float))
    simplex.add_new_point(2, 2, np.array([1, 0, 1], dtype=float))

    barycentric_coordinates = gjk.BarycentricCoordinates()
    barycentric_coordinates.face_coordinates_0(simplex)
    barycentric_coordinates.face_coordinates_1(simplex)
    e123 = barycentric_coordinates.face_coordinates_2(simplex)
    barycentric_coordinates.face_coordinates_3(simplex, e123)
    assert barycentric_coordinates.line_segment_01_of_face_optimal()

    solution, backup = gjk.distance_subalgorithm_with_backup_procedure(
        simplex, gjk.Solution(), True)
    assert backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:2], [0.5, 0.5])

    simplex = gjk.SimplexInfo()
    simplex.set_first_point(0, 0, np.array([1, 0, 1], dtype=float))
    simplex.add_new_point(1, 1, np.array([-1, 0, 1], dtype=float))
    simplex.add_new_point(2, 2, np.array([0, 0, 2], dtype=float))

    barycentric_coordinates = gjk.BarycentricCoordinates()
    barycentric_coordinates.face_coordinates_0(simplex)
    barycentric_coordinates.face_coordinates_1(simplex)
    e123 = barycentric_coordinates.face_coordinates_2(simplex)
    barycentric_coordinates.face_coordinates_3(simplex, e123)
    assert barycentric_coordinates.line_segment_12_of_face_optimal()

    solution, backup = gjk.distance_subalgorithm_with_backup_procedure(
        simplex, gjk.Solution(), True)
    assert backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:2], [0.5, 0.5])

    simplex = gjk.SimplexInfo()
    simplex.set_first_point(0, 0, np.array([0, 0, 2], dtype=float))
    simplex.add_new_point(1, 1, np.array([1, 0, 1], dtype=float))
    simplex.add_new_point(2, 2, np.array([-1, 0, 1], dtype=float))

    barycentric_coordinates = gjk.BarycentricCoordinates()
    barycentric_coordinates.face_coordinates_0(simplex)
    barycentric_coordinates.face_coordinates_1(simplex)
    e123 = barycentric_coordinates.face_coordinates_2(simplex)
    barycentric_coordinates.face_coordinates_3(simplex, e123)
    assert barycentric_coordinates.line_segment_02_of_face_optimal()

    solution, backup = gjk.distance_subalgorithm_with_backup_procedure(
        simplex, gjk.Solution(), True)
    assert backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:2], [0.5, 0.5])

    simplex = gjk.SimplexInfo()
    simplex.set_first_point(0, 0, np.array([0, 1, 1], dtype=float))
    simplex.add_new_point(1, 1, np.array([1, -1, 1], dtype=float))
    simplex.add_new_point(2, 2, np.array([-1, -1, 1], dtype=float))

    barycentric_coordinates = gjk.BarycentricCoordinates()
    barycentric_coordinates.face_coordinates_0(simplex)
    barycentric_coordinates.face_coordinates_1(simplex)
    e123 = barycentric_coordinates.face_coordinates_2(simplex)
    barycentric_coordinates.face_coordinates_3(simplex, e123)
    assert barycentric_coordinates.face_012_of_face_optimal()

    solution, backup = gjk.distance_subalgorithm_with_backup_procedure(
        simplex, gjk.Solution(), True)
    assert backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(
        solution.barycentric_coordinates[:3], [0.25, 0.5, 0.25])


def test_tetrahedron_optimal_point():
    simplex = gjk.SimplexInfo()
    simplex.set_first_point(0, 0, np.array([0, 0, 4], dtype=float))
    simplex.add_new_point(1, 1, np.array([0, 0, 3], dtype=float))
    simplex.add_new_point(2, 2, np.array([0, 0, 2], dtype=float))
    simplex.add_new_point(3, 3, np.array([0, 0, 1], dtype=float))

    barycentric_coordinates = gjk.BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.vertex_0_of_tetrahedron_optimal()

    solution, backup = gjk.distance_subalgorithm_with_backup_procedure(
        simplex, gjk.Solution(), False)
    assert not backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:1], [1])

    simplex = gjk.SimplexInfo()
    simplex.set_first_point(0, 0, np.array([0, 0, 1], dtype=float))
    simplex.add_new_point(1, 1, np.array([0, 0, 4], dtype=float))
    simplex.add_new_point(2, 2, np.array([0, 0, 2], dtype=float))
    simplex.add_new_point(3, 3, np.array([0, 0, 3], dtype=float))

    barycentric_coordinates = gjk.BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.vertex_1_of_tetrahedron_optimal()

    solution, backup = gjk.distance_subalgorithm_with_backup_procedure(
        simplex, gjk.Solution(), False)
    assert not backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:1], [1])

    simplex = gjk.SimplexInfo()
    simplex.set_first_point(0, 0, np.array([0, 0, 4], dtype=float))
    simplex.add_new_point(1, 1, np.array([0, 0, 1], dtype=float))
    simplex.add_new_point(2, 2, np.array([0, 0, 2], dtype=float))
    simplex.add_new_point(3, 3, np.array([0, 0, 3], dtype=float))

    barycentric_coordinates = gjk.BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.vertex_2_of_tetrahedron_optimal()

    solution, backup = gjk.distance_subalgorithm_with_backup_procedure(
        simplex, gjk.Solution(), False)
    assert not backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:1], [1])

    simplex = gjk.SimplexInfo()
    simplex.set_first_point(0, 0, np.array([0, 0, 2], dtype=float))
    simplex.add_new_point(1, 1, np.array([0, 0, 4], dtype=float))
    simplex.add_new_point(2, 2, np.array([0, 0, 1], dtype=float))
    simplex.add_new_point(3, 3, np.array([0, 0, 3], dtype=float))

    barycentric_coordinates = gjk.BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.vertex_3_of_tetrahedron_optimal()

    solution, backup = gjk.distance_subalgorithm_with_backup_procedure(
        simplex, gjk.Solution(), False)
    assert not backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:1], [1])

    simplex = gjk.SimplexInfo()
    simplex.set_first_point(0, 0, np.array([-1, 0, 1], dtype=float))
    simplex.add_new_point(1, 1, np.array([1, 0, 2], dtype=float))
    simplex.add_new_point(2, 2, np.array([0, 0, 2], dtype=float))
    simplex.add_new_point(3, 3, np.array([1, 0, 1], dtype=float))

    barycentric_coordinates = gjk.BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.line_segment_01_of_tetrahedron_optimal()

    solution, backup = gjk.distance_subalgorithm_with_backup_procedure(
        simplex, gjk.Solution(), False)
    assert not backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:2], [0.5, 0.5])

    simplex = gjk.SimplexInfo()
    simplex.set_first_point(0, 0, np.array([1, 0, 2], dtype=float))
    simplex.add_new_point(1, 1, np.array([-1, 0, 1], dtype=float))
    simplex.add_new_point(2, 2, np.array([0, 0, 2], dtype=float))
    simplex.add_new_point(3, 3, np.array([1, 0, 1], dtype=float))

    barycentric_coordinates = gjk.BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.line_segment_02_of_tetrahedron_optimal()

    solution, backup = gjk.distance_subalgorithm_with_backup_procedure(
        simplex, gjk.Solution(), False)
    assert not backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:2], [0.5, 0.5])

    simplex = gjk.SimplexInfo()
    simplex.set_first_point(0, 0, np.array([1, 0, 2], dtype=float))
    simplex.add_new_point(1, 1, np.array([0, 0, 2], dtype=float))
    simplex.add_new_point(2, 2, np.array([-1, 0, 1], dtype=float))
    simplex.add_new_point(3, 3, np.array([1, 0, 1], dtype=float))

    barycentric_coordinates = gjk.BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.line_segment_03_of_tetrahedron_optimal()

    solution, backup = gjk.distance_subalgorithm_with_backup_procedure(
        simplex, gjk.Solution(), False)
    assert not backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:2], [0.5, 0.5])

    simplex = gjk.SimplexInfo()
    simplex.set_first_point(0, 0, np.array([-1, 0, 1], dtype=float))
    simplex.add_new_point(1, 1, np.array([1, 0, 1], dtype=float))
    simplex.add_new_point(2, 2, np.array([1, 0, 2], dtype=float))
    simplex.add_new_point(3, 3, np.array([0, 0, 2], dtype=float))

    barycentric_coordinates = gjk.BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.line_segment_12_of_tetrahedron_optimal()

    solution, backup = gjk.distance_subalgorithm_with_backup_procedure(
        simplex, gjk.Solution(), False)
    assert not backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:2], [0.5, 0.5])

    simplex = gjk.SimplexInfo()
    simplex.set_first_point(0, 0, np.array([-1, 0, 1], dtype=float))
    simplex.add_new_point(1, 1, np.array([1, 0, 2], dtype=float))
    simplex.add_new_point(2, 2, np.array([1, 0, 1], dtype=float))
    simplex.add_new_point(3, 3, np.array([0, 0, 2], dtype=float))

    barycentric_coordinates = gjk.BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.line_segment_13_of_tetrahedron_optimal()

    solution, backup = gjk.distance_subalgorithm_with_backup_procedure(
        simplex, gjk.Solution(), False)
    assert not backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:2], [0.5, 0.5])

    simplex = gjk.SimplexInfo()
    simplex.set_first_point(0, 0, np.array([1, 0, 2], dtype=float))
    simplex.add_new_point(1, 1, np.array([-1, 0, 1], dtype=float))
    simplex.add_new_point(2, 2, np.array([1, 0, 1], dtype=float))
    simplex.add_new_point(3, 3, np.array([0, 0, 2], dtype=float))

    barycentric_coordinates = gjk.BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.line_segment_23_of_tetrahedron_optimal()

    solution, backup = gjk.distance_subalgorithm_with_backup_procedure(
        simplex, gjk.Solution(), False)
    assert not backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:2], [0.5, 0.5])

    simplex = gjk.SimplexInfo()
    simplex.set_first_point(0, 0, np.array([1, -1, 1], dtype=float))
    simplex.add_new_point(1, 1, np.array([-1, -1, 1], dtype=float))
    simplex.add_new_point(2, 2, np.array([1, 0, 2], dtype=float))
    simplex.add_new_point(3, 3, np.array([0, 2, 1], dtype=float))

    barycentric_coordinates = gjk.BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.face_012_of_tetrahedron_optimal()

    solution, backup = gjk.distance_subalgorithm_with_backup_procedure(
        simplex, gjk.Solution(), False)
    assert not backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(
        solution.barycentric_coordinates[:3], [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

    simplex = gjk.SimplexInfo()
    simplex.set_first_point(0, 0, np.array([1, -1, 1], dtype=float))
    simplex.add_new_point(1, 1, np.array([1, 0, 2], dtype=float))
    simplex.add_new_point(2, 2, np.array([-1, -1, 1], dtype=float))
    simplex.add_new_point(3, 3, np.array([0, 2, 1], dtype=float))

    barycentric_coordinates = gjk.BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.face_013_of_tetrahedron_optimal()

    solution, backup = gjk.distance_subalgorithm_with_backup_procedure(
        simplex, gjk.Solution(), False)
    assert not backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(
        solution.barycentric_coordinates[:3], [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

    simplex = gjk.SimplexInfo()
    simplex.set_first_point(0, 0, np.array([1, 0, 2], dtype=float))
    simplex.add_new_point(1, 1, np.array([1, -1, 1], dtype=float))
    simplex.add_new_point(2, 2, np.array([-1, -1, 1], dtype=float))
    simplex.add_new_point(3, 3, np.array([0, 2, 1], dtype=float))

    barycentric_coordinates = gjk.BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.face_023_of_tetrahedron_optimal()

    solution, backup = gjk.distance_subalgorithm_with_backup_procedure(
        simplex, gjk.Solution(), False)
    assert not backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(
        solution.barycentric_coordinates[:3], [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

    simplex = gjk.SimplexInfo()
    simplex.set_first_point(0, 0, np.array([1, -1, 1], dtype=float))
    simplex.add_new_point(1, 1, np.array([-1, -1, 1], dtype=float))
    simplex.add_new_point(2, 2, np.array([0, 2, 1], dtype=float))
    simplex.add_new_point(3, 3, np.array([1, 0, 2], dtype=float))

    barycentric_coordinates = gjk.BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.face_123_of_tetrahedron_optimal()

    solution, backup = gjk.distance_subalgorithm_with_backup_procedure(
        simplex, gjk.Solution(), False)
    assert not backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(
        solution.barycentric_coordinates[:3], [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

    simplex = gjk.SimplexInfo()
    simplex.set_first_point(0, 0, np.array([1, -1, 1], dtype=float))
    simplex.add_new_point(1, 1, np.array([-1, -1, 1], dtype=float))
    simplex.add_new_point(2, 2, np.array([0, 2, 1], dtype=float))
    simplex.add_new_point(3, 3, np.array([0, 0, -3], dtype=float))

    barycentric_coordinates = gjk.BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.convex_hull_of_tetrahedron_optimal()

    solution, backup = gjk.distance_subalgorithm_with_backup_procedure(
        simplex, gjk.Solution(), False)
    assert not backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 0])
    assert approx(solution.distance_squared) == 0
    assert_array_almost_equal(
        solution.barycentric_coordinates, np.ones(4) / 4.0)


def test_tetrahedron_optimal_point_backup():
    simplex = gjk.SimplexInfo()
    simplex.set_first_point(0, 0, np.array([0, 0, 4], dtype=float))
    simplex.add_new_point(1, 1, np.array([0, 0, 3], dtype=float))
    simplex.add_new_point(2, 2, np.array([0, 0, 2], dtype=float))
    simplex.add_new_point(3, 3, np.array([0, 0, 1], dtype=float))

    barycentric_coordinates = gjk.BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.vertex_0_of_tetrahedron_optimal()

    solution, backup = gjk.distance_subalgorithm_with_backup_procedure(
        simplex, gjk.Solution(), True)
    assert backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:1], [1])

    simplex = gjk.SimplexInfo()
    simplex.set_first_point(0, 0, np.array([0, 0, 1], dtype=float))
    simplex.add_new_point(1, 1, np.array([0, 0, 4], dtype=float))
    simplex.add_new_point(2, 2, np.array([0, 0, 2], dtype=float))
    simplex.add_new_point(3, 3, np.array([0, 0, 3], dtype=float))

    barycentric_coordinates = gjk.BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.vertex_1_of_tetrahedron_optimal()

    solution, backup = gjk.distance_subalgorithm_with_backup_procedure(
        simplex, gjk.Solution(), True)
    assert backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:1], [1])

    simplex = gjk.SimplexInfo()
    simplex.set_first_point(0, 0, np.array([0, 0, 4], dtype=float))
    simplex.add_new_point(1, 1, np.array([0, 0, 1], dtype=float))
    simplex.add_new_point(2, 2, np.array([0, 0, 2], dtype=float))
    simplex.add_new_point(3, 3, np.array([0, 0, 3], dtype=float))

    barycentric_coordinates = gjk.BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.vertex_2_of_tetrahedron_optimal()

    solution, backup = gjk.distance_subalgorithm_with_backup_procedure(
        simplex, gjk.Solution(), True)
    assert backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:1], [1])

    simplex = gjk.SimplexInfo()
    simplex.set_first_point(0, 0, np.array([0, 0, 2], dtype=float))
    simplex.add_new_point(1, 1, np.array([0, 0, 4], dtype=float))
    simplex.add_new_point(2, 2, np.array([0, 0, 1], dtype=float))
    simplex.add_new_point(3, 3, np.array([0, 0, 3], dtype=float))

    barycentric_coordinates = gjk.BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.vertex_3_of_tetrahedron_optimal()

    solution, backup = gjk.distance_subalgorithm_with_backup_procedure(
        simplex, gjk.Solution(), True)
    assert backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:1], [1])

    simplex = gjk.SimplexInfo()
    simplex.set_first_point(0, 0, np.array([-1, 0, 1], dtype=float))
    simplex.add_new_point(1, 1, np.array([1, 0, 2], dtype=float))
    simplex.add_new_point(2, 2, np.array([0, 0, 2], dtype=float))
    simplex.add_new_point(3, 3, np.array([1, 0, 1], dtype=float))

    barycentric_coordinates = gjk.BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.line_segment_01_of_tetrahedron_optimal()

    solution, backup = gjk.distance_subalgorithm_with_backup_procedure(
        simplex, gjk.Solution(), True)
    assert backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:2], [0.5, 0.5])

    simplex = gjk.SimplexInfo()
    simplex.set_first_point(0, 0, np.array([1, 0, 2], dtype=float))
    simplex.add_new_point(1, 1, np.array([-1, 0, 1], dtype=float))
    simplex.add_new_point(2, 2, np.array([0, 0, 2], dtype=float))
    simplex.add_new_point(3, 3, np.array([1, 0, 1], dtype=float))

    barycentric_coordinates = gjk.BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.line_segment_02_of_tetrahedron_optimal()

    solution, backup = gjk.distance_subalgorithm_with_backup_procedure(
        simplex, gjk.Solution(), True)
    assert backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:2], [0.5, 0.5])

    simplex = gjk.SimplexInfo()
    simplex.set_first_point(0, 0, np.array([1, 0, 2], dtype=float))
    simplex.add_new_point(1, 1, np.array([0, 0, 2], dtype=float))
    simplex.add_new_point(2, 2, np.array([-1, 0, 1], dtype=float))
    simplex.add_new_point(3, 3, np.array([1, 0, 1], dtype=float))

    barycentric_coordinates = gjk.BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.line_segment_03_of_tetrahedron_optimal()

    solution, backup = gjk.distance_subalgorithm_with_backup_procedure(
        simplex, gjk.Solution(), True)
    assert backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:2], [0.5, 0.5])

    simplex = gjk.SimplexInfo()
    simplex.set_first_point(0, 0, np.array([-1, 0, 1], dtype=float))
    simplex.add_new_point(1, 1, np.array([1, 0, 1], dtype=float))
    simplex.add_new_point(2, 2, np.array([1, 0, 2], dtype=float))
    simplex.add_new_point(3, 3, np.array([0, 0, 2], dtype=float))

    barycentric_coordinates = gjk.BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.line_segment_12_of_tetrahedron_optimal()

    solution, backup = gjk.distance_subalgorithm_with_backup_procedure(
        simplex, gjk.Solution(), True)
    assert backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:2], [0.5, 0.5])

    simplex = gjk.SimplexInfo()
    simplex.set_first_point(0, 0, np.array([-1, 0, 1], dtype=float))
    simplex.add_new_point(1, 1, np.array([1, 0, 2], dtype=float))
    simplex.add_new_point(2, 2, np.array([1, 0, 1], dtype=float))
    simplex.add_new_point(3, 3, np.array([0, 0, 2], dtype=float))

    barycentric_coordinates = gjk.BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.line_segment_13_of_tetrahedron_optimal()

    solution, backup = gjk.distance_subalgorithm_with_backup_procedure(
        simplex, gjk.Solution(), True)
    assert backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:2], [0.5, 0.5])

    simplex = gjk.SimplexInfo()
    simplex.set_first_point(0, 0, np.array([1, 0, 2], dtype=float))
    simplex.add_new_point(1, 1, np.array([-1, 0, 1], dtype=float))
    simplex.add_new_point(2, 2, np.array([1, 0, 1], dtype=float))
    simplex.add_new_point(3, 3, np.array([0, 0, 2], dtype=float))

    barycentric_coordinates = gjk.BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.line_segment_23_of_tetrahedron_optimal()

    solution, backup = gjk.distance_subalgorithm_with_backup_procedure(
        simplex, gjk.Solution(), True)
    assert backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:2], [0.5, 0.5])

    simplex = gjk.SimplexInfo()
    simplex.set_first_point(0, 0, np.array([1, -1, 1], dtype=float))
    simplex.add_new_point(1, 1, np.array([-1, -1, 1], dtype=float))
    simplex.add_new_point(2, 2, np.array([1, 0, 2], dtype=float))
    simplex.add_new_point(3, 3, np.array([0, 2, 1], dtype=float))

    barycentric_coordinates = gjk.BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.face_012_of_tetrahedron_optimal()

    solution, backup = gjk.distance_subalgorithm_with_backup_procedure(
        simplex, gjk.Solution(), True)
    assert backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(
        solution.barycentric_coordinates[:3], [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

    simplex = gjk.SimplexInfo()
    simplex.set_first_point(0, 0, np.array([1, -1, 1], dtype=float))
    simplex.add_new_point(1, 1, np.array([1, 0, 2], dtype=float))
    simplex.add_new_point(2, 2, np.array([-1, -1, 1], dtype=float))
    simplex.add_new_point(3, 3, np.array([0, 2, 1], dtype=float))

    barycentric_coordinates = gjk.BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.face_013_of_tetrahedron_optimal()

    solution, backup = gjk.distance_subalgorithm_with_backup_procedure(
        simplex, gjk.Solution(), True)
    assert backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(
        solution.barycentric_coordinates[:3], [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

    simplex = gjk.SimplexInfo()
    simplex.set_first_point(0, 0, np.array([1, 0, 2], dtype=float))
    simplex.add_new_point(1, 1, np.array([1, -1, 1], dtype=float))
    simplex.add_new_point(2, 2, np.array([-1, -1, 1], dtype=float))
    simplex.add_new_point(3, 3, np.array([0, 2, 1], dtype=float))

    barycentric_coordinates = gjk.BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.face_023_of_tetrahedron_optimal()

    solution, backup = gjk.distance_subalgorithm_with_backup_procedure(
        simplex, gjk.Solution(), True)
    assert backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(
        solution.barycentric_coordinates[:3], [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

    simplex = gjk.SimplexInfo()
    simplex.set_first_point(0, 0, np.array([1, -1, 1], dtype=float))
    simplex.add_new_point(1, 1, np.array([-1, -1, 1], dtype=float))
    simplex.add_new_point(2, 2, np.array([0, 2, 1], dtype=float))
    simplex.add_new_point(3, 3, np.array([1, 0, 2], dtype=float))

    barycentric_coordinates = gjk.BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.face_123_of_tetrahedron_optimal()

    solution, backup = gjk.distance_subalgorithm_with_backup_procedure(
        simplex, gjk.Solution(), True)
    assert backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(
        solution.barycentric_coordinates[:3], [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

    simplex = gjk.SimplexInfo()
    simplex.set_first_point(0, 0, np.array([1, -1, 1], dtype=float))
    simplex.add_new_point(1, 1, np.array([-1, -1, 1], dtype=float))
    simplex.add_new_point(2, 2, np.array([0, 2, 1], dtype=float))
    simplex.add_new_point(3, 3, np.array([0, 0, -3], dtype=float))

    barycentric_coordinates = gjk.BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.convex_hull_of_tetrahedron_optimal()

    solution, backup = gjk.distance_subalgorithm_with_backup_procedure(
        simplex, gjk.Solution(), True)
    assert backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 0])
    assert approx(solution.distance_squared) == 0
    assert_array_almost_equal(
        solution.barycentric_coordinates, np.ones(4) / 4.0)


def test_gjk_boxes():
    box2origin = np.eye(4)
    size = np.ones(3)
    box_collider = colliders.Box(box2origin, size)

    # complete overlap
    dist, closest_point1, closest_point2, _ = gjk.gjk(
        box_collider, box_collider)
    assert approx(dist) == 0.0
    assert_array_almost_equal(closest_point1, np.array([-0.5, -0.5, -0.5]))
    assert_array_almost_equal(closest_point1, closest_point2)

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

                            dist, closest_point1, closest_point2, _ = gjk.gjk(
                                box_collider, box_collider2)
                            assert approx(dist) == 0.0
                            expected = -0.5 * np.ones(3)
                            expected[dim1] = 0.5 * sign1
                            expected[dim2] = 0.5 * sign2
                            expected[dim3] = 0.5 * sign3
                            assert_array_almost_equal(closest_point1, expected)
                            assert_array_almost_equal(
                                closest_point1, closest_point2)

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

    dist, closest_point1, closest_point2, _ = gjk.gjk(
        box_collider, box_collider2)

    assert approx(dist) == 1.7900192730149391


def test_gjk_spheres():
    sphere1 = colliders.Sphere(center=np.array([0, 0, 0], dtype=float), radius=1.0)
    dist, closest_point1, closest_point2, _ = gjk.gjk(
        sphere1, sphere1)
    assert approx(dist) == 0.0
    assert_array_almost_equal(closest_point1, np.array([0, 0, 1]))
    assert_array_almost_equal(closest_point1, closest_point2)

    sphere2 = colliders.Sphere(center=np.array([1, 1, 1], dtype=float), radius=1.0)
    dist, closest_point1, closest_point2, _ = gjk.gjk(
        sphere1, sphere2)
    assert approx(dist) == 0.0
    assert_array_almost_equal(closest_point1, np.array([0.5, 0.5, 0.633975]))
    assert_array_almost_equal(closest_point1, closest_point2)

    sphere1 = colliders.Sphere(center=np.array([0, 0, 0], dtype=float), radius=1.0)
    sphere2 = colliders.Sphere(center=np.array([0, 0, 3], dtype=float), radius=1.0)
    dist, closest_point1, closest_point2, _ = gjk.gjk(
        sphere1, sphere2)
    assert approx(dist) == 1
    assert_array_almost_equal(closest_point1, np.array([0, 0, 1]))
    assert_array_almost_equal(closest_point2, np.array([0, 0, 2]))


def test_gjk_cylinders():
    cylinder1 = colliders.Cylinder(np.eye(4), 1, 1)
    dist, closest_point1, closest_point2, _ = gjk.gjk(
        cylinder1, cylinder1)
    assert approx(dist) == 0
    assert_array_almost_equal(closest_point1, np.array([1, 0, 0.5]))
    assert_array_almost_equal(closest_point2, np.array([1, 0, 0.5]))

    A2B = np.eye(4)
    A2B[:3, 3] = np.array([3, 0, 0])
    cylinder2 = colliders.Cylinder(A2B, 1, 1)
    dist, closest_point1, closest_point2, _ = gjk.gjk(
        cylinder1, cylinder2)
    assert approx(dist) == 1
    assert_array_almost_equal(closest_point1, np.array([1, 0, 0.5]))
    assert_array_almost_equal(closest_point2, np.array([2, 0, 0.5]))

    A2B = np.eye(4)
    A2B[:3, 3] = np.array([0, 0, 4])
    cylinder2 = colliders.Cylinder(A2B, 1, 1)
    dist, closest_point1, closest_point2, _ = gjk.gjk(
        cylinder1, cylinder2)
    assert approx(dist) == 3
    assert_array_almost_equal(closest_point1, np.array([1, 0, 0.5]))
    assert_array_almost_equal(closest_point2, np.array([1, 0, 3.5]))


def test_gjk_capsules():
    capsule1 = colliders.Capsule(np.eye(4), 1, 1)
    dist, closest_point1, closest_point2, _ = gjk.gjk(
        capsule1, capsule1)
    assert approx(dist) == 0
    assert_array_almost_equal(closest_point1, closest_point2)

    A2B = np.eye(4)
    A2B[:3, 3] = np.array([3, 0, 0])
    capsule2 = colliders.Capsule(A2B, 1, 1)
    dist, closest_point1, closest_point2, _ = gjk.gjk(
        capsule1, capsule2)
    assert approx(dist) == 1
    assert_array_almost_equal(closest_point1, np.array([1, 0, -0.5]))
    assert_array_almost_equal(closest_point2, np.array([2, 0, -0.5]))

    A2B = np.eye(4)
    A2B[:3, 3] = np.array([0, 0, 4])
    capsule2 = colliders.Capsule(A2B, 1, 1)
    dist, closest_point1, closest_point2, _ = gjk.gjk(
        capsule1, capsule2)
    assert approx(dist) == 1
    assert_array_almost_equal(closest_point1, np.array([0, 0, 1.5]))
    assert_array_almost_equal(closest_point2, np.array([0, 0, 2.5]))


def test_gjk_points():
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
        dist, closest_point1, closest_point2, _ = gjk.gjk(
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
        dist, closest_point_triangle, closest_point_triangle2, _ = gjk.gjk(
            colliders.ConvexHullVertices(triangle_points), colliders.ConvexHullVertices(triangle_points2))
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
        dist, closest_point_triangle, closest_point_rectangle, _ = gjk.gjk(
            colliders.ConvexHullVertices(triangle_points), colliders.ConvexHullVertices(rectangle_points))
        dist2, closest_point_triangle2, closest_point_rectangle2 = distance.triangle_to_rectangle(
            triangle_points, rectangle_center, rectangle_axes, rectangle_lengths)
        assert approx(dist) == dist2
        assert_array_almost_equal(
            closest_point_triangle, closest_point_triangle2)
        assert_array_almost_equal(
            closest_point_rectangle, closest_point_rectangle2)


def test_gjk_ellipsoids():
    random_state = np.random.RandomState(83)
    for _ in range(10):
        ellipsoid2origin1, radii1 = random.rand_ellipsoid(
            random_state, center_scale=2.0)
        ellipsoid2origin2, radii2 = random.rand_ellipsoid(
            random_state, center_scale=2.0)
        dist, closest_point1, closest_point2, _ = gjk.gjk(
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


def test_gjk_floating_point_accuracy_of_barycentric_coordinates_of_face():
    vertices1 = np.array([
        [0.3302341224102143, 0.19050359211588797, 1.3411105874214977],
        [1.8078177759995258, 0.20620691174863182, 1.0702191106030559],
        [0.7979462705135667, 1.2183177500830087, 0.8211438441965184],
        [1.4524872148245276, 0.7965455117961597, 1.0991113134742683],
        [1.8445208896122431, 1.2876443386590763, 1.5700034764677797],
        [0.7550697105384843, 0.18477518020505856, 1.3471929737602069]])
    vertices2 = np.array([
        [0.3527386248034806, 0.035237715006153913, 0.8756651090484097],
        [0.8866579336737807, 0.2548857692642903, 0.184959080850696],
        [0.2900323943660029, 0.2810156342173158, 0.12167462536248075],
        [0.46786979185146715, 0.1591897910580864, 0.47912132312107114],
        [0.9895873763930331, 0.5549956997906568, 0.6427151977095377],
        [0.07800999236334982, 0.8380559395155408, 0.24411711979617723]])
    dist = gjk.gjk(colliders.ConvexHullVertices(vertices1), colliders.ConvexHullVertices(vertices2))[0]
    assert dist > 0.0


def test_gjk_intersection_vs_gjk():
    random_state = np.random.RandomState(24)

    for _ in range(50):
        vertices1 = random_state.rand(6, 3) * np.array([[2, 2, 2]])
        convex1 = colliders.ConvexHullVertices(vertices1)

        vertices2 = random_state.rand(6, 3) * np.array([[1, 1, 1]])
        convex2 = colliders.ConvexHullVertices(vertices2)

        dist, closest_point1, closest_point2, _ = gjk.gjk(convex1, convex2)
        gjk_intersection = dist < 1e-16
        gjk_intersection2 = gjk.gjk_intersection(convex1, convex2)
        assert gjk_intersection == gjk_intersection2, f"{vertices1=}\n{vertices2=}"
