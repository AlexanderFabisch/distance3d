import numpy as np
from distance3d.gjk._gjk_original import (
    SimplexInfo, Solution, BarycentricCoordinates,
    distance_subalgorithm_with_backup_procedure)
from numpy.testing import assert_array_almost_equal
from pytest import approx


def assert_dot_product_table(simplex):
    for i in range(simplex.n_simplex_points):
        for j in range(i + 1):
            assert approx(simplex.dot_product_table[i, j]) == np.dot(
                simplex.points[i], simplex.points[j])


def test_select_line_segment():
    simplex = SimplexInfo()
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

    simplex_backup = SimplexInfo()
    simplex_backup.copy_from(simplex)

    for i in range(4):
        for j in range(1, 4):
            if i == j:
                continue
            simplex.copy_from(simplex_backup)
            simplex.select_line_segment(i, j)
            assert_dot_product_table(simplex)


def test_select_face():
    simplex = SimplexInfo()
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

    simplex_backup = SimplexInfo()
    simplex_backup.copy_from(simplex)

    for i in range(4):
        for j in range(1, 4):
            for k in range(2, 4):
                if i == j or i == k or j == k:
                    continue
                simplex.copy_from(simplex_backup)
                simplex.select_face(i, j, k)
                assert_dot_product_table(simplex)


def test_simplex_reorder():
    simplex = SimplexInfo()
    simplex.n_simplex_points = 4
    random_state = np.random.RandomState(24)
    simplex.points[:, :] = random_state.randn(*simplex.points.shape)
    simplex.dot_product_table = simplex.points.dot(simplex.points.T)
    simplex.indices_polytope1 = np.arange(4, dtype=int)
    simplex.indices_polytope2 = np.arange(4, dtype=int)
    for i in range(1, 4):
        for j in range(i + 1, 4):
            simplex.dot_product_table[i, j] = float("nan")

    simplex_backup = SimplexInfo()
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
    simplex = SimplexInfo()
    simplex.set_first_point(0, 0, np.array([0, 0, 1], dtype=float))

    solution, backup = distance_subalgorithm_with_backup_procedure(
        simplex, Solution(), False)
    assert not backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:1], [1])


def test_vertex_optimal_point_backup():
    simplex = SimplexInfo()
    simplex.set_first_point(0, 0, np.array([0, 0, 1], dtype=float))

    solution, backup = distance_subalgorithm_with_backup_procedure(
        simplex, Solution(), True)
    assert backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:1], [1])


def test_line_segment_optimal_point():
    simplex = SimplexInfo()
    simplex.set_first_point(0, 0, np.array([0, 0, 2], dtype=float))
    simplex.add_new_point(1, 1, np.array([0, 0, 1], dtype=float))

    barycentric_coordinates = BarycentricCoordinates()
    barycentric_coordinates.line_segment_coordinates_0(simplex)
    barycentric_coordinates.line_segment_coordinates_1(simplex)
    assert barycentric_coordinates.vertex_0_of_line_segment_optimal()

    solution, backup = distance_subalgorithm_with_backup_procedure(
        simplex, Solution(), False)
    assert not backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:1], [1])

    simplex = SimplexInfo()
    simplex.set_first_point(0, 0, np.array([0, 0, 1], dtype=float))
    simplex.add_new_point(1, 1, np.array([0, 0, 2], dtype=float))

    barycentric_coordinates = BarycentricCoordinates()
    barycentric_coordinates.line_segment_coordinates_0(simplex)
    barycentric_coordinates.line_segment_coordinates_1(simplex)
    assert barycentric_coordinates.vertex_1_of_line_segment_optimal()

    solution, backup = distance_subalgorithm_with_backup_procedure(
        simplex, Solution(), False)
    assert not backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:1], [1])

    simplex = SimplexInfo()
    simplex.set_first_point(0, 0, np.array([-1, 0, 1], dtype=float))
    simplex.add_new_point(1, 1, np.array([1, 0, 1], dtype=float))

    barycentric_coordinates = BarycentricCoordinates()
    barycentric_coordinates.line_segment_coordinates_0(simplex)
    barycentric_coordinates.line_segment_coordinates_1(simplex)
    assert barycentric_coordinates.line_segment_01_of_line_segment_optimal()

    solution, backup = distance_subalgorithm_with_backup_procedure(
        simplex, Solution(), False)
    assert not backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:2], [0.5, 0.5])


def test_line_segment_optimal_point_backup():
    simplex = SimplexInfo()
    simplex.set_first_point(0, 0, np.array([0, 0, 2], dtype=float))
    simplex.add_new_point(1, 1, np.array([0, 0, 1], dtype=float))

    barycentric_coordinates = BarycentricCoordinates()
    barycentric_coordinates.line_segment_coordinates_0(simplex)
    barycentric_coordinates.line_segment_coordinates_1(simplex)
    assert barycentric_coordinates.vertex_0_of_line_segment_optimal()

    solution, backup = distance_subalgorithm_with_backup_procedure(
        simplex, Solution(), True)
    assert backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:1], [1])

    simplex = SimplexInfo()
    simplex.set_first_point(0, 0, np.array([0, 0, 1], dtype=float))
    simplex.add_new_point(1, 1, np.array([0, 0, 2], dtype=float))

    barycentric_coordinates = BarycentricCoordinates()
    barycentric_coordinates.line_segment_coordinates_0(simplex)
    barycentric_coordinates.line_segment_coordinates_1(simplex)
    assert barycentric_coordinates.vertex_1_of_line_segment_optimal()

    solution, backup = distance_subalgorithm_with_backup_procedure(
        simplex, Solution(), True)
    assert backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:1], [1])

    simplex = SimplexInfo()
    simplex.set_first_point(0, 0, np.array([-1, 0, 1], dtype=float))
    simplex.add_new_point(1, 1, np.array([1, 0, 1], dtype=float))

    barycentric_coordinates = BarycentricCoordinates()
    barycentric_coordinates.line_segment_coordinates_0(simplex)
    barycentric_coordinates.line_segment_coordinates_1(simplex)
    assert barycentric_coordinates.line_segment_01_of_line_segment_optimal()

    solution, backup = distance_subalgorithm_with_backup_procedure(
        simplex, Solution(), True)
    assert backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:2], [0.5, 0.5])


def test_face_optimal_point():
    simplex = SimplexInfo()
    simplex.set_first_point(0, 0, np.array([1, 0, 2], dtype=float))
    simplex.add_new_point(1, 1, np.array([0, 0, 2], dtype=float))
    simplex.add_new_point(2, 2, np.array([0, 0, 1], dtype=float))

    barycentric_coordinates = BarycentricCoordinates()
    barycentric_coordinates.face_coordinates_0(simplex)
    barycentric_coordinates.face_coordinates_1(simplex)
    e123 = barycentric_coordinates.face_coordinates_2(simplex)
    barycentric_coordinates.face_coordinates_3(simplex, e123)
    assert_array_almost_equal(simplex.points[0], [0, 0, 1])
    assert barycentric_coordinates.vertex_0_of_face_optimal()

    solution, backup = distance_subalgorithm_with_backup_procedure(
        simplex, Solution(), False)
    assert not backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:1], [1])

    simplex = SimplexInfo()
    simplex.set_first_point(0, 0, np.array([0, 0, 1], dtype=float))
    simplex.add_new_point(1, 1, np.array([0, 0, 2], dtype=float))
    simplex.add_new_point(2, 2, np.array([1, 0, 2], dtype=float))

    barycentric_coordinates = BarycentricCoordinates()
    barycentric_coordinates.face_coordinates_0(simplex)
    barycentric_coordinates.face_coordinates_1(simplex)
    e123 = barycentric_coordinates.face_coordinates_2(simplex)
    barycentric_coordinates.face_coordinates_3(simplex, e123)
    assert_array_almost_equal(simplex.points[1], [0, 0, 1])
    assert barycentric_coordinates.vertex_1_of_face_optimal()

    solution, backup = distance_subalgorithm_with_backup_procedure(
        simplex, Solution(), False)
    assert not backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:1], [1])

    simplex = SimplexInfo()
    simplex.set_first_point(0, 0, np.array([0, 0, 2], dtype=float))
    simplex.add_new_point(1, 1, np.array([0, 0, 1], dtype=float))
    simplex.add_new_point(2, 2, np.array([1, 0, 2], dtype=float))

    barycentric_coordinates = BarycentricCoordinates()
    barycentric_coordinates.face_coordinates_0(simplex)
    barycentric_coordinates.face_coordinates_1(simplex)
    e123 = barycentric_coordinates.face_coordinates_2(simplex)
    barycentric_coordinates.face_coordinates_3(simplex, e123)
    assert_array_almost_equal(simplex.points[2], [0, 0, 1])
    assert barycentric_coordinates.vertex_2_of_face_optimal()

    solution, backup = distance_subalgorithm_with_backup_procedure(
        simplex, Solution(), False)
    assert not backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:1], [1])

    simplex = SimplexInfo()
    simplex.set_first_point(0, 0, np.array([-1, 0, 1], dtype=float))
    simplex.add_new_point(1, 1, np.array([0, 0, 2], dtype=float))
    simplex.add_new_point(2, 2, np.array([1, 0, 1], dtype=float))

    barycentric_coordinates = BarycentricCoordinates()
    barycentric_coordinates.face_coordinates_0(simplex)
    barycentric_coordinates.face_coordinates_1(simplex)
    e123 = barycentric_coordinates.face_coordinates_2(simplex)
    barycentric_coordinates.face_coordinates_3(simplex, e123)
    assert barycentric_coordinates.line_segment_01_of_face_optimal()

    solution, backup = distance_subalgorithm_with_backup_procedure(
        simplex, Solution(), False)
    assert not backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:2], [0.5, 0.5])

    simplex = SimplexInfo()
    simplex.set_first_point(0, 0, np.array([1, 0, 1], dtype=float))
    simplex.add_new_point(1, 1, np.array([-1, 0, 1], dtype=float))
    simplex.add_new_point(2, 2, np.array([0, 0, 2], dtype=float))

    barycentric_coordinates = BarycentricCoordinates()
    barycentric_coordinates.face_coordinates_0(simplex)
    barycentric_coordinates.face_coordinates_1(simplex)
    e123 = barycentric_coordinates.face_coordinates_2(simplex)
    barycentric_coordinates.face_coordinates_3(simplex, e123)
    assert barycentric_coordinates.line_segment_12_of_face_optimal()

    solution, backup = distance_subalgorithm_with_backup_procedure(
        simplex, Solution(), False)
    assert not backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:2], [0.5, 0.5])

    simplex = SimplexInfo()
    simplex.set_first_point(0, 0, np.array([0, 0, 2], dtype=float))
    simplex.add_new_point(1, 1, np.array([1, 0, 1], dtype=float))
    simplex.add_new_point(2, 2, np.array([-1, 0, 1], dtype=float))

    barycentric_coordinates = BarycentricCoordinates()
    barycentric_coordinates.face_coordinates_0(simplex)
    barycentric_coordinates.face_coordinates_1(simplex)
    e123 = barycentric_coordinates.face_coordinates_2(simplex)
    barycentric_coordinates.face_coordinates_3(simplex, e123)
    assert barycentric_coordinates.line_segment_02_of_face_optimal()

    solution, backup = distance_subalgorithm_with_backup_procedure(
        simplex, Solution(), False)
    assert not backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:2], [0.5, 0.5])

    simplex = SimplexInfo()
    simplex.set_first_point(0, 0, np.array([0, 1, 1], dtype=float))
    simplex.add_new_point(1, 1, np.array([1, -1, 1], dtype=float))
    simplex.add_new_point(2, 2, np.array([-1, -1, 1], dtype=float))

    barycentric_coordinates = BarycentricCoordinates()
    barycentric_coordinates.face_coordinates_0(simplex)
    barycentric_coordinates.face_coordinates_1(simplex)
    e123 = barycentric_coordinates.face_coordinates_2(simplex)
    barycentric_coordinates.face_coordinates_3(simplex, e123)
    assert barycentric_coordinates.face_012_of_face_optimal()

    solution, backup = distance_subalgorithm_with_backup_procedure(
        simplex, Solution(), False)
    assert not backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(
        solution.barycentric_coordinates[:3], [0.25, 0.5, 0.25])


def test_face_optimal_point_backup():
    simplex = SimplexInfo()
    simplex.set_first_point(0, 0, np.array([1, 0, 2], dtype=float))
    simplex.add_new_point(1, 1, np.array([0, 0, 2], dtype=float))
    simplex.add_new_point(2, 2, np.array([0, 0, 1], dtype=float))

    barycentric_coordinates = BarycentricCoordinates()
    barycentric_coordinates.face_coordinates_0(simplex)
    barycentric_coordinates.face_coordinates_1(simplex)
    e123 = barycentric_coordinates.face_coordinates_2(simplex)
    barycentric_coordinates.face_coordinates_3(simplex, e123)
    assert_array_almost_equal(simplex.points[0], [0, 0, 1])
    assert barycentric_coordinates.vertex_0_of_face_optimal()

    solution, backup = distance_subalgorithm_with_backup_procedure(
        simplex, Solution(), True)
    assert backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:1], [1])

    simplex = SimplexInfo()
    simplex.set_first_point(0, 0, np.array([0, 0, 1], dtype=float))
    simplex.add_new_point(1, 1, np.array([0, 0, 2], dtype=float))
    simplex.add_new_point(2, 2, np.array([1, 0, 2], dtype=float))

    barycentric_coordinates = BarycentricCoordinates()
    barycentric_coordinates.face_coordinates_0(simplex)
    barycentric_coordinates.face_coordinates_1(simplex)
    e123 = barycentric_coordinates.face_coordinates_2(simplex)
    barycentric_coordinates.face_coordinates_3(simplex, e123)
    assert_array_almost_equal(simplex.points[1], [0, 0, 1])
    assert barycentric_coordinates.vertex_1_of_face_optimal()

    solution, backup = distance_subalgorithm_with_backup_procedure(
        simplex, Solution(), True)
    assert backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:1], [1])

    simplex = SimplexInfo()
    simplex.set_first_point(0, 0, np.array([0, 0, 2], dtype=float))
    simplex.add_new_point(1, 1, np.array([0, 0, 1], dtype=float))
    simplex.add_new_point(2, 2, np.array([1, 0, 2], dtype=float))

    barycentric_coordinates = BarycentricCoordinates()
    barycentric_coordinates.face_coordinates_0(simplex)
    barycentric_coordinates.face_coordinates_1(simplex)
    e123 = barycentric_coordinates.face_coordinates_2(simplex)
    barycentric_coordinates.face_coordinates_3(simplex, e123)
    assert_array_almost_equal(simplex.points[2], [0, 0, 1])
    assert barycentric_coordinates.vertex_2_of_face_optimal()

    solution, backup = distance_subalgorithm_with_backup_procedure(
        simplex, Solution(), True)
    assert backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:1], [1])

    simplex = SimplexInfo()
    simplex.set_first_point(0, 0, np.array([-1, 0, 1], dtype=float))
    simplex.add_new_point(1, 1, np.array([0, 0, 2], dtype=float))
    simplex.add_new_point(2, 2, np.array([1, 0, 1], dtype=float))

    barycentric_coordinates = BarycentricCoordinates()
    barycentric_coordinates.face_coordinates_0(simplex)
    barycentric_coordinates.face_coordinates_1(simplex)
    e123 = barycentric_coordinates.face_coordinates_2(simplex)
    barycentric_coordinates.face_coordinates_3(simplex, e123)
    assert barycentric_coordinates.line_segment_01_of_face_optimal()

    solution, backup = distance_subalgorithm_with_backup_procedure(
        simplex, Solution(), True)
    assert backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:2], [0.5, 0.5])

    simplex = SimplexInfo()
    simplex.set_first_point(0, 0, np.array([1, 0, 1], dtype=float))
    simplex.add_new_point(1, 1, np.array([-1, 0, 1], dtype=float))
    simplex.add_new_point(2, 2, np.array([0, 0, 2], dtype=float))

    barycentric_coordinates = BarycentricCoordinates()
    barycentric_coordinates.face_coordinates_0(simplex)
    barycentric_coordinates.face_coordinates_1(simplex)
    e123 = barycentric_coordinates.face_coordinates_2(simplex)
    barycentric_coordinates.face_coordinates_3(simplex, e123)
    assert barycentric_coordinates.line_segment_12_of_face_optimal()

    solution, backup = distance_subalgorithm_with_backup_procedure(
        simplex, Solution(), True)
    assert backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:2], [0.5, 0.5])

    simplex = SimplexInfo()
    simplex.set_first_point(0, 0, np.array([0, 0, 2], dtype=float))
    simplex.add_new_point(1, 1, np.array([1, 0, 1], dtype=float))
    simplex.add_new_point(2, 2, np.array([-1, 0, 1], dtype=float))

    barycentric_coordinates = BarycentricCoordinates()
    barycentric_coordinates.face_coordinates_0(simplex)
    barycentric_coordinates.face_coordinates_1(simplex)
    e123 = barycentric_coordinates.face_coordinates_2(simplex)
    barycentric_coordinates.face_coordinates_3(simplex, e123)
    assert barycentric_coordinates.line_segment_02_of_face_optimal()

    solution, backup = distance_subalgorithm_with_backup_procedure(
        simplex, Solution(), True)
    assert backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:2], [0.5, 0.5])

    simplex = SimplexInfo()
    simplex.set_first_point(0, 0, np.array([0, 1, 1], dtype=float))
    simplex.add_new_point(1, 1, np.array([1, -1, 1], dtype=float))
    simplex.add_new_point(2, 2, np.array([-1, -1, 1], dtype=float))

    barycentric_coordinates = BarycentricCoordinates()
    barycentric_coordinates.face_coordinates_0(simplex)
    barycentric_coordinates.face_coordinates_1(simplex)
    e123 = barycentric_coordinates.face_coordinates_2(simplex)
    barycentric_coordinates.face_coordinates_3(simplex, e123)
    assert barycentric_coordinates.face_012_of_face_optimal()

    solution, backup = distance_subalgorithm_with_backup_procedure(
        simplex, Solution(), True)
    assert backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(
        solution.barycentric_coordinates[:3], [0.25, 0.5, 0.25])


def test_tetrahedron_optimal_point():
    simplex = SimplexInfo()
    simplex.set_first_point(0, 0, np.array([0, 0, 4], dtype=float))
    simplex.add_new_point(1, 1, np.array([0, 0, 3], dtype=float))
    simplex.add_new_point(2, 2, np.array([0, 0, 2], dtype=float))
    simplex.add_new_point(3, 3, np.array([0, 0, 1], dtype=float))

    barycentric_coordinates = BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.vertex_0_of_tetrahedron_optimal()

    solution, backup = distance_subalgorithm_with_backup_procedure(
        simplex, Solution(), False)
    assert not backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:1], [1])

    simplex = SimplexInfo()
    simplex.set_first_point(0, 0, np.array([0, 0, 1], dtype=float))
    simplex.add_new_point(1, 1, np.array([0, 0, 4], dtype=float))
    simplex.add_new_point(2, 2, np.array([0, 0, 2], dtype=float))
    simplex.add_new_point(3, 3, np.array([0, 0, 3], dtype=float))

    barycentric_coordinates = BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.vertex_1_of_tetrahedron_optimal()

    solution, backup = distance_subalgorithm_with_backup_procedure(
        simplex, Solution(), False)
    assert not backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:1], [1])

    simplex = SimplexInfo()
    simplex.set_first_point(0, 0, np.array([0, 0, 4], dtype=float))
    simplex.add_new_point(1, 1, np.array([0, 0, 1], dtype=float))
    simplex.add_new_point(2, 2, np.array([0, 0, 2], dtype=float))
    simplex.add_new_point(3, 3, np.array([0, 0, 3], dtype=float))

    barycentric_coordinates = BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.vertex_2_of_tetrahedron_optimal()

    solution, backup = distance_subalgorithm_with_backup_procedure(
        simplex, Solution(), False)
    assert not backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:1], [1])

    simplex = SimplexInfo()
    simplex.set_first_point(0, 0, np.array([0, 0, 2], dtype=float))
    simplex.add_new_point(1, 1, np.array([0, 0, 4], dtype=float))
    simplex.add_new_point(2, 2, np.array([0, 0, 1], dtype=float))
    simplex.add_new_point(3, 3, np.array([0, 0, 3], dtype=float))

    barycentric_coordinates = BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.vertex_3_of_tetrahedron_optimal()

    solution, backup = distance_subalgorithm_with_backup_procedure(
        simplex, Solution(), False)
    assert not backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:1], [1])

    simplex = SimplexInfo()
    simplex.set_first_point(0, 0, np.array([-1, 0, 1], dtype=float))
    simplex.add_new_point(1, 1, np.array([1, 0, 2], dtype=float))
    simplex.add_new_point(2, 2, np.array([0, 0, 2], dtype=float))
    simplex.add_new_point(3, 3, np.array([1, 0, 1], dtype=float))

    barycentric_coordinates = BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.line_segment_01_of_tetrahedron_optimal()

    solution, backup = distance_subalgorithm_with_backup_procedure(
        simplex, Solution(), False)
    assert not backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:2], [0.5, 0.5])

    simplex = SimplexInfo()
    simplex.set_first_point(0, 0, np.array([1, 0, 2], dtype=float))
    simplex.add_new_point(1, 1, np.array([-1, 0, 1], dtype=float))
    simplex.add_new_point(2, 2, np.array([0, 0, 2], dtype=float))
    simplex.add_new_point(3, 3, np.array([1, 0, 1], dtype=float))

    barycentric_coordinates = BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.line_segment_02_of_tetrahedron_optimal()

    solution, backup = distance_subalgorithm_with_backup_procedure(
        simplex, Solution(), False)
    assert not backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:2], [0.5, 0.5])

    simplex = SimplexInfo()
    simplex.set_first_point(0, 0, np.array([1, 0, 2], dtype=float))
    simplex.add_new_point(1, 1, np.array([0, 0, 2], dtype=float))
    simplex.add_new_point(2, 2, np.array([-1, 0, 1], dtype=float))
    simplex.add_new_point(3, 3, np.array([1, 0, 1], dtype=float))

    barycentric_coordinates = BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.line_segment_03_of_tetrahedron_optimal()

    solution, backup = distance_subalgorithm_with_backup_procedure(
        simplex, Solution(), False)
    assert not backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:2], [0.5, 0.5])

    simplex = SimplexInfo()
    simplex.set_first_point(0, 0, np.array([-1, 0, 1], dtype=float))
    simplex.add_new_point(1, 1, np.array([1, 0, 1], dtype=float))
    simplex.add_new_point(2, 2, np.array([1, 0, 2], dtype=float))
    simplex.add_new_point(3, 3, np.array([0, 0, 2], dtype=float))

    barycentric_coordinates = BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.line_segment_12_of_tetrahedron_optimal()

    solution, backup = distance_subalgorithm_with_backup_procedure(
        simplex, Solution(), False)
    assert not backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:2], [0.5, 0.5])

    simplex = SimplexInfo()
    simplex.set_first_point(0, 0, np.array([-1, 0, 1], dtype=float))
    simplex.add_new_point(1, 1, np.array([1, 0, 2], dtype=float))
    simplex.add_new_point(2, 2, np.array([1, 0, 1], dtype=float))
    simplex.add_new_point(3, 3, np.array([0, 0, 2], dtype=float))

    barycentric_coordinates = BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.line_segment_13_of_tetrahedron_optimal()

    solution, backup = distance_subalgorithm_with_backup_procedure(
        simplex, Solution(), False)
    assert not backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:2], [0.5, 0.5])

    simplex = SimplexInfo()
    simplex.set_first_point(0, 0, np.array([1, 0, 2], dtype=float))
    simplex.add_new_point(1, 1, np.array([-1, 0, 1], dtype=float))
    simplex.add_new_point(2, 2, np.array([1, 0, 1], dtype=float))
    simplex.add_new_point(3, 3, np.array([0, 0, 2], dtype=float))

    barycentric_coordinates = BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.line_segment_23_of_tetrahedron_optimal()

    solution, backup = distance_subalgorithm_with_backup_procedure(
        simplex, Solution(), False)
    assert not backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:2], [0.5, 0.5])

    simplex = SimplexInfo()
    simplex.set_first_point(0, 0, np.array([1, -1, 1], dtype=float))
    simplex.add_new_point(1, 1, np.array([-1, -1, 1], dtype=float))
    simplex.add_new_point(2, 2, np.array([1, 0, 2], dtype=float))
    simplex.add_new_point(3, 3, np.array([0, 2, 1], dtype=float))

    barycentric_coordinates = BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.face_012_of_tetrahedron_optimal()

    solution, backup = distance_subalgorithm_with_backup_procedure(
        simplex, Solution(), False)
    assert not backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(
        solution.barycentric_coordinates[:3], [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

    simplex = SimplexInfo()
    simplex.set_first_point(0, 0, np.array([1, -1, 1], dtype=float))
    simplex.add_new_point(1, 1, np.array([1, 0, 2], dtype=float))
    simplex.add_new_point(2, 2, np.array([-1, -1, 1], dtype=float))
    simplex.add_new_point(3, 3, np.array([0, 2, 1], dtype=float))

    barycentric_coordinates = BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.face_013_of_tetrahedron_optimal()

    solution, backup = distance_subalgorithm_with_backup_procedure(
        simplex, Solution(), False)
    assert not backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(
        solution.barycentric_coordinates[:3], [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

    simplex = SimplexInfo()
    simplex.set_first_point(0, 0, np.array([1, 0, 2], dtype=float))
    simplex.add_new_point(1, 1, np.array([1, -1, 1], dtype=float))
    simplex.add_new_point(2, 2, np.array([-1, -1, 1], dtype=float))
    simplex.add_new_point(3, 3, np.array([0, 2, 1], dtype=float))

    barycentric_coordinates = BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.face_023_of_tetrahedron_optimal()

    solution, backup = distance_subalgorithm_with_backup_procedure(
        simplex, Solution(), False)
    assert not backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(
        solution.barycentric_coordinates[:3], [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

    simplex = SimplexInfo()
    simplex.set_first_point(0, 0, np.array([1, -1, 1], dtype=float))
    simplex.add_new_point(1, 1, np.array([-1, -1, 1], dtype=float))
    simplex.add_new_point(2, 2, np.array([0, 2, 1], dtype=float))
    simplex.add_new_point(3, 3, np.array([1, 0, 2], dtype=float))

    barycentric_coordinates = BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.face_123_of_tetrahedron_optimal()

    solution, backup = distance_subalgorithm_with_backup_procedure(
        simplex, Solution(), False)
    assert not backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(
        solution.barycentric_coordinates[:3], [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

    simplex = SimplexInfo()
    simplex.set_first_point(0, 0, np.array([1, -1, 1], dtype=float))
    simplex.add_new_point(1, 1, np.array([-1, -1, 1], dtype=float))
    simplex.add_new_point(2, 2, np.array([0, 2, 1], dtype=float))
    simplex.add_new_point(3, 3, np.array([0, 0, -3], dtype=float))

    barycentric_coordinates = BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.convex_hull_of_tetrahedron_optimal()

    solution, backup = distance_subalgorithm_with_backup_procedure(
        simplex, Solution(), False)
    assert not backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 0])
    assert approx(solution.distance_squared) == 0
    assert_array_almost_equal(
        solution.barycentric_coordinates, np.ones(4) / 4.0)


def test_tetrahedron_optimal_point_backup():
    simplex = SimplexInfo()
    simplex.set_first_point(0, 0, np.array([0, 0, 4], dtype=float))
    simplex.add_new_point(1, 1, np.array([0, 0, 3], dtype=float))
    simplex.add_new_point(2, 2, np.array([0, 0, 2], dtype=float))
    simplex.add_new_point(3, 3, np.array([0, 0, 1], dtype=float))

    barycentric_coordinates = BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.vertex_0_of_tetrahedron_optimal()

    solution, backup = distance_subalgorithm_with_backup_procedure(
        simplex, Solution(), True)
    assert backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:1], [1])

    simplex = SimplexInfo()
    simplex.set_first_point(0, 0, np.array([0, 0, 1], dtype=float))
    simplex.add_new_point(1, 1, np.array([0, 0, 4], dtype=float))
    simplex.add_new_point(2, 2, np.array([0, 0, 2], dtype=float))
    simplex.add_new_point(3, 3, np.array([0, 0, 3], dtype=float))

    barycentric_coordinates = BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.vertex_1_of_tetrahedron_optimal()

    solution, backup = distance_subalgorithm_with_backup_procedure(
        simplex, Solution(), True)
    assert backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:1], [1])

    simplex = SimplexInfo()
    simplex.set_first_point(0, 0, np.array([0, 0, 4], dtype=float))
    simplex.add_new_point(1, 1, np.array([0, 0, 1], dtype=float))
    simplex.add_new_point(2, 2, np.array([0, 0, 2], dtype=float))
    simplex.add_new_point(3, 3, np.array([0, 0, 3], dtype=float))

    barycentric_coordinates = BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.vertex_2_of_tetrahedron_optimal()

    solution, backup = distance_subalgorithm_with_backup_procedure(
        simplex, Solution(), True)
    assert backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:1], [1])

    simplex = SimplexInfo()
    simplex.set_first_point(0, 0, np.array([0, 0, 2], dtype=float))
    simplex.add_new_point(1, 1, np.array([0, 0, 4], dtype=float))
    simplex.add_new_point(2, 2, np.array([0, 0, 1], dtype=float))
    simplex.add_new_point(3, 3, np.array([0, 0, 3], dtype=float))

    barycentric_coordinates = BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.vertex_3_of_tetrahedron_optimal()

    solution, backup = distance_subalgorithm_with_backup_procedure(
        simplex, Solution(), True)
    assert backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:1], [1])

    simplex = SimplexInfo()
    simplex.set_first_point(0, 0, np.array([-1, 0, 1], dtype=float))
    simplex.add_new_point(1, 1, np.array([1, 0, 2], dtype=float))
    simplex.add_new_point(2, 2, np.array([0, 0, 2], dtype=float))
    simplex.add_new_point(3, 3, np.array([1, 0, 1], dtype=float))

    barycentric_coordinates = BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.line_segment_01_of_tetrahedron_optimal()

    solution, backup = distance_subalgorithm_with_backup_procedure(
        simplex, Solution(), True)
    assert backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:2], [0.5, 0.5])

    simplex = SimplexInfo()
    simplex.set_first_point(0, 0, np.array([1, 0, 2], dtype=float))
    simplex.add_new_point(1, 1, np.array([-1, 0, 1], dtype=float))
    simplex.add_new_point(2, 2, np.array([0, 0, 2], dtype=float))
    simplex.add_new_point(3, 3, np.array([1, 0, 1], dtype=float))

    barycentric_coordinates = BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.line_segment_02_of_tetrahedron_optimal()

    solution, backup = distance_subalgorithm_with_backup_procedure(
        simplex, Solution(), True)
    assert backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:2], [0.5, 0.5])

    simplex = SimplexInfo()
    simplex.set_first_point(0, 0, np.array([1, 0, 2], dtype=float))
    simplex.add_new_point(1, 1, np.array([0, 0, 2], dtype=float))
    simplex.add_new_point(2, 2, np.array([-1, 0, 1], dtype=float))
    simplex.add_new_point(3, 3, np.array([1, 0, 1], dtype=float))

    barycentric_coordinates = BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.line_segment_03_of_tetrahedron_optimal()

    solution, backup = distance_subalgorithm_with_backup_procedure(
        simplex, Solution(), True)
    assert backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:2], [0.5, 0.5])

    simplex = SimplexInfo()
    simplex.set_first_point(0, 0, np.array([-1, 0, 1], dtype=float))
    simplex.add_new_point(1, 1, np.array([1, 0, 1], dtype=float))
    simplex.add_new_point(2, 2, np.array([1, 0, 2], dtype=float))
    simplex.add_new_point(3, 3, np.array([0, 0, 2], dtype=float))

    barycentric_coordinates = BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.line_segment_12_of_tetrahedron_optimal()

    solution, backup = distance_subalgorithm_with_backup_procedure(
        simplex, Solution(), True)
    assert backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:2], [0.5, 0.5])

    simplex = SimplexInfo()
    simplex.set_first_point(0, 0, np.array([-1, 0, 1], dtype=float))
    simplex.add_new_point(1, 1, np.array([1, 0, 2], dtype=float))
    simplex.add_new_point(2, 2, np.array([1, 0, 1], dtype=float))
    simplex.add_new_point(3, 3, np.array([0, 0, 2], dtype=float))

    barycentric_coordinates = BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.line_segment_13_of_tetrahedron_optimal()

    solution, backup = distance_subalgorithm_with_backup_procedure(
        simplex, Solution(), True)
    assert backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:2], [0.5, 0.5])

    simplex = SimplexInfo()
    simplex.set_first_point(0, 0, np.array([1, 0, 2], dtype=float))
    simplex.add_new_point(1, 1, np.array([-1, 0, 1], dtype=float))
    simplex.add_new_point(2, 2, np.array([1, 0, 1], dtype=float))
    simplex.add_new_point(3, 3, np.array([0, 0, 2], dtype=float))

    barycentric_coordinates = BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.line_segment_23_of_tetrahedron_optimal()

    solution, backup = distance_subalgorithm_with_backup_procedure(
        simplex, Solution(), True)
    assert backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(solution.barycentric_coordinates[:2], [0.5, 0.5])

    simplex = SimplexInfo()
    simplex.set_first_point(0, 0, np.array([1, -1, 1], dtype=float))
    simplex.add_new_point(1, 1, np.array([-1, -1, 1], dtype=float))
    simplex.add_new_point(2, 2, np.array([1, 0, 2], dtype=float))
    simplex.add_new_point(3, 3, np.array([0, 2, 1], dtype=float))

    barycentric_coordinates = BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.face_012_of_tetrahedron_optimal()

    solution, backup = distance_subalgorithm_with_backup_procedure(
        simplex, Solution(), True)
    assert backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(
        solution.barycentric_coordinates[:3], [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

    simplex = SimplexInfo()
    simplex.set_first_point(0, 0, np.array([1, -1, 1], dtype=float))
    simplex.add_new_point(1, 1, np.array([1, 0, 2], dtype=float))
    simplex.add_new_point(2, 2, np.array([-1, -1, 1], dtype=float))
    simplex.add_new_point(3, 3, np.array([0, 2, 1], dtype=float))

    barycentric_coordinates = BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.face_013_of_tetrahedron_optimal()

    solution, backup = distance_subalgorithm_with_backup_procedure(
        simplex, Solution(), True)
    assert backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(
        solution.barycentric_coordinates[:3], [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

    simplex = SimplexInfo()
    simplex.set_first_point(0, 0, np.array([1, 0, 2], dtype=float))
    simplex.add_new_point(1, 1, np.array([1, -1, 1], dtype=float))
    simplex.add_new_point(2, 2, np.array([-1, -1, 1], dtype=float))
    simplex.add_new_point(3, 3, np.array([0, 2, 1], dtype=float))

    barycentric_coordinates = BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.face_023_of_tetrahedron_optimal()

    solution, backup = distance_subalgorithm_with_backup_procedure(
        simplex, Solution(), True)
    assert backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(
        solution.barycentric_coordinates[:3], [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

    simplex = SimplexInfo()
    simplex.set_first_point(0, 0, np.array([1, -1, 1], dtype=float))
    simplex.add_new_point(1, 1, np.array([-1, -1, 1], dtype=float))
    simplex.add_new_point(2, 2, np.array([0, 2, 1], dtype=float))
    simplex.add_new_point(3, 3, np.array([1, 0, 2], dtype=float))

    barycentric_coordinates = BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.face_123_of_tetrahedron_optimal()

    solution, backup = distance_subalgorithm_with_backup_procedure(
        simplex, Solution(), True)
    assert backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 1])
    assert approx(solution.distance_squared) == 1
    assert_array_almost_equal(
        solution.barycentric_coordinates[:3], [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

    simplex = SimplexInfo()
    simplex.set_first_point(0, 0, np.array([1, -1, 1], dtype=float))
    simplex.add_new_point(1, 1, np.array([-1, -1, 1], dtype=float))
    simplex.add_new_point(2, 2, np.array([0, 2, 1], dtype=float))
    simplex.add_new_point(3, 3, np.array([0, 0, -3], dtype=float))

    barycentric_coordinates = BarycentricCoordinates()
    barycentric_coordinates.tetrahedron_coordinates_0(simplex)
    e132, e142 = barycentric_coordinates.tetrahedron_coordinates_1(simplex)
    e123, e143 = barycentric_coordinates.tetrahedron_coordinates_2(simplex)
    e213 = barycentric_coordinates.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    e124, e134 = barycentric_coordinates.tetrahedron_coordinates_4(simplex)
    e214 = barycentric_coordinates.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    barycentric_coordinates.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    barycentric_coordinates.tetrahedron_coordinates_7(simplex, e213, e214)
    assert barycentric_coordinates.convex_hull_of_tetrahedron_optimal()

    solution, backup = distance_subalgorithm_with_backup_procedure(
        simplex, Solution(), True)
    assert backup
    assert_array_almost_equal(solution.search_direction, [0, 0, 0])
    assert approx(solution.distance_squared) == 0
    assert_array_almost_equal(
        solution.barycentric_coordinates, np.ones(4) / 4.0)
