"""Gilbert-Johnson-Keerthi (GJK) for distance calculation of convex shapes."""
import math
import numpy as np
from .colliders import Convex


def gjk(vertices1, vertices2):
    """Gilbert-Johnson-Keerthi algorithm for distance calculation.

    Parameters
    ----------
    vertices1 : array, shape (n_vertices1, 3)
        Vertices of the first convex shape.

    vertices2 : array, shape (n_vertices2, 3)
        Vertices of the second convex shape.

    Returns
    -------
    distance : float
        The shortest distance between two convex shapes.

    closest_point1 : array, shape (3,)
        Closest point on first convex shape.

    closest_point2 : array, shape (3,)
        Closest point on second convex shape.
    """
    return gjk_with_simplex(Convex(vertices1), Convex(vertices2))[:3]


def gjk_with_simplex(collider1, collider2):
    """Gilbert-Johnson-Keerthi (GJK) algorithm for distance calculation.

    The GJK algorithm only works for convex shapes. Concave objects have to be
    decomposed into convex shapes first.

    Based on the translation to C of the original Fortran implementation:
    Ruspini, Diego. gilbert.c, a C version of the original Fortran
    implementation of the GJK algorithm.
    ftp://labrea.stanford.edu/cs/robotics/sean/distance/gilbert.c,
    also available from http://realtimecollisiondetection.net/files/gilbert.c

    The original publication describing the algorithm is:
    E.G. Gilbert, D.W. Johnson, S.S. Keerthi: A fast procedure for computing
    the distance between complex objects in three-dimensional space, IEEE
    Journal of Robotics and Automation (1988),
    https://graphics.stanford.edu/courses/cs448b-00-winter/papers/gilbert.pdf

    Parameters
    ----------
    collider1 : ConvexCollider
        Convex collider 1.

    collider2 : ConvexCollider
        Convex collider 2.

    Returns
    -------
    distance : float
        The shortest distance between two convex shapes.

    closest_point1 : array, shape (3,)
        Closest point on first convex shape.

    closest_point2 : array, shape (3,)
        Closest point on second convex shape.

    simplex : array, shape (4, 3)
        Simplex defined by 4 points of the Minkowski difference between
        vertices of the two colliders.
    """
    solution = Solution()
    simplex = Simplex()
    old_simplex = Simplex()
    backup = False

    simplex.initialize_with_point(
        collider1.first_vertex() - collider2.first_vertex())

    solution.barycentric_coordinates[0] = 1.0
    solution.dstsq = simplex.dot_product_table[0, 0] + simplex.dot_product_table[0, 0] + 1.0
    iteration = 0
    while True:
        iteration += 1

        new_solution, backup = distance_subalgorithm(simplex, solution, backup)

        if new_solution.dstsq >= solution.dstsq or len(simplex) == 4:
            if backup:
                closest_point1 = collider1.compute_point(
                    solution.barycentric_coordinates[:len(simplex)],
                    simplex.indices_polytope1[:len(simplex)])
                closest_point2 = collider2.compute_point(
                    solution.barycentric_coordinates[:len(simplex)],
                    simplex.indices_polytope2[:len(simplex)])

                if len(simplex) == 4:
                    # Make sure intersection has zero distance
                    closest_point1[:] = 0.5 * (closest_point1 + closest_point2)
                    closest_point2[:] = closest_point1
                    distance = 0.0
                else:
                    distance = math.sqrt(new_solution.dstsq)

                return distance, closest_point1, closest_point2, simplex.simplex

            backup = True
            if iteration != 1:
                simplex.copy_from(old_simplex)
            continue

        solution = new_solution

        # Find new supporting point in direction -search_direction:
        # s_(A-B)(-search_direction) = s_A(-search_direction) - s_B(search_direction)
        new_index1, new_vertex1 = collider1.support_function(-solution.search_direction)
        new_index2, new_vertex2 = collider2.support_function(solution.search_direction)
        new_simplex_point = new_vertex1 - new_vertex2

        simplex.add_new_point(new_index1, new_index2, new_simplex_point)
        old_simplex.copy_from(simplex)
        if len(simplex) == 4:
            _reorder_simplex_nondecreasing_order(simplex, old_simplex)

    raise RuntimeError("Solution should be found in loop.")


class Solution:
    """Represents current best solution and search direction.

    Attributes
    ----------
    search_direction : array, shape (3,)
        Near point to the convex hull of the points in simplex.

    barycentric_coordinates : array, shape (n_simplex_points,)
        The barycentric coordinates of search_direction, i.e.,
        search_direction = barycentric_coordinates[0]*simplex[1] + ...
        + barycentric_coordinates(n_simplex_points)*simplex[n_simplex_points-1],
        barycentric_coordinates[k] > 0.0 for k=0,...,n_simplex_points-1, and,
        barycentric_coordinates[0] + ...
        + barycentric_coordinates[n_simplex_points-1] = 1.0.

    dstsq : float
        Squared distance to origin.
    """
    def __init__(self):
        self.barycentric_coordinates = np.zeros(4, dtype=float)
        self.search_direction = np.zeros(3, dtype=float)
        self.dstsq = np.inf

    def from_vertex(self, vertex_idx, a, simplex):
        self.barycentric_coordinates[vertex_idx] = a
        self.search_direction = simplex.simplex[vertex_idx]
        self.dstsq = simplex.dot_product_table[vertex_idx, vertex_idx]

    def from_line_segment(self, vi1, vi2, a, b, simplex, bci1=0, bci2=1):
        coords_sum = a + b
        self.barycentric_coordinates[bci1] = a / coords_sum
        self.barycentric_coordinates[bci2] = 1.0 - self.barycentric_coordinates[bci1]
        self.search_direction = simplex.search_direction_line(
            vi1, vi2, self.barycentric_coordinates[bci1])
        self.dstsq = np.dot(self.search_direction, self.search_direction)

    def from_face(self, vi1, vi2, vi3, a, b, c, simplex, bci1=0, bci2=1, bci3=2):
        coords_sum = a + b + c
        self.barycentric_coordinates[bci1] = a / coords_sum
        self.barycentric_coordinates[bci2] = b / coords_sum
        self.barycentric_coordinates[bci3] = (
            1.0 - self.barycentric_coordinates[bci1]
            - self.barycentric_coordinates[bci2])
        self.search_direction = simplex.search_direction_face(
            vi1, vi2, vi3, self.barycentric_coordinates[bci1],
            self.barycentric_coordinates[bci2])
        self.dstsq = np.dot(self.search_direction, self.search_direction)

    def from_simplex(self, a, b, c, d, simplex):
        coords_sum = a + b + c + d
        self.barycentric_coordinates[0] = a / coords_sum
        self.barycentric_coordinates[1] = b / coords_sum
        self.barycentric_coordinates[2] = c / coords_sum
        self.barycentric_coordinates[3] = 1.0 - sum(self.barycentric_coordinates[:3])
        self.search_direction = simplex.search_direction_simplex(
            self.barycentric_coordinates)
        self.dstsq = np.dot(self.search_direction, self.search_direction)

    def copy_from(self, solution, n_simplex_points):
        self.barycentric_coordinates[:n_simplex_points] = \
            solution.barycentric_coordinates[:n_simplex_points]
        self.search_direction = solution.search_direction
        self.dstsq = solution.dstsq


class Simplex:
    """Simplex and additional data.

    Attributes
    ----------
    n_simplex_points : int
        Number of current simplex points.

    simplex : array, shape (n_simplex_points, 3)
      Current simplex.

    dot_product_table : array, shape (n_simplex_points, n_simplex_points)
        dot_product_table[i, j] = Inner product of simplex[i] and simplex[j].

    indices_polytope1 : array, shape (n_simplex_points,)
        Index vector for first polytope. For k = 1, ..., n_simplex_points,
        simplex[k] = vertices1[indices_polytope1[k]]
        - vertices2[indices_polytope2[k]].

    indices_polytope2 : array, shape (n_simplex_points,)
        Index vectors for first and second polytope. For k = 1, ...,
        n_simplex_points, simplex[k] = vertices1[indices_polytope1[k]]
        - vertices2[indices_polytope2[k]].
    """
    def __init__(self):
        self.n_simplex_points = 0
        self.simplex = np.empty((4, 3), dtype=float)
        self.dot_product_table = np.empty((4, 4), dtype=float)
        self.indices_polytope1 = np.empty(4, dtype=int)
        self.indices_polytope2 = np.empty(4, dtype=int)

    def initialize_with_point(self, point):
        self.n_simplex_points = 1
        self.simplex[0] = point
        self.dot_product_table[0, 0] = np.dot(self.simplex[0], self.simplex[0])
        self.indices_polytope1[0] = 0
        self.indices_polytope2[0] = 0

    def copy_from(self, simplex):
        self.n_simplex_points = len(simplex)
        self.simplex[:len(simplex)] = simplex.simplex[:len(simplex)]
        self.indices_polytope1[:len(simplex)] = simplex.indices_polytope1[:len(simplex)]
        self.indices_polytope2[:len(simplex)] = simplex.indices_polytope2[:len(simplex)]
        self.dot_product_table[:self.n_simplex_points, :self.n_simplex_points] = simplex.dot_product_table[
            :self.n_simplex_points, :self.n_simplex_points]

    def reorder(self, ordered_indices):
        indices_polytope1 = np.copy(self.indices_polytope1[:self.n_simplex_points])
        indices_polytope2 = np.copy(self.indices_polytope2[:self.n_simplex_points])
        simplex = np.copy(self.simplex[:self.n_simplex_points])
        dot_product_table = np.empty((4, 4), dtype=float)
        for k in range(self.n_simplex_points):
            dot_product_table[k, :k + 1] = self.dot_product_table[k, :k + 1]
        self.n_simplex_points = len(ordered_indices)
        for k in range(self.n_simplex_points):
            kk = ordered_indices[k]
            self.indices_polytope1[k] = indices_polytope1[kk]
            self.indices_polytope2[k] = indices_polytope2[kk]
            self.simplex[k] = simplex[kk]
            for l in range(k):
                ll = ordered_indices[l]
                if kk >= ll:
                    self.dot_product_table[k, l] = dot_product_table[kk, ll]
                else:
                    self.dot_product_table[k, l] = dot_product_table[ll, kk]
            self.dot_product_table[k, k] = dot_product_table[kk, kk]

    def add_new_point(self, new_index1, new_index2, new_simplex_point):
        self._move_first_point_to_last_spot()
        self._put_new_point_in_first_spot(new_index1, new_index2, new_simplex_point)

    def _move_first_point_to_last_spot(self):
        self.indices_polytope1[self.n_simplex_points] = self.indices_polytope1[0]
        self.indices_polytope2[self.n_simplex_points] = self.indices_polytope2[0]
        self.simplex[self.n_simplex_points] = self.simplex[0]
        self.dot_product_table[self.n_simplex_points, :self.n_simplex_points] = self.dot_product_table[
                                                                                :self.n_simplex_points, 0]
        self.dot_product_table[self.n_simplex_points, self.n_simplex_points] = self.dot_product_table[0, 0]

    def _put_new_point_in_first_spot(self, new_index1, new_index2, new_simplex_point):
        self.indices_polytope1[0] = new_index1
        self.indices_polytope2[0] = new_index2
        self.simplex[0] = new_simplex_point
        self.n_simplex_points += 1
        self.dot_product_table[:self.n_simplex_points, 0] = np.dot(
            self.simplex[:self.n_simplex_points], self.simplex[0])

    def reduce_to_optimal_vertex(self, vertex_index):
        self.n_simplex_points = 1
        self.move_vertex(vertex_index, 0)
        self.dot_product_table[0, 0] = self.dot_product_table[
            vertex_index, vertex_index]

    def move_vertex(self, old_index, new_index):
        if old_index == new_index:
            return
        self.indices_polytope1[new_index] = self.indices_polytope1[old_index]
        self.indices_polytope2[new_index] = self.indices_polytope2[old_index]
        self.simplex[new_index] = self.simplex[old_index]

    def search_direction_line(self, vi1, vi2, a):
        return self.simplex[vi1] + a * (self.simplex[vi2] - self.simplex[vi1])

    def search_direction_face(self, vi1, vi2, vi3, a, b):
        return (
            self.simplex[vi1]
            + a * (self.simplex[vi2] - self.simplex[vi1])
            + b * (self.simplex[vi3] - self.simplex[vi1]))

    def search_direction_simplex(self, barycentric_coordinates):
        return barycentric_coordinates.dot(self.simplex)

    def select_line_segment_13(self):
        self.n_simplex_points = 2
        self.move_vertex(2, 1)
        self.dot_product_table[1, 0] = self.dot_product_table[2, 0]
        self.dot_product_table[1, 1] = self.dot_product_table[2, 2]

    def select_line_segment_14(self):
        self.n_simplex_points = 2
        self.move_vertex(3, 1)
        self.dot_product_table[1, 0] = self.dot_product_table[3, 0]
        self.dot_product_table[1, 1] = self.dot_product_table[3, 3]

    def select_line_segment_23(self):
        self.n_simplex_points = 2
        self.move_vertex(2, 0)
        self.dot_product_table[1, 0] = self.dot_product_table[2, 1]
        self.dot_product_table[0, 0] = self.dot_product_table[2, 2]

    def select_line_segment_24(self):
        self.n_simplex_points = 2
        self.move_vertex(3, 0)
        self.dot_product_table[1, 0] = self.dot_product_table[3, 1]
        self.dot_product_table[0, 0] = self.dot_product_table[3, 3]

    def select_line_segment_34(self):
        self.n_simplex_points = 2
        self.move_vertex(2, 0)
        self.move_vertex(3, 1)
        self.dot_product_table[0, 0] = self.dot_product_table[2, 2]
        self.dot_product_table[1, 0] = self.dot_product_table[3, 2]
        self.dot_product_table[1, 1] = self.dot_product_table[3, 3]

    def select_face_124(self):
        self.n_simplex_points = 3
        self.move_vertex(3, 2)
        self.dot_product_table[2, 0] = self.dot_product_table[3, 0]
        self.dot_product_table[2, 1] = self.dot_product_table[3, 1]
        self.dot_product_table[2, 2] = self.dot_product_table[3, 3]

    def select_face_134(self):
        self.n_simplex_points = 3
        self.move_vertex(3, 1)
        self.dot_product_table[1, 0] = self.dot_product_table[3, 0]
        self.dot_product_table[1, 1] = self.dot_product_table[3, 3]
        self.dot_product_table[2, 1] = self.dot_product_table[3, 2]

    def select_face_234(self):
        self.n_simplex_points = 3
        self.move_vertex(3, 0)
        self.dot_product_table[0, 0] = self.dot_product_table[3, 3]
        self.dot_product_table[1, 0] = self.dot_product_table[3, 1]
        self.dot_product_table[2, 0] = self.dot_product_table[3, 2]

    def __len__(self):
        return self.n_simplex_points


def distance_subalgorithm(simplex, solution, backup):
    """Distance subalgorithm.

    Computes point of minimum norm in the convex hull of the simplex.

    Implements, in a very efficient way, the distance subalgorithm
    of finding the near point to the convex hull of four or less points
    in 3D space. The procedure and its efficient FORTRAN implementation
    are both due to D.W. Johnson. Although this subroutine is quite long,
    only a very small part of it will be executed on each call. Refer to
    sections 5 and 6 of the report mentioned in routine DIST3 for details
    concerning the distance subalgorithm. Converted to C be Diego C. Ruspini
    3/25/93.

    This function also determines an affinely independent subset of the
    points such that search_direction is a near point to the affine hull of
    the points in the subset. The variable simplex is modified so that, on
    output, it corresponds to this subset of points.

    Parameters
    ----------
    simplex : array, shape (n_simplex_points, 3)
      Current simplex.

    solution : Solution
        Represents current best solution and search direction.

    backup : bool
        Perform backup procedure.

    Returns
    -------
    solution : Solution
        Current solution.

    backup : int
        Perform backup procedure.
    """
    d1 = np.empty(15, dtype=float)
    d2 = np.empty(15, dtype=float)
    d3 = np.empty(15, dtype=float)
    d4 = np.empty(15, dtype=float)

    d1[0] = 1.0
    d2[1] = 1.0
    d3[3] = 1.0
    d4[7] = 1.0

    if not backup:
        new_solution = _regular_distance_subalgorithm(simplex, d1, d2, d3, d4)
        if new_solution is not None:
            return new_solution, backup

    return _backup_procedure(simplex, solution, d1, d2, d3, d4, backup)


def _regular_distance_subalgorithm(simplex, d1, d2, d3, d4):
    if len(simplex) == 1:
        solution = Solution()
        solution.from_vertex(0, d1[0], simplex)
        return solution
    elif len(simplex) == 2:
        return _distance_subalgorithm_line_segment(simplex, d1, d2)
    elif len(simplex) == 3:
        return _distance_subalgorithm_face(simplex, d1, d2, d3)
    elif len(simplex) == 4:
        return _distance_subalgorithm_simplex(simplex, d1, d2, d3, d4)


def _distance_subalgorithm_line_segment(simplex, d1, d2):
    solution = Solution()
    d2[2] = simplex.dot_product_table[0, 0] - simplex.dot_product_table[1, 0]
    vertex_1_optimal = d2[2] <= 0.0
    if vertex_1_optimal:
        simplex.reduce_to_optimal_vertex(0)
        solution.from_vertex(0, d1[0], simplex)
        return solution
    d1[2] = simplex.dot_product_table[1, 1] - simplex.dot_product_table[1, 0]
    line_segment_12_optimal = not (d1[2] <= 0.0 or d2[2] <= 0.0)
    if line_segment_12_optimal:
        solution.from_line_segment(1, 0, d1[2], d2[2], simplex)
        return solution
    vertex_2_optimal = d1[2] <= 0.0
    if vertex_2_optimal:
        simplex.reduce_to_optimal_vertex(1)
        solution.from_vertex(0, d2[1], simplex)
        return solution
    return None


def _distance_subalgorithm_face(simplex, d1, d2, d3):
    solution = Solution()
    _compute_face_distances_0(simplex, d2, d3)
    vertex_1_optimal = not (d2[2] > 0.0 or d3[4] > 0.0)
    if vertex_1_optimal:
        simplex.reduce_to_optimal_vertex(0)
        solution.from_vertex(0, d1[0], simplex)
        return solution
    _compute_face_distances_1(simplex, d1, d2, d3)
    line_segment_12_optimal = not (d1[2] <= 0.0 or d2[2] <= 0.0 or d3[6] > 0.0)
    if line_segment_12_optimal:
        simplex.n_simplex_points = 2
        solution.from_line_segment(1, 0, d1[2], d2[2], simplex)
        return solution
    e123 = _compute_face_distances_2(simplex, d1, d2, d3)
    line_segment_13_optimal = not (d1[4] <= 0.0 or d2[6] > 0.0 or d3[4] <= 0.0)
    if line_segment_13_optimal:
        simplex.select_line_segment_13()
        solution.from_line_segment(1, 0, d1[4], d3[4], simplex)
        return solution
    _compute_face_distances_3(simplex, d1, d2, d3, e123)
    face_123_optimal = not (d1[6] <= 0.0 or d2[6] <= 0.0 or d3[6] <= 0.0)
    if face_123_optimal:
        solution.from_face(2, 0, 1, d1[6], d2[6], d3[6], simplex)
        return solution
    vertex_2_optimal = not (d1[2] > 0.0 or d3[5] > 0.0)
    if vertex_2_optimal:
        simplex.reduce_to_optimal_vertex(1)
        solution.from_vertex(0, d2[1], simplex)
        return solution
    vertex_3_optimal = not (d1[4] > 0.0 or d2[5] > 0.0)
    if vertex_3_optimal:
        simplex.reduce_to_optimal_vertex(2)
        solution.from_vertex(0, d3[3], simplex)
        return solution
    line_segment_23_optimal = not (d1[6] > 0.0 or d2[5] <= 0.0 or d3[5] <= 0.0)
    if line_segment_23_optimal:
        simplex.select_line_segment_23()
        solution.from_line_segment(0, 1, d2[5], d3[5], simplex)
        return solution
    return None


def _distance_subalgorithm_simplex(simplex, d1, d2, d3, d4):
    solution = Solution()
    _compute_face_distances_0(simplex, d2, d3)
    d4[8] = simplex.dot_product_table[0, 0] - simplex.dot_product_table[3, 0]
    vertex_1_optimal = not (d2[2] > 0.0 or d3[4] > 0.0 or d4[8] > 0.0)
    if vertex_1_optimal:
        simplex.reduce_to_optimal_vertex(0)
        solution.from_vertex(0, d1[0], simplex)
        return solution
    e132 = simplex.dot_product_table[1, 0] - simplex.dot_product_table[2, 1]
    e142 = simplex.dot_product_table[1, 0] - simplex.dot_product_table[3, 1]
    d1[2] = simplex.dot_product_table[1, 1] - simplex.dot_product_table[1, 0]
    d3[6] = d1[2] * d3[4] + d2[2] * e132
    d4[11] = d1[2] * d4[8] + d2[2] * e142
    line_segment_12_optimal = not (d1[2] <= 0.0 or d2[2] <= 0.0 or d3[6] > 0.0 or d4[11] > 0.0)
    if line_segment_12_optimal:
        simplex.n_simplex_points = 2
        solution.from_line_segment(1, 0, d1[2], d2[2], simplex)
        return solution
    e123 = simplex.dot_product_table[2, 0] - simplex.dot_product_table[2, 1]
    e143 = simplex.dot_product_table[2, 0] - simplex.dot_product_table[3, 2]
    d1[4] = simplex.dot_product_table[2, 2] - simplex.dot_product_table[2, 0]
    d2[6] = d1[4] * d2[2] + d3[4] * e123
    d4[12] = d1[4] * d4[8] + d3[4] * e143
    line_segment_13_optimal = not (d1[4] <= 0.0 or d2[6] > 0.0 or d3[4] <= 0.0 or d4[12] > 0.0)
    if line_segment_13_optimal:
        simplex.select_line_segment_13()
        solution.from_line_segment(1, 0, d1[4], d3[4], simplex)
        return solution
    d2[5] = simplex.dot_product_table[2, 2] - simplex.dot_product_table[2, 1]
    d3[5] = simplex.dot_product_table[1, 1] - simplex.dot_product_table[2, 1]
    e213 = -e123
    d1[6] = d2[5] * d1[2] + d3[5] * e213
    d4[14] = d1[6] * d4[8] + d2[6] * e142 + d3[6] * e143
    face_123_optimal = not (d1[6] <= 0.0 or d2[6] <= 0.0 or d3[6] <= 0.0 or d4[14] > 0.0)
    if face_123_optimal:
        simplex.n_simplex_points = 3
        solution.from_face(2, 0, 1, d1[6], d2[6], d3[6], simplex)
        return solution
    e124, e134 = _compute_simplex_distances_0(simplex, d1, d2, d3, d4)
    line_segment_14_optimal = not (d1[8] <= 0.0 or d2[11] > 0.0 or d3[12] > 0.0 or d4[8] <= 0.0)
    if line_segment_14_optimal:
        simplex.select_line_segment_14()
        solution.from_line_segment(1, 0, d1[8], d4[8], simplex)
        return solution
    e214 = _compute_simplex_distances_1(simplex, d1, d2, d3, d4, e124, e132, e134)
    face_124_optimal = not (d1[11] <= 0.0 or d2[11] <= 0.0 or d3[14] > 0.0 or d4[11] <= 0.0)
    if face_124_optimal:
        simplex.select_face_124()
        solution.from_face(2, 0, 1, d1[11], d2[11], d4[11], simplex)
        return solution
    _compute_simplex_distances_2(simplex, d1, d2, d3, d4, e123, e124, e134)
    face_134_optimal = not (d1[12] <= 0.0 or d2[14] > 0.0 or d3[12] <= 0.0 or d4[12] <= 0.0)
    if face_134_optimal:
        simplex.select_face_134()
        solution.from_face(1, 0, 2, d1[12], d3[12], d4[12], simplex, 0, 2, 1)
        return solution
    _compute_simplex_distances_3(simplex, d1, d2, d3, d4, e213, e214)
    convex_hull_optimal = not (d1[14] <= 0.0 or d2[14] <= 0.0 or d3[14] <= 0.0 or d4[14] <= 0.0)
    if convex_hull_optimal:
        solution.from_simplex(d1[14], d2[14], d3[14], d4[14], simplex)
        return solution
    vertex_2_optimal = not (d1[2] > 0.0 or d3[5] > 0.0 or d4[9] > 0.0)
    if vertex_2_optimal:
        simplex.reduce_to_optimal_vertex(1)
        solution.from_vertex(0, d2[1], simplex)
        return solution
    vertex_3_optimal = not (d1[4] > 0.0 or d2[5] > 0.0 or d4[10] > 0.0)
    if vertex_3_optimal:
        simplex.reduce_to_optimal_vertex(2)
        solution.from_vertex(0, d3[3], simplex)
        return solution
    vertex_4_optimal = not (d1[8] > 0.0 or d2[9] > 0.0 or d3[10] > 0.0)
    if vertex_4_optimal:
        simplex.reduce_to_optimal_vertex(3)
        solution.from_vertex(0, d4[7], simplex)
        return solution
    line_segment_23_optimal = not (d1[6] > 0.0 or d2[5] <= 0.0 or d3[5] <= 0.0 or d4[13] > 0.0)
    if line_segment_23_optimal:
        simplex.select_line_segment_23()
        solution.from_line_segment(0, 1, d2[5], d3[5], simplex)
        return solution
    line_segment_24_optimal = not (d1[11] > 0.0 or d2[9] <= 0.0 or d3[13] > 0.0 or d4[9] <= 0.0)
    if line_segment_24_optimal:
        simplex.select_line_segment_24()
        solution.from_line_segment(0, 1, d2[9], d4[9], simplex, 1, 0)
        return solution
    line_segment_34_optimal = not (d1[12] > 0.0 or d2[13] > 0.0 or d3[10] <= 0.0 or d4[10] <= 0.0)
    if line_segment_34_optimal:
        simplex.select_line_segment_34()
        solution.from_line_segment(1, 0, d3[10], d4[10], simplex)
        return solution
    face_234_optimal = not (d1[14] > 0.0 or d2[13] <= 0.0 or d3[13] <= 0.0 or d4[13] <= 0.0)
    if face_234_optimal:
        simplex.select_face_234()
        solution.from_face(0, 1, 2, d2[13], d3[13], d4[13], simplex, 1, 2, 0)
        return solution
    return None


def _compute_face_distances_0(simplex, d2, d3):
    d2[2] = simplex.dot_product_table[0, 0] - simplex.dot_product_table[1, 0]
    d3[4] = simplex.dot_product_table[0, 0] - simplex.dot_product_table[2, 0]


def _compute_face_distances_1(simplex, d1, d2, d3):
    e132 = simplex.dot_product_table[1, 0] - simplex.dot_product_table[2, 1]
    d1[2] = simplex.dot_product_table[1, 1] - simplex.dot_product_table[1, 0]
    d3[6] = d1[2] * d3[4] + d2[2] * e132


def _compute_face_distances_2(simplex, d1, d2, d3):
    e123 = simplex.dot_product_table[2, 0] - simplex.dot_product_table[2, 1]
    d1[4] = simplex.dot_product_table[2, 2] - simplex.dot_product_table[2, 0]
    d2[6] = d1[4] * d2[2] + d3[4] * e123
    return e123


def _compute_face_distances_3(simplex, d1, d2, d3, e123):
    e213 = -e123
    d2[5] = simplex.dot_product_table[2, 2] - simplex.dot_product_table[2, 1]
    d3[5] = simplex.dot_product_table[1, 1] - simplex.dot_product_table[2, 1]
    d1[6] = d2[5] * d1[2] + d3[5] * e213
    return e213


def _compute_simplex_distances_0(simplex, d1, d2, d3, d4):
    e124 = simplex.dot_product_table[3, 0] - simplex.dot_product_table[3, 1]
    e134 = simplex.dot_product_table[3, 0] - simplex.dot_product_table[3, 2]
    d1[8] = simplex.dot_product_table[3, 3] - simplex.dot_product_table[3, 0]
    d2[11] = d1[8] * d2[2] + d4[8] * e124
    d3[12] = d1[8] * d3[4] + d4[8] * e134
    return e124, e134


def _compute_simplex_distances_1(simplex, d1, d2, d3, d4, e124, e132, e134):
    d2[9] = simplex.dot_product_table[3, 3] - simplex.dot_product_table[3, 1]
    d4[9] = simplex.dot_product_table[1, 1] - simplex.dot_product_table[3, 1]
    e214 = -e124
    d1[11] = d2[9] * d1[2] + d4[9] * e214
    d3[14] = d1[11] * d3[4] + d2[11] * e132 + d4[11] * e134
    return e214


def _compute_simplex_distances_2(simplex, d1, d2, d3, d4, e123, e124, e134):
    d3[10] = simplex.dot_product_table[3, 3] - simplex.dot_product_table[3, 2]
    d4[10] = simplex.dot_product_table[2, 2] - simplex.dot_product_table[3, 2]
    e314 = -e134
    d1[12] = d3[10] * d1[4] + d4[10] * e314
    d2[14] = d1[12] * d2[2] + d3[12] * e123 + d4[12] * e124


def _compute_simplex_distances_3(simplex, d1, d2, d3, d4, e213, e214):
    e243 = simplex.dot_product_table[2, 1] - simplex.dot_product_table[3, 2]
    d4[13] = d2[5] * d4[9] + d3[5] * e243
    e234 = simplex.dot_product_table[3, 1] - simplex.dot_product_table[3, 2]
    d3[13] = d2[9] * d3[5] + d4[9] * e234
    e324 = -e234
    d2[13] = d3[10] * d2[5] + d4[10] * e324
    d1[14] = d2[13] * d1[2] + d3[13] * e213 + d4[13] * e214


def _backup_procedure(simplex, solution, d1, d2, d3, d4, backup):
    ordered_indices = np.empty(4, dtype=int)
    solution_d = Solution()
    if len(simplex) == 1:
        solution.from_vertex(0, d1[0], simplex)
        return solution, True
    elif len(simplex) == 2:
        n_simplex_points = _backup_procedure_line_segment(
            simplex, backup, d1, d2, ordered_indices, solution, solution_d)
    elif len(simplex) == 3:
        n_simplex_points = _backup_procedure_face(
            simplex, backup, d1, d2, d3, ordered_indices, solution, solution_d)
    elif len(simplex) == 4:
        n_simplex_points = _backup_procedure_simplex(
            simplex, backup, d1, d2, d3, d4, ordered_indices, solution,
            solution_d)

    simplex.reorder(ordered_indices[:n_simplex_points])
    return solution, True


def _backup_procedure_line_segment(
        simplex, backup, d1, d2, ordered_indices, solution, solution_d):
    if backup:
        _backup_line_segments(simplex, d1, d2)
    # check vertex 1
    solution.from_vertex(0, d1[0], simplex)
    n_simplex_points = 1
    ordered_indices[0] = 0
    check_line_segment_12 = not (d1[2] <= 0.0 or d2[2] <= 0.0)
    if check_line_segment_12:
        solution_d.from_line_segment(1, 0, d1[2], d2[2], simplex)
        if solution_d.dstsq < solution.dstsq:
            n_simplex_points = 2
            solution.copy_from(solution_d, n_simplex_points)
            ordered_indices[:2] = 0, 1
    check_vertex_2 = simplex.dot_product_table[1, 1] < solution.dstsq
    if check_vertex_2:
        n_simplex_points = 1
        solution.from_vertex(1, d2[1], simplex)
        ordered_indices[0] = 1
    return n_simplex_points


def _backup_procedure_face(
        simplex, backup, d1, d2, d3, ordered_indices, solution, solution_d):
    if backup:
        _backup_faces(simplex, d1, d2, d3)
    # check vertex 1
    n_simplex_points = 1
    solution.from_vertex(0, d1[0], simplex)
    ordered_indices[0] = 0
    check_line_segment_12 = not (d1[2] <= 0.0 or d2[2] <= 0.0)
    if check_line_segment_12:
        solution_d.from_line_segment(1, 0, d1[2], d2[2], simplex)
        if solution_d.dstsq < solution.dstsq:
            n_simplex_points = 2
            solution.copy_from(solution_d, n_simplex_points)
            ordered_indices[:2] = 0, 1
    check_line_segment_13 = not (d1[4] <= 0.0 or d3[4] <= 0.0)
    if check_line_segment_13:
        solution_d.from_line_segment(2, 0, d1[4], d3[4], simplex)
        if solution_d.dstsq < solution.dstsq:
            n_simplex_points = 2
            solution.copy_from(solution_d, n_simplex_points)
            ordered_indices[:2] = 0, 2
    check_face_123 = not (d1[6] <= 0.0 or d2[6] <= 0.0 or d3[6] <= 0.0)
    if check_face_123:
        solution_d.from_face(2, 0, 1, d1[6], d2[6], d3[6], simplex)
        if solution_d.dstsq < solution.dstsq:
            n_simplex_points = 3
            solution.copy_from(solution_d, n_simplex_points)
            ordered_indices[:3] = 0, 1, 2
    check_vertex_2 = simplex.dot_product_table[1, 1] < solution.dstsq
    if check_vertex_2:
        n_simplex_points = 1
        solution.from_vertex(1, d2[1], simplex)
        ordered_indices[0] = 1
    check_vertex_3 = simplex.dot_product_table[2, 2] < solution.dstsq
    if check_vertex_3:
        n_simplex_points = 1
        solution.from_vertex(2, d3[3], simplex)
        ordered_indices[0] = 2
    check_line_segment_23 = not (d2[5] <= 0.0 or d3[5] <= 0.0)
    if check_line_segment_23:
        solution_d.from_line_segment(2, 1, d2[5], d3[5], simplex, 1, 0)
        if solution_d.dstsq < solution.dstsq:
            n_simplex_points = 2
            solution.copy_from(solution_d, n_simplex_points)
            ordered_indices[:2] = 2, 1
    return n_simplex_points


def _backup_procedure_simplex(
        simplex, backup, d1, d2, d3, d4, ordered_indices, solution, solution_d):
    if backup:
        _backup_simplex(simplex, d1, d2, d3, d4)
    # check vertex 1
    n_simplex_points = 1
    solution.from_vertex(0, d1[0], simplex)
    ordered_indices[0] = 0
    check_line_segment_12 = not (d1[2] <= 0.0 or d2[2] <= 0.0)
    if check_line_segment_12:
        solution_d.from_line_segment(1, 0, d1[2], d2[2], simplex)
        if solution_d.dstsq < solution.dstsq:
            n_simplex_points = 2
            solution.copy_from(solution_d, n_simplex_points)
            ordered_indices[:2] = 0, 1
    check_line_segment_13 = not (d1[4] <= 0.0 or d3[4] <= 0.0)
    if check_line_segment_13:
        solution_d.from_line_segment(2, 0, d1[4], d3[4], simplex)
        if solution_d.dstsq < solution.dstsq:
            n_simplex_points = 2
            solution.copy_from(solution_d, n_simplex_points)
            ordered_indices[:2] = 0, 2
    check_face_123 = not (d1[6] <= 0.0 or d2[6] <= 0.0 or d3[6] <= 0.0)
    if check_face_123:
        solution_d.from_face(2, 0, 1, d1[6], d2[6], d3[6], simplex)
        if solution_d.dstsq < solution.dstsq:
            n_simplex_points = 3
            solution.copy_from(solution_d, n_simplex_points)
            ordered_indices[:3] = 0, 1, 2
    check_line_segment_14 = not (d1[8] <= 0.0 or d4[8] <= 0.0)
    if check_line_segment_14:
        solution_d.from_line_segment(3, 0, d1[8], d4[8], simplex)
        if solution_d.dstsq < solution.dstsq:
            n_simplex_points = 2
            solution.copy_from(solution_d, n_simplex_points)
            ordered_indices[:2] = 0, 3
    check_face_124 = not (d1[11] <= 0.0 or d2[11] <= 0.0 or d4[11] <= 0.0)
    if check_face_124:
        solution_d.from_face(3, 0, 1, d1[11], d2[11], d4[11], simplex)
        if solution_d.dstsq < solution.dstsq:
            n_simplex_points = 3
            solution.copy_from(solution_d, n_simplex_points)
            ordered_indices[:3] = 0, 1, 3
    check_face_134 = not (d1[12] <= 0.0 or d3[12] <= 0.0 or d4[12] <= 0.0)
    if check_face_134:
        solution_d.from_face(3, 0, 2, d1[12], d3[12], d4[12], simplex, 0, 2, 1)
        if solution_d.dstsq < solution.dstsq:
            n_simplex_points = 3
            solution.copy_from(solution_d, n_simplex_points)
            ordered_indices[:3] = 0, 3, 2
    check_convex_hull = not (d1[14] <= 0.0 or d2[14] <= 0.0 or d3[14] <= 0.0 or d4[14] <= 0.0)
    if check_convex_hull:
        solution_d.from_simplex(d1[14], d2[14], d3[14], d4[14], simplex)
        if solution_d.dstsq < solution.dstsq:
            n_simplex_points = 4
            solution.copy_from(solution_d, n_simplex_points)
            ordered_indices[:] = 0, 1, 2, 3
    check_vertex_2 = simplex.dot_product_table[1, 1] < solution.dstsq
    if check_vertex_2:
        n_simplex_points = 1
        solution.from_vertex(1, d2[1], simplex)
        ordered_indices[0] = 1
    check_vertex_3 = simplex.dot_product_table[2, 2] < solution.dstsq
    if check_vertex_3:
        n_simplex_points = 1
        solution.from_vertex(2, d3[3], simplex)
        ordered_indices[0] = 2
    check_vertex_4 = simplex.dot_product_table[3, 3] < solution.dstsq
    if check_vertex_4:
        n_simplex_points = 1
        solution.from_vertex(3, d4[7], simplex)
        ordered_indices[0] = 3
    check_line_segment_23 = not (d2[5] <= 0.0 or d3[5] <= 0.0)
    if check_line_segment_23:
        solution_d.from_line_segment(1, 2, d2[5], d3[5], simplex, 1, 0)
        if solution_d.dstsq < solution.dstsq:
            n_simplex_points = 2
            solution.copy_from(solution_d, n_simplex_points)
            ordered_indices[:2] = 2, 1
    check_line_segment_24 = not (d2[9] <= 0.0 or d4[9] <= 0.0)
    if check_line_segment_24:
        solution_d.from_line_segment(3, 1, d2[9], d4[9], simplex, 1, 0)
        if solution_d.dstsq < solution.dstsq:
            n_simplex_points = 2
            solution.copy_from(solution_d, n_simplex_points)
            ordered_indices[:2] = 3, 1
    check_line_segment_34 = not (d3[10] <= 0.0 or d4[10] <= 0.0)
    if check_line_segment_34:
        solution_d.from_line_segment(3, 2, d3[10], d4[10], simplex)
        if solution_d.dstsq < solution.dstsq:
            n_simplex_points = 2
            solution.copy_from(solution_d, n_simplex_points)
            ordered_indices[:2] = 2, 3
    check_face_234 = not (d2[13] <= 0.0 or d3[13] <= 0.0 or d4[13] <= 0.0)
    if check_face_234:
        solution_d.from_face(3, 1, 2, d2[13], d3[13], d4[13], simplex, 1, 2, 0)
        if solution_d.dstsq < solution.dstsq:
            n_simplex_points = 3
            solution.copy_from(solution_d, n_simplex_points)
            ordered_indices[:3] = 3, 1, 2
    return n_simplex_points


def _backup_line_segments(simplex, d1, d2):
    d2[2] = simplex.dot_product_table[0, 0] - simplex.dot_product_table[1, 0]
    d1[2] = simplex.dot_product_table[1, 1] - simplex.dot_product_table[1, 0]


def _backup_faces(simplex, d1, d2, d3):
    _backup_line_segments(simplex, d1, d2)
    d3[4] = simplex.dot_product_table[0, 0] - simplex.dot_product_table[2, 0]
    e132 = simplex.dot_product_table[1, 0] - simplex.dot_product_table[2, 1]
    d3[6] = d1[2] * d3[4] + d2[2] * e132
    e123 = _compute_face_distances_2(simplex, d1, d2, d3)
    e213 = _compute_face_distances_3(simplex, d1, d2, d3, e123)
    return e132, e123, e213


def _backup_simplex(simplex, d1, d2, d3, d4):
    e132, e123, e213 = _backup_faces(simplex, d1, d2, d3)
    d4[8] = simplex.dot_product_table[0, 0] - simplex.dot_product_table[3, 0]
    e142 = simplex.dot_product_table[1, 0] - simplex.dot_product_table[3, 1]
    d4[11] = d1[2] * d4[8] + d2[2] * e142
    e143 = simplex.dot_product_table[2, 0] - simplex.dot_product_table[3, 2]
    d4[12] = d1[4] * d4[8] + d3[4] * e143
    d4[14] = d1[6] * d4[8] + d2[6] * e142 + d3[6] * e143
    e124, e134 = _compute_simplex_distances_0(simplex, d1, d2, d3, d4)
    e214 = _compute_simplex_distances_1(simplex, d1, d2, d3, d4, e124, e132, e134)
    _compute_simplex_distances_2(simplex, d1, d2, d3, d4, e123, e124, e134)
    _compute_simplex_distances_3(simplex, d1, d2, d3, d4, e213, e214)


def _reorder_simplex_nondecreasing_order(simplex, old_simplex):
    ordered_indices = np.zeros(4, dtype=int)
    ordered_indices[:3] = 0, 1, 2
    if simplex.dot_product_table[2, 0] < simplex.dot_product_table[1, 0]:
        ordered_indices[1] = 2
        ordered_indices[2] = 1
    ii = ordered_indices[1]
    if simplex.dot_product_table[3, 0] < simplex.dot_product_table[ii, 0]:
        ordered_indices[3] = ordered_indices[2]
        ordered_indices[2] = ordered_indices[1]
        ordered_indices[1] = 3
    else:
        ii = ordered_indices[2]
        if simplex.dot_product_table[3, 0] < simplex.dot_product_table[ii, 0]:
            ordered_indices[3] = ordered_indices[2]
            ordered_indices[2] = 3
        else:
            ordered_indices[3] = 3
    # Reorder indices_polytope1, indices_polytope2 simplex and dot_product_table
    for k in range(1, len(simplex)):
        kk = ordered_indices[k]
        simplex.indices_polytope1[k] = old_simplex.indices_polytope1[kk]
        simplex.indices_polytope2[k] = old_simplex.indices_polytope2[kk]
        simplex.simplex[k] = old_simplex.simplex[kk]
        for l in range(k):
            ll = ordered_indices[l]
            if kk >= ll:
                simplex.dot_product_table[k, l] = old_simplex.dot_product_table[kk, ll]
            else:
                simplex.dot_product_table[k, l] = old_simplex.dot_product_table[ll, kk]
        simplex.dot_product_table[k, k] = old_simplex.dot_product_table[kk, kk]


def minkowski_sum(vertices1, vertices2):
    """Minkowski sum of two sets of vertices.

    Parameters
    ----------
    vertices1 : array, shape (n_vertices1, 3)
        First set of vertices.

    vertices2 : array, shape (n_vertices2, 3)
        Second set of vertices.

    Returns
    -------
    ms : array, shape (n_vertices1 * n_vertices2, 3)
        Sums of all pairs of vertices from first and second set.
    """
    return np.array([v1 + v2 for v1 in vertices1 for v2 in vertices2])
