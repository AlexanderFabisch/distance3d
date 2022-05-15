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

    # Initialize simplex to difference of first points of the objects
    ncy = 0

    simplex.initialize_with_point(
        collider1.first_vertex() - collider2.first_vertex())

    solution.barycentric_coordinates[0] = 1.0

    solution.dstsq = simplex.dot_product_table[0, 0] + simplex.dot_product_table[0, 0] + 1.0
    while True:
        ncy += 1

        # Compute point of minimum norm in the convex hull of the simplex
        new_solution, backup = distance_subalgorithm(simplex, solution, backup)

        if new_solution.dstsq >= solution.dstsq or len(simplex) == 4:
            if backup:
                closest_point1 = collider1.compute_point(
                    solution.barycentric_coordinates[:len(simplex)],
                    simplex.indices_polytope1[:len(simplex)])
                closest_point2 = collider2.compute_point(
                    solution.barycentric_coordinates[:len(simplex)],
                    simplex.indices_polytope2[:len(simplex)])

                # Make sure intersection has zero distance
                if len(simplex) == 4:
                    closest_point1[:] = 0.5 * (closest_point1 + closest_point2)
                    closest_point2[:] = closest_point1
                    distance = 0.0
                else:
                    distance = math.sqrt(new_solution.dstsq)

                return distance, closest_point1, closest_point2, simplex.simplex

            backup = True
            if ncy != 1:
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

    def reorder(self, ordered_indices, n_simplex_points):
        indices_polytope1 = np.copy(self.indices_polytope1[:self.n_simplex_points])
        indices_polytope2 = np.copy(self.indices_polytope2[:self.n_simplex_points])
        simplex = np.copy(self.simplex[:self.n_simplex_points])
        dot_product_table = np.empty((4, 4), dtype=float)
        for k in range(self.n_simplex_points):
            dot_product_table[k, :k + 1] = self.dot_product_table[k, :k + 1]
        self.n_simplex_points = n_simplex_points
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

    def search_direction_line_01(self, a):
        return self.simplex[0] + a * (self.simplex[1] - self.simplex[0])

    def search_direction_line_10(self, a):
        return self.simplex[1] + a * (self.simplex[0] - self.simplex[1])

    def search_direction_line_12(self, a):
        return self.simplex[1] + a * (self.simplex[2] - self.simplex[1])

    def search_direction_line_20(self, a):
        return self.simplex[2] + a * (self.simplex[0] - self.simplex[2])

    def search_direction_line_21(self, a):
        return self.simplex[2] + a * (self.simplex[1] - self.simplex[2])

    def search_direction_line_30(self, a):
        return self.simplex[3] + a * (self.simplex[0] - self.simplex[3])

    def search_direction_line_31(self, a):
        return self.simplex[3] + a * (self.simplex[1] - self.simplex[3])

    def search_direction_line_32(self, a):
        return self.simplex[3] + a * (self.simplex[2] - self.simplex[3])

    def search_direction_face_012(self, a, b):
        return (
            self.simplex[0]
            + a * (self.simplex[1] - self.simplex[0])
            + b * (self.simplex[2] - self.simplex[0]))

    def search_direction_face_102(self, a, b):
        return (
            self.simplex[1]
            + a * (self.simplex[0] - self.simplex[1])
            + b * (self.simplex[2] - self.simplex[1]))

    def search_direction_face_201(self, a, b):
        return (
            self.simplex[2]
            + a * (self.simplex[0] - self.simplex[2])
            + b * (self.simplex[1] - self.simplex[2]))

    def search_direction_face_301(self, a, b):
        return (
            self.simplex[3]
            + a * (self.simplex[0] - self.simplex[3])
            + b * (self.simplex[1] - self.simplex[3]))

    def search_direction_face_302(self, a, b):
        return (
            self.simplex[3]
            + a * (self.simplex[0] - self.simplex[3])
            + b * (self.simplex[2] - self.simplex[3]))

    def search_direction_face_312(self, a, b):
        return (
            self.simplex[3]
            + a * (self.simplex[1] - self.simplex[3])
            + b * (self.simplex[2] - self.simplex[3]))

    def search_direction_simplex(self, a, b, c, d):
        return (a * self.simplex[0] + b * self.simplex[1]
                + c * self.simplex[2] + d * self.simplex[3])

    def __len__(self):
        return self.n_simplex_points


def distance_subalgorithm(simplex, solution, backup):
    """Distance subalgorithm.

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
    dstsq : float
        Squared distance.

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
    solution = Solution()
    if len(simplex) == 1:
        solution.from_vertex(0, d1[0], simplex)
        return solution
    elif len(simplex) == 2:
        d2[2] = simplex.dot_product_table[0, 0] - simplex.dot_product_table[1, 0]
        vertex_1_optimal = d2[2] <= 0.0
        if vertex_1_optimal:
            simplex.reduce_to_optimal_vertex(0)
            solution.from_vertex(0, d1[0], simplex)
            return solution
        d1[2] = simplex.dot_product_table[1, 1] - simplex.dot_product_table[1, 0]
        line_segment_12_optimal = not (d1[2] <= 0.0 or d2[2] <= 0.0)
        if line_segment_12_optimal:
            coords_sum = d1[2] + d2[2]
            solution.barycentric_coordinates[0] = d1[2] / coords_sum
            solution.barycentric_coordinates[1] = 1.0 - solution.barycentric_coordinates[0]
            solution.search_direction = simplex.search_direction_line_10(
                solution.barycentric_coordinates[0])
            solution.dstsq = np.dot(solution.search_direction, solution.search_direction)
            return solution
        vertex_2_optimal = d1[2] <= 0.0
        if vertex_2_optimal:
            simplex.reduce_to_optimal_vertex(1)
            solution.from_vertex(0, d2[1], simplex)
            return solution
    elif len(simplex) == 3:
        d2[2] = simplex.dot_product_table[0, 0] - simplex.dot_product_table[1, 0]
        d3[4] = simplex.dot_product_table[0, 0] - simplex.dot_product_table[2, 0]
        vertex_1_optimal = not (d2[2] > 0.0 or d3[4] > 0.0)
        if vertex_1_optimal:
            simplex.reduce_to_optimal_vertex(0)
            solution.from_vertex(0, d1[0], simplex)
            return solution
        e132 = simplex.dot_product_table[1, 0] - simplex.dot_product_table[2, 1]
        d1[2] = simplex.dot_product_table[1, 1] - simplex.dot_product_table[1, 0]
        d3[6] = d1[2] * d3[4] + d2[2] * e132
        line_segment_12_optimal = not (d1[2] <= 0.0 or d2[2] <= 0.0 or d3[6] > 0.0)
        if line_segment_12_optimal:
            simplex.n_simplex_points = 2
            coords_sum = d1[2] + d2[2]
            solution.barycentric_coordinates[0] = d1[2] / coords_sum
            solution.barycentric_coordinates[1] = 1.0 - solution.barycentric_coordinates[0]
            solution.search_direction = simplex.search_direction_line_10(
                solution.barycentric_coordinates[0])
            solution.dstsq = np.dot(solution.search_direction, solution.search_direction)
            return solution
        e123 = simplex.dot_product_table[2, 0] - simplex.dot_product_table[2, 1]
        d1[4] = simplex.dot_product_table[2, 2] - simplex.dot_product_table[2, 0]
        d2[6] = d1[4] * d2[2] + d3[4] * e123
        line_segment_13_optimal = not (d1[4] <= 0.0 or d2[6] > 0.0 or d3[4] <= 0.0)
        if line_segment_13_optimal:
            simplex.n_simplex_points = 2
            simplex.move_vertex(2, 1)
            simplex.dot_product_table[1, 0] = simplex.dot_product_table[2, 0]
            simplex.dot_product_table[1, 1] = simplex.dot_product_table[2, 2]
            coords_sum = d1[4] + d3[4]
            solution.barycentric_coordinates[0] = d1[4] / coords_sum
            solution.barycentric_coordinates[1] = 1.0 - solution.barycentric_coordinates[0]
            solution.search_direction = simplex.search_direction_line_10(
                solution.barycentric_coordinates[0])
            solution.dstsq = np.dot(solution.search_direction, solution.search_direction)
            return solution
        e213 = -e123
        d2[5] = simplex.dot_product_table[2, 2] - simplex.dot_product_table[2, 1]
        d3[5] = simplex.dot_product_table[1, 1] - simplex.dot_product_table[2, 1]
        d1[6] = d2[5] * d1[2] + d3[5] * e213
        face_123_optimal = not (d1[6] <= 0.0 or d2[6] <= 0.0 or d3[6] <= 0.0)
        if face_123_optimal:
            coords_sum = d1[6] + d2[6] + d3[6]
            solution.barycentric_coordinates[0] = d1[6] / coords_sum
            solution.barycentric_coordinates[1] = d2[6] / coords_sum
            solution.barycentric_coordinates[2] = 1.0 - sum(solution.barycentric_coordinates[:2])
            solution.search_direction = simplex.search_direction_face_201(
                solution.barycentric_coordinates[0], solution.barycentric_coordinates[1])
            solution.dstsq = np.dot(solution.search_direction, solution.search_direction)
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
            simplex.n_simplex_points = 2
            simplex.move_vertex(2, 0)
            coords_sum = d2[5] + d3[5]
            solution.barycentric_coordinates[1] = d2[5] / coords_sum
            solution.barycentric_coordinates[0] = 1.0 - solution.barycentric_coordinates[1]
            solution.search_direction = simplex.search_direction_line_01(
                solution.barycentric_coordinates[1])
            simplex.dot_product_table[1, 0] = simplex.dot_product_table[2, 1]
            simplex.dot_product_table[0, 0] = simplex.dot_product_table[2, 2]
            solution.dstsq = np.dot(solution.search_direction, solution.search_direction)
            return solution
    elif len(simplex) == 4:
        d2[2] = simplex.dot_product_table[0, 0] - simplex.dot_product_table[1, 0]
        d3[4] = simplex.dot_product_table[0, 0] - simplex.dot_product_table[2, 0]
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
            coords_sum = d1[2] + d2[2]
            solution.barycentric_coordinates[0] = d1[2] / coords_sum
            solution.barycentric_coordinates[1] = 1.0 - solution.barycentric_coordinates[0]
            solution.search_direction = simplex.search_direction_line_10(
                solution.barycentric_coordinates[0])
            solution.dstsq = np.dot(solution.search_direction, solution.search_direction)
            return solution
        e123 = simplex.dot_product_table[2, 0] - simplex.dot_product_table[2, 1]
        e143 = simplex.dot_product_table[2, 0] - simplex.dot_product_table[3, 2]
        d1[4] = simplex.dot_product_table[2, 2] - simplex.dot_product_table[2, 0]
        d2[6] = d1[4] * d2[2] + d3[4] * e123
        d4[12] = d1[4] * d4[8] + d3[4] * e143
        line_segment_13_optimal = not (d1[4] <= 0.0 or d2[6] > 0.0 or d3[4] <= 0.0 or d4[12] > 0.0)
        if line_segment_13_optimal:
            simplex.n_simplex_points = 2
            simplex.move_vertex(2, 1)
            coords_sum = d1[4] + d3[4]
            solution.barycentric_coordinates[0] = d1[4] / coords_sum
            solution.barycentric_coordinates[1] = 1.0 - solution.barycentric_coordinates[0]
            solution.search_direction = simplex.search_direction_line_10(
                solution.barycentric_coordinates[0])
            simplex.dot_product_table[1, 0] = simplex.dot_product_table[2, 0]
            simplex.dot_product_table[1, 1] = simplex.dot_product_table[2, 2]
            solution.dstsq = np.dot(solution.search_direction, solution.search_direction)
            return solution
        d2[5] = simplex.dot_product_table[2, 2] - simplex.dot_product_table[2, 1]
        d3[5] = simplex.dot_product_table[1, 1] - simplex.dot_product_table[2, 1]
        e213 = -e123
        d1[6] = d2[5] * d1[2] + d3[5] * e213
        d4[14] = d1[6] * d4[8] + d2[6] * e142 + d3[6] * e143
        face_123_optimal = not (d1[6] <= 0.0 or d2[6] <= 0.0 or d3[6] <= 0.0 or d4[14] > 0.0)
        if face_123_optimal:
            simplex.n_simplex_points = 3
            coords_sum = d1[6] + d2[6] + d3[6]
            solution.barycentric_coordinates[0] = d1[6] / coords_sum
            solution.barycentric_coordinates[1] = d2[6] / coords_sum
            solution.barycentric_coordinates[2] = 1.0 - sum(solution.barycentric_coordinates[:2])
            solution.search_direction = simplex.search_direction_face_201(
                solution.barycentric_coordinates[0], solution.barycentric_coordinates[1])
            solution.dstsq = np.dot(solution.search_direction, solution.search_direction)
            return solution
        e124 = simplex.dot_product_table[3, 0] - simplex.dot_product_table[3, 1]
        e134 = simplex.dot_product_table[3, 0] - simplex.dot_product_table[3, 2]
        d1[8] = simplex.dot_product_table[3, 3] - simplex.dot_product_table[3, 0]
        d2[11] = d1[8] * d2[2] + d4[8] * e124
        d3[12] = d1[8] * d3[4] + d4[8] * e134
        line_segment_14_optimal = not (d1[8] <= 0.0 or d2[11] > 0.0 or d3[12] > 0.0 or d4[8] <= 0.0)
        if line_segment_14_optimal:
            simplex.n_simplex_points = 2
            simplex.move_vertex(3, 1)
            coords_sum = d1[8] + d4[8]
            solution.barycentric_coordinates[0] = d1[8] / coords_sum
            solution.barycentric_coordinates[1] = 1.0 - solution.barycentric_coordinates[0]
            solution.search_direction = simplex.search_direction_line_10(
                solution.barycentric_coordinates[0])
            simplex.dot_product_table[1, 0] = simplex.dot_product_table[3, 0]
            simplex.dot_product_table[1, 1] = simplex.dot_product_table[3, 3]
            solution.dstsq = np.dot(solution.search_direction, solution.search_direction)
            return solution
        d2[9] = simplex.dot_product_table[3, 3] - simplex.dot_product_table[3, 1]
        d4[9] = simplex.dot_product_table[1, 1] - simplex.dot_product_table[3, 1]
        e214 = -e124
        d1[11] = d2[9] * d1[2] + d4[9] * e214
        d3[14] = d1[11] * d3[4] + d2[11] * e132 + d4[11] * e134
        face_124_optimal = not (d1[11] <= 0.0 or d2[11] <= 0.0 or d3[14] > 0.0 or d4[11] <= 0.0)
        if face_124_optimal:
            simplex.n_simplex_points = 3
            simplex.move_vertex(3, 2)
            coords_sum = d1[11] + d2[11] + d4[11]
            solution.barycentric_coordinates[0] = d1[11] / coords_sum
            solution.barycentric_coordinates[1] = d2[11] / coords_sum
            solution.barycentric_coordinates[2] = 1.0 - sum(solution.barycentric_coordinates[:2])
            solution.search_direction = simplex.search_direction_face_201(
                solution.barycentric_coordinates[0], solution.barycentric_coordinates[1])
            simplex.dot_product_table[2, 0] = simplex.dot_product_table[3, 0]
            simplex.dot_product_table[2, 1] = simplex.dot_product_table[3, 1]
            simplex.dot_product_table[2, 2] = simplex.dot_product_table[3, 3]
            solution.dstsq = np.dot(solution.search_direction, solution.search_direction)
            return solution
        d3[10] = simplex.dot_product_table[3, 3] - simplex.dot_product_table[3, 2]
        d4[10] = simplex.dot_product_table[2, 2] - simplex.dot_product_table[3, 2]
        e314 = -e134
        d1[12] = d3[10] * d1[4] + d4[10] * e314
        d2[14] = d1[12] * d2[2] + d3[12] * e123 + d4[12] * e124
        face_134_optimal = not (d1[12] <= 0.0 or d2[14] > 0.0 or d3[12] <= 0.0 or d4[12] <= 0.0)
        if face_134_optimal:
            simplex.n_simplex_points = 3
            simplex.move_vertex(3, 1)
            coords_sum = d1[12] + d3[12] + d4[12]
            solution.barycentric_coordinates[0] = d1[12] / coords_sum
            solution.barycentric_coordinates[2] = d3[12] / coords_sum
            solution.barycentric_coordinates[1] = 1.0 - solution.barycentric_coordinates[0] - \
                                                  solution.barycentric_coordinates[2]
            solution.search_direction = simplex.search_direction_face_102(
                solution.barycentric_coordinates[0], solution.barycentric_coordinates[2])
            simplex.dot_product_table[1, 0] = simplex.dot_product_table[3, 0]
            simplex.dot_product_table[1, 1] = simplex.dot_product_table[3, 3]
            simplex.dot_product_table[2, 1] = simplex.dot_product_table[3, 2]
            solution.dstsq = np.dot(solution.search_direction, solution.search_direction)
            return solution
        e243 = simplex.dot_product_table[2, 1] - simplex.dot_product_table[3, 2]
        d4[13] = d2[5] * d4[9] + d3[5] * e243
        e234 = simplex.dot_product_table[3, 1] - simplex.dot_product_table[3, 2]
        d3[13] = d2[9] * d3[5] + d4[9] * e234
        e324 = -e234
        d2[13] = d3[10] * d2[5] + d4[10] * e324
        d1[14] = d2[13] * d1[2] + d3[13] * e213 + d4[13] * e214
        convex_hull_optimal = not (d1[14] <= 0.0 or d2[14] <= 0.0 or d3[14] <= 0.0 or d4[14] <= 0.0)
        if convex_hull_optimal:
            coords_sum = d1[14] + d2[14] + d3[14] + d4[14]
            solution.barycentric_coordinates[0] = d1[14] / coords_sum
            solution.barycentric_coordinates[1] = d2[14] / coords_sum
            solution.barycentric_coordinates[2] = d3[14] / coords_sum
            solution.barycentric_coordinates[3] = 1.0 - sum(solution.barycentric_coordinates[:3])
            solution.search_direction = simplex.search_direction_simplex(
                *(solution.barycentric_coordinates))
            solution.dstsq = np.dot(solution.search_direction, solution.search_direction)
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
            simplex.n_simplex_points = 2
            simplex.move_vertex(2, 0)
            coords_sum = d2[5] + d3[5]
            solution.barycentric_coordinates[1] = d2[5] / coords_sum
            solution.barycentric_coordinates[0] = 1.0 - solution.barycentric_coordinates[1]
            solution.search_direction = simplex.search_direction_line_01(
                solution.barycentric_coordinates[1])
            simplex.dot_product_table[1, 0] = simplex.dot_product_table[2, 1]
            simplex.dot_product_table[0, 0] = simplex.dot_product_table[2, 2]
            solution.dstsq = np.dot(solution.search_direction, solution.search_direction)
            return solution
        line_segment_24_optimal = not (d1[11] > 0.0 or d2[9] <= 0.0 or d3[13] > 0.0 or d4[9] <= 0.0)
        if line_segment_24_optimal:
            simplex.n_simplex_points = 2
            simplex.move_vertex(3, 0)
            coords_sum = d2[9] + d4[9]
            solution.barycentric_coordinates[1] = d2[9] / coords_sum
            solution.barycentric_coordinates[0] = 1.0 - solution.barycentric_coordinates[1]
            solution.search_direction = simplex.search_direction_line_01(
                solution.barycentric_coordinates[1])
            simplex.dot_product_table[1, 0] = simplex.dot_product_table[3, 1]
            simplex.dot_product_table[0, 0] = simplex.dot_product_table[3, 3]
            solution.dstsq = np.dot(solution.search_direction, solution.search_direction)
            return solution
        line_segment_34_optimal = not (d1[12] > 0.0 or d2[13] > 0.0 or d3[10] <= 0.0 or d4[10] <= 0.0)
        if line_segment_34_optimal:
            simplex.n_simplex_points = 2
            simplex.move_vertex(2, 0)
            simplex.move_vertex(3, 1)
            coords_sum = d3[10] + d4[10]
            solution.barycentric_coordinates[0] = d3[10] / coords_sum
            solution.barycentric_coordinates[1] = 1.0 - solution.barycentric_coordinates[0]
            solution.search_direction = simplex.search_direction_line_10(
                solution.barycentric_coordinates[0])
            simplex.dot_product_table[0, 0] = simplex.dot_product_table[2, 2]
            simplex.dot_product_table[1, 0] = simplex.dot_product_table[3, 2]
            simplex.dot_product_table[1, 1] = simplex.dot_product_table[3, 3]
            solution.dstsq = np.dot(solution.search_direction, solution.search_direction)
            return solution
        face_234_optimal = not (d1[14] > 0.0 or d2[13] <= 0.0 or d3[13] <= 0.0 or d4[13] <= 0.0)
        if face_234_optimal:
            simplex.n_simplex_points = 3
            simplex.move_vertex(3, 0)
            coords_sum = d2[13] + d3[13] + d4[13]
            solution.barycentric_coordinates[1] = d2[13] / coords_sum
            solution.barycentric_coordinates[2] = d3[13] / coords_sum
            solution.barycentric_coordinates[0] = 1.0 - sum(solution.barycentric_coordinates[1:3])
            solution.search_direction = simplex.search_direction_face_012(
                solution.barycentric_coordinates[1], solution.barycentric_coordinates[2])
            simplex.dot_product_table[0, 0] = simplex.dot_product_table[3, 3]
            simplex.dot_product_table[1, 0] = simplex.dot_product_table[3, 1]
            simplex.dot_product_table[2, 0] = simplex.dot_product_table[3, 2]
            solution.dstsq = np.dot(solution.search_direction, solution.search_direction)
            return solution
    return None


def _backup_procedure(simplex, solution, d1, d2, d3, d4, backup):
    ordered_indices = np.empty(4, dtype=int)
    solution_d = Solution()
    if len(simplex) == 1:
        solution.barycentric_coordinates[0] = d1[0]
        solution.search_direction = simplex.simplex[0]
        solution.dstsq = simplex.dot_product_table[0, 0]
        return solution, True
    elif len(simplex) == 2:
        if backup:
            d2[2] = simplex.dot_product_table[0, 0] - simplex.dot_product_table[1, 0]
            d1[2] = simplex.dot_product_table[1, 1] - simplex.dot_product_table[1, 0]
        # check vertex 1
        solution.dstsq = simplex.dot_product_table[0, 0]
        n_simplex_points = 1
        solution.barycentric_coordinates[0] = d1[0]
        solution.search_direction = simplex.simplex[0]
        ordered_indices[0] = 0
        check_line_segment_12 = not (d1[2] <= 0.0 or d2[2] <= 0.0)
        if check_line_segment_12:
            coords_sum = d1[2] + d2[2]
            solution_d.barycentric_coordinates[0] = d1[2] / coords_sum
            solution_d.barycentric_coordinates[1] = 1.0 - solution_d.barycentric_coordinates[0]
            solution_d.search_direction = simplex.search_direction_line_10(
                solution_d.barycentric_coordinates[0])
            solution_d.dstsq = np.dot(solution_d.search_direction, solution_d.search_direction)
            if solution_d.dstsq < solution.dstsq:
                solution.dstsq = solution_d.dstsq
                n_simplex_points = 2
                solution.barycentric_coordinates[:n_simplex_points] = solution_d.barycentric_coordinates[
                                                                      :n_simplex_points]
                solution.search_direction = solution_d.search_direction
                ordered_indices[:2] = 0, 1
        check_vertex_2 = simplex.dot_product_table[1, 1] < solution.dstsq
        if check_vertex_2:
            solution.dstsq = simplex.dot_product_table[1, 1]
            n_simplex_points = 1
            solution.barycentric_coordinates[0] = d2[1]
            solution.search_direction = simplex.simplex[1]
            ordered_indices[0] = 1
    elif len(simplex) == 3:
        if backup:
            d2[2] = simplex.dot_product_table[0, 0] - simplex.dot_product_table[1, 0]
            d3[4] = simplex.dot_product_table[0, 0] - simplex.dot_product_table[2, 0]
            e132 = simplex.dot_product_table[1, 0] - simplex.dot_product_table[2, 1]
            d1[2] = simplex.dot_product_table[1, 1] - simplex.dot_product_table[1, 0]
            d3[6] = d1[2] * d3[4] + d2[2] * e132
            e123 = simplex.dot_product_table[2, 0] - simplex.dot_product_table[2, 1]
            d1[4] = simplex.dot_product_table[2, 2] - simplex.dot_product_table[2, 0]
            d2[6] = d1[4] * d2[2] + d3[4] * e123
            e213 = -e123
            d2[5] = simplex.dot_product_table[2, 2] - simplex.dot_product_table[2, 1]
            d3[5] = simplex.dot_product_table[1, 1] - simplex.dot_product_table[2, 1]
            d1[6] = d2[5] * d1[2] + d3[5] * e213
        # check vertex 1
        solution.dstsq = simplex.dot_product_table[0, 0]
        n_simplex_points = 1
        solution.barycentric_coordinates[0] = d1[0]
        solution.search_direction = simplex.simplex[0]
        ordered_indices[0] = 0
        check_line_segment_12 = not (d1[2] <= 0.0 or d2[2] <= 0.0)
        if check_line_segment_12:
            coords_sum = d1[2] + d2[2]
            solution_d.barycentric_coordinates[0] = d1[2] / coords_sum
            solution_d.barycentric_coordinates[1] = 1.0 - solution_d.barycentric_coordinates[0]
            solution_d.search_direction = simplex.search_direction_line_10(solution_d.barycentric_coordinates[0])
            solution_d.dstsq = np.dot(solution_d.search_direction, solution_d.search_direction)
            if solution_d.dstsq < solution.dstsq:
                solution.dstsq = solution_d.dstsq
                n_simplex_points = 2
                solution.barycentric_coordinates[:n_simplex_points] = solution_d.barycentric_coordinates[
                                                                      :n_simplex_points]
                solution.search_direction = solution_d.search_direction
                ordered_indices[:2] = 0, 1
        check_line_segment_13 = not (d1[4] <= 0.0 or d3[4] <= 0.0)
        if check_line_segment_13:
            coords_sum = d1[4] + d3[4]
            solution_d.barycentric_coordinates[0] = d1[4] / coords_sum
            solution_d.barycentric_coordinates[1] = 1.0 - solution_d.barycentric_coordinates[0]
            solution_d.search_direction = simplex.search_direction_line_20(solution_d.barycentric_coordinates[0])
            solution_d.dstsq = np.dot(solution_d.search_direction, solution_d.search_direction)
            if solution_d.dstsq < solution.dstsq:
                solution.dstsq = solution_d.dstsq
                n_simplex_points = 2
                solution.barycentric_coordinates[:n_simplex_points] = solution_d.barycentric_coordinates[
                                                                      :n_simplex_points]
                solution.search_direction = solution_d.search_direction
                ordered_indices[:2] = 0, 2
        check_face_123 = not (d1[6] <= 0.0 or d2[6] <= 0.0 or d3[6] <= 0.0)
        if check_face_123:
            coords_sum = d1[6] + d2[6] + d3[6]
            solution_d.barycentric_coordinates[0] = d1[6] / coords_sum
            solution_d.barycentric_coordinates[1] = d2[6] / coords_sum
            solution_d.barycentric_coordinates[2] = 1.0 - sum(solution_d.barycentric_coordinates[:2])
            solution_d.search_direction = simplex.search_direction_face_201(
                solution_d.barycentric_coordinates[0], solution_d.barycentric_coordinates[1])
            solution_d.dstsq = np.dot(solution_d.search_direction, solution_d.search_direction)
            if solution_d.dstsq < solution.dstsq:
                solution.dstsq = solution_d.dstsq
                n_simplex_points = 3
                solution.barycentric_coordinates[:] = solution_d.barycentric_coordinates
                solution.search_direction = solution_d.search_direction
                ordered_indices[:3] = 0, 1, 2
        check_vertex_2 = simplex.dot_product_table[1, 1] < solution.dstsq
        if check_vertex_2:
            n_simplex_points = 1
            solution.dstsq = simplex.dot_product_table[1, 1]
            solution.barycentric_coordinates[0] = d2[1]
            solution.search_direction = simplex.simplex[1]
            ordered_indices[0] = 1
        check_vertex_3 = simplex.dot_product_table[2, 2] < solution.dstsq
        if check_vertex_3:
            n_simplex_points = 1
            solution.dstsq = simplex.dot_product_table[2, 2]
            solution.barycentric_coordinates[0] = d3[3]
            solution.search_direction = simplex.simplex[2]
            ordered_indices[0] = 2
        check_line_segment_23 = not (d2[5] <= 0.0 or d3[5] <= 0.0)
        if check_line_segment_23:
            coords_sum = d2[5] + d3[5]
            solution_d.barycentric_coordinates[1] = d2[5] / coords_sum
            solution_d.barycentric_coordinates[0] = 1.0 - solution_d.barycentric_coordinates[1]
            solution_d.search_direction = simplex.search_direction_line_21(solution_d.barycentric_coordinates[1])
            solution_d.dstsq = np.dot(solution_d.search_direction, solution_d.search_direction)
            if solution_d.dstsq < solution.dstsq:
                solution.dstsq = solution_d.dstsq
                n_simplex_points = 2
                solution.barycentric_coordinates[:n_simplex_points] = solution_d.barycentric_coordinates[
                                                                      :n_simplex_points]
                solution.search_direction = solution_d.search_direction
                ordered_indices[:2] = 2, 1
    elif len(simplex) == 4:
        if backup:
            d2[2] = simplex.dot_product_table[0, 0] - simplex.dot_product_table[1, 0]
            d3[4] = simplex.dot_product_table[0, 0] - simplex.dot_product_table[2, 0]
            d4[8] = simplex.dot_product_table[0, 0] - simplex.dot_product_table[3, 0]
            e132 = simplex.dot_product_table[1, 0] - simplex.dot_product_table[2, 1]
            e142 = simplex.dot_product_table[1, 0] - simplex.dot_product_table[3, 1]
            d1[2] = simplex.dot_product_table[1, 1] - simplex.dot_product_table[1, 0]
            d3[6] = d1[2] * d3[4] + d2[2] * e132
            d4[11] = d1[2] * d4[8] + d2[2] * e142
            e123 = simplex.dot_product_table[2, 0] - simplex.dot_product_table[2, 1]
            e143 = simplex.dot_product_table[2, 0] - simplex.dot_product_table[3, 2]
            d1[4] = simplex.dot_product_table[2, 2] - simplex.dot_product_table[2, 0]
            d2[6] = d1[4] * d2[2] + d3[4] * e123
            d4[12] = d1[4] * d4[8] + d3[4] * e143
            d2[5] = simplex.dot_product_table[2, 2] - simplex.dot_product_table[2, 1]
            d3[5] = simplex.dot_product_table[1, 1] - simplex.dot_product_table[2, 1]
            e213 = -e123
            d1[6] = d2[5] * d1[2] + d3[5] * e213
            d4[14] = d1[6] * d4[8] + d2[6] * e142 + d3[6] * e143
            e124 = simplex.dot_product_table[3, 0] - simplex.dot_product_table[3, 1]
            e134 = simplex.dot_product_table[3, 0] - simplex.dot_product_table[3, 2]
            d1[8] = simplex.dot_product_table[3, 3] - simplex.dot_product_table[3, 0]
            d2[11] = d1[8] * d2[2] + d4[8] * e124
            d3[12] = d1[8] * d3[4] + d4[8] * e134
            d2[9] = simplex.dot_product_table[3, 3] - simplex.dot_product_table[3, 1]
            d4[9] = simplex.dot_product_table[1, 1] - simplex.dot_product_table[3, 1]
            e214 = -e124
            d1[11] = d2[9] * d1[2] + d4[9] * e214
            d3[14] = d1[11] * d3[4] + d2[11] * e132 + d4[11] * e134
            d3[10] = simplex.dot_product_table[3, 3] - simplex.dot_product_table[3, 2]
            d4[10] = simplex.dot_product_table[2, 2] - simplex.dot_product_table[3, 2]
            e314 = -e134
            d1[12] = d3[10] * d1[4] + d4[10] * e314
            d2[14] = d1[12] * d2[2] + d3[12] * e123 + d4[12] * e124
            e243 = simplex.dot_product_table[2, 1] - simplex.dot_product_table[3, 2]
            d4[13] = d2[5] * d4[9] + d3[5] * e243
            e234 = simplex.dot_product_table[3, 1] - simplex.dot_product_table[3, 2]
            d3[13] = d2[9] * d3[5] + d4[9] * e234
            e324 = -e234
            d2[13] = d3[10] * d2[5] + d4[10] * e324
            d1[14] = d2[13] * d1[2] + d3[13] * e213 + d4[13] * e214
        # check vertex 1
        solution.dstsq = simplex.dot_product_table[0, 0]
        n_simplex_points = 1
        solution.barycentric_coordinates[0] = d1[0]
        solution.search_direction = simplex.simplex[0]
        ordered_indices[0] = 0
        check_line_segment_12 = not (d1[2] <= 0.0 or d2[2] <= 0.0)
        if check_line_segment_12:
            coords_sum = d1[2] + d2[2]
            solution_d.barycentric_coordinates[0] = d1[2] / coords_sum
            solution_d.barycentric_coordinates[1] = 1.0 - solution_d.barycentric_coordinates[0]
            solution_d.search_direction = simplex.search_direction_line_10(solution_d.barycentric_coordinates[0])
            solution_d.dstsq = np.dot(solution_d.search_direction, solution_d.search_direction)
            if solution_d.dstsq < solution.dstsq:
                solution.dstsq = solution_d.dstsq
                n_simplex_points = 2
                solution.barycentric_coordinates[:n_simplex_points] = solution_d.barycentric_coordinates[
                                                                      :n_simplex_points]
                solution.search_direction = solution_d.search_direction
                ordered_indices[:2] = 0, 1
        check_line_segment_13 = not (d1[4] <= 0.0 or d3[4] <= 0.0)
        if check_line_segment_13:
            coords_sum = d1[4] + d3[4]
            solution_d.barycentric_coordinates[0] = d1[4] / coords_sum
            solution_d.barycentric_coordinates[1] = 1.0 - solution_d.barycentric_coordinates[0]
            solution_d.search_direction = simplex.search_direction_line_20(solution_d.barycentric_coordinates[0])
            solution_d.dstsq = np.dot(solution_d.search_direction, solution_d.search_direction)
            if solution_d.dstsq < solution.dstsq:
                solution.dstsq = solution_d.dstsq
                n_simplex_points = 2
                solution.barycentric_coordinates[:n_simplex_points] = solution_d.barycentric_coordinates[
                                                                      :n_simplex_points]
                solution.search_direction = solution_d.search_direction
                ordered_indices[:2] = 0, 2
        check_face_123 = not (d1[6] <= 0.0 or d2[6] <= 0.0 or d3[6] <= 0.0)
        if check_face_123:
            coords_sum = d1[6] + d2[6] + d3[6]
            solution_d.barycentric_coordinates[0] = d1[6] / coords_sum
            solution_d.barycentric_coordinates[1] = d2[6] / coords_sum
            solution_d.barycentric_coordinates[2] = 1.0 - sum(solution_d.barycentric_coordinates[:2])
            solution_d.search_direction = simplex.search_direction_face_201(
                solution_d.barycentric_coordinates[0], solution_d.barycentric_coordinates[1])
            solution_d.dstsq = np.dot(solution_d.search_direction, solution_d.search_direction)
            if solution_d.dstsq < solution.dstsq:
                solution.dstsq = solution_d.dstsq
                n_simplex_points = 3
                solution.barycentric_coordinates[:] = solution_d.barycentric_coordinates
                solution.search_direction = solution_d.search_direction
                ordered_indices[:3] = 0, 1, 2
        check_line_segment_14 = not (d1[8] <= 0.0 or d4[8] <= 0.0)
        if check_line_segment_14:
            coords_sum = d1[8] + d4[8]
            solution_d.barycentric_coordinates[0] = d1[8] / coords_sum
            solution_d.barycentric_coordinates[1] = 1.0 - solution_d.barycentric_coordinates[0]
            solution_d.search_direction = simplex.search_direction_line_30(solution_d.barycentric_coordinates[0])
            solution_d.dstsq = np.dot(solution_d.search_direction, solution_d.search_direction)
            if solution_d.dstsq < solution.dstsq:
                solution.dstsq = solution_d.dstsq
                n_simplex_points = 2
                solution.barycentric_coordinates[:n_simplex_points] = solution_d.barycentric_coordinates[
                                                                      :n_simplex_points]
                solution.search_direction = solution_d.search_direction
                ordered_indices[:2] = 0, 3
        check_face_124 = not (d1[11] <= 0.0 or d2[11] <= 0.0 or d4[11] <= 0.0)
        if check_face_124:
            coords_sum = d1[11] + d2[11] + d4[11]
            solution_d.barycentric_coordinates[0] = d1[11] / coords_sum
            solution_d.barycentric_coordinates[1] = d2[11] / coords_sum
            solution_d.barycentric_coordinates[2] = 1.0 - sum(solution_d.barycentric_coordinates[:2])
            solution_d.search_direction = simplex.search_direction_face_301(solution_d.barycentric_coordinates[0],
                                                                               solution_d.barycentric_coordinates[1])
            solution_d.dstsq = np.dot(solution_d.search_direction, solution_d.search_direction)
            if solution_d.dstsq < solution.dstsq:
                solution.dstsq = solution_d.dstsq
                n_simplex_points = 3
                solution.barycentric_coordinates[:n_simplex_points] = solution_d.barycentric_coordinates[
                                                                      :n_simplex_points]
                solution.search_direction = solution_d.search_direction
                ordered_indices[:3] = 0, 1, 3
        check_face_134 = not (d1[12] <= 0.0 or d3[12] <= 0.0 or d4[12] <= 0.0)
        if check_face_134:
            coords_sum = d1[12] + d3[12] + d4[12]
            solution_d.barycentric_coordinates[0] = d1[12] / coords_sum
            solution_d.barycentric_coordinates[2] = d3[12] / coords_sum
            solution_d.barycentric_coordinates[1] = 1.0 - solution_d.barycentric_coordinates[0] - \
                                                    solution_d.barycentric_coordinates[2]
            solution_d.search_direction = simplex.search_direction_face_302(solution_d.barycentric_coordinates[0],
                                                                               solution_d.barycentric_coordinates[2])
            solution_d.dstsq = np.dot(solution_d.search_direction, solution_d.search_direction)
            if solution_d.dstsq < solution.dstsq:
                solution.dstsq = solution_d.dstsq
                n_simplex_points = 3
                solution.barycentric_coordinates[:n_simplex_points] = solution_d.barycentric_coordinates[
                                                                      :n_simplex_points]
                solution.search_direction = solution_d.search_direction
                ordered_indices[:3] = 0, 3, 2
        check_hull = not (d1[14] <= 0.0 or d2[14] <= 0.0 or d3[14] <= 0.0 or d4[14] <= 0.0)
        if check_hull:
            coords_sum = d1[14] + d2[14] + d3[14] + d4[14]
            solution_d.barycentric_coordinates[0] = d1[14] / coords_sum
            solution_d.barycentric_coordinates[1] = d2[14] / coords_sum
            solution_d.barycentric_coordinates[2] = d3[14] / coords_sum
            solution_d.barycentric_coordinates[3] = 1.0 - sum(solution_d.barycentric_coordinates[:3])
            solution_d.search_direction = simplex.search_direction_simplex(*(solution_d.barycentric_coordinates))
            solution_d.dstsq = np.dot(solution_d.search_direction, solution_d.search_direction)
            if solution_d.dstsq < solution.dstsq:
                solution.dstsq = solution_d.dstsq
                n_simplex_points = 4
                solution.barycentric_coordinates[:] = solution_d.barycentric_coordinates
                solution.search_direction = solution_d.search_direction
                ordered_indices[:] = 0, 1, 2, 3
        check_vertex_2 = simplex.dot_product_table[1, 1] < solution.dstsq
        if check_vertex_2:
            n_simplex_points = 1
            solution.dstsq = simplex.dot_product_table[1, 1]
            solution.barycentric_coordinates[0] = d2[1]
            solution.search_direction = simplex.simplex[1]
            ordered_indices[0] = 1
        check_vertex_3 = simplex.dot_product_table[2, 2] < solution.dstsq
        if check_vertex_3:
            n_simplex_points = 1
            solution.dstsq = simplex.dot_product_table[2, 2]
            solution.barycentric_coordinates[0] = d3[3]
            solution.search_direction = simplex.simplex[2]
            ordered_indices[0] = 2
        check_vertex_4 = simplex.dot_product_table[3, 3] < solution.dstsq
        if check_vertex_4:
            n_simplex_points = 1
            solution.dstsq = simplex.dot_product_table[3, 3]
            solution.barycentric_coordinates[0] = d4[7]
            solution.search_direction = simplex.simplex[3]
            ordered_indices[0] = 3
        check_line_segment_23 = not (d2[5] <= 0.0 or d3[5] <= 0.0)
        if check_line_segment_23:
            coords_sum = d2[5] + d3[5]
            solution_d.barycentric_coordinates[1] = d2[5] / coords_sum
            solution_d.barycentric_coordinates[0] = 1.0 - solution_d.barycentric_coordinates[1]
            solution_d.search_direction = simplex.search_direction_line_12(solution_d.barycentric_coordinates[1])
            solution_d.dstsq = np.dot(solution_d.search_direction, solution_d.search_direction)
            if solution_d.dstsq < solution.dstsq:
                solution.dstsq = solution_d.dstsq
                n_simplex_points = 2
                solution.barycentric_coordinates[:n_simplex_points] = solution_d.barycentric_coordinates[
                                                                      :n_simplex_points]
                solution.search_direction = solution_d.search_direction
                ordered_indices[:2] = 2, 1
        check_line_segment_24 = not (d2[9] <= 0.0 or d4[9] <= 0.0)
        if check_line_segment_24:
            coords_sum = d2[9] + d4[9]
            solution_d.barycentric_coordinates[1] = d2[9] / coords_sum
            solution_d.barycentric_coordinates[0] = 1.0 - solution_d.barycentric_coordinates[1]
            solution_d.search_direction = simplex.search_direction_line_31(solution_d.barycentric_coordinates[1])
            solution_d.dstsq = np.dot(solution_d.search_direction, solution_d.search_direction)
            if solution_d.dstsq < solution.dstsq:
                solution.dstsq = solution_d.dstsq
                n_simplex_points = 2
                solution.barycentric_coordinates[:n_simplex_points] = solution_d.barycentric_coordinates[
                                                                      :n_simplex_points]
                solution.search_direction = solution_d.search_direction
                ordered_indices[:2] = 3, 1
        check_line_segment_34 = not (d3[10] <= 0.0 or d4[10] <= 0.0)
        if check_line_segment_34:
            coords_sum = d3[10] + d4[10]
            solution_d.barycentric_coordinates[0] = d3[10] / coords_sum
            solution_d.barycentric_coordinates[1] = 1.0 - solution_d.barycentric_coordinates[0]
            solution_d.search_direction = simplex.search_direction_line_32(solution_d.barycentric_coordinates[0])
            solution_d.dstsq = np.dot(solution_d.search_direction, solution_d.search_direction)
            if solution_d.dstsq < solution.dstsq:
                solution.dstsq = solution_d.dstsq
                n_simplex_points = 2
                solution.barycentric_coordinates[:n_simplex_points] = solution_d.barycentric_coordinates[
                                                                      :n_simplex_points]
                solution.search_direction = solution_d.search_direction
                ordered_indices[:2] = 2, 3
        check_face_234 = not (d2[13] <= 0.0 or d3[13] <= 0.0 or d4[13] <= 0.0)
        if check_face_234:
            coords_sum = d2[13] + d3[13] + d4[13]
            solution_d.barycentric_coordinates[1] = d2[13] / coords_sum
            solution_d.barycentric_coordinates[2] = d3[13] / coords_sum
            solution_d.barycentric_coordinates[0] = 1.0 - sum(solution_d.barycentric_coordinates[1:3])
            solution_d.search_direction = simplex.search_direction_face_312(
                solution_d.barycentric_coordinates[1], solution_d.barycentric_coordinates[2])
            solution_d.dstsq = np.dot(solution_d.search_direction, solution_d.search_direction)
            if solution_d.dstsq < solution.dstsq:
                solution.dstsq = solution_d.dstsq
                n_simplex_points = 3
                solution.barycentric_coordinates[:n_simplex_points] = solution_d.barycentric_coordinates[
                                                                      :n_simplex_points]
                solution.search_direction = solution_d.search_direction
                ordered_indices[:3] = 3, 1, 2

    simplex.reorder(ordered_indices, n_simplex_points)
    return solution, True


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
