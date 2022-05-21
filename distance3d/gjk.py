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

    simplex.initialize_with_point(
        collider1.first_vertex() - collider2.first_vertex())
    solution.initialize(simplex)

    iteration = 0
    backup = False
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
            else:
                backup = True
                if iteration != 1:
                    simplex.copy_from(old_simplex)
        else:
            solution = new_solution

            _find_new_supporting_point(collider1, collider2, simplex, solution)

            old_simplex.copy_from(simplex)
            if len(simplex) == 4:
                _reorder_simplex_nondecreasing_order(simplex, old_simplex)


def _find_new_supporting_point(collider1, collider2, simplex, solution):
    new_index1, new_vertex1 = collider1.support_function(-solution.search_direction)
    new_index2, new_vertex2 = collider2.support_function(solution.search_direction)
    new_simplex_point = new_vertex1 - new_vertex2
    simplex.add_new_point(new_index1, new_index2, new_simplex_point)


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

    def from_vertex(self, simplex, vertex_idx, a):
        self.barycentric_coordinates[vertex_idx] = a
        self.search_direction = simplex.simplex[vertex_idx]
        self.dstsq = simplex.dot_product_table[vertex_idx, vertex_idx]

    def from_line_segment(self, simplex, vi1, vi2, a, b, bci1=0, bci2=1):
        coords_sum = a + b
        self.barycentric_coordinates[bci1] = a / coords_sum
        self.barycentric_coordinates[bci2] = 1.0 - self.barycentric_coordinates[bci1]
        self.search_direction = simplex.search_direction_line(
            vi1, vi2, self.barycentric_coordinates[bci1])
        self.dstsq = np.dot(self.search_direction, self.search_direction)

    def from_face(self, simplex, vi1, vi2, vi3, a, b, c, bci1=0, bci2=1, bci3=2):
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

    def from_simplex(self, simplex, a, b, c, d):
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

    def initialize(self, simplex):
        self.barycentric_coordinates[0] = 1.0
        self.dstsq = simplex.dot_product_table[0, 0] + simplex.dot_product_table[0, 0] + 1.0


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
    d = BarycentricCoordinates()

    if not backup:
        new_solution = _regular_distance_subalgorithm(simplex, d)
        if new_solution is not None:
            return new_solution, backup

    return _backup_procedure(simplex, solution, d, backup)


class BarycentricCoordinates:
    """Barycentric coordinates.

    Attributes
    ----------
    d : array, shape (4, 15)
        All barycentric coordinates of all vertices in all 15 cases. Each row
        corresponds to one vertex. The column index identifies one of the 15
        cases:
        * 0: vertex 0
        * 1: vertex 1
        * 2: line segment 0-1
        * 3: vertex 3
        * 4: line segment 0-2
        * 5: line segment 1-2
        * 6: face 0-1-2
        * 7: vertex 4
        * 8: line segment 0-3
        * 9: line segment 1-3
        * 10: line segment 2-3
        * 11: face 0-1-3
        * 12: face 0-2-3
        * 13: face 1-2-3
        * 14: simplex 0-1-2-3
    """
    def __init__(self):
        self.d = np.empty((4, 15), dtype=float)
        self.d[0, 0] = 1.0
        self.d[1, 1] = 1.0
        self.d[2, 3] = 1.0
        self.d[3, 7] = 1.0

    def face_coordinates_0(self, simplex):
        self.d[1, 2] = simplex.dot_product_table[0, 0] - simplex.dot_product_table[1, 0]
        self.d[2, 4] = simplex.dot_product_table[0, 0] - simplex.dot_product_table[2, 0]

    def face_coordinates_1(self, simplex):
        e132 = simplex.dot_product_table[1, 0] - simplex.dot_product_table[2, 1]
        self.d[0, 2] = simplex.dot_product_table[1, 1] - simplex.dot_product_table[1, 0]
        self.d[2, 6] = self.d[0, 2] * self.d[2, 4] + self.d[1, 2] * e132

    def face_coordinates_2(self, simplex):
        e123 = simplex.dot_product_table[2, 0] - simplex.dot_product_table[2, 1]
        self.d[0, 4] = simplex.dot_product_table[2, 2] - simplex.dot_product_table[2, 0]
        self.d[1, 6] = self.d[0, 4] * self.d[1, 2] + self.d[2, 4] * e123
        return e123

    def face_coordinates_3(self, simplex, e123):
        e213 = -e123
        self.d[1, 5] = simplex.dot_product_table[2, 2] - simplex.dot_product_table[2, 1]
        self.d[2, 5] = simplex.dot_product_table[1, 1] - simplex.dot_product_table[2, 1]
        self.d[0, 6] = self.d[1, 5] * self.d[0, 2] + self.d[2, 5] * e213
        return e213

    def compute_simplex_distances_0(self, simplex):
        e124 = simplex.dot_product_table[3, 0] - simplex.dot_product_table[3, 1]
        e134 = simplex.dot_product_table[3, 0] - simplex.dot_product_table[3, 2]
        self.d[0, 8] = simplex.dot_product_table[3, 3] - simplex.dot_product_table[3, 0]
        self.d[1, 11] = self.d[0, 8] * self.d[1, 2] + self.d[3, 8] * e124
        self.d[2, 12] = self.d[0, 8] * self.d[2, 4] + self.d[3, 8] * e134
        return e124, e134

    def compute_simplex_distances_1(self, simplex, e124, e132, e134):
        self.d[1, 9] = simplex.dot_product_table[3, 3] - simplex.dot_product_table[3, 1]
        self.d[3, 9] = simplex.dot_product_table[1, 1] - simplex.dot_product_table[3, 1]
        e214 = -e124
        self.d[0, 11] = self.d[1, 9] * self.d[0, 2] + self.d[3, 9] * e214
        self.d[2, 14] = self.d[0, 11] * self.d[2, 4] + self.d[1, 11] * e132 + self.d[3, 11] * e134
        return e214

    def compute_simplex_distances_2(self, simplex, e123, e124, e134):
        self.d[2, 10] = simplex.dot_product_table[3, 3] - simplex.dot_product_table[3, 2]
        self.d[3, 10] = simplex.dot_product_table[2, 2] - simplex.dot_product_table[3, 2]
        e314 = -e134
        self.d[0, 12] = self.d[2, 10] * self.d[0, 4] + self.d[3, 10] * e314
        self.d[1, 14] = self.d[0, 12] * self.d[1, 2] + self.d[2, 12] * e123 + self.d[3, 12] * e124

    def compute_simplex_distances_3(self, simplex, e213, e214):
        e243 = simplex.dot_product_table[2, 1] - simplex.dot_product_table[3, 2]
        self.d[3, 13] = self.d[1, 5] * self.d[3, 9] + self.d[2, 5] * e243
        e234 = simplex.dot_product_table[3, 1] - simplex.dot_product_table[3, 2]
        self.d[2, 13] = self.d[1, 9] * self.d[2, 5] + self.d[3, 9] * e234
        e324 = -e234
        self.d[1, 13] = self.d[2, 10] * self.d[1, 5] + self.d[3, 10] * e324
        self.d[0, 14] = self.d[1, 13] * self.d[0, 2] + self.d[2, 13] * e213 + self.d[3, 13] * e214

    def backup_line_segments(self, simplex):
        self.d[1, 2] = simplex.dot_product_table[0, 0] - simplex.dot_product_table[1, 0]
        self.d[0, 2] = simplex.dot_product_table[1, 1] - simplex.dot_product_table[1, 0]

    def backup_faces(self, simplex):
        self.backup_line_segments(simplex)
        self.d[2, 4] = simplex.dot_product_table[0, 0] - simplex.dot_product_table[2, 0]
        e132 = simplex.dot_product_table[1, 0] - simplex.dot_product_table[2, 1]
        self.d[2, 6] = self.d[0, 2] * self.d[2, 4] + self.d[1, 2] * e132
        e123 = self.face_coordinates_2(simplex)
        e213 = self.face_coordinates_3(simplex, e123)
        return e132, e123, e213

    def backup_simplex(self, simplex):
        e132, e123, e213 = self.backup_faces(simplex)
        self.d[3, 8] = simplex.dot_product_table[0, 0] - simplex.dot_product_table[3, 0]
        e142 = simplex.dot_product_table[1, 0] - simplex.dot_product_table[3, 1]
        self.d[3, 11] = self.d[0, 2] * self.d[3, 8] + self.d[1, 2] * e142
        e143 = simplex.dot_product_table[2, 0] - simplex.dot_product_table[3, 2]
        self.d[3, 12] = self.d[0, 4] * self.d[3, 8] + self.d[2, 4] * e143
        self.d[3, 14] = self.d[0, 6] * self.d[3, 8] + self.d[1, 6] * e142 + self.d[2, 6] * e143
        e124, e134 = self.compute_simplex_distances_0(simplex)
        e214 = self.compute_simplex_distances_1(simplex, e124, e132, e134)
        self.compute_simplex_distances_2(simplex, e123, e124, e134)
        self.compute_simplex_distances_3(simplex, e213, e214)


def _regular_distance_subalgorithm(simplex, d):
    if len(simplex) == 1:
        solution = Solution()
        solution.from_vertex(simplex, 0, d.d[0, 0])
        return solution
    elif len(simplex) == 2:
        return _distance_subalgorithm_line_segment(simplex, d)
    elif len(simplex) == 3:
        return _distance_subalgorithm_face(simplex, d)
    elif len(simplex) == 4:
        return _distance_subalgorithm_simplex(simplex, d)


def _distance_subalgorithm_line_segment(simplex, d):
    solution = Solution()
    d.d[1, 2] = simplex.dot_product_table[0, 0] - simplex.dot_product_table[1, 0]
    vertex_1_optimal = d.d[1, 2] <= 0.0
    if vertex_1_optimal:
        simplex.reduce_to_optimal_vertex(0)
        solution.from_vertex(simplex, 0, d.d[0, 0])
        return solution
    d.d[0, 2] = simplex.dot_product_table[1, 1] - simplex.dot_product_table[1, 0]
    line_segment_12_optimal = not (d.d[0, 2] <= 0.0 or d.d[1, 2] <= 0.0)
    if line_segment_12_optimal:
        solution.from_line_segment(simplex, 1, 0, d.d[0, 2], d.d[1, 2])
        return solution
    vertex_2_optimal = d.d[0, 2] <= 0.0
    if vertex_2_optimal:
        simplex.reduce_to_optimal_vertex(1)
        solution.from_vertex(simplex, 0, d.d[1, 1])
        return solution
    return None


def _distance_subalgorithm_face(simplex, d):
    solution = Solution()
    d.face_coordinates_0(simplex)
    vertex_1_optimal = not (d.d[1, 2] > 0.0 or d.d[2, 4] > 0.0)
    if vertex_1_optimal:
        simplex.reduce_to_optimal_vertex(0)
        solution.from_vertex(simplex, 0, d.d[0, 0])
        return solution
    d.face_coordinates_1(simplex)
    line_segment_12_optimal = not (d.d[0, 2] <= 0.0 or d.d[1, 2] <= 0.0 or d.d[2, 6] > 0.0)
    if line_segment_12_optimal:
        simplex.n_simplex_points = 2
        solution.from_line_segment(simplex, 1, 0, d.d[0, 2], d.d[1, 2])
        return solution
    e123 = d.face_coordinates_2(simplex)
    line_segment_13_optimal = not (d.d[0, 4] <= 0.0 or d.d[1, 6] > 0.0 or d.d[2, 4] <= 0.0)
    if line_segment_13_optimal:
        simplex.select_line_segment_13()
        solution.from_line_segment(simplex, 1, 0, d.d[0, 4], d.d[2, 4])
        return solution
    d.face_coordinates_3(simplex, e123)
    face_123_optimal = not (d.d[0, 6] <= 0.0 or d.d[1, 6] <= 0.0 or d.d[2, 6] <= 0.0)
    if face_123_optimal:
        solution.from_face(simplex, 2, 0, 1, d.d[0, 6], d.d[1, 6], d.d[2, 6])
        return solution
    vertex_2_optimal = not (d.d[0, 2] > 0.0 or d.d[2, 5] > 0.0)
    if vertex_2_optimal:
        simplex.reduce_to_optimal_vertex(1)
        solution.from_vertex(simplex, 0, d.d[1, 1])
        return solution
    vertex_3_optimal = not (d.d[0, 4] > 0.0 or d.d[1, 5] > 0.0)
    if vertex_3_optimal:
        simplex.reduce_to_optimal_vertex(2)
        solution.from_vertex(simplex, 0, d.d[2, 3])
        return solution
    line_segment_23_optimal = not (d.d[0, 6] > 0.0 or d.d[1, 5] <= 0.0 or d.d[2, 5] <= 0.0)
    if line_segment_23_optimal:
        simplex.select_line_segment_23()
        solution.from_line_segment(simplex, 0, 1, d.d[1, 5], d.d[2, 5])
        return solution
    return None


def _distance_subalgorithm_simplex(simplex, d):
    solution = Solution()
    d.face_coordinates_0(simplex)
    d.d[3, 8] = simplex.dot_product_table[0, 0] - simplex.dot_product_table[3, 0]
    vertex_1_optimal = not (d.d[1, 2] > 0.0 or d.d[2, 4] > 0.0 or d.d[3, 8] > 0.0)
    if vertex_1_optimal:
        simplex.reduce_to_optimal_vertex(0)
        solution.from_vertex(simplex, 0, d.d[0, 0])
        return solution
    e132 = simplex.dot_product_table[1, 0] - simplex.dot_product_table[2, 1]
    e142 = simplex.dot_product_table[1, 0] - simplex.dot_product_table[3, 1]
    d.d[0, 2] = simplex.dot_product_table[1, 1] - simplex.dot_product_table[1, 0]
    d.d[2, 6] = d.d[0, 2] * d.d[2, 4] + d.d[1, 2] * e132
    d.d[3, 11] = d.d[0, 2] * d.d[3, 8] + d.d[1, 2] * e142
    line_segment_12_optimal = not (d.d[0, 2] <= 0.0 or d.d[1, 2] <= 0.0 or d.d[2, 6] > 0.0 or d.d[3, 11] > 0.0)
    if line_segment_12_optimal:
        simplex.n_simplex_points = 2
        solution.from_line_segment(simplex, 1, 0, d.d[0, 2], d.d[1, 2])
        return solution
    e123 = simplex.dot_product_table[2, 0] - simplex.dot_product_table[2, 1]
    e143 = simplex.dot_product_table[2, 0] - simplex.dot_product_table[3, 2]
    d.d[0, 4] = simplex.dot_product_table[2, 2] - simplex.dot_product_table[2, 0]
    d.d[1, 6] = d.d[0, 4] * d.d[1, 2] + d.d[2, 4] * e123
    d.d[3, 12] = d.d[0, 4] * d.d[3, 8] + d.d[2, 4] * e143
    line_segment_13_optimal = not (d.d[0, 4] <= 0.0 or d.d[1, 6] > 0.0 or d.d[2, 4] <= 0.0 or d.d[3, 12] > 0.0)
    if line_segment_13_optimal:
        simplex.select_line_segment_13()
        solution.from_line_segment(simplex, 1, 0, d.d[0, 4], d.d[2, 4])
        return solution
    d.d[1, 5] = simplex.dot_product_table[2, 2] - simplex.dot_product_table[2, 1]
    d.d[2, 5] = simplex.dot_product_table[1, 1] - simplex.dot_product_table[2, 1]
    e213 = -e123
    d.d[0, 6] = d.d[1, 5] * d.d[0, 2] + d.d[2, 5] * e213
    d.d[3, 14] = d.d[0, 6] * d.d[3, 8] + d.d[1, 6] * e142 + d.d[2, 6] * e143
    face_123_optimal = not (d.d[0, 6] <= 0.0 or d.d[1, 6] <= 0.0 or d.d[2, 6] <= 0.0 or d.d[3, 14] > 0.0)
    if face_123_optimal:
        simplex.n_simplex_points = 3
        solution.from_face(simplex, 2, 0, 1, d.d[0, 6], d.d[1, 6], d.d[2, 6])
        return solution
    e124, e134 = d.compute_simplex_distances_0(simplex)
    line_segment_14_optimal = not (d.d[0, 8] <= 0.0 or d.d[1, 11] > 0.0 or d.d[2, 12] > 0.0 or d.d[3, 8] <= 0.0)
    if line_segment_14_optimal:
        simplex.select_line_segment_14()
        solution.from_line_segment(simplex, 1, 0, d.d[0, 8], d.d[3, 8])
        return solution
    e214 = d.compute_simplex_distances_1(simplex, e124, e132, e134)
    face_124_optimal = not (d.d[0, 11] <= 0.0 or d.d[1, 11] <= 0.0 or d.d[2, 14] > 0.0 or d.d[3, 11] <= 0.0)
    if face_124_optimal:
        simplex.select_face_124()
        solution.from_face(simplex, 2, 0, 1, d.d[0, 11], d.d[1, 11], d.d[3, 11])
        return solution
    d.compute_simplex_distances_2(simplex, e123, e124, e134)
    face_134_optimal = not (d.d[0, 12] <= 0.0 or d.d[1, 14] > 0.0 or d.d[2, 12] <= 0.0 or d.d[3, 12] <= 0.0)
    if face_134_optimal:
        simplex.select_face_134()
        solution.from_face(simplex, 1, 0, 2, d.d[0, 12], d.d[2, 12], d.d[3, 12], 0, 2, 1)
        return solution
    d.compute_simplex_distances_3(simplex, e213, e214)
    convex_hull_optimal = not (d.d[0, 14] <= 0.0 or d.d[1, 14] <= 0.0 or d.d[2, 14] <= 0.0 or d.d[3, 14] <= 0.0)
    if convex_hull_optimal:
        solution.from_simplex(simplex, d.d[0, 14], d.d[1, 14], d.d[2, 14], d.d[3, 14])
        return solution
    vertex_2_optimal = not (d.d[0, 2] > 0.0 or d.d[2, 5] > 0.0 or d.d[3, 9] > 0.0)
    if vertex_2_optimal:
        simplex.reduce_to_optimal_vertex(1)
        solution.from_vertex(simplex, 0, d.d[1, 1])
        return solution
    vertex_3_optimal = not (d.d[0, 4] > 0.0 or d.d[1, 5] > 0.0 or d.d[3, 10] > 0.0)
    if vertex_3_optimal:
        simplex.reduce_to_optimal_vertex(2)
        solution.from_vertex(simplex, 0, d.d[2, 3])
        return solution
    vertex_4_optimal = not (d.d[0, 8] > 0.0 or d.d[1, 9] > 0.0 or d.d[2, 10] > 0.0)
    if vertex_4_optimal:
        simplex.reduce_to_optimal_vertex(3)
        solution.from_vertex(simplex, 0, d.d[3, 7])
        return solution
    line_segment_23_optimal = not (d.d[0, 6] > 0.0 or d.d[1, 5] <= 0.0 or d.d[2, 5] <= 0.0 or d.d[3, 13] > 0.0)
    if line_segment_23_optimal:
        simplex.select_line_segment_23()
        solution.from_line_segment(simplex, 0, 1, d.d[1, 5], d.d[2, 5])
        return solution
    line_segment_24_optimal = not (d.d[0, 11] > 0.0 or d.d[1, 9] <= 0.0 or d.d[2, 13] > 0.0 or d.d[3, 9] <= 0.0)
    if line_segment_24_optimal:
        simplex.select_line_segment_24()
        solution.from_line_segment(simplex, 0, 1, d.d[1, 9], d.d[3, 9], 1, 0)
        return solution
    line_segment_34_optimal = not (d.d[0, 12] > 0.0 or d.d[1, 13] > 0.0 or d.d[2, 10] <= 0.0 or d.d[3, 10] <= 0.0)
    if line_segment_34_optimal:
        simplex.select_line_segment_34()
        solution.from_line_segment(simplex, 1, 0, d.d[2, 10], d.d[3, 10])
        return solution
    face_234_optimal = not (d.d[0, 14] > 0.0 or d.d[1, 13] <= 0.0 or d.d[2, 13] <= 0.0 or d.d[3, 13] <= 0.0)
    if face_234_optimal:
        simplex.select_face_234()
        solution.from_face(simplex, 0, 1, 2, d.d[1, 13], d.d[2, 13], d.d[3, 13], 1, 2, 0)
        return solution
    return None


def _backup_procedure(simplex, solution, d, backup):
    ordered_indices = np.empty(4, dtype=int)
    solution_d = Solution()
    if len(simplex) == 1:
        solution.from_vertex(simplex, 0, d.d[0, 0])
        return solution, True
    elif len(simplex) == 2:
        n_simplex_points = _backup_procedure_line_segment(
            simplex, backup, d, ordered_indices, solution, solution_d)
    elif len(simplex) == 3:
        n_simplex_points = _backup_procedure_face(
            simplex, backup, d, ordered_indices, solution, solution_d)
    else:
        assert len(simplex) == 4
        n_simplex_points = _backup_procedure_simplex(
            simplex, backup, d, ordered_indices, solution,
            solution_d)

    simplex.reorder(ordered_indices[:n_simplex_points])
    return solution, True


def _backup_procedure_line_segment(
        simplex, backup, d, ordered_indices, solution, solution_d):
    if backup:
        d.backup_line_segments(simplex)
    # check vertex 1
    solution.from_vertex(simplex, 0, d.d[0, 0])
    n_simplex_points = 1
    ordered_indices[0] = 0
    check_line_segment_12 = not (d.d[0, 2] <= 0.0 or d.d[1, 2] <= 0.0)
    if check_line_segment_12:
        solution_d.from_line_segment(simplex, 1, 0, d.d[0, 2], d.d[1, 2])
        if solution_d.dstsq < solution.dstsq:
            n_simplex_points = 2
            solution.copy_from(solution_d, n_simplex_points)
            ordered_indices[:2] = 0, 1
    check_vertex_2 = simplex.dot_product_table[1, 1] < solution.dstsq
    if check_vertex_2:
        n_simplex_points = 1
        solution.from_vertex(simplex, 1, d.d[1, 1])
        ordered_indices[0] = 1
    return n_simplex_points


def _backup_procedure_face(
        simplex, backup, d, ordered_indices, solution, solution_d):
    if backup:
        d.backup_faces(simplex)
    # check vertex 1
    n_simplex_points = 1
    solution.from_vertex(simplex, 0, d.d[0, 0])
    ordered_indices[0] = 0
    check_line_segment_12 = not (d.d[0, 2] <= 0.0 or d.d[1, 2] <= 0.0)
    if check_line_segment_12:
        solution_d.from_line_segment(simplex, 1, 0, d.d[0, 2], d.d[1, 2])
        if solution_d.dstsq < solution.dstsq:
            n_simplex_points = 2
            solution.copy_from(solution_d, n_simplex_points)
            ordered_indices[:2] = 0, 1
    check_line_segment_13 = not (d.d[0, 4] <= 0.0 or d.d[2, 4] <= 0.0)
    if check_line_segment_13:
        solution_d.from_line_segment(simplex, 2, 0, d.d[0, 4], d.d[2, 4])
        if solution_d.dstsq < solution.dstsq:
            n_simplex_points = 2
            solution.copy_from(solution_d, n_simplex_points)
            ordered_indices[:2] = 0, 2
    check_face_123 = not (d.d[0, 6] <= 0.0 or d.d[1, 6] <= 0.0 or d.d[2, 6] <= 0.0)
    if check_face_123:
        solution_d.from_face(simplex, 2, 0, 1, d.d[0, 6], d.d[1, 6], d.d[2, 6])
        if solution_d.dstsq < solution.dstsq:
            n_simplex_points = 3
            solution.copy_from(solution_d, n_simplex_points)
            ordered_indices[:3] = 0, 1, 2
    check_vertex_2 = simplex.dot_product_table[1, 1] < solution.dstsq
    if check_vertex_2:
        n_simplex_points = 1
        solution.from_vertex(simplex, 1, d.d[1, 1])
        ordered_indices[0] = 1
    check_vertex_3 = simplex.dot_product_table[2, 2] < solution.dstsq
    if check_vertex_3:
        n_simplex_points = 1
        solution.from_vertex(simplex, 2, d.d[2, 3])
        ordered_indices[0] = 2
    check_line_segment_23 = not (d.d[1, 5] <= 0.0 or d.d[2, 5] <= 0.0)
    if check_line_segment_23:
        solution_d.from_line_segment(simplex, 2, 1, d.d[1, 5], d.d[2, 5], 1, 0)
        if solution_d.dstsq < solution.dstsq:
            n_simplex_points = 2
            solution.copy_from(solution_d, n_simplex_points)
            ordered_indices[:2] = 2, 1
    return n_simplex_points


def _backup_procedure_simplex(
        simplex, backup, d, ordered_indices, solution, solution_d):
    if backup:
        d.backup_simplex(simplex)
    # check vertex 1
    n_simplex_points = 1
    solution.from_vertex(simplex, 0, d.d[0, 0])
    ordered_indices[0] = 0
    check_line_segment_12 = not (d.d[0, 2] <= 0.0 or d.d[1, 2] <= 0.0)
    if check_line_segment_12:
        solution_d.from_line_segment(simplex, 1, 0, d.d[0, 2], d.d[1, 2])
        if solution_d.dstsq < solution.dstsq:
            n_simplex_points = 2
            solution.copy_from(solution_d, n_simplex_points)
            ordered_indices[:2] = 0, 1
    check_line_segment_13 = not (d.d[0, 4] <= 0.0 or d.d[2, 4] <= 0.0)
    if check_line_segment_13:
        solution_d.from_line_segment(simplex, 2, 0, d.d[0, 4], d.d[2, 4])
        if solution_d.dstsq < solution.dstsq:
            n_simplex_points = 2
            solution.copy_from(solution_d, n_simplex_points)
            ordered_indices[:2] = 0, 2
    check_face_123 = not (d.d[0, 6] <= 0.0 or d.d[1, 6] <= 0.0 or d.d[2, 6] <= 0.0)
    if check_face_123:
        solution_d.from_face(simplex, 2, 0, 1, d.d[0, 6], d.d[1, 6], d.d[2, 6])
        if solution_d.dstsq < solution.dstsq:
            n_simplex_points = 3
            solution.copy_from(solution_d, n_simplex_points)
            ordered_indices[:3] = 0, 1, 2
    check_line_segment_14 = not (d.d[0, 8] <= 0.0 or d.d[3, 8] <= 0.0)
    if check_line_segment_14:
        solution_d.from_line_segment(simplex, 3, 0, d.d[0, 8], d.d[3, 8])
        if solution_d.dstsq < solution.dstsq:
            n_simplex_points = 2
            solution.copy_from(solution_d, n_simplex_points)
            ordered_indices[:2] = 0, 3
    check_face_124 = not (d.d[0, 11] <= 0.0 or d.d[1, 11] <= 0.0 or d.d[3, 11] <= 0.0)
    if check_face_124:
        solution_d.from_face(simplex, 3, 0, 1, d.d[0, 11], d.d[1, 11], d.d[3, 11])
        if solution_d.dstsq < solution.dstsq:
            n_simplex_points = 3
            solution.copy_from(solution_d, n_simplex_points)
            ordered_indices[:3] = 0, 1, 3
    check_face_134 = not (d.d[0, 12] <= 0.0 or d.d[2, 12] <= 0.0 or d.d[3, 12] <= 0.0)
    if check_face_134:
        solution_d.from_face(simplex, 3, 0, 2, d.d[0, 12], d.d[2, 12], d.d[3, 12], 0, 2, 1)
        if solution_d.dstsq < solution.dstsq:
            n_simplex_points = 3
            solution.copy_from(solution_d, n_simplex_points)
            ordered_indices[:3] = 0, 3, 2
    check_convex_hull = not (d.d[0, 14] <= 0.0 or d.d[1, 14] <= 0.0 or d.d[2, 14] <= 0.0 or d.d[3, 14] <= 0.0)
    if check_convex_hull:
        solution_d.from_simplex(simplex, d.d[0, 14], d.d[1, 14], d.d[2, 14], d.d[3, 14])
        if solution_d.dstsq < solution.dstsq:
            n_simplex_points = 4
            solution.copy_from(solution_d, n_simplex_points)
            ordered_indices[:] = 0, 1, 2, 3
    check_vertex_2 = simplex.dot_product_table[1, 1] < solution.dstsq
    if check_vertex_2:
        n_simplex_points = 1
        solution.from_vertex(simplex, 1, d.d[1, 1])
        ordered_indices[0] = 1
    check_vertex_3 = simplex.dot_product_table[2, 2] < solution.dstsq
    if check_vertex_3:
        n_simplex_points = 1
        solution.from_vertex(simplex, 2, d.d[2, 3])
        ordered_indices[0] = 2
    check_vertex_4 = simplex.dot_product_table[3, 3] < solution.dstsq
    if check_vertex_4:
        n_simplex_points = 1
        solution.from_vertex(simplex, 3, d.d[3, 7])
        ordered_indices[0] = 3
    check_line_segment_23 = not (d.d[1, 5] <= 0.0 or d.d[2, 5] <= 0.0)
    if check_line_segment_23:
        solution_d.from_line_segment(simplex, 1, 2, d.d[1, 5], d.d[2, 5], 1, 0)
        if solution_d.dstsq < solution.dstsq:
            n_simplex_points = 2
            solution.copy_from(solution_d, n_simplex_points)
            ordered_indices[:2] = 2, 1
    check_line_segment_24 = not (d.d[1, 9] <= 0.0 or d.d[3, 9] <= 0.0)
    if check_line_segment_24:
        solution_d.from_line_segment(simplex, 3, 1, d.d[1, 9], d.d[3, 9], 1, 0)
        if solution_d.dstsq < solution.dstsq:
            n_simplex_points = 2
            solution.copy_from(solution_d, n_simplex_points)
            ordered_indices[:2] = 3, 1
    check_line_segment_34 = not (d.d[2, 10] <= 0.0 or d.d[3, 10] <= 0.0)
    if check_line_segment_34:
        solution_d.from_line_segment(simplex, 3, 2, d.d[2, 10], d.d[3, 10])
        if solution_d.dstsq < solution.dstsq:
            n_simplex_points = 2
            solution.copy_from(solution_d, n_simplex_points)
            ordered_indices[:2] = 2, 3
    check_face_234 = not (d.d[1, 13] <= 0.0 or d.d[2, 13] <= 0.0 or d.d[3, 13] <= 0.0)
    if check_face_234:
        solution_d.from_face(simplex, 3, 1, 2, d.d[1, 13], d.d[2, 13], d.d[3, 13], 1, 2, 0)
        if solution_d.dstsq < solution.dstsq:
            n_simplex_points = 3
            solution.copy_from(solution_d, n_simplex_points)
            ordered_indices[:3] = 3, 1, 2
    return n_simplex_points


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
