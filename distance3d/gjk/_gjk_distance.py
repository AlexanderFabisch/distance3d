import math
import numpy as np


EPSILON = 10.0 * np.finfo(float).eps


def gjk(collider1, collider2):
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
    collider1 = VertexCachedCollider(collider1)
    collider2 = VertexCachedCollider(collider2)

    solution = Solution()
    simplex = SimplexInfo()
    old_simplex = SimplexInfo()

    new_idx1, new_vertex1 = collider1.first_vertex()
    new_idx2, new_vertex2 = collider2.first_vertex()
    simplex.set_first_point(new_idx1, new_idx2, new_vertex1 - new_vertex2)

    iteration = 0
    backup = False
    while True:
        iteration += 1

        new_solution, backup = distance_subalgorithm_with_backup_procedure(
            simplex, solution, backup)

        no_improvement = new_solution.distance_squared >= solution.distance_squared
        simplex_is_tetrahedron = len(simplex) == 4
        if no_improvement or simplex_is_tetrahedron:
            if backup:
                closest_point1 = compute_point(
                    collider1, solution.barycentric_coordinates[:len(simplex)],
                    simplex.indices_polytope1[:len(simplex)])
                closest_point2 = compute_point(
                    collider2, solution.barycentric_coordinates[:len(simplex)],
                    simplex.indices_polytope2[:len(simplex)])

                if simplex_is_tetrahedron:
                    # Make sure intersection has zero distance
                    closest_point1[:] = 0.5 * (closest_point1 + closest_point2)
                    closest_point2[:] = closest_point1
                    distance = 0.0
                else:
                    distance = new_solution.distance

                return distance, closest_point1, closest_point2, simplex.points
            else:
                backup = True
                if iteration != 1:
                    simplex.copy_from(old_simplex)
        else:
            solution = new_solution

            _find_new_supporting_point(collider1, collider2, simplex, solution)

            old_simplex.copy_from(simplex)
            if len(simplex) == 4:
                simplex.reorder(simplex.nondecreasing_ordered_indices())


gjk_with_simplex = gjk  # for backward compatibility


class VertexCachedCollider:
    """Caches vertices from a collider.

    Many colliders do not store their vertices explicitly. However, GJK only
    stores indices to the vertices. Hence, we will cache requested vertices
    of a collider here.

    Parameters
    ----------
    collider : ConvexCollider
        Collider object.
    """
    def __init__(self, collider):
        self.collider = collider
        self.vertices_ = []

    def first_vertex(self):
        """Get vertex from collider to initialize GJK algorithm.

        Returns
        -------
        vertex_idx : int
            Index in vertex cache.

        vertex : array, shape (3,)
            Vertex from collider.
        """
        vertex = self.collider.first_vertex()
        vertex_idx = len(self.vertices_)
        self.vertices_.append(vertex)
        return vertex_idx, vertex

    def support_function(self, search_direction):
        """Support function for collider.

        Parameters
        ----------
        search_direction : array, shape (3,)
            Direction in which we search for support point of the collider.

        Returns
        -------
        vertex_idx : int
            Index in vertex cache.

        support_point : array, shape (3,)
            Extreme point along search direction.
        """
        vertex = self.collider.support_function(search_direction)
        vertex_idx = len(self.vertices_)
        self.vertices_.append(vertex)
        return vertex_idx, vertex


def _find_new_supporting_point(collider1, collider2, simplex, solution):
    new_index1, new_vertex1 = collider1.support_function(-solution.search_direction)
    new_index2, new_vertex2 = collider2.support_function(solution.search_direction)
    new_simplex_point = new_vertex1 - new_vertex2
    simplex.add_new_point(new_index1, new_index2, new_simplex_point)


def compute_point(vertex_cache, barycentric_coordinates, indices):
    """Compute point from barycentric coordinates.

    Parameters
    ----------
    vertex_cache : VertexCachedCollider
        Collider that contains cached vertices.

    barycentric_coordinates : array, shape (n_vertices,)
        Barycentric coordinates of the point that we compute.

    indices : array, shape (n_vertices,)
        Vertex indices to which the barycentric coordinates apply.

    Returns
    -------
    point : array, shape (3,)
        Point that we compute from barycentric coordinates.
    """
    return np.dot(barycentric_coordinates,
                  [vertex_cache.vertices_[i] for i in indices])


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

    distance_squared : float
        Squared distance to origin.
    """
    def __init__(self):
        self.barycentric_coordinates = np.empty(4, dtype=float)
        self.barycentric_coordinates[0] = 1.0
        self.search_direction = np.empty(3, dtype=float)
        self.distance_squared = np.inf

    @property
    def distance(self):
        return math.sqrt(self.distance_squared)

    def from_vertex(self, simplex, vi):
        self.barycentric_coordinates[0] = 1.0
        self.search_direction = simplex.points[vi]
        self.distance_squared = simplex.dot_product_table[vi, vi]

    def from_line_segment(self, simplex, vi, a, b):
        coords_sum = a + b
        self.barycentric_coordinates[0] = a / coords_sum
        self.barycentric_coordinates[1] = 1.0 - self.barycentric_coordinates[0]
        self.search_direction = self.barycentric_coordinates[:2].dot(simplex.points[vi])
        self.distance_squared = np.dot(self.search_direction, self.search_direction)

    def from_face(self, simplex, vi, a, b, c):
        coords_sum = a + b + c
        self.barycentric_coordinates[0] = a / coords_sum
        self.barycentric_coordinates[1] = b / coords_sum
        # more stable floating point operation than c / coords_sum:
        self.barycentric_coordinates[2] = 1.0 - (
            self.barycentric_coordinates[0] + self.barycentric_coordinates[1])
        self.search_direction = self.barycentric_coordinates[:3].dot(simplex.points[vi])
        self.distance_squared = np.dot(self.search_direction, self.search_direction)

    def from_tetrahedron(self, simplex, barycentric_coordinates):
        self.barycentric_coordinates[:] = barycentric_coordinates / sum(barycentric_coordinates)
        self.search_direction = self.barycentric_coordinates.dot(simplex.points)
        self.distance_squared = np.dot(self.search_direction, self.search_direction)

    def copy_from(self, solution, n_simplex_points):
        self.barycentric_coordinates[:n_simplex_points] = \
            solution.barycentric_coordinates[:n_simplex_points]
        self.search_direction = solution.search_direction
        self.distance_squared = solution.distance_squared


class SimplexInfo:
    """Simplex and additional data.

    Attributes
    ----------
    n_simplex_points : int
        Number of current simplex points.

    points : array, shape (n_simplex_points, 3)
      Current simplex.

    dot_product_table : array, shape (n_simplex_points, n_simplex_points)
        dot_product_table[i, j] = Inner product of simplex[i] and simplex[j].
        Note that only elements i >= j are used.

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
        self.points = np.empty((4, 3), dtype=float)
        self.dot_product_table = np.empty((4, 4), dtype=float)
        self.indices_polytope1 = np.empty(4, dtype=int)
        self.indices_polytope2 = np.empty(4, dtype=int)

    def set_first_point(self, new_index1, new_index2, new_simplex_point):
        self.indices_polytope1[0] = new_index1
        self.indices_polytope2[0] = new_index2
        self.points[0] = new_simplex_point
        self.n_simplex_points += 1
        self.dot_product_table[:self.n_simplex_points, 0] = np.dot(
            self.points[:self.n_simplex_points], self.points[0])

    def copy_from(self, simplex):
        self.n_simplex_points = len(simplex)
        self.points[:len(simplex)] = simplex.points[:len(simplex)]
        self.indices_polytope1[:len(simplex)] = simplex.indices_polytope1[:len(simplex)]
        self.indices_polytope2[:len(simplex)] = simplex.indices_polytope2[:len(simplex)]
        self.dot_product_table[:self.n_simplex_points, :self.n_simplex_points] = simplex.dot_product_table[
            :self.n_simplex_points, :self.n_simplex_points]

    def reorder(self, ordered_indices):
        self.n_simplex_points = len(ordered_indices)
        self.indices_polytope1[:self.n_simplex_points] = self.indices_polytope1[ordered_indices]
        self.indices_polytope2[:self.n_simplex_points] = self.indices_polytope2[ordered_indices]
        self.points[:self.n_simplex_points] = self.points[ordered_indices]
        self.dot_product_table = self.points.dot(self.points.T)

    def add_new_point(self, new_index1, new_index2, new_simplex_point):
        self._move_first_point_to_last_spot()
        self.set_first_point(new_index1, new_index2, new_simplex_point)

    def _move_first_point_to_last_spot(self):
        self.indices_polytope1[self.n_simplex_points] = self.indices_polytope1[0]
        self.indices_polytope2[self.n_simplex_points] = self.indices_polytope2[0]
        self.points[self.n_simplex_points] = self.points[0]
        self.dot_product_table[self.n_simplex_points, :self.n_simplex_points] = self.dot_product_table[
                                                                                :self.n_simplex_points, 0]
        self.dot_product_table[self.n_simplex_points, self.n_simplex_points] = self.dot_product_table[0, 0]

    def _move_vertex(self, old_index, new_index):
        if old_index == new_index:
            return
        self.indices_polytope1[new_index] = self.indices_polytope1[old_index]
        self.indices_polytope2[new_index] = self.indices_polytope2[old_index]
        self.points[new_index] = self.points[old_index]

    def select_vertex(self, i):
        self.n_simplex_points = 1
        self._move_vertex(i, 0)
        self.dot_product_table[0, 0] = self.dot_product_table[i, i]

    def select_line_segment(self, i, j):
        self.n_simplex_points = 2
        if i != 0:
            self._move_vertex(i, 0)
            self.dot_product_table[0, 0] = self.dot_product_table[i, i]
            idx1, idx2 = (j, i) if i < j else (i, j)
            self.dot_product_table[1, 0] = self.dot_product_table[idx1, idx2]
        if j != 1:
            self._move_vertex(j, 1)
            idx1, idx2 = (j, i) if i < j else (i, j)
            self.dot_product_table[1, 0] = self.dot_product_table[idx1, idx2]
            self.dot_product_table[1, 1] = self.dot_product_table[j, j]

    def select_face(self, i, j, k):
        self.n_simplex_points = 3
        if i != 0:
            self._move_vertex(i, 0)
            self.dot_product_table[0, 0] = self.dot_product_table[i, i]
            idx1, idx2 = (j, i) if i < j else (i, j)
            self.dot_product_table[1, 0] = self.dot_product_table[idx1, idx2]
            idx1, idx2 = (k, i) if i < k else (i, k)
            self.dot_product_table[2, 0] = self.dot_product_table[idx1, idx2]
        if j != 1:
            self._move_vertex(j, 1)
            idx1, idx2 = (j, i) if i < j else (i, j)
            self.dot_product_table[1, 0] = self.dot_product_table[idx1, idx2]
            self.dot_product_table[1, 1] = self.dot_product_table[j, j]
            idx1, idx2 = (j, k) if k < j else (k, j)
            self.dot_product_table[2, 1] = self.dot_product_table[idx1, idx2]
        if k != 2:
            self._move_vertex(k, 2)
            idx1, idx2 = (i, k) if k < i else (k, i)
            self.dot_product_table[2, 0] = self.dot_product_table[idx1, idx2]
            idx1, idx2 = (j, k) if k < j else (k, j)
            self.dot_product_table[2, 1] = self.dot_product_table[idx1, idx2]
            self.dot_product_table[2, 2] = self.dot_product_table[k, k]

    def nondecreasing_ordered_indices(self):
        # fast version of np.hstack(((0,), 1 + np.argsort(self.dot_product_table[1:, 0])))
        ordered_indices = np.empty(4, dtype=int)
        ordered_indices[:3] = 0, 1, 2
        if self.dot_product_table[2, 0] < self.dot_product_table[1, 0]:
            ordered_indices[1] = 2
            ordered_indices[2] = 1
        ii = ordered_indices[1]
        if self.dot_product_table[3, 0] < self.dot_product_table[ii, 0]:
            ordered_indices[3] = ordered_indices[2]
            ordered_indices[2] = ordered_indices[1]
            ordered_indices[1] = 3
        else:
            ii = ordered_indices[2]
            if self.dot_product_table[3, 0] < self.dot_product_table[ii, 0]:
                ordered_indices[3] = ordered_indices[2]
                ordered_indices[2] = 3
            else:
                ordered_indices[3] = 3
        return ordered_indices

    def __len__(self):
        return self.n_simplex_points


def distance_subalgorithm_with_backup_procedure(simplex, solution, backup=False):
    """Johnson's distance subalgorithm.

    Implements, in a very efficient way, the distance subalgorithm of finding
    the point of minimum norm in the convex hull of four or less points
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

    backup : bool, optional (default: False)
        Perform backup procedure.

    Returns
    -------
    solution : Solution
        Current solution.

    backup : int
        Perform backup procedure.
    """
    d = BarycentricCoordinates()

    if backup:
        return backup_procedure(simplex, solution, d, backup), backup
    else:
        try:
            return distance_subalgorithm(simplex, d), backup
        except np.linalg.LinAlgError:
            backup = True
            return backup_procedure(simplex, solution, d, backup), backup


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
        * 14: tetrahedron 0-1-2-3
    """
    def __init__(self):
        self.d = np.empty((4, 15), dtype=float)
        self.d[0, 0] = 1.0
        self.d[1, 1] = 1.0
        self.d[2, 3] = 1.0
        self.d[3, 7] = 1.0

    def line_segment_coordinates_0(self, simplex):
        self.d[1, 2] = simplex.dot_product_table[0, 0] - simplex.dot_product_table[1, 0]

    def line_segment_coordinates_1(self, simplex):
        self.d[0, 2] = simplex.dot_product_table[1, 1] - simplex.dot_product_table[1, 0]

    def face_coordinates_0(self, simplex):
        self.line_segment_coordinates_0(simplex)
        self.d[2, 4] = simplex.dot_product_table[0, 0] - simplex.dot_product_table[2, 0]

    def face_coordinates_1(self, simplex):
        e132 = simplex.dot_product_table[1, 0] - simplex.dot_product_table[2, 1]
        self.line_segment_coordinates_1(simplex)
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

    def tetrahedron_coordinates_0(self, simplex):
        self.face_coordinates_0(simplex)
        self.d[3, 8] = simplex.dot_product_table[0, 0] - simplex.dot_product_table[3, 0]

    def tetrahedron_coordinates_1(self, simplex):
        e132 = simplex.dot_product_table[1, 0] - simplex.dot_product_table[2, 1]
        e142 = simplex.dot_product_table[1, 0] - simplex.dot_product_table[3, 1]
        self.d[0, 2] = simplex.dot_product_table[1, 1] - simplex.dot_product_table[1, 0]
        self.d[2, 6] = self.d[0, 2] * self.d[2, 4] + self.d[1, 2] * e132
        self.d[3, 11] = self.d[0, 2] * self.d[3, 8] + self.d[1, 2] * e142
        return e132, e142

    def tetrahedron_coordinates_2(self, simplex):
        e123 = simplex.dot_product_table[2, 0] - simplex.dot_product_table[2, 1]
        e143 = simplex.dot_product_table[2, 0] - simplex.dot_product_table[3, 2]
        self.d[0, 4] = simplex.dot_product_table[2, 2] - simplex.dot_product_table[2, 0]
        self.d[1, 6] = self.d[0, 4] * self.d[1, 2] + self.d[2, 4] * e123
        self.d[3, 12] = self.d[0, 4] * self.d[3, 8] + self.d[2, 4] * e143
        return e123, e143

    def tetrahedron_coordinates_3(self, simplex, e123, e142, e143):
        self.d[1, 5] = simplex.dot_product_table[2, 2] - simplex.dot_product_table[2, 1]
        self.d[2, 5] = simplex.dot_product_table[1, 1] - simplex.dot_product_table[2, 1]
        e213 = -e123
        self.d[0, 6] = self.d[1, 5] * self.d[0, 2] + self.d[2, 5] * e213
        self.d[3, 14] = self.d[0, 6] * self.d[3, 8] + self.d[1, 6] * e142 + self.d[2, 6] * e143
        return e213

    def tetrahedron_coordinates_4(self, simplex):
        e124 = simplex.dot_product_table[3, 0] - simplex.dot_product_table[3, 1]
        e134 = simplex.dot_product_table[3, 0] - simplex.dot_product_table[3, 2]
        self.d[0, 8] = simplex.dot_product_table[3, 3] - simplex.dot_product_table[3, 0]
        self.d[1, 11] = self.d[0, 8] * self.d[1, 2] + self.d[3, 8] * e124
        self.d[2, 12] = self.d[0, 8] * self.d[2, 4] + self.d[3, 8] * e134
        return e124, e134

    def tetrahedron_coordinates_5(self, simplex, e124, e132, e134):
        self.d[1, 9] = simplex.dot_product_table[3, 3] - simplex.dot_product_table[3, 1]
        self.d[3, 9] = simplex.dot_product_table[1, 1] - simplex.dot_product_table[3, 1]
        e214 = -e124
        self.d[0, 11] = self.d[1, 9] * self.d[0, 2] + self.d[3, 9] * e214
        self.d[2, 14] = self.d[0, 11] * self.d[2, 4] + self.d[1, 11] * e132 + self.d[3, 11] * e134
        return e214

    def tetrahedron_coordinates_6(self, simplex, e123, e124, e134):
        self.d[2, 10] = simplex.dot_product_table[3, 3] - simplex.dot_product_table[3, 2]
        self.d[3, 10] = simplex.dot_product_table[2, 2] - simplex.dot_product_table[3, 2]
        e314 = -e134
        self.d[0, 12] = self.d[2, 10] * self.d[0, 4] + self.d[3, 10] * e314
        self.d[1, 14] = self.d[0, 12] * self.d[1, 2] + self.d[2, 12] * e123 + self.d[3, 12] * e124

    def tetrahedron_coordinates_7(self, simplex, e213, e214):
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

    def backup_tetrahedron(self, simplex):
        e132, e123, e213 = self.backup_faces(simplex)
        self.d[3, 8] = simplex.dot_product_table[0, 0] - simplex.dot_product_table[3, 0]
        e142 = simplex.dot_product_table[1, 0] - simplex.dot_product_table[3, 1]
        self.d[3, 11] = self.d[0, 2] * self.d[3, 8] + self.d[1, 2] * e142
        e143 = simplex.dot_product_table[2, 0] - simplex.dot_product_table[3, 2]
        self.d[3, 12] = self.d[0, 4] * self.d[3, 8] + self.d[2, 4] * e143
        self.d[3, 14] = self.d[0, 6] * self.d[3, 8] + self.d[1, 6] * e142 + self.d[2, 6] * e143
        e124, e134 = self.tetrahedron_coordinates_4(simplex)
        e214 = self.tetrahedron_coordinates_5(simplex, e124, e132, e134)
        self.tetrahedron_coordinates_6(simplex, e123, e124, e134)
        self.tetrahedron_coordinates_7(simplex, e213, e214)

    def vertex_0_of_line_segment_optimal(self):
        return self.d[1, 2] <= 0.0

    def line_segment_01_of_line_segment_optimal(self):
        return not (self.vertex_1_of_line_segment_optimal() or self.vertex_0_of_line_segment_optimal())

    def vertex_1_of_line_segment_optimal(self):
        return self.d[0, 2] <= 0.0

    def vertex_0_of_face_optimal(self):
        return not (self.d[1, 2] > 0.0 or self.d[2, 4] > 0.0)

    def line_segment_01_of_face_optimal(self):
        return self.line_segment_01_of_line_segment_optimal() and not self.d[2, 6] > 0.0

    def line_segment_02_of_face_optimal(self):
        return not (self.d[0, 4] <= 0.0 or self.d[1, 6] > 0.0 or self.d[2, 4] <= 0.0)

    def face_012_of_face_optimal(self):
        return not (self.d[0, 6] <= 0.0 or self.d[1, 6] <= 0.0 or self.d[2, 6] <= 0.0)

    def vertex_1_of_face_optimal(self):
        return not (self.d[0, 2] > 0.0 or self.d[2, 5] > 0.0)

    def vertex_2_of_face_optimal(self):
        return not (self.d[0, 4] > 0.0 or self.d[1, 5] > 0.0)

    def line_segment_12_of_face_optimal(self):
        return not self.d[0, 6] > 0.0 and self.check_line_segment_12_of_face()

    def vertex_0_of_tetrahedron_optimal(self):
        return self.vertex_0_of_face_optimal() and not self.d[3, 8] > 0.0

    def line_segment_01_of_tetrahedron_optimal(self):
        return self.line_segment_01_of_face_optimal() and not self.d[3, 11] > 0.0

    def line_segment_02_of_tetrahedron_optimal(self):
        return self.line_segment_02_of_face_optimal() and not self.d[3, 12] > 0.0

    def face_012_of_tetrahedron_optimal(self):
        return self.face_012_of_face_optimal() and not self.d[3, 14] > EPSILON

    def line_segment_03_of_tetrahedron_optimal(self):
        return not (self.d[1, 11] > 0.0 or self.d[2, 12] > 0.0) and self.check_line_segment_03_of_tetrahedron()

    def face_013_of_tetrahedron_optimal(self):
        return not self.d[2, 14] > EPSILON and self.check_face_013_of_tetrahedron()

    def face_023_of_tetrahedron_optimal(self):
        return not self.d[1, 14] > EPSILON and self.check_face_023_of_tetrahedron()

    def convex_hull_of_tetrahedron_optimal(self):
        return not (self.d[0, 14] <= EPSILON or self.d[1, 14] <= EPSILON
                    or self.d[2, 14] <= EPSILON or self.d[3, 14] <= EPSILON)

    def vertex_1_of_tetrahedron_optimal(self):
        return self.vertex_1_of_face_optimal() and not self.d[3, 9] > 0.0

    def vertex_2_of_tetrahedron_optimal(self):
        return self.vertex_2_of_face_optimal() and not self.d[3, 10] > 0.0

    def vertex_3_of_tetrahedron_optimal(self):
        return not (self.d[0, 8] > 0.0 or self.d[1, 9] > 0.0 or self.d[2, 10] > 0.0)

    def line_segment_12_of_tetrahedron_optimal(self):
        return self.line_segment_12_of_face_optimal() and not self.d[3, 13] > 0.0

    def line_segment_13_of_tetrahedron_optimal(self):
        return not (self.d[0, 11] > 0.0 or self.d[2, 13] > 0.0) and self.check_line_segment_13_of_tetrahedron()

    def line_segment_23_of_tetrahedron_optimal(self):
        return not (self.d[0, 12] > 0.0 or self.d[1, 13] > 0.0) and self.check_line_segment_23_of_tetrahedron()

    def face_123_of_tetrahedron_optimal(self):
        return not self.d[0, 14] > EPSILON and self.check_face_123_of_tetrahedron()

    def check_line_segment_02_of_face(self):
        return not (self.d[0, 4] <= 0.0 or self.d[2, 4] <= 0.0)

    def check_face_012_of_face(self):
        return not (self.d[0, 6] <= 0.0 or self.d[1, 6] <= 0.0 or self.d[2, 6] <= 0.0)

    def check_line_segment_12_of_face(self):
        return not (self.d[1, 5] <= 0.0 or self.d[2, 5] <= 0.0)

    def check_line_segment_03_of_tetrahedron(self):
        return not (self.d[0, 8] <= 0.0 or self.d[3, 8] <= 0.0)

    def check_face_013_of_tetrahedron(self):
        return not (self.d[0, 11] <= 0.0 or self.d[1, 11] <= 0.0 or self.d[3, 11] <= 0.0)

    def check_face_023_of_tetrahedron(self):
        return not (self.d[0, 12] <= 0.0 or self.d[2, 12] <= 0.0 or self.d[3, 12] <= 0.0)

    def check_line_segment_13_of_tetrahedron(self):
        return not (self.d[1, 9] <= 0.0 or self.d[3, 9] <= 0.0)

    def check_line_segment_23_of_tetrahedron(self):
        return not (self.d[2, 10] <= 0.0 or self.d[3, 10] <= 0.0)

    def check_face_123_of_tetrahedron(self):
        return not (self.d[1, 13] <= 0.0 or self.d[2, 13] <= 0.0 or self.d[3, 13] <= 0.0)


def distance_subalgorithm(simplex, d):
    """Johnson's distance subalgorithm."""
    if len(simplex) == 1:
        solution = Solution()
        solution.from_vertex(simplex, 0)
        return solution
    elif len(simplex) == 2:
        return _distance_subalgorithm_line_segment(simplex, d)
    elif len(simplex) == 3:
        return _distance_subalgorithm_face(simplex, d)
    else:
        assert len(simplex) == 4
        return _distance_subalgorithm_tetrahedron(simplex, d)


def _distance_subalgorithm_line_segment(simplex, d):
    solution = Solution()
    d.line_segment_coordinates_0(simplex)
    if d.vertex_0_of_line_segment_optimal():
        simplex.select_vertex(0)
        solution.from_vertex(simplex, 0)
        return solution
    d.line_segment_coordinates_1(simplex)
    if d.line_segment_01_of_line_segment_optimal():
        solution.from_line_segment(simplex, [0, 1], d.d[0, 2], d.d[1, 2])
        return solution
    else:
        assert d.vertex_1_of_line_segment_optimal()
        simplex.select_vertex(1)
        solution.from_vertex(simplex, 0)
        return solution


def _distance_subalgorithm_face(simplex, d):
    solution = Solution()
    d.face_coordinates_0(simplex)
    if d.vertex_0_of_face_optimal():
        simplex.select_vertex(0)
        solution.from_vertex(simplex, 0)
        return solution
    d.face_coordinates_1(simplex)
    if d.line_segment_01_of_face_optimal():
        simplex.select_line_segment(0, 1)
        solution.from_line_segment(simplex, [0, 1], d.d[0, 2], d.d[1, 2])
        return solution
    e123 = d.face_coordinates_2(simplex)
    if d.line_segment_02_of_face_optimal():
        simplex.select_line_segment(0, 2)
        solution.from_line_segment(simplex, [0, 1], d.d[0, 4], d.d[2, 4])
        return solution
    d.face_coordinates_3(simplex, e123)
    if d.face_012_of_face_optimal():
        solution.from_face(simplex, [0, 1, 2], d.d[0, 6], d.d[1, 6], d.d[2, 6])
        return solution
    if d.vertex_1_of_face_optimal():
        simplex.select_vertex(1)
        solution.from_vertex(simplex, 0)
        return solution
    if d.vertex_2_of_face_optimal():
        simplex.select_vertex(2)
        solution.from_vertex(simplex, 0)
        return solution
    if d.line_segment_12_of_face_optimal():
        simplex.select_line_segment(2, 1)
        solution.from_line_segment(simplex, [1, 0], d.d[1, 5], d.d[2, 5])
        return solution
    raise np.linalg.LinAlgError("Numerical problem, backup procedure required")


def _distance_subalgorithm_tetrahedron(simplex, d):
    solution = Solution()
    d.tetrahedron_coordinates_0(simplex)
    if d.vertex_0_of_tetrahedron_optimal():
        simplex.select_vertex(0)
        solution.from_vertex(simplex, 0)
        return solution
    e132, e142 = d.tetrahedron_coordinates_1(simplex)
    if d.line_segment_01_of_tetrahedron_optimal():
        simplex.select_line_segment(0, 1)
        solution.from_line_segment(simplex, [0, 1], d.d[0, 2], d.d[1, 2])
        return solution
    e123, e143 = d.tetrahedron_coordinates_2(simplex)
    if d.line_segment_02_of_tetrahedron_optimal():
        simplex.select_line_segment(0, 2)
        solution.from_line_segment(simplex, [0, 1], d.d[0, 4], d.d[2, 4])
        return solution
    e213 = d.tetrahedron_coordinates_3(simplex, e123, e142, e143)
    if d.face_012_of_tetrahedron_optimal():
        simplex.select_face(0, 1, 2)
        solution.from_face(simplex, [0, 1, 2], d.d[0, 6], d.d[1, 6], d.d[2, 6])
        return solution
    e124, e134 = d.tetrahedron_coordinates_4(simplex)
    if d.line_segment_03_of_tetrahedron_optimal():
        simplex.select_line_segment(0, 3)
        solution.from_line_segment(simplex, [0, 1], d.d[0, 8], d.d[3, 8])
        return solution
    e214 = d.tetrahedron_coordinates_5(simplex, e124, e132, e134)
    if d.face_013_of_tetrahedron_optimal():
        simplex.select_face(0, 1, 3)
        solution.from_face(simplex, [0, 1, 2], d.d[0, 11], d.d[1, 11], d.d[3, 11])
        return solution
    d.tetrahedron_coordinates_6(simplex, e123, e124, e134)
    if d.face_023_of_tetrahedron_optimal():
        simplex.select_face(0, 3, 2)
        solution.from_face(simplex, [0, 1, 2], d.d[0, 12], d.d[3, 12], d.d[2, 12])
        return solution
    d.tetrahedron_coordinates_7(simplex, e213, e214)
    if d.convex_hull_of_tetrahedron_optimal():
        solution.from_tetrahedron(simplex, d.d[:, 14])
        return solution
    if d.vertex_1_of_tetrahedron_optimal():
        simplex.select_vertex(1)
        solution.from_vertex(simplex, 0)
        return solution
    if d.vertex_2_of_tetrahedron_optimal():
        simplex.select_vertex(2)
        solution.from_vertex(simplex, 0)
        return solution
    if d.vertex_3_of_tetrahedron_optimal():
        simplex.select_vertex(3)
        solution.from_vertex(simplex, 0)
        return solution
    if d.line_segment_12_of_tetrahedron_optimal():
        simplex.select_line_segment(2, 1)
        solution.from_line_segment(simplex, [1, 0], d.d[1, 5], d.d[2, 5])
        return solution
    if d.line_segment_13_of_tetrahedron_optimal():
        simplex.select_line_segment(3, 1)
        solution.from_line_segment(simplex, [1, 0], d.d[3, 9], d.d[1, 9])
        return solution
    if d.line_segment_23_of_tetrahedron_optimal():
        simplex.select_line_segment(2, 3)
        solution.from_line_segment(simplex, [0, 1], d.d[2, 10], d.d[3, 10])
        return solution
    if d.face_123_of_tetrahedron_optimal():
        simplex.select_face(3, 1, 2)
        solution.from_face(simplex, [1, 2, 0], d.d[2, 13], d.d[3, 13], d.d[1, 13])
        return solution
    raise np.linalg.LinAlgError("Numerical problem, backup procedure required")


def backup_procedure(simplex, solution, d, backup=True):
    """Backup procedure.

    Johnson's distance subalgorithm is affected by rounding errors in floating
    point operations in cases of degenerate simplices. This backup procedure
    addresses the problem. It will always succeed, but is computationally more
    expensive.
    """
    if len(simplex) == 1:
        solution.from_vertex(simplex, 0)
        return solution
    elif len(simplex) == 2:
        ordered_indices = _backup_procedure_line_segment(
            simplex, backup, d, solution)
    elif len(simplex) == 3:
        ordered_indices = _backup_procedure_face(
            simplex, backup, d, solution)
    else:
        assert len(simplex) == 4
        ordered_indices = _backup_procedure_tetrahedron(
            simplex, backup, d, solution)

    simplex.reorder(ordered_indices)
    return solution


def _backup_procedure_line_segment(simplex, backup, d, solution):
    if backup:
        d.backup_line_segments(simplex)
    ordered_indices = np.empty(2, dtype=int)
    # check vertex 1
    solution.from_vertex(simplex, 0)
    n_simplex_points = 1
    ordered_indices[0] = 0
    if d.line_segment_01_of_line_segment_optimal():
        solution_d = Solution()
        solution_d.from_line_segment(simplex, [0, 1], d.d[0, 2], d.d[1, 2])
        if solution_d.distance_squared < solution.distance_squared:
            n_simplex_points = 2
            solution.copy_from(solution_d, n_simplex_points)
            ordered_indices[:2] = 0, 1
    check_vertex_2 = simplex.dot_product_table[1, 1] < solution.distance_squared
    if check_vertex_2:
        n_simplex_points = 1
        solution.from_vertex(simplex, 1)
        ordered_indices[0] = 1
    return ordered_indices[:n_simplex_points]


def _backup_procedure_face(simplex, backup, d, solution):
    if backup:
        d.backup_faces(simplex)
    ordered_indices = np.empty(3, dtype=int)
    # check vertex 1
    n_simplex_points = 1
    solution.from_vertex(simplex, 0)
    ordered_indices[0] = 0
    solution_d = Solution()
    if d.line_segment_01_of_line_segment_optimal():
        solution_d.from_line_segment(simplex, [0, 1], d.d[0, 2], d.d[1, 2])
        if solution_d.distance_squared < solution.distance_squared:
            n_simplex_points = 2
            solution.copy_from(solution_d, n_simplex_points)
            ordered_indices[:2] = 0, 1
    if d.check_line_segment_02_of_face():
        solution_d.from_line_segment(simplex, [0, 2], d.d[0, 4], d.d[2, 4])
        if solution_d.distance_squared < solution.distance_squared:
            n_simplex_points = 2
            solution.copy_from(solution_d, n_simplex_points)
            ordered_indices[:2] = 0, 2
    if d.check_face_012_of_face():
        solution_d.from_face(simplex, [0, 1, 2], d.d[0, 6], d.d[1, 6], d.d[2, 6])
        if solution_d.distance_squared < solution.distance_squared:
            n_simplex_points = 3
            solution.copy_from(solution_d, n_simplex_points)
            ordered_indices[:3] = 0, 1, 2
    check_vertex_2 = simplex.dot_product_table[1, 1] < solution.distance_squared
    if check_vertex_2:
        n_simplex_points = 1
        solution.from_vertex(simplex, 1)
        ordered_indices[0] = 1
    check_vertex_3 = simplex.dot_product_table[2, 2] < solution.distance_squared
    if check_vertex_3:
        n_simplex_points = 1
        solution.from_vertex(simplex, 2)
        ordered_indices[0] = 2
    if d.check_line_segment_12_of_face():
        solution_d.from_line_segment(simplex, [2, 1], d.d[2, 5], d.d[1, 5])
        if solution_d.distance_squared < solution.distance_squared:
            n_simplex_points = 2
            solution.copy_from(solution_d, n_simplex_points)
            ordered_indices[:2] = 2, 1
    return ordered_indices[:n_simplex_points]


def _backup_procedure_tetrahedron(simplex, backup, d, solution):
    if backup:
        d.backup_tetrahedron(simplex)
    ordered_indices = np.empty(4, dtype=int)
    # check vertex 1
    n_simplex_points = 1
    solution.from_vertex(simplex, 0)
    ordered_indices[0] = 0
    solution_d = Solution()
    if d.line_segment_01_of_line_segment_optimal():
        solution_d.from_line_segment(simplex, [0, 1], d.d[0, 2], d.d[1, 2])
        if solution_d.distance_squared < solution.distance_squared:
            n_simplex_points = 2
            solution.copy_from(solution_d, n_simplex_points)
            ordered_indices[:2] = 0, 1
    if d.check_line_segment_02_of_face():
        solution_d.from_line_segment(simplex, [0, 2], d.d[0, 4], d.d[2, 4])
        if solution_d.distance_squared < solution.distance_squared:
            n_simplex_points = 2
            solution.copy_from(solution_d, n_simplex_points)
            ordered_indices[:2] = 0, 2
    if d.check_face_012_of_face():
        solution_d.from_face(simplex, [0, 1, 2], d.d[0, 6], d.d[1, 6], d.d[2, 6])
        if solution_d.distance_squared < solution.distance_squared:
            n_simplex_points = 3
            solution.copy_from(solution_d, n_simplex_points)
            ordered_indices[:3] = 0, 1, 2
    if d.check_line_segment_03_of_tetrahedron():
        solution_d.from_line_segment(simplex, [0, 3], d.d[0, 8], d.d[3, 8])
        if solution_d.distance_squared < solution.distance_squared:
            n_simplex_points = 2
            solution.copy_from(solution_d, n_simplex_points)
            ordered_indices[:2] = 0, 3
    if d.check_face_013_of_tetrahedron():
        solution_d.from_face(simplex, [0, 1, 3], d.d[0, 11], d.d[1, 11], d.d[3, 11])
        if solution_d.distance_squared < solution.distance_squared:
            n_simplex_points = 3
            solution.copy_from(solution_d, n_simplex_points)
            ordered_indices[:3] = 0, 1, 3
    if d.check_face_023_of_tetrahedron():
        solution_d.from_face(simplex, [0, 3, 2], d.d[0, 12], d.d[3, 12], d.d[2, 12])
        if solution_d.distance_squared < solution.distance_squared:
            n_simplex_points = 3
            solution.copy_from(solution_d, n_simplex_points)
            ordered_indices[:3] = 0, 3, 2
    if d.convex_hull_of_tetrahedron_optimal():
        solution_d.from_tetrahedron(simplex, d.d[:, 14])
        if solution_d.distance_squared < solution.distance_squared:
            n_simplex_points = 4
            solution.copy_from(solution_d, n_simplex_points)
            ordered_indices[:] = 0, 1, 2, 3
    check_vertex_2 = simplex.dot_product_table[1, 1] < solution.distance_squared
    if check_vertex_2:
        n_simplex_points = 1
        solution.from_vertex(simplex, 1)
        ordered_indices[0] = 1
    check_vertex_3 = simplex.dot_product_table[2, 2] < solution.distance_squared
    if check_vertex_3:
        n_simplex_points = 1
        solution.from_vertex(simplex, 2)
        ordered_indices[0] = 2
    check_vertex_4 = simplex.dot_product_table[3, 3] < solution.distance_squared
    if check_vertex_4:
        n_simplex_points = 1
        solution.from_vertex(simplex, 3)
        ordered_indices[0] = 3
    if d.check_line_segment_12_of_face():
        solution_d.from_line_segment(simplex, [2, 1], d.d[2, 5], d.d[1, 5])
        if solution_d.distance_squared < solution.distance_squared:
            n_simplex_points = 2
            solution.copy_from(solution_d, n_simplex_points)
            ordered_indices[:2] = 2, 1
    if d.check_line_segment_13_of_tetrahedron():
        solution_d.from_line_segment(simplex, [3, 1], d.d[3, 9], d.d[1, 9])
        if solution_d.distance_squared < solution.distance_squared:
            n_simplex_points = 2
            solution.copy_from(solution_d, n_simplex_points)
            ordered_indices[:2] = 3, 1
    if d.check_line_segment_23_of_tetrahedron():
        solution_d.from_line_segment(simplex, [2, 3], d.d[2, 10], d.d[3, 10])
        if solution_d.distance_squared < solution.distance_squared:
            n_simplex_points = 2
            solution.copy_from(solution_d, n_simplex_points)
            ordered_indices[:2] = 2, 3
    if d.check_face_123_of_tetrahedron():
        solution_d.from_face(simplex, [3, 1, 2], d.d[3, 13], d.d[1, 13], d.d[2, 13])
        if solution_d.distance_squared < solution.distance_squared:
            n_simplex_points = 3
            solution.copy_from(solution_d, n_simplex_points)
            ordered_indices[:3] = 3, 1, 2
    return ordered_indices[:n_simplex_points]
