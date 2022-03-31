import math
import numpy as np
from .geometry import (
    capsule_extreme_along_direction, cylinder_extreme_along_direction)


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

    contact_point1 : array, shape (3,)
        Contact point on first convex shape.

    contact_point2 : array, shape (3,)
        Contact point on second convex shape.
    """
    return gjk_with_simplex(Convex(vertices1), Convex(vertices2))[:3]


class Convex:
    """Wraps convex hull of a set of vertices for GJK algorithm.

    Parameters
    ----------
    vertices : array, shape (n_vertices, 3)
        Vertices of the convex shape.
    """
    def __init__(self, vertices):
        self.vertices = vertices

    def first_vertex(self):
        return self.vertices[0]

    def support_function(self, search_direction):
        idx = np.argmax(self.vertices.dot(search_direction))
        return idx, self.vertices[idx]

    def compute_point(self, barycentric_coordinates, indices):
        return np.dot(barycentric_coordinates, self.vertices[indices])


class Cylinder:
    """Wraps cylinder for GJK algorithm."""
    def __init__(self, cylinder2origin, radius, length):
        self.cylinder2origin = cylinder2origin
        self.radius = radius
        self.length = length
        self.vertices = []

    def first_vertex(self):
        vertex = self.cylinder2origin[:3, 3] + 0.5 * self.length * self.cylinder2origin[:3, 2]
        self.vertices.append(vertex)
        return vertex

    def support_function(self, search_direction):
        vertex = cylinder_extreme_along_direction(
            search_direction, self.cylinder2origin, self.radius, self.length)
        vertex_idx = len(self.vertices)
        self.vertices.append(vertex)
        return vertex_idx, vertex

    def compute_point(self, barycentric_coordinates, indices):
        return np.dot(barycentric_coordinates, np.array([self.vertices[i] for i in indices]))


class Capsule:
    """Wraps capsule for GJK algorithm."""
    def __init__(self, capsule2origin, radius, height):
        self.capsule2origin = capsule2origin
        self.radius = radius
        self.height = height
        self.vertices = []

    def first_vertex(self):
        vertex = self.capsule2origin[:3, 3] - (self.radius + 0.5 * self.height) * self.capsule2origin[:3, 2]
        self.vertices.append(vertex)
        return vertex

    def support_function(self, search_direction):
        vertex = capsule_extreme_along_direction(
            search_direction, self.capsule2origin, self.radius, self.height)
        vertex_idx = len(self.vertices)
        self.vertices.append(vertex)
        return vertex_idx, vertex

    def compute_point(self, barycentric_coordinates, indices):
        return np.dot(barycentric_coordinates, np.array([self.vertices[i] for i in indices]))


class Sphere:
    """Wraps sphere for GJK algorithm."""
    # https://github.com/kevinmoran/GJK/blob/master/Collider.h#L33
    def __init__(self, center, radius):
        self.c = center
        self.radius = radius
        self.vertices = []

    def first_vertex(self):
        vertex = self.c + np.array([0, 0, self.radius])
        self.vertices.append(vertex)
        return vertex

    def support_function(self, search_direction):
        s_norm = np.linalg.norm(search_direction)
        if s_norm == 0.0:
            vertex = self.c + np.array([0, 0, self.radius])
        else:
            vertex = self.c + search_direction / s_norm * self.radius
        vertex_idx = len(self.vertices)
        self.vertices.append(vertex)
        return vertex_idx, vertex

    def compute_point(self, barycentric_coordinates, indices):
        return np.dot(barycentric_coordinates, np.array([self.vertices[i] for i in indices]))


def gjk_with_simplex(collider1, collider2):
    """Gilbert-Johnson-Keerthi algorithm for distance calculation.

    The GJK algorithm only works for convex shapes. Concave objects have to be
    decomposed into convex shapes first.

    Based on the translation to C of the original Fortran implementation:
    Ruspini, Diego. gilbert.c, a C version of the original Fortran
    implementation of the GJK algorithm.
    ftp://labrea.stanford.edu/cs/robotics/sean/distance/gilbert.c,
    also available from http://realtimecollisiondetection.net/files/gilbert.c

    Parameters
    ----------
    collider1 : Collider
        Convex collider 1.

    collider2 : Collider
        Convex collider 2.

    Returns
    -------
    distance : float
        The shortest distance between two convex shapes.

    contact_point1 : array, shape (3,)
        Contact point on first convex shape.

    contact_point2 : array, shape (3,)
        Contact point on second convex shape.

    simplex : array, shape (4, 3)
        Simplex defined by 4 points of the Minkowski difference between
        vertices of the two colliders.
    """
    indices_polytope1 = np.array([0, 0, 0, 0], dtype=int)
    indices_polytope2 = np.array([0, 0, 0, 0], dtype=int)

    barycentric_coordinates = np.zeros(4, dtype=float)
    simplex = np.zeros((4, 3), dtype=float)
    old_simplex = np.zeros((4, 3), dtype=float)
    dot_product_table = np.zeros((4, 4), dtype=float)
    old_dot_product_table = np.zeros((4, 4), dtype=float)
    old_indices_polytope1 = np.zeros(4, dtype=int)
    old_indices_polytope2 = np.zeros(4, dtype=int)
    iord = np.zeros(4, dtype=int)
    search_direction = np.zeros(3, dtype=float)
    backup = 0

    # Initialize simplex to difference of first points of the objects
    ncy = 0
    n_simplex_points = 1
    barycentric_coordinates[0] = 1.0

    simplex[0] = collider1.first_vertex() - collider2.first_vertex()
    dot_product_table[0, 0] = np.dot(simplex[0], simplex[0])

    lastdstsq = dot_product_table[0, 0] + dot_product_table[0, 0] + 1.0
    while True:
        ncy += 1

        # Compute point of minimum norm in the convex hull of the simplex
        dstsq, n_simplex_points, backup = distance_subalgorithm(
            n_simplex_points, indices_polytope1, indices_polytope2, simplex,
            dot_product_table, search_direction, barycentric_coordinates, backup)

        if dstsq >= lastdstsq or n_simplex_points == 4:
            if backup:
                closest_point1 = collider1.compute_point(
                    barycentric_coordinates[:n_simplex_points],
                    indices_polytope1[:n_simplex_points])
                closest_point2 = collider2.compute_point(
                    barycentric_coordinates[:n_simplex_points],
                    indices_polytope2[:n_simplex_points])

                # Make sure intersection has zero distance
                if n_simplex_points == 4:
                    closest_point1[:] = 0.5 * (closest_point1 + closest_point2)
                    closest_point2[:] = closest_point1
                    distance = 0.0
                else:
                    distance = math.sqrt(dstsq)

                return distance, closest_point1, closest_point2, simplex

            backup = 1
            if ncy != 1:
                n_simplex_points = _revert_to_old_simplex(
                    dot_product_table, indices_polytope1, indices_polytope2,
                    old_dot_product_table, old_indices_polytope1,
                    old_indices_polytope2, old_simplex, n_old_simplex_points,
                    simplex)
            continue

        lastdstsq = dstsq

        # Find new supporting point in direction -search_direction:
        # s_(A-B)(-search_direction) = s_A(-search_direction) - s_B(search_direction)
        new_index1, new_vertex1 = collider1.support_function(-search_direction)
        new_index2, new_vertex2 = collider2.support_function(search_direction)
        new_simplex_point = new_vertex1 - new_vertex2

        n_simplex_points = _add_new_point(
            dot_product_table, indices_polytope1, indices_polytope2,
            n_simplex_points, new_index1, new_index2, simplex,
            new_simplex_point)
        n_old_simplex_points = _save_old_simplex(
            dot_product_table, indices_polytope1, indices_polytope2,
            n_simplex_points, old_dot_product_table, old_indices_polytope1,
            old_indices_polytope2, old_simplex, simplex)
        _reorder_simplex(
            dot_product_table, indices_polytope1, indices_polytope2, iord,
            n_simplex_points, old_dot_product_table, old_indices_polytope1,
            old_indices_polytope2, old_simplex, simplex)

    raise RuntimeError("Solution should be found in loop.")


def distance_subalgorithm(
        n_simplex_points, old_indices_polytope1, old_indices_polytope2,
        simplex, dot_product_table, search_direction, barycentric_coordinates,
        backup):
    """Distance subalgorithm.

    Implements, in a very efficient way, the distance subalgorithm
    of finding the near point to the convex hull of four or less points
    in 3-D space. The procedure and its efficient FORTRAN implementation
    are both due to D.W. Johnson. Although this subroutine is quite long,
    only a very small part of it will be executed on each call. Refer to
    sections 5 and 6 of the report mentioned in routine DIST3 for details
    concerning the distance subalgorithm. Converted to C be Diego C. Ruspini
    3/25/93.

    This function also determines an affinely independent subset of the
    points such that zsol= near point to the affine hull of the points
    in the subset. The variables nvs, y, ris, rjs and dell are modified
    so that, on output, they correspond to this subset of points.

    Parameters
    ----------
    n_simplex_points : int
      The number of points. 1 <= nvs <= 4 .

    simplex : array, shape (n_points, 3)
      The array containing the points.

    old_indices_polytope1 : array, shape (n_points,)
        Index vector for Polytope-I. For k = 1,...,nvs,
        y[k] = zbi[ris[k]] - zbj[rjs[k]].

    old_indices_polytope2 : array, shape (n_points,)
        Index vectors for Polytope-I and Polytope-J. For k = 1,...,nvs,
        y[k] = zbi[ris[k]] - zbj[rjs[k]].

    dot_product_table : array, shape (n_points, n_points)
        dell[i, j] = Inner product of y[i] and y[j].

    search_direction : array, shape (3,)
        Near point to the convex hull of the points in y.

    barycentric_coordinates : array, shape (n_points,)
        The barycentric coordinates of zsol, i.e.,
        zsol = als[0]*y[1] + ... + ALS(nvs)*y[nvs-1],
        als[k] > 0.0 for k=0,...,nvs-1, and, als[0] + ... + als[nvs-1] = 1.0 .

    backup : int
        TODO

    Returns
    -------
    dstsq : float
        Squared distance.

    nvs : int
        The new number of points. 1 <= nvs <= 4 .

    backup : int
        TODO
    """
    risd = np.empty(4, dtype=int)
    rjsd = np.empty(4, dtype=int)
    iord = np.empty(4, dtype=int)
    d1 = np.empty(15, dtype=float)
    d2 = np.empty(15, dtype=float)
    d3 = np.empty(15, dtype=float)
    d4 = np.empty(15, dtype=float)
    yd = np.empty((4, 3), dtype=float)
    delld = np.empty((4, 4), dtype=float)
    zsold = np.empty(3, dtype=float)
    alsd = np.empty(4, dtype=float)

    d1[0] = 1.0
    d2[1] = 1.0
    d3[3] = 1.0
    d4[7] = 1.0

    if not backup:
        # regular distance subalgorithm begins
        if n_simplex_points == 1:
            # case  of  a  single  point ...
            barycentric_coordinates[0] = d1[0]
            search_direction[:] = simplex[0]
            dstsq = dot_product_table[0, 0]
            return dstsq, n_simplex_points, backup
        elif n_simplex_points == 2:
            # case of two points ...
            # check optimality of vertex 1
            d2[2] = dot_product_table[0, 0] - dot_product_table[1, 0]
            if d2[2] <= 0.0:
                n_simplex_points = 1
                barycentric_coordinates[0] = d1[0]
                search_direction[:] = simplex[0]
                dstsq = dot_product_table[0, 0]
                return dstsq, n_simplex_points, backup
            # check optimality of line segment 1-2
            d1[2] = dot_product_table[1, 1] - dot_product_table[1, 0]
            if not (d1[2] <= 0.0 or d2[2] <= 0.0):
                sum = d1[2] + d2[2]
                barycentric_coordinates[0] = d1[2] / sum
                barycentric_coordinates[1] = 1.0 - barycentric_coordinates[0]
                search_direction[:] = simplex[1] + barycentric_coordinates[0] * (simplex[0] - simplex[1])
                dstsq = np.dot(search_direction, search_direction)
                return dstsq, n_simplex_points, backup
            # check optimality of vertex 2
            if d1[2] <= 0.0:
                n_simplex_points = 1
                old_indices_polytope1[0] = old_indices_polytope1[1]
                old_indices_polytope2[0] = old_indices_polytope2[1]
                barycentric_coordinates[0] = d2[1]
                search_direction[:] = simplex[1]
                dstsq = dot_product_table[1, 1]
                simplex[0, :] = simplex[1, :]
                dot_product_table[0, 0] = dot_product_table[1, 1]
                return dstsq, n_simplex_points, backup
        elif n_simplex_points == 3:
            # case of three points ...
            # check optimality of vertex 1
            d2[2] = dot_product_table[0, 0] - dot_product_table[1, 0]
            d3[4] = dot_product_table[0, 0] - dot_product_table[2, 0]
            if not (d2[2] > 0.0 or d3[4] > 0.0):
                n_simplex_points = 1
                barycentric_coordinates[0] = d1[0]
                search_direction[:] = simplex[0]
                dstsq = dot_product_table[0, 0]
                return dstsq, n_simplex_points, backup
            # check optimality of line segment 1-2
            e132 = dot_product_table[1, 0] - dot_product_table[2, 1]
            d1[2] = dot_product_table[1, 1] - dot_product_table[1, 0]
            d3[6] = d1[2] * d3[4] + d2[2] * e132
            if not (d1[2] <= 0.0 or d2[2] <= 0.0 or d3[6] > 0.0):
                n_simplex_points = 2
                sum = d1[2] + d2[2]
                barycentric_coordinates[0] = d1[2] / sum
                barycentric_coordinates[1] = 1.0 - barycentric_coordinates[0]
                search_direction[:] = simplex[1] + barycentric_coordinates[0] * (simplex[0] - simplex[1])
                dstsq = np.dot(search_direction, search_direction)
                return dstsq, n_simplex_points, backup
            # check optimality of line segment 1-3
            e123 = dot_product_table[2, 0] - dot_product_table[2, 1]
            d1[4] = dot_product_table[2, 2] - dot_product_table[2, 0]
            d2[6] = d1[4] * d2[2] + d3[4] * e123
            if not (d1[4] <= 0.0 or d2[6] > 0.0 or d3[4] <= 0.0):
                n_simplex_points = 2
                old_indices_polytope1[1] = old_indices_polytope1[2]
                old_indices_polytope2[1] = old_indices_polytope2[2]
                sum = d1[4] + d3[4]
                barycentric_coordinates[0] = d1[4] / sum
                barycentric_coordinates[1] = 1.0 - barycentric_coordinates[0]
                search_direction[:] = simplex[2] + barycentric_coordinates[0] * (simplex[0] - simplex[2])
                dstsq = np.dot(search_direction, search_direction)
                simplex[1, 0] = simplex[2, 0]
                simplex[1, 1] = simplex[2, 1]
                simplex[1, 2] = simplex[2, 2]
                dot_product_table[1, 0] = dot_product_table[2, 0]
                dot_product_table[1, 1] = dot_product_table[2, 2]
                return dstsq, n_simplex_points, backup
            # check optimality of face 123
            e213 = -e123
            d2[5] = dot_product_table[2, 2] - dot_product_table[2, 1]
            d3[5] = dot_product_table[1, 1] - dot_product_table[2, 1]
            d1[6] = d2[5] * d1[2] + d3[5] * e213
            if not (d1[6] <= 0.0 or d2[6] <= 0.0 or d3[6] <= 0.0):
                sum = d1[6] + d2[6] + d3[6]
                barycentric_coordinates[0] = d1[6] / sum
                barycentric_coordinates[1] = d2[6] / sum
                barycentric_coordinates[2] = 1.0 - barycentric_coordinates[0] - barycentric_coordinates[1]
                search_direction[:] = simplex[2] + barycentric_coordinates[0] * (simplex[0] - simplex[2]) + barycentric_coordinates[1] * (simplex[1] - simplex[2])
                dstsq = np.dot(search_direction, search_direction)
                return dstsq, n_simplex_points, backup
            # check optimality of vertex 2
            if not (d1[2] > 0.0 or d3[5] > 0.0):
                n_simplex_points = 1
                old_indices_polytope1[0] = old_indices_polytope1[1]
                old_indices_polytope2[0] = old_indices_polytope2[1]
                barycentric_coordinates[0] = d2[1]
                search_direction[:] = simplex[1]
                dstsq = dot_product_table[1, 1]
                simplex[0, :] = simplex[1, :]
                dot_product_table[0, 0] = dot_product_table[1, 1]
                return dstsq, n_simplex_points, backup
            # check optimality of vertex 3
            if not (d1[4] > 0.0 or d2[5] > 0.0):
                n_simplex_points = 1
                old_indices_polytope1[0] = old_indices_polytope1[2]
                old_indices_polytope2[0] = old_indices_polytope2[2]
                barycentric_coordinates[0] = d3[3]
                search_direction[:] = simplex[2]
                dstsq = dot_product_table[2, 2]
                simplex[0] = simplex[2]
                dot_product_table[0, 0] = dot_product_table[2, 2]
                return dstsq, n_simplex_points, backup
            # check optimality of line segment 2-3
            if not (d1[6] > 0.0 or d2[5] <= 0.0 or d3[5] <= 0.0):
                n_simplex_points = 2
                old_indices_polytope1[0] = old_indices_polytope1[2]
                old_indices_polytope2[0] = old_indices_polytope2[2]
                sum = d2[5] + d3[5]
                barycentric_coordinates[1] = d2[5] / sum
                barycentric_coordinates[0] = 1.0 - barycentric_coordinates[1]
                search_direction[:] = simplex[2] + barycentric_coordinates[1] * (simplex[1] - simplex[2])
                dstsq = np.dot(search_direction, search_direction)
                simplex[0] = simplex[2]
                dot_product_table[1, 0] = dot_product_table[2, 1]
                dot_product_table[0, 0] = dot_product_table[2, 2]
                return dstsq, n_simplex_points, backup
        elif n_simplex_points == 4:
            # case of four points ...
            # check optimality of vertex 1
            d2[2] = dot_product_table[0, 0] - dot_product_table[1, 0]
            d3[4] = dot_product_table[0, 0] - dot_product_table[2, 0]
            d4[8] = dot_product_table[0, 0] - dot_product_table[3, 0]
            if not (d2[2] > 0.0 or d3[4] > 0.0 or d4[8] > 0.0):
                n_simplex_points = 1
                barycentric_coordinates[0] = d1[0]
                search_direction[:] = simplex[0]
                dstsq = dot_product_table[0, 0]
                return dstsq, n_simplex_points, backup
            # check optimality of line segment 1-2
            e132 = dot_product_table[1, 0] - dot_product_table[2, 1]
            e142 = dot_product_table[1, 0] - dot_product_table[3, 1]
            d1[2] = dot_product_table[1, 1] - dot_product_table[1, 0]
            d3[6] = d1[2] * d3[4] + d2[2] * e132
            d4[11] = d1[2] * d4[8] + d2[2] * e142
            if not (d1[2] <= 0.0 or d2[2] <= 0.0 or d3[6] > 0.0 or d4[11] > 0.0):
                n_simplex_points = 2
                sum = d1[2] + d2[2]
                barycentric_coordinates[0] = d1[2] / sum
                barycentric_coordinates[1] = 1.0 - barycentric_coordinates[0]
                search_direction[:] = simplex[1] + barycentric_coordinates[0] * (simplex[0] - simplex[1])
                dstsq = np.dot(search_direction, search_direction)
                return dstsq, n_simplex_points, backup
            # check optimality of line segment 1-3
            e123 = dot_product_table[2, 0] - dot_product_table[2, 1]
            e143 = dot_product_table[2, 0] - dot_product_table[3, 2]
            d1[4] = dot_product_table[2, 2] - dot_product_table[2, 0]
            d2[6] = d1[4] * d2[2] + d3[4] * e123
            d4[12] = d1[4] * d4[8] + d3[4] * e143
            if not (d1[4] <= 0.0 or d2[6] > 0.0 or d3[4] <= 0.0 or d4[12] > 0.0):
                n_simplex_points = 2
                old_indices_polytope1[1] = old_indices_polytope1[2]
                old_indices_polytope2[1] = old_indices_polytope2[2]
                sum = d1[4] + d3[4]
                barycentric_coordinates[0] = d1[4] / sum
                barycentric_coordinates[1] = 1.0 - barycentric_coordinates[0]
                search_direction[:] = simplex[2] + barycentric_coordinates[0] * (simplex[0] - simplex[2])
                dstsq = np.dot(search_direction, search_direction)
                simplex[1] = simplex[2]
                dot_product_table[1, 0] = dot_product_table[2, 0]
                dot_product_table[1, 1] = dot_product_table[2, 2]
                return dstsq, n_simplex_points, backup
            # check optimality of face 123
            d2[5] = dot_product_table[2, 2] - dot_product_table[2, 1]
            d3[5] = dot_product_table[1, 1] - dot_product_table[2, 1]
            e213 = -e123
            d1[6] = d2[5] * d1[2] + d3[5] * e213
            d4[14] = d1[6] * d4[8] + d2[6] * e142 + d3[6] * e143
            if not (d1[6] <= 0.0 or d2[6] <= 0.0 or d3[6] <= 0.0 or d4[14] > 0.0):
                n_simplex_points = 3
                sum = d1[6] + d2[6] + d3[6]
                barycentric_coordinates[0] = d1[6] / sum
                barycentric_coordinates[1] = d2[6] / sum
                barycentric_coordinates[2] = 1.0 - barycentric_coordinates[0] - barycentric_coordinates[1]
                search_direction[:] = simplex[2] + barycentric_coordinates[0] * (simplex[0] - simplex[2]) + barycentric_coordinates[1] * (simplex[1] - simplex[2])
                dstsq = np.dot(search_direction, search_direction)
                return dstsq, n_simplex_points, backup
            # check optimality of line segment 1-4
            e124 = dot_product_table[3, 0] - dot_product_table[3, 1]
            e134 = dot_product_table[3, 0] - dot_product_table[3, 2]
            d1[8] = dot_product_table[3, 3] - dot_product_table[3, 0]
            d2[11] = d1[8] * d2[2] + d4[8] * e124
            d3[12] = d1[8] * d3[4] + d4[8] * e134
            if not (d1[8] <= 0.0 or d2[11] > 0.0 or d3[12] > 0.0 or d4[8] <= 0.0):
                n_simplex_points = 2
                old_indices_polytope1[1] = old_indices_polytope1[3]
                old_indices_polytope2[1] = old_indices_polytope2[3]
                sum = d1[8] + d4[8]
                barycentric_coordinates[0] = d1[8] / sum
                barycentric_coordinates[1] = 1.0 - barycentric_coordinates[0]
                search_direction[:] = simplex[3] + barycentric_coordinates[0] * (simplex[0] - simplex[3])
                dstsq = np.dot(search_direction, search_direction)
                simplex[1] = simplex[3]
                dot_product_table[1, 0] = dot_product_table[3, 0]
                dot_product_table[1, 1] = dot_product_table[3, 3]
                return dstsq, n_simplex_points, backup
            # check optimality of face 1-2-4
            d2[9] = dot_product_table[3, 3] - dot_product_table[3, 1]
            d4[9] = dot_product_table[1, 1] - dot_product_table[3, 1]
            e214 = -e124
            d1[11] = d2[9] * d1[2] + d4[9] * e214
            d3[14] = d1[11] * d3[4] + d2[11] * e132 + d4[11] * e134
            if not (d1[11] <= 0.0 or d2[11] <= 0.0 or d3[14] > 0.0 or d4[11] <= 0.0):
                n_simplex_points = 3
                old_indices_polytope1[2] = old_indices_polytope1[3]
                old_indices_polytope2[2] = old_indices_polytope2[3]
                sum = d1[11] + d2[11] + d4[11]
                barycentric_coordinates[0] = d1[11] / sum
                barycentric_coordinates[1] = d2[11] / sum
                barycentric_coordinates[2] = 1.0 - barycentric_coordinates[0] - barycentric_coordinates[1]
                search_direction[:] = simplex[3] + barycentric_coordinates[0] * (simplex[0] - simplex[3]) + barycentric_coordinates[1] * (simplex[1] - simplex[3])
                dstsq = np.dot(search_direction, search_direction)
                simplex[2] = simplex[3]
                dot_product_table[2, 0] = dot_product_table[3, 0]
                dot_product_table[2, 1] = dot_product_table[3, 1]
                dot_product_table[2, 2] = dot_product_table[3, 3]
                return dstsq, n_simplex_points, backup
            # check optimality of face 1-3-4
            d3[10] = dot_product_table[3, 3] - dot_product_table[3, 2]
            d4[10] = dot_product_table[2, 2] - dot_product_table[3, 2]
            e314 = -e134
            d1[12] = d3[10] * d1[4] + d4[10] * e314
            d2[14] = d1[12] * d2[2] + d3[12] * e123 + d4[12] * e124
            if not (d1[12] <= 0.0 or d2[14] > 0.0 or d3[12] <= 0.0 or d4[12] <= 0.0):
                n_simplex_points = 3
                old_indices_polytope1[1] = old_indices_polytope1[3]
                old_indices_polytope2[1] = old_indices_polytope2[3]
                sum = d1[12] + d3[12] + d4[12]
                barycentric_coordinates[0] = d1[12] / sum
                barycentric_coordinates[2] = d3[12] / sum
                barycentric_coordinates[1] = 1.0 - barycentric_coordinates[0] - barycentric_coordinates[2]
                search_direction[:] = simplex[3] + barycentric_coordinates[0] * (simplex[0] - simplex[3]) + barycentric_coordinates[2] * (simplex[2] - simplex[3])
                dstsq = np.dot(search_direction, search_direction)
                simplex[1] = simplex[3]
                dot_product_table[1, 0] = dot_product_table[3, 0]
                dot_product_table[1, 1] = dot_product_table[3, 3]
                dot_product_table[2, 1] = dot_product_table[3, 2]
                return dstsq, n_simplex_points, backup
            # check optimality of the hull of all 4 points
            e243 = dot_product_table[2, 1] - dot_product_table[3, 2]
            d4[13] = d2[5] * d4[9] + d3[5] * e243
            e234 = dot_product_table[3, 1] - dot_product_table[3, 2]
            d3[13] = d2[9] * d3[5] + d4[9] * e234
            e324 = -e234
            d2[13] = d3[10] * d2[5] + d4[10] * e324
            d1[14] = d2[13] * d1[2] + d3[13] * e213 + d4[13] * e214
            if not (d1[14] <= 0.0 or d2[14] <= 0.0 or d3[14] <= 0.0 or d4[14] <= 0.0):
                sum = d1[14] + d2[14] + d3[14] + d4[14]
                barycentric_coordinates[0] = d1[14] / sum
                barycentric_coordinates[1] = d2[14] / sum
                barycentric_coordinates[2] = d3[14] / sum
                barycentric_coordinates[3] = 1.0 - barycentric_coordinates[0] - barycentric_coordinates[1] - barycentric_coordinates[2]
                search_direction[:] = barycentric_coordinates[0] * simplex[0] + barycentric_coordinates[1] * simplex[1] + barycentric_coordinates[2] * simplex[2] + barycentric_coordinates[3] * simplex[3]
                dstsq = np.dot(search_direction, search_direction)
                return dstsq, n_simplex_points, backup
            # check optimality of vertex 2
            if not (d1[2] > 0.0 or d3[5] > 0.0 or d4[9] > 0.0):
                n_simplex_points = 1
                old_indices_polytope1[0] = old_indices_polytope1[1]
                old_indices_polytope2[0] = old_indices_polytope2[1]
                barycentric_coordinates[0] = d2[1]
                search_direction[:] = simplex[1]
                dstsq = dot_product_table[1, 1]
                simplex[0] = simplex[1]
                dot_product_table[0, 0] = dot_product_table[1, 1]
                return dstsq, n_simplex_points, backup
            # check optimality of vertex 3
            if not (d1[4] > 0.0 or d2[5] > 0.0 or d4[10] > 0.0):
                n_simplex_points = 1
                old_indices_polytope1[0] = old_indices_polytope1[2]
                old_indices_polytope2[0] = old_indices_polytope2[2]
                barycentric_coordinates[0] = d3[3]
                search_direction[:] = simplex[2]
                dstsq = dot_product_table[2, 2]
                simplex[0] = simplex[2]
                dot_product_table[0, 0] = dot_product_table[2, 2]
                return dstsq, n_simplex_points, backup
            # check optimality of vertex 4
            if not (d1[8] > 0.0 or d2[9] > 0.0 or d3[10] > 0.0):
                n_simplex_points = 1
                old_indices_polytope1[0] = old_indices_polytope1[3]
                old_indices_polytope2[0] = old_indices_polytope2[3]
                barycentric_coordinates[0] = d4[7]
                search_direction[:] = simplex[3]
                dstsq = dot_product_table[3, 3]
                simplex[0] = simplex[3]
                dot_product_table[0, 0] = dot_product_table[3, 3]
                return dstsq, n_simplex_points, backup
            # check optimality of line segment 2-3
            if not (d1[6] > 0.0 or d2[5] <= 0.0 or d3[5] <= 0.0 or d4[13] > 0.0):
                n_simplex_points = 2
                old_indices_polytope1[0] = old_indices_polytope1[2]
                old_indices_polytope2[0] = old_indices_polytope2[2]
                sum = d2[5] + d3[5]
                barycentric_coordinates[1] = d2[5] / sum
                barycentric_coordinates[0] = 1.0 - barycentric_coordinates[1]
                search_direction[:] = simplex[2] + barycentric_coordinates[1] * (simplex[1] - simplex[2])
                dstsq = np.dot(search_direction, search_direction)
                simplex[0] = simplex[2]
                dot_product_table[1, 0] = dot_product_table[2, 1]
                dot_product_table[0, 0] = dot_product_table[2, 2]
                return dstsq, n_simplex_points, backup
            # check optimality of line segment 2-4
            if not (d1[11] > 0.0 or d2[9] <= 0.0 or d3[13] > 0.0 or d4[9] <= 0.0):
                n_simplex_points = 2
                old_indices_polytope1[0] = old_indices_polytope1[3]
                old_indices_polytope2[0] = old_indices_polytope2[3]
                sum = d2[9] + d4[9]
                barycentric_coordinates[1] = d2[9] / sum
                barycentric_coordinates[0] = 1.0 - barycentric_coordinates[1]
                search_direction[:] = simplex[3] + barycentric_coordinates[1] * (simplex[1] - simplex[3])
                dstsq = np.dot(search_direction, search_direction)
                simplex[0] = simplex[3]
                dot_product_table[1, 0] = dot_product_table[3, 1]
                dot_product_table[0, 0] = dot_product_table[3, 3]
                return dstsq, n_simplex_points, backup
            # check optimality of line segment 3-4
            if not (d1[12] > 0.0 or d2[13] > 0.0 or d3[10] <= 0.0 or d4[10] <= 0.0):
                n_simplex_points = 2
                old_indices_polytope1[0] = old_indices_polytope1[2]
                old_indices_polytope1[1] = old_indices_polytope1[3]
                old_indices_polytope2[0] = old_indices_polytope2[2]
                old_indices_polytope2[1] = old_indices_polytope2[3]
                sum = d3[10] + d4[10]
                barycentric_coordinates[0] = d3[10] / sum
                barycentric_coordinates[1] = 1.0 - barycentric_coordinates[0]
                search_direction[:] = simplex[3] + barycentric_coordinates[0] * (simplex[2] - simplex[3])
                dstsq = np.dot(search_direction, search_direction)
                simplex[0] = simplex[2]
                simplex[1] = simplex[3]
                dot_product_table[0, 0] = dot_product_table[2, 2]
                dot_product_table[1, 0] = dot_product_table[3, 2]
                dot_product_table[1, 1] = dot_product_table[3, 3]
                return dstsq, n_simplex_points, backup
            # check optimality of face 2-3-4
            if not (d1[14] > 0.0 or d2[13] <= 0.0 or d3[13] <= 0.0 or d4[13] <= 0.0):
                n_simplex_points = 3
                old_indices_polytope1[0] = old_indices_polytope1[3]
                old_indices_polytope2[0] = old_indices_polytope2[3]
                sum = d2[13] + d3[13] + d4[13]
                barycentric_coordinates[1] = d2[13] / sum
                barycentric_coordinates[2] = d3[13] / sum
                barycentric_coordinates[0] = 1.0 - barycentric_coordinates[1] - barycentric_coordinates[2]
                search_direction[:] = simplex[3] + barycentric_coordinates[1] * (simplex[1] - simplex[3]) + barycentric_coordinates[2] * (simplex[2] - simplex[3])
                dstsq = np.dot(search_direction, search_direction)
                simplex[0] = simplex[3]
                dot_product_table[0, 0] = dot_product_table[3, 3]
                dot_product_table[1, 0] = dot_product_table[3, 1]
                dot_product_table[2, 0] = dot_product_table[3, 2]
                return dstsq, n_simplex_points, backup
        else:
            raise ValueError("Invalid value for nvs %d given" % n_simplex_points)

    # ======================================================================
    # The backup procedure  begins ...
    # ======================================================================
    if n_simplex_points == 1:
        # case of a single point ...
        dstsq = dot_product_table[0, 0]
        barycentric_coordinates[0] = d1[0]
        search_direction[:] = simplex[0]
        backup = 1
        return dstsq, n_simplex_points, backup
    elif n_simplex_points == 2:
        # case of two points ...
        if backup:
            d2[2] = dot_product_table[0, 0] - dot_product_table[1, 0]
            d1[2] = dot_product_table[1, 1] - dot_product_table[1, 0]
        # check vertex 1
        dstsq = dot_product_table[0, 0]
        nvsd = 1
        barycentric_coordinates[0] = d1[0]
        search_direction[:] = simplex[0]
        iord[0] = 0
        # check line segment 1-2
        if not (d1[2] <= 0.0 or d2[2] <= 0.0):
            sum = d1[2] + d2[2]
            alsd[0] = d1[2] / sum
            alsd[1] = 1.0 - alsd[0]
            zsold[:] = simplex[1, :] + alsd[0] * (simplex[0, :] - simplex[1, :])
            dstsqd = zsold[0] * zsold[0] + zsold[1] * zsold[1] + zsold[2] * zsold[2]
            if dstsqd < dstsq:
                dstsq = dstsqd
                nvsd = 2
                barycentric_coordinates[:nvsd] = alsd[:nvsd]
                search_direction[:] = zsold[:]
                iord[0] = 0
                iord[1] = 1
        # check vertex 2
        if dot_product_table[1, 1] < dstsq:
            dstsq = dot_product_table[1, 1]
            nvsd = 1
            barycentric_coordinates[0] = d2[1]
            search_direction[:] = simplex[1]
            iord[0] = 1
    elif n_simplex_points == 3:
        # case of three points
        if backup:
            d2[2] = dot_product_table[0, 0] - dot_product_table[1, 0]
            d3[4] = dot_product_table[0, 0] - dot_product_table[2, 0]
            e132 = dot_product_table[1, 0] - dot_product_table[2, 1]
            d1[2] = dot_product_table[1, 1] - dot_product_table[1, 0]
            d3[6] = d1[2] * d3[4] + d2[2] * e132
            e123 = dot_product_table[2, 0] - dot_product_table[2, 1]
            d1[4] = dot_product_table[2, 2] - dot_product_table[2, 0]
            d2[6] = d1[4] * d2[2] + d3[4] * e123
            e213 = -e123
            d2[5] = dot_product_table[2, 2] - dot_product_table[2, 1]
            d3[5] = dot_product_table[1, 1] - dot_product_table[2, 1]
            d1[6] = d2[5] * d1[2] + d3[5] * e213
        # check vertex 1
        dstsq = dot_product_table[0, 0]
        nvsd = 1
        barycentric_coordinates[0] = d1[0]
        search_direction[:] = simplex[0]
        iord[0] = 0
        # check line segment 1-2
        if not (d1[2] <= 0.0 or d2[2] <= 0.0):
            sum = d1[2] + d2[2]
            alsd[0] = d1[2] / sum
            alsd[1] = 1.0 - alsd[0]
            zsold[:] = simplex[1] + alsd[0] * (simplex[0] - simplex[1])
            dstsqd = np.dot(zsold, zsold)
            if dstsqd < dstsq:
                dstsq = dstsqd
                nvsd = 2
                barycentric_coordinates[:nvsd] = alsd[:nvsd]
                search_direction[:] = zsold
                iord[0] = 0
                iord[1] = 1
        # check line segment 1-3
        if not (d1[4] <= 0.0 or d3[4] <= 0.0):
            sum = d1[4] + d3[4]
            alsd[0] = d1[4] / sum
            alsd[1] = 1.0 - alsd[0]
            zsold[:] = simplex[2] + alsd[0] * (simplex[0] - simplex[2])
            dstsqd = np.dot(zsold, zsold)
            if dstsqd < dstsq:
                dstsq = dstsqd
                nvsd = 2
                barycentric_coordinates[:nvsd] = alsd[:nvsd]
                search_direction[:] = zsold
                iord[0] = 0
                iord[1] = 2
        # check face 1-2-3
        if not (d1[6] <= 0.0 or d2[6] <= 0.0 or d3[6] <= 0.0):
            sum = d1[6] + d2[6] + d3[6]
            alsd[0] = d1[6] / sum
            alsd[1] = d2[6] / sum
            alsd[2] = 1.0 - alsd[0] - alsd[1]
            zsold[:] = simplex[2] + alsd[0] * (simplex[0] - simplex[2]) + alsd[1] * (simplex[1] - simplex[2])
            dstsqd = np.dot(zsold, zsold)
            if dstsqd < dstsq:
                dstsq = dstsqd
                nvsd = 3
                barycentric_coordinates[:] = alsd
                search_direction[:] = zsold
                iord[0] = 0
                iord[1] = 1
                iord[2] = 2
        # check vertex 2
        if dot_product_table[1, 1] < dstsq:
            nvsd = 1
            dstsq = dot_product_table[1, 1]
            barycentric_coordinates[0] = d2[1]
            search_direction[:] = simplex[1]
            iord[0] = 1
        # check vertex 3
        if dot_product_table[2, 2] < dstsq:
            nvsd = 1
            dstsq = dot_product_table[2, 2]
            barycentric_coordinates[0] = d3[3]
            search_direction[:] = simplex[2]
            iord[0] = 2
        # check line segment 2-3
        if not (d2[5] <= 0.0 or d3[5] <= 0.0):
            sum = d2[5] + d3[5]
            alsd[1] = d2[5] / sum
            alsd[0] = 1.0 - alsd[1]
            zsold[:] = simplex[2] + alsd[1] * (simplex[1] - simplex[2])
            dstsqd = np.dot(zsold, zsold)
            if dstsqd < dstsq:
                dstsq = dstsqd
                nvsd = 2
                barycentric_coordinates[:nvsd] = alsd[:nvsd]
                search_direction[:] = zsold
                iord[0] = 2
                iord[1] = 1
    elif n_simplex_points == 4:
        # case of four points
        if backup:
            d2[2] = dot_product_table[0, 0] - dot_product_table[1, 0]
            d3[4] = dot_product_table[0, 0] - dot_product_table[2, 0]
            d4[8] = dot_product_table[0, 0] - dot_product_table[3, 0]
            e132 = dot_product_table[1, 0] - dot_product_table[2, 1]
            e142 = dot_product_table[1, 0] - dot_product_table[3, 1]
            d1[2] = dot_product_table[1, 1] - dot_product_table[1, 0]
            d3[6] = d1[2] * d3[4] + d2[2] * e132
            d4[11] = d1[2] * d4[8] + d2[2] * e142
            e123 = dot_product_table[2, 0] - dot_product_table[2, 1]
            e143 = dot_product_table[2, 0] - dot_product_table[3, 2]
            d1[4] = dot_product_table[2, 2] - dot_product_table[2, 0]
            d2[6] = d1[4] * d2[2] + d3[4] * e123
            d4[12] = d1[4] * d4[8] + d3[4] * e143
            d2[5] = dot_product_table[2, 2] - dot_product_table[2, 1]
            d3[5] = dot_product_table[1, 1] - dot_product_table[2, 1]
            e213 = -e123
            d1[6] = d2[5] * d1[2] + d3[5] * e213
            d4[14] = d1[6] * d4[8] + d2[6] * e142 + d3[6] * e143
            e124 = dot_product_table[3, 0] - dot_product_table[3, 1]
            e134 = dot_product_table[3, 0] - dot_product_table[3, 2]
            d1[8] = dot_product_table[3, 3] - dot_product_table[3, 0]
            d2[11] = d1[8] * d2[2] + d4[8] * e124
            d3[12] = d1[8] * d3[4] + d4[8] * e134
            d2[9] = dot_product_table[3, 3] - dot_product_table[3, 1]
            d4[9] = dot_product_table[1, 1] - dot_product_table[3, 1]
            e214 = -e124
            d1[11] = d2[9] * d1[2] + d4[9] * e214
            d3[14] = d1[11] * d3[4] + d2[11] * e132 + d4[11] * e134
            d3[10] = dot_product_table[3, 3] - dot_product_table[3, 2]
            d4[10] = dot_product_table[2, 2] - dot_product_table[3, 2]
            e314 = -e134
            d1[12] = d3[10] * d1[4] + d4[10] * e314
            d2[14] = d1[12] * d2[2] + d3[12] * e123 + d4[12] * e124
            e243 = dot_product_table[2, 1] - dot_product_table[3, 2]
            d4[13] = d2[5] * d4[9] + d3[5] * e243
            e234 = dot_product_table[3, 1] - dot_product_table[3, 2]
            d3[13] = d2[9] * d3[5] + d4[9] * e234
            e324 = -e234
            d2[13] = d3[10] * d2[5] + d4[10] * e324
            d1[14] = d2[13] * d1[2] + d3[13] * e213 + d4[13] * e214
        # check vertex 1
        dstsq = dot_product_table[0, 0]
        nvsd = 1
        barycentric_coordinates[0] = d1[0]
        search_direction[:] = simplex[0]
        iord[0] = 0
        # check line segment 1-2
        if not (d1[2] <= 0.0 or d2[2] <= 0.0):
            sum = d1[2] + d2[2]
            alsd[0] = d1[2] / sum
            alsd[1] = 1.0 - alsd[0]
            zsold[:] = simplex[1] + alsd[0] * (simplex[0] - simplex[1])
            dstsqd = np.dot(zsold, zsold)
            if dstsqd < dstsq:
                dstsq = dstsqd
                nvsd = 2
                barycentric_coordinates[:nvsd] = alsd[:nvsd]
                search_direction[:] = zsold
                iord[0] = 0
                iord[1] = 1
        # check line segment 1-3
        if not (d1[4] <= 0.0 or d3[4] <= 0.0):
            sum = d1[4] + d3[4]
            alsd[0] = d1[4] / sum
            alsd[1] = 1.0 - alsd[0]
            zsold[:] = simplex[2] + alsd[0] * (simplex[0] - simplex[2])
            dstsqd = np.dot(zsold, zsold)
            if dstsqd < dstsq:
                dstsq = dstsqd
                nvsd = 2
                barycentric_coordinates[:nvsd] = alsd[:nvsd]
                search_direction[:] = zsold
                iord[0] = 0
                iord[1] = 2
        # check face 1-2-3
        if not (d1[6] <= 0.0 or d2[6] <= 0.0 or d3[6] <= 0.0):
            sum = d1[6] + d2[6] + d3[6]
            alsd[0] = d1[6] / sum
            alsd[1] = d2[6] / sum
            alsd[2] = 1.0 - alsd[0] - alsd[1]
            zsold[:] = simplex[2] + alsd[0] * (simplex[0] - simplex[2]) + alsd[1] * (simplex[1] - simplex[2])
            dstsqd = np.dot(zsold, zsold)
            if dstsqd < dstsq:
                dstsq = dstsqd
                nvsd = 3
                barycentric_coordinates[:] = alsd
                search_direction[:] = zsold
                iord[0] = 0
                iord[1] = 1
                iord[2] = 2
        # check line segment 1-4
        if not (d1[8] <= 0.0 or d4[8] <= 0.0):
            sum = d1[8] + d4[8]
            alsd[0] = d1[8] / sum
            alsd[1] = 1.0 - alsd[0]
            zsold[:] = simplex[3] + alsd[0] * (simplex[0] - simplex[3])
            dstsqd = np.dot(zsold, zsold)
            if dstsqd < dstsq:
                dstsq = dstsqd
                nvsd = 2
                barycentric_coordinates[:nvsd] = alsd[:nvsd]
                search_direction[:] = zsold
                iord[0] = 0
                iord[1] = 3
        # check face 1-2-4
        if not (d1[11] <= 0.0 or d2[11] <= 0.0 or d4[11] <= 0.0):
            sum = d1[11] + d2[11] + d4[11]
            alsd[0] = d1[11] / sum
            alsd[1] = d2[11] / sum
            alsd[2] = 1.0 - alsd[0] - alsd[1]
            zsold[:] = simplex[3] + alsd[0] * (simplex[0] - simplex[3]) + alsd[1] * (simplex[1] - simplex[3])
            dstsqd = np.dot(zsold, zsold)
            if dstsqd < dstsq:
                dstsq = dstsqd
                nvsd = 3
                barycentric_coordinates[:nvsd] = alsd[:nvsd]
                search_direction[:] = zsold
                iord[0] = 0
                iord[1] = 1
                iord[2] = 3
        # check face 1-3-4
        if not (d1[12] <= 0.0 or d3[12] <= 0.0 or d4[12] <= 0.0):
            sum = d1[12] + d3[12] + d4[12]
            alsd[0] = d1[12] / sum
            alsd[2] = d3[12] / sum
            alsd[1] = 1.0 - alsd[0] - alsd[2]
            zsold[:] = simplex[3] + alsd[0] * (simplex[0] - simplex[3]) + alsd[2] * (simplex[2] - simplex[3])
            dstsqd = np.dot(zsold, zsold)
            if dstsqd < dstsq:
                dstsq = dstsqd
                nvsd = 3
                barycentric_coordinates[:nvsd] = alsd[:nvsd]
                search_direction[:] = zsold
                iord[0] = 0
                iord[1] = 3
                iord[2] = 2
        # check the hull of all 4 points
        if not (d1[14] <= 0.0 or d2[14] <= 0.0 or d3[14] <= 0.0 or d4[14] <= 0.0):
            sum = d1[14] + d2[14] + d3[14] + d4[14]
            alsd[0] = d1[14] / sum
            alsd[1] = d2[14] / sum
            alsd[2] = d3[14] / sum
            alsd[3] = 1.0 - alsd[0] - alsd[1] - alsd[2]
            zsold[:] = alsd[0] * simplex[0] + alsd[1] * simplex[1] + alsd[2] * simplex[2] + alsd[3] * simplex[3]
            dstsqd = np.dot(zsold, zsold)
            if dstsqd < dstsq:
                dstsq = dstsqd
                nvsd = 4
                barycentric_coordinates[:] = alsd
                search_direction[:] = zsold
                iord[0] = 0
                iord[1] = 1
                iord[2] = 2
                iord[3] = 3
        # check vertex 2
        if dot_product_table[1, 1] < dstsq:
            nvsd = 1
            dstsq = dot_product_table[1, 1]
            barycentric_coordinates[0] = d2[1]
            search_direction[:] = simplex[1]
            iord[0] = 1
        # check vertex 3
        if dot_product_table[2, 2] < dstsq:
            nvsd = 1
            dstsq = dot_product_table[2, 2]
            barycentric_coordinates[0] = d3[3]
            search_direction[:] = simplex[2]
            iord[0] = 2
        # check vertex 4
        if dot_product_table[3, 3] < dstsq:
            nvsd = 1
            dstsq = dot_product_table[3, 3]
            barycentric_coordinates[0] = d4[7]
            search_direction[:] = simplex[3]
            iord[0] = 3
        # check line segment 2-3
        if not (d2[5] <= 0.0 or d3[5] <= 0.0):
            sum = d2[5] + d3[5]
            alsd[1] = d2[5] / sum
            alsd[0] = 1.0 - alsd[1]
            zsold[:] = simplex[2] + alsd[1] * (simplex[1] - simplex[2])
            dstsqd = np.dot(zsold, zsold)
            if dstsqd < dstsq:
                dstsq = dstsqd
                nvsd = 2
                barycentric_coordinates[:nvsd] = alsd[:nvsd]
                search_direction[:] = zsold
                iord[0] = 2
                iord[1] = 1
        # check line segment 2-4
        if not (d2[9] <= 0.0 or d4[9] <= 0.0):
            sum = d2[9] + d4[9]
            alsd[1] = d2[9] / sum
            alsd[0] = 1.0 - alsd[1]
            zsold[:] = simplex[3] + alsd[1] * (simplex[1] - simplex[3])
            dstsqd = np.dot(zsold, zsold)
            if dstsqd < dstsq:
                dstsq = dstsqd
                nvsd = 2
                barycentric_coordinates[:nvsd] = alsd[:nvsd]
                search_direction[:] = zsold
                iord[0] = 3
                iord[1] = 1
        # check line segment 3-4
        if not (d3[10] <= 0.0 or d4[10] <= 0.0):
            sum = d3[10] + d4[10]
            alsd[0] = d3[10] / sum
            alsd[1] = 1.0 - alsd[0]
            zsold[:] = simplex[3] + alsd[0] * (simplex[2] - simplex[3])
            dstsqd = np.dot(zsold, zsold)
            if dstsqd < dstsq:
                dstsq = dstsqd
                nvsd = 2
                barycentric_coordinates[:nvsd] = alsd[:nvsd]
                search_direction[:] = zsold
                iord[0] = 2
                iord[1] = 3
        # check face 2-3-4
        if not (d2[13] <= 0.0 or d3[13] <= 0.0 or d4[13] <= 0.0):
            sum = d2[13] + d3[13] + d4[13]
            alsd[1] = d2[13] / sum
            alsd[2] = d3[13] / sum
            alsd[0] = 1.0 - alsd[1] - alsd[2]
            zsold[:] = simplex[3] + alsd[1] * (simplex[1] - simplex[3]) + alsd[2] * (simplex[2] - simplex[3])
            dstsqd = np.dot(zsold, zsold)
            if dstsqd < dstsq:
                dstsq = dstsqd
                nvsd = 3
                barycentric_coordinates[:nvsd] = alsd[:nvsd]
                search_direction[:] = zsold
                iord[0] = 3
                iord[1] = 1
                iord[2] = 2
    # final reordering
    risd[:n_simplex_points] = old_indices_polytope1[:n_simplex_points]
    rjsd[:n_simplex_points] = old_indices_polytope2[:n_simplex_points]
    yd[:n_simplex_points] = simplex[:n_simplex_points]
    for k in range(n_simplex_points):
        delld[k, :k + 1] = dot_product_table[k, :k + 1]

    n_simplex_points = nvsd
    for k in range(n_simplex_points):
        kk = iord[k]
        old_indices_polytope1[k] = risd[kk]
        old_indices_polytope2[k] = rjsd[kk]
        simplex[k] = yd[kk]
        for l in range(k):
            ll = iord[l]
            if kk >= ll:
                dot_product_table[k, l] = delld[kk, ll]
            else:
                dot_product_table[k, l] = delld[ll, kk]
        dot_product_table[k, k] = delld[kk, kk]
    backup = 1
    return dstsq, n_simplex_points, backup


def _revert_to_old_simplex(
        dot_product_table, indices_polytope1, indices_polytope2,
        old_dot_product_table, old_indices_polytope1, old_indices_polytope2,
        old_simplex, n_old_simplex_points, simplex):
    simplex[:n_old_simplex_points] = old_simplex[:n_old_simplex_points]
    indices_polytope1[:n_old_simplex_points] = old_indices_polytope1[:n_old_simplex_points]
    indices_polytope2[:n_old_simplex_points] = old_indices_polytope2[:n_old_simplex_points]
    dot_product_table[:n_old_simplex_points] = old_dot_product_table[:n_old_simplex_points]
    return n_old_simplex_points


def _add_new_point(
        dot_product_table, indices_polytope1, indices_polytope2,
        n_simplex_points, new_index1, new_index2, simplex, new_simplex_point):
    # Move first point to last spot
    indices_polytope1[n_simplex_points] = indices_polytope1[0]
    indices_polytope2[n_simplex_points] = indices_polytope2[0]
    simplex[n_simplex_points] = simplex[0]
    dot_product_table[n_simplex_points, :n_simplex_points] = dot_product_table[:n_simplex_points, 0]
    dot_product_table[n_simplex_points, n_simplex_points] = dot_product_table[0, 0]
    # Put new point in first spot
    indices_polytope1[0] = new_index1
    indices_polytope2[0] = new_index2
    simplex[0] = new_simplex_point
    # Update dot product table
    n_simplex_points += 1
    dot_product_table[:n_simplex_points, 0] = np.dot(simplex[:n_simplex_points], simplex[0])
    return n_simplex_points


def _save_old_simplex(
        dot_product_table, indices_polytope1, indices_polytope2,
        n_simplex_points, old_dot_product_table, old_indices_polytope1,
        old_indices_polytope2, old_simplex, simplex):
    # Save old values of n_simplex_points, indices_polytope1,
    # indices_polytope2, simplex and dot_product_table
    oldnvs = n_simplex_points
    old_simplex[:n_simplex_points] = simplex[:n_simplex_points]
    old_indices_polytope1[:n_simplex_points] = indices_polytope1[:n_simplex_points]
    old_indices_polytope2[:n_simplex_points] = indices_polytope2[:n_simplex_points]
    for k in range(n_simplex_points):
        old_dot_product_table[k, :k + 1] = dot_product_table[k, :k + 1]
    return oldnvs


def _reorder_simplex(
        dot_product_table, indices_polytope1, indices_polytope2, iord,
        n_simplex_points, old_dot_product_table, old_indices_polytope1,
        old_indices_polytope2, old_simplex, simplex):
    # If n_simplex_points == 4, rearrange dot_product_table[1, 0],
    # dot_product_table[2, 1] and dot_product_table[3, 0] in non decreasing
    # order
    if n_simplex_points == 4:
        iord[:3] = 0, 1, 2
        if dot_product_table[2, 0] < dot_product_table[1, 0]:
            iord[1] = 2
            iord[2] = 1
        ii = iord[1]
        if dot_product_table[3, 0] < dot_product_table[ii, 0]:
            iord[3] = iord[2]
            iord[2] = iord[1]
            iord[1] = 3
        else:
            ii = iord[2]
            if dot_product_table[3, 0] < dot_product_table[ii, 0]:
                iord[3] = iord[2]
                iord[2] = 3
            else:
                iord[3] = 3
        # Reorder indices_polytope1, indices_polytope2 simplex and dot_product_table
        for k in range(1, n_simplex_points):
            kk = iord[k]
            indices_polytope1[k] = old_indices_polytope1[kk]
            indices_polytope2[k] = old_indices_polytope2[kk]
            simplex[k] = old_simplex[kk]
            for l in range(k):
                ll = iord[l]
                if kk >= ll:
                    dot_product_table[k, l] = old_dot_product_table[kk, ll]
                else:
                    dot_product_table[k, l] = old_dot_product_table[ll, kk]
            dot_product_table[k, k] = old_dot_product_table[kk, kk]


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
