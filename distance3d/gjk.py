import math
import numpy as np
from .colliders import Convex
try:
    from ._gjk import distance_subalgorithm as default_dsa
except ImportError:
    from ._gjk_python import distance_subalgorithm_python as default_dsa


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


def gjk_with_simplex(collider1, collider2, distance_subalgorithm=default_dsa):
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
    collider1 : ConvexCollider
        Convex collider 1.

    collider2 : ConvexCollider
        Convex collider 2.

    distance_subalgorithm : callable, optional (default: C function if available)
        Distance subalgorithm.

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
                    distance = math.sqrt(max(dstsq, 0))

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
