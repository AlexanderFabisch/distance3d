import numpy as np


def distance_subalgorithm_python(
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
    iord = np.empty(4, dtype=int)
    d1 = np.empty(15, dtype=float)
    d2 = np.empty(15, dtype=float)
    d3 = np.empty(15, dtype=float)
    d4 = np.empty(15, dtype=float)
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
    risd = np.empty(4, dtype=int)
    risd[:n_simplex_points] = old_indices_polytope1[:n_simplex_points]
    rjsd = np.empty(4, dtype=int)
    rjsd[:n_simplex_points] = old_indices_polytope2[:n_simplex_points]
    yd = np.empty((4, 3), dtype=float)
    yd[:n_simplex_points] = simplex[:n_simplex_points]
    delld = np.empty((4, 4), dtype=float)
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
