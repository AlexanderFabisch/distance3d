import math
from enum import Enum
import numba
import numpy as np

from distance3d.utils import EPSILON
from distance3d.minkowski import Simplex, support_function, make_support_point
from distance3d.distance import point_to_triangle


EPSILON_SQRT = math.sqrt(EPSILON)


def gjk_intersection(collider1, collider2, max_iterations=100):
    """Intersection test with Gilbert-Johnson-Keerthi (GJK) algorithm.

    This implementation of GJK is based on libccd (for details, see
    https://github.com/danfis/libccd). For the original code the copyright is
    of Daniel Fiser <danfis@danfis.cz>. It has been released under 3-clause BSD
    license.

    Parameters
    ----------
    collider1 : ConvexCollider
        Convex collider 1.

    collider2 : ConvexCollider
        Convex collider 2.

    max_iterations : int, optional (default: 100)
        Maximum number of iterations.

    Returns
    -------
    intersection : bool
        Do the two colliders intersect?
    """
    return _gjk(collider1, collider2, Simplex(), max_iterations)[0]


def _gjk(collider1, collider2, simplex, max_iterations):
    support_point = make_support_point(
        collider1.first_vertex(), collider2.first_vertex())
    simplex.add_point(*support_point)
    search_direction = -support_point[0]

    for _ in range(max_iterations):
        support_point = support_function(collider1, collider2, search_direction)
        is_origin = np.dot(support_point[0], support_point[0]) < EPSILON
        if is_origin:
            _set_point(simplex.v, simplex.v1, simplex.v2, 0, *support_point)
            simplex.n_points = 1
            return True, simplex

        is_before_origin = np.dot(support_point[0], search_direction) < -EPSILON_SQRT
        if is_before_origin:
            return False, simplex

        simplex.add_point(*support_point)

        gjk_state, search_direction, simplex.n_points = _refine_simplex(
            simplex.v, simplex.v1, simplex.v2, len(simplex))

        if gjk_state == GjkState.CONTACT:
            return True, simplex
        elif gjk_state == GjkState.NO_CONTACT:
            return False, simplex
        elif abs(np.dot(search_direction, search_direction)) < EPSILON:
            return False, simplex

    return False, simplex


class GjkState(Enum):
    NO_CONTACT = -1
    CONTINUE = 0
    CONTACT = 1


@numba.njit(cache=True)
def _refine_simplex(v, v1, v2, n_points):
    if n_points == 2:
        return _line_segment(v, v1, v2)
    elif n_points == 3:
        return _triangle(v, v1, v2)
    else:
        return _tetrahedron(v, v1, v2)


@numba.njit(cache=True)
def _set_point(v, v1, v2, idx, new_v, new_v1, new_v2):
    v[idx] = new_v
    v1[idx] = new_v1
    v2[idx] = new_v2


@numba.njit(cache=True)
def _line_segment(v, v1, v2):
    A = v[1], v1[1], v2[1]
    B = v[0]

    AB = B - A[0]
    AO = -A[0]
    dot = np.dot(AB, AO)

    tmp = np.cross(AB, AO)
    origin_on_AB_segment = abs(np.dot(tmp, tmp)) < EPSILON and dot > 0.0
    if origin_on_AB_segment:
        return GjkState.CONTACT, None, 0

    origin_is_outside_of_A = dot < EPSILON
    if origin_is_outside_of_A:
        _set_point(v, v1, v2, 0, *A)
        n_points = 1
        search_direction = AO
    else:  # origin is closer to line segment
        search_direction = _triple_cross(AB, AO, AB)
        n_points = 2

    return GjkState.CONTINUE, search_direction, n_points


@numba.njit(cache=True)
def _triangle(v, v1, v2):
    A = np.copy(v[2]), np.copy(v1[2]), np.copy(v1[2])
    B = np.copy(v[1]), np.copy(v1[1]), np.copy(v2[1])
    C = np.copy(v[0]), np.copy(v1[0]), np.copy(v1[0])

    touching_contact = abs(point_to_triangle(
        np.zeros(3), np.row_stack((A[0], B[0], C[0])))[0]) < EPSILON_SQRT
    if touching_contact:
        return GjkState.CONTACT, None, 0

    degenerated_triangle = (np.all(np.abs(A[0] - B[0]) < EPSILON)
                            or np.all(np.abs(A[0] - C[0]) < EPSILON))
    if degenerated_triangle:
        return GjkState.NO_CONTACT, None, 0

    AO = -A[0]

    AB = B[0] - A[0]
    AC = C[0] - A[0]
    ABC = np.cross(AB, AC)

    if np.dot(np.cross(ABC, AC), AO) > -EPSILON:
        if np.dot(AC, AO) > -EPSILON:
            _set_point(v, v1, v2, 1, *A)
            n_points = 2
            search_direction = _triple_cross(AC, AO, AC)
        else:
            n_points, search_direction = _triangle_ab(A, B, AB, AO, v, v1, v2)
    else:
        if np.dot(np.cross(AB, ABC), AO) > -EPSILON:
            n_points, search_direction = _triangle_ab(A, B, AB, AO, v, v1, v2)
        else:
            if np.dot(ABC, AO) > -EPSILON:
                n_points = 3
                search_direction = ABC
            else:
                _set_point(v, v1, v2, 0, *B)
                _set_point(v, v1, v2, 1, *C)
                n_points = 3
                search_direction = -ABC

    return GjkState.CONTINUE, search_direction, n_points


@numba.njit(cache=True)
def _triangle_ab(A, B, AB, AO, v, v1, v2):
    if np.dot(AB, AO) > -EPSILON:
        _set_point(v, v1, v2, 0, *B)
        _set_point(v, v1, v2, 1, *A)
        n_points = 2
        search_direction = _triple_cross(AB, AO, AB)
    else:
        _set_point(v, v1, v2, 0, *A)
        n_points = 1
        search_direction = AO
    return n_points, search_direction


@numba.njit(cache=True)
def _tetrahedron(v, v1, v2):
    A = np.copy(v[3]), np.copy(v1[3]), np.copy(v1[3])
    B = np.copy(v[2]), np.copy(v1[2]), np.copy(v1[2])
    C = np.copy(v[1]), np.copy(v1[1]), np.copy(v2[1])
    D = np.copy(v[0]), np.copy(v1[0]), np.copy(v1[0])

    degenerated_tetrahedron = abs(point_to_triangle(
        A[0], np.row_stack((B[0], C[0], D[0])))[0]) < EPSILON_SQRT
    if degenerated_tetrahedron:
        return GjkState.NO_CONTACT, None, 0

    origin = np.zeros(3)
    origin_lies_on_tetrahedrons_face = (
        point_to_triangle(origin, np.row_stack((A[0], B[0], C[0])))[0] < EPSILON_SQRT
        or point_to_triangle(origin, np.row_stack((A[0], C[0], D[0])))[0] < EPSILON_SQRT
        or point_to_triangle(origin, np.row_stack((A[0], B[0], D[0])))[0] < EPSILON_SQRT
        or point_to_triangle(origin, np.row_stack((B[0], C[0], D[0])))[0] < EPSILON_SQRT
    )
    if origin_lies_on_tetrahedrons_face:
        return GjkState.CONTACT, None, 0

    # compute AO, AB, AC, AD segments and ABC, ACD, ADB normal vectors
    AO = -A[0]
    AB = B[0] - A[0]
    AC = C[0] - A[0]
    AD = D[0] - A[0]
    ABC = np.cross(AB, AC)
    ACD = np.cross(AC, AD)
    ADB = np.cross(AD, AB)

    # side (positive or negative) of B, C, D relative to planes ACD, ADB
    # and ABC respectively
    B_on_ACD = np.sign(np.dot(ACD, AB))
    C_on_ADB = np.sign(np.dot(ADB, AC))
    D_on_ABC = np.sign(np.dot(ABC, AD))

    # whether origin is on same side of ACD, ADB, ABC as B, C, D
    # respectively
    AB_O = np.sign(np.dot(ACD, AO)) == B_on_ACD
    AC_O = np.sign(np.dot(ADB, AO)) == C_on_ADB
    AD_O = np.sign(np.dot(ABC, AO)) == D_on_ABC

    origin_is_in_tetrahedron = AB_O and AC_O and AD_O
    if origin_is_in_tetrahedron:
        return GjkState.CONTACT, None, 0

    _rearrange_simplex_to_triangle(A, B, C, D, AB_O, AC_O, v, v1, v2)

    return _triangle(v, v1, v2)


@numba.njit(cache=True)
def _rearrange_simplex_to_triangle(A, B, C, D, AB_O, AC_O, v, v1, v2):
    if not AB_O:
        _set_point(v, v1, v2, 2, *A)
    elif not AC_O:
        _set_point(v, v1, v2, 1, *D)
        _set_point(v, v1, v2, 0, *B)
        _set_point(v, v1, v2, 2, *A)
    else:
        _set_point(v, v1, v2, 0, *C)
        _set_point(v, v1, v2, 1, *B)
        _set_point(v, v1, v2, 2, *A)


@numba.njit(cache=True)
def _triple_cross(a, b, c):
    """d = a x b x c"""
    return np.cross(np.cross(a, b), c)
