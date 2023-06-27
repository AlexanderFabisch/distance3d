"""GJK implementation of Jolt Physics.

Based on: A Fast and Robust GJK Implementation for Collision Detection of
Convex Objects - Gino van den Bergen,
http://www.dtecta.com/papers/jgt98convex.pdf

Copyright 2021 Jorrit Rouwe, MIT license
Source: https://github.com/jrouwe/JoltPhysics/blob/master/Jolt/Geometry/GJKClosestPoint.h
"""
import math
from enum import Enum

import numba
import numpy as np
from ..utils import EPSILON, MAX_FLOAT, scalar_triple_product


EPSILON_SQR = EPSILON * EPSILON
ALL_TRUE = np.array([True, True, True, True], dtype=np.dtype("bool"))


class GjkState(Enum):
    NoIntersection = 0
    Intersection = 1
    Unknown = 2
    Clipped = 3

def gjk_intersection_jolt(collider1, collider2, tolerance=1e-10):
    """Intersection test with Gilbert-Johnson-Keerthi (GJK) algorithm.

    This implementation differs in several ways from the libccd version:

    * Support points will be interpolated
    * A configurable numerical tolerance is used to check convergence
    * Simplex selection is done via an integer of which the first 4 bits are
      used to select or deselect points of the current simplex

    Implementation based on Jolt Physics, Copyright 2021 Jorrit Rouwe, MIT
    license.

    Based on: A Fast and Robust GJK Implementation for Collision Detection of
    Convex Objects - Gino van den Bergen,
    http://www.dtecta.com/papers/jgt98convex.pdf

    Parameters
    ----------
    collider1 : ConvexCollider
        Convex collider 1.

    collider2 : ConvexCollider
        Convex collider 2.

    tolerance : float, optional (default: 1e-10)
        Minimal distance between objects when the objects are considered to be
        colliding.

    Returns
    -------
    intersection : bool
        Do the two colliders intersect?
    """
    Y = np.empty((4, 3))  # Support points on A - B
    n_points = 0  # Number of points in Y that are valid

    tolerance_sq = tolerance * tolerance

    prev_v_len_sq = MAX_FLOAT
    search_direction = np.array([1.0, 0.0, 0.0])

    while True:
        # Get support points for shape A and B in search direction
        p = collider1.support_function(search_direction)
        q = collider2.support_function(-search_direction)
        state, n_points, prev_v_len_sq = _intersection_loop(
            p, q, Y, n_points, tolerance_sq, prev_v_len_sq, search_direction)
        if state == GjkState.Unknown:
            continue
        else:
            return state == GjkState.Intersection


@numba.njit(cache=True)
def _intersection_loop(
        p, q, Y, n_points, tolerance_sq, prev_v_len_sq, search_direction):
    # Get support point of the minkowski sum A - B of v
    support_point = p - q

    # If the support point is in the opposite direction as search_direction,
    # then we have found a separating axis and there is no intersection
    if search_direction.dot(support_point) < -EPSILON:
        # Separating axis found
        return GjkState.NoIntersection, n_points, prev_v_len_sq

    # Store the point for later use
    Y[n_points] = support_point
    n_points += 1

    # Determine the new closest point
    success, search_direction[:], v_len_sq, simplex = get_closest_point_to_origin(
        Y, n_points, prev_v_len_sq)
    if not success:
        return GjkState.NoIntersection, n_points, prev_v_len_sq

    # If there are 4 points, the origin is inside the tetrahedron and we're done
    if simplex == 0xf:
        return GjkState.Intersection, n_points, prev_v_len_sq

    # If v is very close to zero, we consider this a collision
    if v_len_sq <= tolerance_sq:
        return GjkState.Intersection, n_points, prev_v_len_sq

    # If v is very small compared to the length of y, we also consider this a
    # collision
    if v_len_sq <= EPSILON * max_y_length_squared(Y, n_points):
        return GjkState.Intersection, n_points, prev_v_len_sq

    # The next separation axis to test is the negative of the closest point of
    # the Minkowski sum to the origin.
    # Note: This must be done before terminating as converged since the
    # separating axis is -search_direction.
    search_direction *= -1.0

    # If the squared length of search_direction is not changing enough, we've
    # converged and there is no collision.
    assert prev_v_len_sq >= v_len_sq
    if prev_v_len_sq - v_len_sq <= EPSILON * prev_v_len_sq:
        # search_direction is a separating axis
        return GjkState.NoIntersection, n_points, prev_v_len_sq
    prev_v_len_sq = v_len_sq

    # Update the points of the simplex
    n_points = update_simplex_y(Y, n_points, simplex)

    return GjkState.Unknown, n_points, prev_v_len_sq


def gjk_distance_jolt(
        collider1, collider2, tolerance=1e-10, max_distance_squared=100000.0, sanity_check=1e-8):
    """Gilbert-Johnson-Keerthi (GJK) algorithm for distance calculation.

    This implementation extends the intersection test by closest point
    calculation after convergence. It is also possible to clip colliders that
    are too far away early.

    Implementation based on Jolt Physics, Copyright 2021 Jorrit Rouwe, MIT
    license.

    Based on: A Fast and Robust GJK Implementation for Collision Detection of
    Convex Objects - Gino van den Bergen,
    http://www.dtecta.com/papers/jgt98convex.pdf

    Parameters
    ----------
    collider1 : ConvexCollider
        Convex collider 1.

    collider2 : ConvexCollider
        Convex collider 2.

    tolerance : float, optional (default: 1e-10)
        Minimal distance between objects when the objects are considered to be
        colliding.

    max_distance_squared : float, optional (default: 100000)
        The maximum squared distance between colliders before the objects are
        considered infinitely far away and processing is terminated.

    sanity_check : float, optional (default: 1e-8)
        Use `sanity_check=float("inf")` to bypass sanity check.

    Returns
    -------
    distance : float
        The shortest distance between two convex shapes.

    closest_point1 : array, shape (3,)
        Closest point on first convex shape.
        If the distance is MAX_FLOAT the points are invalid.

    closest_point2 : array, shape (3,)
        Closest point on second convex shape.
        If the distance is MAX_FLOAT the points are invalid.

    simplex : array, shape (4, 3)
        Simplex defined by 4 points of the Minkowski difference between
        vertices of the two colliders.
    """
    Y = np.empty((4, 3))  # Support points on A - B
    P = np.empty((4, 3))  # Support points on A
    Q = np.empty((4, 3))  # Support points on B
    n_points = 0  # Number of points in Y, P and Q that are valid

    tolerance_sq = tolerance * tolerance

    search_direction = np.array([1.0, 0.0, 0.0])
    v_len_sq = np.dot(search_direction, search_direction)
    prev_v_len_sq = MAX_FLOAT

    while True:
        # Get support points for shape A and B in search direction
        p = collider1.support_function(search_direction)
        q = collider2.support_function(-search_direction)
        state, n_points, prev_v_len_sq, v_len_sq = _distance_loop(
            p, q, Y, P, Q, n_points, tolerance_sq, prev_v_len_sq, v_len_sq,
            search_direction, max_distance_squared)
        if state == GjkState.Unknown:
            continue
        elif state == GjkState.Clipped:
            return MAX_FLOAT, None, None, None
        else:
            # Get the closest points
            a, b = calculate_closest_points(Y, P, Q, n_points)

            check_value = abs(np.dot(search_direction, search_direction) - v_len_sq)
            assert check_value < sanity_check, f"Sanity check failed: {check_value}"
            
            dist = math.sqrt(v_len_sq)
            if dist < EPSILON:
                a = b = 0.5 * (a + b)
            return dist, a, b, Y


@numba.njit(cache=True)
def _distance_loop(
        p, q, Y, P, Q, n_points, tolerance_sq, prev_v_len_sq, v_len_sq,
        search_direction, max_distance_squared):
    # Get support point of the minkowski sum A - B of v
    support_point = p - q

    dot = search_direction.dot(support_point)

    # Test if we have a separation of more than max_distance_squared,
    # in which case we terminate early
    if dot < 0.0 and dot * dot > v_len_sq * max_distance_squared:
        return GjkState.Clipped, None, None, None

    # Store the point for later use
    Y[n_points] = support_point
    P[n_points] = p
    Q[n_points] = q
    n_points += 1

    success, ioV_new, v_len_sq_new, new_set = get_closest_point_to_origin(
        Y, n_points, prev_v_len_sq)
    if success:
        search_direction[:], v_len_sq, simplex = ioV_new, v_len_sq_new, new_set
    else:
        n_points -= 1  # Undo add last point
        simplex = 0b0000
        for i in range(n_points):
            simplex |= 1 << i

    # If there are 4 points, the origin is inside the tetrahedron,
    # and we're done
    if simplex == 0xf:
        v_len_sq = 0.0
        return GjkState.Intersection, n_points, prev_v_len_sq, v_len_sq

    # Update the points of the simplex
    n_points = update_simplex_ypq(Y, P, Q, n_points, simplex)

    # If v is very close to zero, we consider this a collision
    if v_len_sq <= tolerance_sq:
        v_len_sq = 0.0
        return GjkState.Intersection, n_points, prev_v_len_sq, v_len_sq

    # If v is very small compared to the length of y, we also consider
    # this a collision
    if v_len_sq <= EPSILON * max_y_length_squared(Y, n_points):
        v_len_sq = 0.0
        return GjkState.Intersection, n_points, prev_v_len_sq, v_len_sq

    # The next separation axis to test is the negative of the closest point of
    # the Minkowski sum to the origin
    # Note: This must be done before terminating as converged since the
    # separating axis is -v
    search_direction *= -1.0

    # If the squared length of v is not changing enough, we've converged and
    # there is no collision
    assert prev_v_len_sq >= v_len_sq
    if prev_v_len_sq - v_len_sq <= EPSILON * prev_v_len_sq:
        # search_direction is a separating axis
        return GjkState.NoIntersection, n_points, prev_v_len_sq, v_len_sq

    prev_v_len_sq = v_len_sq
    return GjkState.Unknown, n_points, prev_v_len_sq, v_len_sq


@numba.njit(cache=True)
def get_barycentric_coordinates_line(a, b):
    """Barycentric coordinates of the closest point to origin for infinite line.

    Point can then be computed as a * u + b * v.
    """
    ab = b - a
    denominator = np.dot(ab, ab)
    if denominator < EPSILON_SQR:
        # Degenerate line segment, fallback to points
        if np.dot(a, a) < np.dot(b, b):
            # A closest
            u = 1.0
            v = 0.0
        else:
            # B closest
            u = 0.0
            v = 1.0
    else:
        v = -a.dot(ab) / denominator
        u = 1.0 - v
    return u, v


@numba.njit(cache=True)
def get_barycentric_coordinates_plane(a, b, c):
    """Barycentric coordinates of closest point to origin for plane.

    Point can then be computed as a * u + b * v + c * w.
    Taken from: Real-Time Collision Detection - Christer Ericson
    (Section: Barycentric Coordinates) with p = 0.
    Adjusted to always include the shortest edge of the triangle in the
    calculation to improve numerical accuracy.
    """
    # First calculate the three edges
    v0 = b - a
    v1 = c - a
    v2 = c - b

    # Make sure that the shortest edge is included in the calculation to keep
    # the products a * b - c * d as small as possible to preserve accuracy
    d00 = v0.dot(v0)
    d11 = v1.dot(v1)
    d22 = v2.dot(v2)
    if d00 <= d22:
        # Use v0 and v1 to calculate barycentric coordinates
        d01 = v0.dot(v1)
        denominator = d00 * d11 - d01 * d01
        if abs(denominator) < EPSILON:
            # Degenerate triangle, return coordinates along longest edge
            if d00 > d11:
                u, v = get_barycentric_coordinates_line(a, b)
                w = 0.0
            else:
                u, w = get_barycentric_coordinates_line(a, c)
                v = 0.0
        else:
            a0 = a.dot(v0)
            a1 = a.dot(v1)
            v = (d01 * a1 - d11 * a0) / denominator
            w = (d01 * a0 - d00 * a1) / denominator
            u = 1.0 - v - w
    else:
        # Use v1 and v2 to calculate barycentric coordinates
        d12 = v1.dot(v2)

        denominator = d11 * d22 - d12 * d12
        if abs(denominator) < EPSILON:
            # Degenerate triangle, return coordinates along longest edge
            if d11 > d22:
                u, w = get_barycentric_coordinates_line(a, c)
                v = 0.0
            else:
                v, w = get_barycentric_coordinates_line(b, c)
                u = 0.0
        else:
            c1 = c.dot(v1)
            c2 = c.dot(v2)
            u = (d22 * c1 - d12 * c2) / denominator
            v = (d11 * c2 - d12 * c1) / denominator
            w = 1.0 - u - v
    return u, v, w


@numba.njit(cache=True)
def get_barycentric_coordinates_tetrahedron(a, b, c, d):
    """Barycentric coordinates of the closest point to origin for tetrahedron.

    Source: https://stackoverflow.com/a/38546111/915743
    """
    vab = b - a
    vac = c - a
    vad = d - a

    va6 = -scalar_triple_product(b, d - b, c - b)
    vb6 = -scalar_triple_product(a, vac, vad)
    vc6 = -scalar_triple_product(a, vad, vab)
    vd6 = -scalar_triple_product(a, vab, vac)
    v6 = 1.0 / scalar_triple_product(vab, vac, vad)
    return va6 * v6, vb6 * v6, vc6 * v6, vd6 * v6


@numba.njit(cache=True)
def closest_point_line(a, b):
    """Get the closest point to the origin of line.

    Output set describes which features are closest:

    * 1 = a
    * 2 = b
    * 3 = line segment ab
    """
    u, v = get_barycentric_coordinates_line(a, b)
    if v <= 0.0:
        # a is closest point
        return a, 0b0001
    elif u <= 0.0:
        # b is closest point
        return b, 0b0010
    else:
        # Closest point lies on line
        return u * a + v * b, 0b0011


@numba.njit(cache=True)
def closest_point_triangle(a, b, c):
    """Get the closest point to the origin of triangle.

    Output set describes which features are closest:

    * 1 = a
    * 2 = b
    * 4 = c
    * 5 = line segment ac
    * 7 = triangle interior etc.

    Taken from: Real-Time Collision Detection - Christer Ericson (Section:
    Closest Point on Triangle to Point) with p = 0.
    """
    # Calculate edges
    ab = b - a
    ac = c - a
    bc = c - b

    # The most accurate normal is calculated by using the two shortest edges
    # See: https://box2d.org/posts/2014/01/troublesome-triangle/
    # The difference in normals is most pronounced when one edge is much
    # smaller than the others (in which case the other 2 must have roughly the
    # same length). Therefore, we can suffice by just picking the shortest from
    # 2 edges and use that with the 3rd edge to calculate the normal. We first
    # check which of the edges is shorter.
    bc_shorter_than_ac = bc.dot(bc) < ac.dot(ac)
    if bc_shorter_than_ac:
        n = np.cross(ab, bc)
    else:
        n = np.cross(ab, ac)
    n_len_sq = np.dot(n, n)

    # Check degenerate
    if n_len_sq < EPSILON_SQR:
        # Degenerate, fallback to edges

        # Edge AB
        closest_point, closest_set = closest_point_line(a, b)
        best_dist_sq = np.dot(closest_point, closest_point)

        # Edge AC
        q, new_set = closest_point_line(a, c)
        dist_sq = np.dot(q, q)
        if dist_sq < best_dist_sq:
            closest_point = q
            best_dist_sq = dist_sq
            closest_set = (new_set & 0b0001) + ((new_set & 0b0010) << 1)

        # Edge BC
        q, new_set = closest_point_line(b, c)
        dist_sq = np.dot(q, q)
        if dist_sq < best_dist_sq:
            closest_point = q
            closest_set = new_set << 1

        return closest_point, closest_set

    # Check if P in vertex region outside A
    ap = -a
    d1 = ab.dot(ap)
    d2 = ac.dot(ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a, 0b0001  # barycentric coordinates (1,0,0)

    # Check if P in vertex region outside B
    bp = -b
    d3 = ab.dot(bp)
    d4 = ac.dot(bp)
    if d3 >= 0.0 and d4 <= d3:
        return b, 0b0010  # barycentric coordinates (0,1,0)

    # Check if P in edge region of AB, if so return projection of P onto AB
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 <= d1 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return a + v * ab, 0b0011  # barycentric coordinates (1-v,v,0)

    # Check if P in vertex region outside C
    cp = -c
    d5 = ab.dot(cp)
    d6 = ac.dot(cp)
    if d6 >= 0.0 and d5 <= d6:
        return c, 0b0100  # barycentric coordinates (0,0,1)

    # Check if P in edge region of AC, if so return projection of P onto AC
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 <= d2 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return a + w * ac, 0b0101  # barycentric coordinates (1-w,0,w)

    # Check if P in edge region of BC, if so return projection of P onto BC
    va = d3 * d6 - d5 * d4
    d4_d3 = d4 - d3
    d5_d6 = d5 - d6
    if va <= 0.0 <= d4_d3 and d5_d6 >= 0.0:
        w = d4_d3 / (d4_d3 + d5_d6)
        return b + w * bc, 0b0110  # barycentric coordinates (0,1-w,w)

    # P inside face region.
    # Here we deviate from Christer Ericson's article to improve accuracy.
    # Determine distance between triangle and origin:
    # distance = (centroid - origin) . normal / |normal|
    # Closest point to origin is then: distance . normal / |normal|
    # Note that this way of calculating the closest point is much more accurate
    # than first calculating barycentric coordinates and then calculating the
    # closest point based on those coordinates.
    return n * (a + b + c).dot(n) / (3.0 * n_len_sq), 0b0111


@numba.njit(cache=True)
def origin_outside_of_tetrahedron_planes(a, b, c, d):
    """Returns for each plane of the tetrahedron if the origin is inside.

    Roughly equivalent to:
       [OriginOutsideOfPlane(a, b, c, d),
        OriginOutsideOfPlane(a, c, d, b),
        OriginOutsideOfPlane(a, d, b, c),
        OriginOutsideOfPlane(b, d, c, a)]
    """
    ab = b - a
    ac = c - a
    ad = d - a
    bd = d - b
    bc = c - b

    ab_cross_ac = np.cross(ab, ac)
    ac_cross_ad = np.cross(ac, ad)
    ad_cross_ab = np.cross(ad, ab)
    bd_cross_bc = np.cross(bd, bc)

    # For each plane get the side on which the origin is
    signp0 = a.dot(ab_cross_ac)  # ABC
    signp1 = a.dot(ac_cross_ad)  # ACD
    signp2 = a.dot(ad_cross_ab)  # ADB
    signp3 = b.dot(bd_cross_bc)  # BDC
    signp = np.array([signp0, signp1, signp2, signp3])

    # For each plane get the side that is outside (determined by the 4th point)
    signd0 = ad.dot(ab_cross_ac)  # D
    signd1 = ab.dot(ac_cross_ad)  # B
    signd2 = ac.dot(ad_cross_ab)  # C
    signd3 = -ab.dot(bd_cross_bc)  # A
    signd = np.array([signd0, signd1, signd2, signd3])

    # The winding of all triangles has been chosen so that signd should have
    # the same sign for all components. If this is not the case the tetrahedron
    # is degenerate and we return that the origin is in front of all sides.
    if np.all(signd > 0.0):
        return signp >= -EPSILON
    elif np.all(signd < 0.0):
        return signp <= EPSILON
    else:
        # Mixed signs, degenerate tetrahedron
        return ALL_TRUE


@numba.njit(cache=True)
def closest_point_tetrahedron(a, b, c, d):
    """Get the closest point between tetrahedron to the origin.

    Output set specifies which feature was closest:

    * 1 = a
    * 2 = b
    * 4 = c
    * 8 = d

    Edges have 2 bits set, triangles 3, and if the point is in the interior
    4 bits are set.

    Taken from: Real-Time Collision Detection - Christer Ericson (Section:
    Closest Point on Tetrahedron to Point) with p = 0.
    """
    # Start out assuming point inside all halfspaces, so closest to itself
    closest_set = 0b1111
    closest_point = np.zeros(3)
    best_dist_sq = MAX_FLOAT

    # Determine for each of the faces of the tetrahedron if the origin is in
    # front of the plane
    origin_out_of_planes = origin_outside_of_tetrahedron_planes(a, b, c, d)

    # If point outside face abc then compute the closest point on abc
    if origin_out_of_planes[0]:  # OriginOutsideOfPlane(a, b, c, d)
        # Update best closest point because distance is less than MAX_FLOAT
        closest_point, closest_set = closest_point_triangle(a, b, c)
        best_dist_sq = np.dot(closest_point, closest_point)

    # Repeat test for face acd
    if origin_out_of_planes[1]:  # OriginOutsideOfPlane(a, c, d, b)
        q, new_set = closest_point_triangle(a, c, d)
        dist_sq = np.dot(q, q)
        if dist_sq < best_dist_sq:
            best_dist_sq = dist_sq
            closest_point = q
            closest_set = (new_set & 0b0001) + ((new_set & 0b0110) << 1)

    # Repeat test for face adb
    if origin_out_of_planes[2]:  # OriginOutsideOfPlane(a, d, b, c)
        q, new_set = closest_point_triangle(a, d, b)
        dist_sq = np.dot(q, q)
        if dist_sq < best_dist_sq:
            best_dist_sq = dist_sq
            closest_point = q
            closest_set = (new_set & 0b0001) + ((new_set & 0b0010) << 2) + ((new_set & 0b0100) >> 1)

    # Repeat test for face bdc
    if origin_out_of_planes[3]:  # OriginOutsideOfPlane(b, d, c, a)
        q, new_set = closest_point_triangle(b, d, c)
        dist_sq = np.dot(q, q)
        if dist_sq < best_dist_sq:
            closest_point = q
            closest_set = ((new_set & 0b0001) << 1) + ((new_set & 0b0010) << 2) + (new_set & 0b0100)

    return closest_point, closest_set


@numba.njit(cache=True)
def max_y_length_squared(y, n_points):
    """Get max(|Y_0|^2 .. |Y_n|^2)."""
    y_len_sq = np.dot(y[0], y[0])
    for i in range(1, n_points):
        y_len_sq = max(y_len_sq, np.dot(y[i], y[i]))
    return y_len_sq


@numba.njit(cache=True)
def update_simplex_y(Y, n_points, simplex):
    """Remove points that are not in the set, only updates Y."""
    n_new_points = 0
    for i in range(n_points):
        if (simplex & (1 << i)) != 0:
            Y[n_new_points] = Y[i]
            n_new_points += 1
    return n_new_points


@numba.njit(cache=True)
def update_simplex_ypq(Y, P, Q, n_points, simplex):
    """Remove points that are not in the set, updates Y, P and Q."""
    n_new_points = 0
    for i in range(n_points):
        if (simplex & (1 << i)) != 0:
            Y[n_new_points] = Y[i]
            P[n_new_points] = P[i]
            Q[n_new_points] = Q[i]
            n_new_points += 1
    return n_new_points


@numba.njit(cache=True)
def calculate_closest_points(Y, P, Q, n_points):
    """Calculate the closest points on A and B."""
    if n_points == 1:
        a = P[0]
        b = Q[0]
    elif n_points == 2:
        u, v = get_barycentric_coordinates_line(Y[0], Y[1])
        a = u * P[0] + v * P[1]
        b = u * Q[0] + v * Q[1]
    elif n_points == 3:
        u, v, w = get_barycentric_coordinates_plane(Y[0], Y[1], Y[2])
        a = u * P[0] + v * P[1] + w * P[2]
        b = u * Q[0] + v * Q[1] + w * Q[2]
    elif n_points == 4:  # intersection
        u, v, w, x = get_barycentric_coordinates_tetrahedron(Y[0], Y[1], Y[2], Y[3])
        a = u * P[0] + v * P[1] + w * P[2] + x * P[3]
        b = u * Q[0] + v * Q[1] + w * Q[2] + x * Q[3]
    else:
        a, b = None, None
    return a, b


@numba.njit(cache=True)
def get_closest_point_to_origin(Y, n_points, prev_v_len_sqr):
    """Get new closest point to origin given simplex Y of n_points points."""
    if n_points == 1:
        simplex = 0b0001
        v = Y[0]
    elif n_points == 2:
        v, simplex = closest_point_line(Y[0], Y[1])
    elif n_points == 3:
        v, simplex = closest_point_triangle(Y[0], Y[1], Y[2])
    elif n_points == 4:
        v, simplex = closest_point_tetrahedron(Y[0], Y[1], Y[2], Y[3])
    else:
        assert False

    v_len_sq = np.dot(v, v)
    # Note, comparison order important: If v_len_sq is NaN then this expression
    # will be False, so we will return False
    if v_len_sq < prev_v_len_sqr:
        return True, v, v_len_sq, simplex

    return False, None, None, None

def gjk_distance_jolt_iterations(
        collider1, collider2, tolerance=1e-10, max_distance_squared=100000.0):
    """Gilbert-Johnson-Keerthi (GJK) algorithm for distance calculation.

    This implementation extends the intersection test by closest point
    calculation after convergence. It is also possible to clip colliders that
    are too far away early.

    Implementation based on Jolt Physics, Copyright 2021 Jorrit Rouwe, MIT
    license.

    Based on: A Fast and Robust GJK Implementation for Collision Detection of
    Convex Objects - Gino van den Bergen,
    http://www.dtecta.com/papers/jgt98convex.pdf

    Parameters
    ----------
    collider1 : ConvexCollider
        Convex collider 1.

    collider2 : ConvexCollider
        Convex collider 2.

    tolerance : float, optional (default: 1e-10)
        Minimal distance between objects when the objects are considered to be
        colliding.

    max_distance_squared : float, optional (default: 100000)
        The maximum squared distance between colliders before the objects are
        considered infinitely far away and processing is terminated.


    Returns
    -------
    distance : float
        The shortest distance between two convex shapes.

    closest_point1 : array, shape (3,)
        Closest point on first convex shape.
        If the distance is MAX_FLOAT the points are invalid.

    closest_point2 : array, shape (3,)
        Closest point on second convex shape.
        If the distance is MAX_FLOAT the points are invalid.

    simplex : array, shape (4, 3)
        Simplex defined by 4 points of the Minkowski difference between
        vertices of the two colliders.
    """
    Y = np.empty((4, 3))  # Support points on A - B
    P = np.empty((4, 3))  # Support points on A
    Q = np.empty((4, 3))  # Support points on B
    n_points = 0  # Number of points in Y, P and Q that are valid

    tolerance_sq = tolerance * tolerance

    search_direction = np.array([1.0, 0.0, 0.0])
    v_len_sq = np.dot(search_direction, search_direction)
    prev_v_len_sq = MAX_FLOAT

    iterations = 0
    while True:
        iterations += 1

        # Get support points for shape A and B in search direction
        p = collider1.support_function(search_direction)
        q = collider2.support_function(-search_direction)
        state, n_points, prev_v_len_sq, v_len_sq = _distance_loop(
            p, q, Y, P, Q, n_points, tolerance_sq, prev_v_len_sq, v_len_sq,
            search_direction, max_distance_squared)
        if state == GjkState.Unknown:
            continue
        elif state == GjkState.Clipped:
            return iterations
        else:
            # Get the closest points
            a, b = calculate_closest_points(Y, P, Q, n_points)

            assert abs(np.dot(search_direction, search_direction) - v_len_sq) < 1e-12
            dist = math.sqrt(v_len_sq)
            if dist < EPSILON:
                a = b = 0.5 * (a + b)
            return iterations

