"""Minkowski portal refinement (MPR) for collision detection.

Another name for the algorithm is XenoCollide (for details, see
http://xenocollide.snethen.com/). It has been presented in "Game Programming
Gems 7" (Gem 2.5: XenoCollide: Complex Collision Made Simple).

This implementation of MPR is based on libccd (for details, see
https://github.com/danfis/libccd). For the original code the copyright is of
Daniel Fiser <danfis@danfis.cz>. It has been released under 3-clause BSD
license.
"""
from enum import Enum
import numpy as np
import numba

from .minkowski import Simplex, support_function, make_support_point
from .utils import norm_vector, EPSILON
from .distance import point_to_triangle


def mpr_intersection(collider1, collider2, mpr_tolerance=0.0001):
    """Intersection test with Minkowski Portal Refinement (MPR).

    Parameters
    ----------
    collider1 : ConvexCollider
        Convex collider 1.

    collider2 : ConvexCollider
        Convex collider 2.

    mpr_tolerance : float
        Boundary tolerance for MPR algorithm

    Returns
    -------
    intersection : bool
        Do the two colliders intersect?
    """
    res, portal = _discover_portal(collider1, collider2)
    if res == PortalState.ORIGIN_OUTSIDE_PORTAL:
        return False
    elif res == PortalState.PORTAL_WAS_BUILT:
        return _refine_portal(collider1, collider2, portal, mpr_tolerance)
    else:
        return True


def mpr_penetration(collider1, collider2, mpr_tolerance=0.0001, max_iterations=100):
    """Minkowski Portal Refinement (MPR) with penetration info.

    The returned penetration direction and contact position form a line along
    which the penetration happens. The penetration direction is computed such
    that adding depth + penetration_direction to each vertex of the second
    collider separates the two colliders.

    Parameters
    ----------
    collider1 : ConvexCollider
        Convex collider 1.

    collider2 : ConvexCollider
        Convex collider 2.

    mpr_tolerance : float, optional (default: 0.0001)
        Boundary tolerance for MPR algorithm

    max_iterations : int, optional (default: 100)
        Maximum number of iterations to compute penetration info.

    Returns
    -------
    intersection : bool
        Do the two colliders intersect?

    depth : float or None
        Penetration depth.

    penetration_direction : array, shape (3,) or None
        Penetration direction. Vector will be either a unit vector or zero
        vector in case of touching colliders.

    contact_position : array, shape (3,) or None
        Contact position.
    """
    res, portal = _discover_portal(collider1, collider2)
    depth, penetration_direction, contact_position = None, None, None
    if res == PortalState.ORIGIN_OUTSIDE_PORTAL:
        intersection = False
    elif res == PortalState.ORIGIN_ON_V1:
        intersection = True
        depth, penetration_direction, contact_position = _find_penetration_touch(
            portal.v1, portal.v2)
    elif res == PortalState.ORIGIN_ON_V0V1_SEGMENT:
        intersection = True
        depth, penetration_direction, contact_position = _find_penetration_segment(
            portal.v, portal.v1, portal.v2)
    else:
        assert res == PortalState.PORTAL_WAS_BUILT
        intersection = _refine_portal(collider1, collider2, portal, mpr_tolerance)
        if intersection:
            depth, penetration_direction, contact_position = _find_penetration_info(
                collider1, collider2, portal, mpr_tolerance, max_iterations)
    return intersection, depth, penetration_direction, contact_position


class PortalState(Enum):
    CONTINUE_BUILDING_PORTAL = -2
    ORIGIN_OUTSIDE_PORTAL = -1
    PORTAL_WAS_BUILT = 0
    ORIGIN_ON_V1 = 1
    ORIGIN_ON_V0V1_SEGMENT = 2


def _discover_portal(collider1, collider2):
    """Discover initial portal.

    The initial portal is a tetrahedron that intersects with the ray from
    the center of the Minkowski difference to the origin.
    """
    portal = Simplex()
    _find_origin_ray(portal, collider1, collider2)

    origin_outside_v1 = _find_support_in_direction_of_origin_ray(
        portal, collider1, collider2)
    if origin_outside_v1:
        return PortalState.ORIGIN_OUTSIDE_PORTAL, portal

    res = _find_support_perpendicular_to_plane_containing_origin_v01(
        portal, collider1, collider2)
    if res != PortalState.CONTINUE_BUILDING_PORTAL:
        return res, portal
    portal.n_points = 3

    search_direction = _search_direction_perpendicular_to_plane_containing_v012(
        portal.v, portal.v1, portal.v2)
    while portal.n_points < 4:
        portal.v[3], portal.v1[3], portal.v2[3] = support_function(
            collider1, collider2, search_direction)
        if portal.v[3].dot(search_direction) < EPSILON:
            return PortalState.ORIGIN_OUTSIDE_PORTAL, portal
        search_direction, portal.n_points = _iterate_discover_portal(
            portal.v, portal.v1, portal.v2, search_direction, portal.n_points)

    return PortalState.PORTAL_WAS_BUILT, portal


def _find_origin_ray(portal, collider1, collider2):
    """Find origin ray.

    The origin ray starts at v0 and passes through the origin. v0 must be a
    point that is known to be in the interior of the Minkowski difference.
    Such a point can be obtained from points inside the colliders. Center
    points are good candidates.
    """
    portal.v[0], portal.v1[0], portal.v2[0] = make_support_point(
        collider1.center(), collider2.center())
    portal.n_points = 1
    portals_center_is_origin = all(portal.v[0] == 0.0)
    if portals_center_is_origin:
        # Colliders intersect, but we need penetration info.
        portal.v[0, 0] += EPSILON * 10.0


def _find_support_in_direction_of_origin_ray(portal, collider1, collider2):
    search_direction = norm_vector(-portal.v[0])
    portal.v[1], portal.v1[1], portal.v2[1] = support_function(
        collider1, collider2, search_direction)
    portal.n_points = 2
    origin_outside_v1 = (
        any(portal.v[1] != 0.0) and
        portal.v[1].dot(search_direction) < EPSILON)
    return origin_outside_v1


def _find_support_perpendicular_to_plane_containing_origin_v01(
        portal, collider1, collider2):
    search_direction = np.cross(portal.v[0], portal.v[1])
    if search_direction.dot(search_direction) < EPSILON:
        if all(portal.v[1] == 0.0):
            return PortalState.ORIGIN_ON_V1
        else:
            return PortalState.ORIGIN_ON_V0V1_SEGMENT
    search_direction = norm_vector(search_direction)
    portal.v[2], portal.v1[2], portal.v2[2] = support_function(
        collider1, collider2, search_direction)
    if portal.v[2].dot(search_direction) < EPSILON:
        return PortalState.ORIGIN_OUTSIDE_PORTAL
    return PortalState.CONTINUE_BUILDING_PORTAL


@numba.njit(cache=True)
def _search_direction_perpendicular_to_plane_containing_v012(v, v1, v2):
    search_direction = norm_vector(np.cross(v[1] - v[0], v[2] - v[0]))
    if search_direction.dot(v[0]) > 0.0:
        _swap_vertices(v, v1, v2, 1, 2)
        search_direction *= -1.0
    return search_direction


@numba.njit(cache=True)
def _iterate_discover_portal(v, v1, v2, search_direction, portal_size):
    cont = False
    origin_outside_v103 = np.cross(v[1], v[3]).dot(v[0]) < EPSILON
    if origin_outside_v103:
        v[2], v1[2], v2[2] = v[3], v1[3], v2[3]
        cont = True
    if not cont:
        origin_outside_v302 = np.cross(v[3], v[2]).dot(v[0]) < EPSILON
        if origin_outside_v302:
            v[1], v1[1], v2[1] = v[3], v1[3], v2[3]
            cont = True
    if cont:
        search_direction = norm_vector(np.cross(v[1] - v[0], v[2] - v[0]))
    else:
        portal_size = 4
    return search_direction, portal_size


@numba.njit(cache=True)
def _swap_vertices(v, v1, v2, idx1, idx2):
    tmp = v[idx1]
    tmp1 = v1[idx1]
    tmp2 = v2[idx1]
    v[idx1] = v[idx2]
    v1[idx1] = v1[idx2]
    v2[idx1] = v2[idx2]
    v[idx2] = tmp
    v1[idx2] = tmp1
    v2[idx2] = tmp2


def _refine_portal(collider1, collider2, portal, mpr_tolerance):
    """Expands portal towards origin and determine if objects intersect.

    Parameters
    ----------
    portal : Simplex
        Already established portal.

    Returns
    -------
    intersection_found : bool
        Was an intersection found?
    """
    while True:
        search_direction = _portal_direction(portal.v)
        if _encapsulates_origin(portal.v[1], search_direction):
            return True
        next_support_point, next_support_point1, next_support_point2 = support_function(
            collider1, collider2, search_direction)
        if (not _encapsulates_origin(next_support_point, search_direction)
                or _portal_reach_tolerance(portal.v, next_support_point,
                                           search_direction, mpr_tolerance)):
            return False
        _expand_portal(portal.v, portal.v1, portal.v2, next_support_point,
                       next_support_point1, next_support_point2)


@numba.njit(cache=True)
def _portal_direction(v):
    """Compute direction outside portal (from v0 through v1-v2-v3 face).

    Portal's v1-v2-v3 face must be arranged in correct order!
    """
    return norm_vector(np.cross(v[2] - v[1], v[3] - v[1]))


@numba.njit(cache=True)
def _encapsulates_origin(v, search_direction):
    """Does the portal encapsulate the origin?"""
    return v.dot(search_direction) > -10.0 * EPSILON


@numba.njit(cache=True)
def _portal_reach_tolerance(v, v4, search_direction, mpr_tolerance):
    """Returns if portal with new point v4 would reach a specified tolerance.

    Returns true if portal can't significantly expand within Minkowski
    difference. v4 is candidate for new point in portal, search_direction is
    direction in which v4 was obtained.
    """
    return min(v4.dot(search_direction) - v[1:].dot(search_direction)) < mpr_tolerance + EPSILON


@numba.njit(cache=True)
def _expand_portal(v, v1, v2, v4, v14, v24):
    """Extends portal with new support point.

    Portal must have face v1-v2-v3 arranged to face outside portal.
    """
    v4v0 = np.cross(v4, v[0])
    if v[1].dot(v4v0) > 0.0:
        if v[2].dot(v4v0) > 0.0:
            v[1], v1[1], v2[1] = v4, v14, v24
        else:
            v[3], v1[3], v2[3] = v4, v14, v24
    else:
        if v[3].dot(v4v0) > 0.0:
            v[2], v1[2], v2[2] = v4, v14, v24
        else:
            v[1], v1[1], v2[1] = v4, v14, v24


def _find_penetration_info(collider1, collider2, portal, mpr_tolerance, max_iterations):
    """Finds penetration info by expanding the portal."""
    iterations = 0
    while True:
        search_direction = _portal_direction(portal.v)
        next_support_point, next_support_point1, next_support_point2 = support_function(
            collider1, collider2, search_direction)
        tolerance_reached = (
                _portal_reach_tolerance(
                    portal.v, next_support_point, search_direction, mpr_tolerance)
                or iterations > max_iterations)
        if tolerance_reached:
            depth, pdir, pos = _penetration_info(portal.v, portal.v1, portal.v2)
            return depth, norm_vector(pdir), pos
        _expand_portal(portal.v, portal.v1, portal.v2, next_support_point,
                       next_support_point1, next_support_point2)
        iterations += 1


@numba.njit(cache=True)
def _penetration_info(v, v1, v2):
    depth, penetration_direction = point_to_triangle(np.zeros(3), v[1:])
    if abs(depth) < EPSILON:
        # Touching contact. Direction does not matter.
        penetration_direction = np.zeros(3)
    return depth, penetration_direction, _contact_position(
        v, v1, v2, _portal_direction(v))


@numba.njit(cache=True)
def _find_penetration_touch(v1, v2):
    """Finds penetration info if origin lies on v1 (touching contact)."""
    depth = 0.0
    penetration_direction = np.zeros(3)
    contact_position = 0.5 * (v1[1] + v2[1])
    return depth, penetration_direction, contact_position


@numba.njit(cache=True)
def _find_penetration_segment(v, v1, v2):
    """Find penetration info if origin lies on segment v0-v1.

    Depth is distance to v1. Direction also and position must be computed.
    """
    contact_position = 0.5 * (v1[1] + v2[1])
    penetration_direction = v[1]
    depth = np.linalg.norm(penetration_direction)
    return depth, norm_vector(penetration_direction), contact_position


@numba.njit(cache=True)
def _contact_position(v, v1, v2, search_direction):
    """Compute contact position from fully established portal."""
    barycentric_coordinates = np.empty(4)
    barycentric_coordinates[0] = np.cross(v[1], v[2]).dot(v[3])
    barycentric_coordinates[1] = np.cross(v[3], v[2]).dot(v[0])
    barycentric_coordinates[2] = np.cross(v[0], v[1]).dot(v[3])
    barycentric_coordinates[3] = np.cross(v[2], v[1]).dot(v[0])

    coords_sum = np.sum(barycentric_coordinates)

    if coords_sum < EPSILON:
        barycentric_coordinates = np.array([
            0.0,
            np.cross(v[2], v[3]).dot(search_direction),
            np.cross(v[3], v[1]).dot(search_direction),
            np.cross(v[1], v[2]).dot(search_direction)
        ])
        coords_sum = np.sum(barycentric_coordinates)

    barycentric_coordinates /= coords_sum

    v1 = barycentric_coordinates.dot(v1)
    v2 = barycentric_coordinates.dot(v2)

    return 0.5 * (v1 + v2)
