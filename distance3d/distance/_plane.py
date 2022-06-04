import math

import numpy as np
import numba
from ..geometry import (
    hesse_normal_form, convert_segment_to_line, line_from_pluecker,
    convert_rectangle_to_vertices, convert_box_to_vertices)


def point_to_plane(point, plane_point, plane_normal, signed=False):
    """Compute the shortest distance between a point and a plane.

    Parameters
    ----------
    point : array, shape (3,)
        3D point.

    plane_point : array, shape (3,)
        Point on the plane.

    plane_normal : array, shape (3,)
        Normal of the plane. We assume unit length.

    signed : bool, optional (default: False)
        Should the distance have a sign?

    Returns
    -------
    dist : float
        The shortest distance between point and plane. A sign indicates
        the direction along the normal.

    closest_point_plane : array, shape (3,)
        Closest point on plane.
    """
    return _point_to_plane(point, plane_point, plane_normal, signed)


@numba.njit(
    numba.types.Tuple((numba.float64, numba.float64[::1]))
    (numba.float64[::1], numba.float64[::1], numba.float64[::1], numba.boolean),
    cache=True)
def _point_to_plane(point, plane_point, plane_normal, signed):
    t = np.dot(plane_normal, point - plane_point)
    closest_point_plane = point - t * plane_normal
    if not signed:
        t = abs(t)
    return t, closest_point_plane


def line_to_plane(
        line_point, line_direction, plane_point, plane_normal, epsilon=1e-6):
    """Compute the shortest distance from line to plane.

    Parameters
    ----------
    line_point : array, shape (3,)
        Point on line.

    line_direction : array, shape (3,)
        Direction of the line. This is assumed to be of unit length.

    plane_point : array, shape (3,)
        Point on the plane.

    plane_normal : array, shape (3,)
        Normal of the plane. We assume unit length.

    epsilon : float, optional (default: 1e-6)
        Values smaller than epsilon are considered to be 0.

    Returns
    -------
    dist : float
        The shortest distance between line and plane.

    closest_point_line : array, shape (3,)
        Closest point on line.

    closest_point_plane : array, shape (3,)
        Closest point on plane.
    """
    intersection, t = _line_to_plane(
        line_point, line_direction, plane_point, plane_normal, epsilon)

    if intersection:
        closest_point_line = line_point + t * line_direction
        return 0.0, closest_point_line, closest_point_line

    dist, closest_point_plane = point_to_plane(
        line_point, plane_point, plane_normal)
    return dist, line_point, closest_point_plane


@numba.njit(
    numba.types.Tuple((numba.boolean, numba.float64))
    (numba.float64[::1], numba.float64[::1], numba.float64[::1],
     numba.float64[::1], numba.float64),
    cache=False)
def _line_to_plane(line_point, line_direction, plane_point, plane_normal,
                   epsilon):
    length_of_dir_projected_on_normal = np.dot(line_direction, plane_normal)
    if length_of_dir_projected_on_normal * length_of_dir_projected_on_normal < epsilon:
        # line parallel to plane
        return False, 0.0

    _, d = hesse_normal_form(plane_point, plane_normal)
    t = (d - np.dot(plane_normal, line_point)) / np.dot(plane_normal, line_direction)
    return True, t


def line_segment_to_plane(
        segment_start, segment_end, plane_point, plane_normal, epsilon=1e-6):
    """Compute the shortest distance between line segment and plane.

    Parameters
    ----------
    segment_start : array, shape (3,)
        Start point of segment.

    segment_end : array, shape (3,)
        End point of segment.

    plane_point : array, shape (3,)
        Point on the plane.

    plane_normal : array, shape (3,)
        Normal of the plane. We assume unit length.

    epsilon : float, optional (default: 1e-6)
        Values smaller than epsilon are considered to be 0.

    Returns
    -------
    dist : float
        The shortest distance between line segment and plane.

    closest_point_segment : array, shape (3,)
        Closest point on line segment.

    closest_point_plane : array, shape (3,)
        Closest point on plane.
    """
    return _line_segment_to_plane(
        segment_start, segment_end, plane_point, plane_normal, epsilon)


@numba.njit(cache=True)
def _line_segment_to_plane(
        segment_start, segment_end, plane_point, plane_normal, epsilon):
    segment_direction, segment_length = convert_segment_to_line(
        segment_start, segment_end)
    line_intersects, t = _line_to_plane(
        segment_start, segment_direction, plane_point, plane_normal, epsilon)

    if line_intersects:
        if 0 <= t <= segment_length:
            closest_point_segment = segment_start + t * segment_direction
            return 0.0, closest_point_segment, closest_point_segment

        if t < 0.0:
            closest_point_segment = segment_start
        else:
            closest_point_segment = segment_end
    else:
        closest_point_segment = segment_start

    dist, closest_point_plane = _point_to_plane(
        closest_point_segment, plane_point, plane_normal, False)
    return dist, closest_point_segment, closest_point_plane


def plane_to_plane(plane_point1, plane_normal1, plane_point2, plane_normal2,
                   epsilon=1e-6):
    """Compute the shortest distance between two planes.

    Parameters
    ----------
    plane_point1 : array, shape (3,)
        Point on the plane.

    plane_normal1 : array, shape (3,)
        Normal of the plane. We assume unit length.

    plane_point2 : array, shape (3,)
        Point on the plane.

    plane_normal2 : array, shape (3,)
        Normal of the plane. We assume unit length.

    epsilon : float, optional (default: 1e-6)
        Values smaller than epsilon are considered to be 0.

    Returns
    -------
    dist : float
        The shortest distance between line segment and plane.

    closest_point_plane1 : array, shape (3,)
        Closest point on first plane.

    closest_point_plane2 : array, shape (3,)
        Closest point on second plane.
    """
    line_direction = np.cross(plane_normal1, plane_normal2)
    if np.linalg.norm(line_direction) > epsilon:
        line_direction, line_moment = plane_intersects_plane(
            plane_point1, plane_normal1, plane_point2, plane_normal2,
            line_direction=line_direction)
        line_point, _ = line_from_pluecker(line_direction, line_moment)
        return 0.0, line_point, line_point
    else:
        dist, closest_point_plane2 = point_to_plane(
            plane_point1, plane_point2, plane_normal2)
        return dist, plane_point1, closest_point_plane2


def plane_intersects_plane(
        plane_point1, plane_normal1, plane_point2, plane_normal2,
        line_direction=None):
    _, d1 = hesse_normal_form(plane_point1, plane_normal1)
    _, d2 = hesse_normal_form(plane_point2, plane_normal2)
    if line_direction is None:
        line_direction = np.cross(plane_normal1, plane_normal2)
    line_moment = plane_normal1 * d2 - plane_normal2 * d1
    return line_direction, line_moment


def plane_to_triangle(plane_point, plane_normal, triangle_points):
    """Compute the shortest distance between a plane and a triangle.

    Parameters
    ----------
    plane_point : array, shape (3,)
        Point on the plane.

    plane_normal : array, shape (3,)
        Normal of the plane. We assume unit length.

    triangle_points : array, shape (3, 3)
        Each row contains a point of the triangle (A, B, C).

    Returns
    -------
    dist : float
        The shortest distance between triangle and plane.

    closest_point_plane : array, shape (3,)
        Closest point on plane.

    closest_point_triangle : array, shape (3,)
        Closest point on triangle.
    """
    return _plane_to_convex_hull_points(plane_point, plane_normal, triangle_points)


@numba.njit(cache=True)
def _plane_to_convex_hull_points(plane_point, plane_normal, points):
    ts = np.dot(points - plane_point.reshape(1, -1), plane_normal)
    min_idx = np.argmin(ts)
    max_idx = np.argmax(ts)

    if ts[min_idx] * ts[max_idx] < 0:  # on opposite sides, intersection
        return _line_segment_to_plane(
            points[min_idx], points[max_idx], plane_point, plane_normal, 1e-6)

    closest_idx = np.argmin(np.abs(ts))
    closest_point = points[closest_idx]
    t = ts[closest_idx]
    closest_point_plane = closest_point - t * plane_normal

    return abs(t), closest_point_plane, closest_point


def plane_to_rectangle(
        plane_point, plane_normal, rectangle_center, rectangle_axes,
        rectangle_lengths):
    """Compute the shortest distance between a plane and a rectangle.

    Parameters
    ----------
    plane_point : array, shape (3,)
        Point on the plane.

    plane_normal : array, shape (3,)
        Normal of the plane. We assume unit length.

    rectangle_center : array, shape (3,)
        Center point of the rectangle.

    rectangle_axes : array, shape (2, 3)
        Each row is a vector of unit length, indicating the direction of one
        axis of the rectangle. Both vectors are orthogonal.

    rectangle_lengths : array, shape (2,)
        Lengths of the two sides of the rectangle.

    Returns
    -------
    dist : float
        The shortest distance between rectangle and plane.

    closest_point_plane : array, shape (3,)
        Closest point on plane.

    closest_point_rectangle : array, shape (3,)
        Closest point on rectangle.
    """
    points = convert_rectangle_to_vertices(
        rectangle_center, rectangle_axes, rectangle_lengths)
    return _plane_to_convex_hull_points(plane_point, plane_normal, points)


def plane_to_box(plane_point, plane_normal, box2origin, size):
    """Compute the shortest distance between a plane and a box.

    Parameters
    ----------
    plane_point : array, shape (3,)
        Point on the plane.

    plane_normal : array, shape (3,)
        Normal of the plane. We assume unit length.

    box2origin : array, shape (4, 4)
        Pose of the box.

    size : array, shape (3,)
        Size of the box along its axes.

    Returns
    -------
    dist : float
        The shortest distance between rectangle and plane.

    closest_point_plane : array, shape (3,)
        Closest point on plane.

    closest_point_box : array, shape (3,)
        Closest point on box.
    """
    points = convert_box_to_vertices(box2origin, size)
    return _plane_to_convex_hull_points(plane_point, plane_normal, points)


@numba.njit(cache=True)
def plane_to_ellipsoid(plane_point, plane_normal, ellipsoid2origin, radii):
    """Compute the shortest distance between a plane and an ellipsoid.

    Parameters
    ----------
    plane_point : array, shape (3,)
        Point on the plane.

    plane_normal : array, shape (3,)
        Normal of the plane. We assume unit length.

    ellipsoid2origin : array, shape (4, 4)
        Pose of the ellipsoid.

    radii : array, shape (3,)
        Radii of the ellipsoid.

    Returns
    -------
    dist : float
        The shortest distance between rectangle and plane.

    closest_point_plane : array, shape (3,)
        Closest point on plane.

    closest_point_box : array, shape (3,)
        Closest point on box.
    """
    # https://gamedev.net/forums/topic/457831-distance-between-ellipsoid-plane/4028358/
    # Let the ellipsoid be represented in standard form,
    # (X-C)^T*M*(X-C) = 1, where C is the center point, M is a positive
    # definite matrix, and X is any point on the ellipsoid. The superscript T
    # denotes the transpose operator. Let N be a normal vector to the plane.
    # When the ellipsoid is not intersected by the plane, the closest and
    # farthest points are
    # X = C +/- M^{-1}*N/sqrt(N^T*M^{-1}*N)
    # where M^{-1} is the inverse of M.
    C = ellipsoid2origin[:3, 3]
    M = _ellipsoid_quadric_matrix(ellipsoid2origin, radii)

    M_inv = np.linalg.inv(M)
    M_inv_normal = M_inv.dot(plane_normal)
    extent_along_plane_normal = (
        M_inv_normal / math.sqrt(plane_normal.dot(M_inv_normal)))
    point1 = C - extent_along_plane_normal
    point2 = C + extent_along_plane_normal
    dist, closest_point_plane, closest_point_ellipsoid = _plane_to_convex_hull_points(
        plane_point, plane_normal, np.vstack((point1, point2)))
    return dist, closest_point_plane, closest_point_ellipsoid


@numba.njit(cache=True)
def _ellipsoid_quadric_matrix(ellipsoid2origin, radii):
    R_scaled = ellipsoid2origin[:3, :3] / radii
    return R_scaled.dot(R_scaled.T)
