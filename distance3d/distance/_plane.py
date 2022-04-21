import numpy as np
from ..geometry import hesse_normal_form, convert_segment_to_line


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


def _line_to_plane(line_point, line_direction, plane_point, plane_normal,
                   epsilon=1e-6):
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

    dist, closest_point_plane = point_to_plane(
        closest_point_segment, plane_point, plane_normal)
    return dist, closest_point_segment, closest_point_plane
