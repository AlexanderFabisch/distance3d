import math
import numpy as np


def convert_rectangle_to_segment(rectangle_center, rectangle_extents, i0, i1):
    """Extract line segment from rectangle.

    Parameters
    ----------
    rectangle_center : array, shape (3,)
        Center point of the rectangle.

    rectangle_extents : array, shape (3, 2)
        Extents along axes of the rectangles:
        0.5 * rectangle_sizes * rectangle_axes.

    i0 : int
        Either 0 or 1, selecting line segment.

    i1 : int
        Either 0 or 1, selecting line segment.

    Returns
    -------
    segment_start : array, shape (3,)
        Start point of segment.

    segment_end : array, shape (3,)
        End point of segment.
    """
    segment_middle = rectangle_center + (2 * i0 - 1) * rectangle_extents[i1]
    segment_start = segment_middle - rectangle_extents[1 - i1]
    segment_end = segment_middle + rectangle_extents[1 - i1]
    return segment_end, segment_start


def convert_segment_to_line(segment_start, segment_end):
    """Convert line segment to line.

    Parameters
    ----------
    segment_start : array, shape (3,)
        Start point of segment.

    segment_end : array, shape (3,)
        End point of segment.

    Returns
    -------
    segment_direction : array, shape (3,)
        Line direction with unit length (or 0).

    segment_length : float
        Length of the line segment.
    """
    segment_direction = segment_end - segment_start
    segment_length = np.linalg.norm(segment_direction)
    if segment_length > 0:
        segment_direction /= segment_length
    return segment_direction, segment_length


def cylinder_extreme_along_direction(search_direction, cylinder2origin, radius, length):
    # https://github.com/kevinmoran/GJK/blob/master/Collider.h#L42
    # https://github.com/bulletphysics/bullet3/blob/master/src/BulletCollision/CollisionShapes/btConvexShape.cpp#L167
    local_dir = np.dot(cylinder2origin[:3, :3].T, search_direction)

    s = math.sqrt(local_dir[0] ** 2 + local_dir[1] ** 2)
    if local_dir[2] < 0.0:
        z = -0.5 * length
    else:
        z = 0.5 * length
    if s != 0.0:
        d = radius / s
        local_vertex = np.array([local_dir[0] * d, local_dir[1] * d, z])
    else:
        local_vertex = np.array([radius, 0.0, z])
    return np.dot(cylinder2origin[:3, :3], local_vertex) + cylinder2origin[:3, 3]


def capsule_extreme_along_direction(search_direction, capsule2origin, radius, height):  # TODO error with axis-aligned capsules
    # https://github.com/kevinmoran/GJK/blob/master/Collider.h#L42/kevinmoran/GJK/blob/master/Collider.h#L42
    # https://github.com/bulletphysics/bullet3/blob/master/src/BulletCollision/CollisionShapes/btConvexShape.cpp#L228
    local_dir = np.dot(capsule2origin[:3, :3].T, search_direction)

    s = np.linalg.norm(local_dir)
    if s == 0.0:
        local_vertex = np.zeros(3)
    else:
        local_vertex = local_dir / s * radius
    if local_dir[2] > 0.0:
        local_vertex[2] += 0.5 * height
    else:
        local_vertex[2] -= 0.5 * height

    return np.dot(capsule2origin[:3, :3], local_vertex) + capsule2origin[:3, 3]
