import numpy as np
from ..geometry import (
    convert_rectangle_to_segment, convert_segment_to_line)
from ._line import _line_to_line_segment
from ..utils import plane_basis_from_normal


def point_to_rectangle(point, rectangle_center, rectangle_axes,
                       rectangle_lengths):
    """Compute the shortest distance from point to rectangle.

    Parameters
    ----------
    point : array, shape (3,)
        3D point.

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
        The shortest distance between the point and the rectangle.

    closest_point_rectangle : array, shape (3,)
        Closest point on the rectangle.
    """
    rectangle_coordinates = rectangle_axes.dot(point - rectangle_center)

    rectangle_half_lengths = 0.5 * rectangle_lengths
    rectangle_coordinates = np.clip(
        rectangle_coordinates, -rectangle_half_lengths, rectangle_half_lengths)

    closest_point_rectangle = (
        rectangle_center + rectangle_coordinates.dot(rectangle_axes))

    return (np.linalg.norm(point - closest_point_rectangle),
            closest_point_rectangle)


def line_to_rectangle(
        line_point, line_direction,
        rectangle_center, rectangle_axes, rectangle_lengths, epsilon=1e-6):
    """Compute the shortest distance between line and rectangle.

    Implementation adapted from 3D Game Engine Design by David H. Eberly.

    Geometric Tools, Inc.
    http://www.geometrictools.com
    Copyright (c) 1998-2006.  All Rights Reserved

    The Wild Magic Version 4 Foundation Library source code is supplied
    under the terms of the license agreement
    (http://www.geometrictools.com/License/Wm4FoundationLicense.pdf)
    and may not be copied or disclosed except in accordance with the terms
    of that agreement.

    Parameters
    ----------
    line_point : array, shape (3,)
        Point on line.

    line_direction : array, shape (3,)
        Direction of the line. This is assumed to be of unit length.

    rectangle_center : array, shape (3,)
        Center point of the rectangle.

    rectangle_axes : array, shape (2, 3)
        Each row is a vector of unit length, indicating the direction of one
        axis of the rectangle. Both vectors are orthogonal.

    rectangle_lengths : array, shape (2,)
        Lengths of the two sides of the rectangle.

    epsilon : float, optional (default: 1e-6)
        Values smaller than epsilon are considered to be 0.

    Returns
    -------
    dist : float
        The shortest distance between line and rectangle.

    closest_point_line : array, shape (3,)
        Closest point on the line.

    closest_point_rectangle : array, shape (3,)
        Closest point on the rectangle.
    """
    return _line_to_rectangle(
        line_point, line_direction,
        rectangle_center, rectangle_axes, rectangle_lengths, epsilon)[:3]


def _line_to_rectangle(
        line_point, line_direction,
        rectangle_center, rectangle_axes, rectangle_lengths, epsilon=1e-6):
    rectangle_half_lengths = 0.5 * rectangle_lengths

    # Test if line intersects rectangle. If so, the squared distance is zero.
    intersects, result = _line_intersects_rectangle(
        line_point, line_direction, rectangle_center,
        rectangle_axes, rectangle_half_lengths, epsilon)
    if intersects:
        return result

    rectangle_extents = rectangle_half_lengths[:, np.newaxis] * rectangle_axes
    # Either (1) the line is not parallel to the rectangle and the point of
    # intersection of the line and the plane of the rectangle is outside the
    # rectangle or (2) the line and rectangle are parallel. Regardless, the
    # closest point on the rectangle is on an edge of the rectangle. Compare
    # the line to all four edges of the rectangle.
    best_dist = np.finfo(float).max
    for i1 in range(2):
        for i0 in range(2):
            segment_end, segment_start = convert_rectangle_to_segment(
                rectangle_center, rectangle_extents, i0, i1)
            dist, closest_point_line, closest_point_segment, l_closest, s_closest = _line_to_line_segment(
                line_point, line_direction, segment_start, segment_end, epsilon)

            if dist < best_dist:
                best_closest_point_line = closest_point_line
                best_closest_point_rectangle = closest_point_segment
                best_dist = dist
                best_line_parameter = l_closest
            if best_dist < epsilon:
                break

    return best_dist, best_closest_point_line, best_closest_point_rectangle, best_line_parameter


def _line_intersects_rectangle(
        line_point, line_direction, rectangle_center, rectangle_axes,
        rectangle_half_lengths, epsilon):
    rectangle_normal = np.cross(rectangle_axes[0], rectangle_axes[1])
    if abs(rectangle_normal.dot(line_direction)) > epsilon:
        # The line and rectangle are not parallel, so the line intersects
        # the plane of the rectangle.
        diff = line_point - rectangle_center
        u, v = plane_basis_from_normal(line_direction)
        udd = rectangle_axes.dot(u)
        vdd = rectangle_axes.dot(v)
        uddiff = u.dot(diff)
        vddiff = v.dot(diff)
        det = udd[0] * vdd[1] - udd[1] * vdd[0]

        # Rectangle coordinates for the point of intersection.
        s = np.array([(vdd[1] * uddiff - udd[1] * vddiff),
                      (udd[0] * vddiff - vdd[0] * uddiff)]) / det

        if abs(s[0]) <= rectangle_half_lengths[0] and abs(s[1]) <= rectangle_half_lengths[1]:
            # Line parameter for the point of intersection.
            line_direction_d_d = rectangle_axes.dot(line_direction)
            line_direction_dot_diff = line_direction.dot(diff)
            line_parameter = np.dot(s, line_direction_d_d) - line_direction_dot_diff

            # The intersection point is inside or on the rectangle.
            closest_point_line = line_point + line_parameter * line_direction
            closest_point_rectangle = rectangle_center + s.dot(rectangle_axes)
            return True, (0.0, closest_point_line, closest_point_rectangle, line_parameter)
    return False, None


def line_segment_to_rectangle(
        segment_start, segment_end,
        rectangle_center, rectangle_axes, rectangle_lengths, epsilon=1e-6):
    """Compute the shortest distance between line segment and rectangle.

    Parameters
    ----------
    segment_start : array, shape (3,)
        Start point of segment.

    segment_end : array, shape (3,)
        End point of segment.

    rectangle_center : array, shape (3,)
        Center point of the rectangle.

    rectangle_axes : array, shape (2, 3)
        Each row is a vector of unit length, indicating the direction of one
        axis of the rectangle. Both vectors are orthogonal.

    rectangle_lengths : array, shape (2,)
        Lengths of the two sides of the rectangle.

    epsilon : float, optional (default: 1e-6)
        Values smaller than epsilon are considered to be 0.

    Returns
    -------
    dist : float
        The shortest distance between line segment and rectangle.

    closest_point_line_segment : array, shape (3,)
        Closest point on the line segment.

    closest_point_rectangle : array, shape (3,)
        Closest point on the rectangle.
    """
    segment_direction, segment_length = convert_segment_to_line(
        segment_start, segment_end)

    distance, closest_point_line_segment, closest_point_rectangle, t = \
        _line_to_rectangle(
            segment_start, segment_direction,
            rectangle_center, rectangle_axes, rectangle_lengths, epsilon)

    if t < 0:
        distance, closest_point_rectangle = point_to_rectangle(
            segment_start, rectangle_center, rectangle_axes, rectangle_lengths)
        closest_point_line_segment = segment_start
    elif t > segment_length:
        distance, closest_point_rectangle = point_to_rectangle(
            segment_end, rectangle_center, rectangle_axes, rectangle_lengths)
        closest_point_line_segment = segment_end

    return distance, closest_point_line_segment, closest_point_rectangle


def rectangle_to_rectangle(
        rectangle_center1, rectangle_axes1, rectangle_lengths1,
        rectangle_center2, rectangle_axes2, rectangle_lengths2, epsilon=1e-6):
    """Compute the shortest distance between two rectangles.

    Implementation adapted from 3D Game Engine Design by David H. Eberly.

    Geometric Tools, Inc.
    http://www.geometrictools.com
    Copyright (c) 1998-2006.  All Rights Reserved

    The Wild Magic Version 4 Foundation Library source code is supplied
    under the terms of the license agreement
    (http://www.geometrictools.com/License/Wm4FoundationLicense.pdf)
    and may not be copied or disclosed except in accordance with the terms
    of that agreement.

    Parameters
    ----------
    rectangle_center1 : array, shape (3,)
        Center point of the rectangle.

    rectangle_axes1 : array, shape (2, 3)
        Each row is a vector of unit length, indicating the direction of one
        axis of the rectangle. Both vectors are orthogonal.

    rectangle_lengths1 : array, shape (2,)
        Lengths of the two sides of the rectangle.

    rectangle_center2 : array, shape (3,)
        Center point of the rectangle.

    rectangle_axes2 : array, shape (2, 3)
        Each row is a vector of unit length, indicating the direction of one
        axis of the rectangle. Both vectors are orthogonal.

    rectangle_lengths2 : array, shape (2,)
        Lengths of the two sides of the rectangle.

    epsilon : float, optional (default: 1e-6)
        Values smaller than epsilon are considered to be 0.

    Returns
    -------
    dist : float
        The shortest distance between two rectangles.

    closest_point_line_segment : array, shape (3,)
        Closest point on the line segment.

    closest_point_rectangle : array, shape (3,)
        Closest point on the rectangle.
    """
    # compare edges of rectangle0 to the interior of rectangle1
    best_dist = np.finfo(float).max

    rectangle_half_lengths1 = 0.5 * rectangle_lengths1
    rectangle_extents1 = rectangle_half_lengths1[:, np.newaxis] * rectangle_axes1
    for i1 in range(2):
        for i0 in range(2):
            segment_end, segment_start = convert_rectangle_to_segment(
                rectangle_center1, rectangle_extents1, i0, i1)

            dist, closest_point_rectangle1, closest_point_rectangle2 = line_segment_to_rectangle(
                segment_start, segment_end, rectangle_center2, rectangle_axes2, rectangle_lengths2)

            if dist < best_dist:
                best_closest_point_rectangle1 = closest_point_rectangle1
                best_closest_point_rectangle2 = closest_point_rectangle2
                best_dist = dist
            if dist <= epsilon:
                break

    # compare edges of rectangle1 to the interior of rectangle0
    rectangle_half_lengths2 = 0.5 * rectangle_lengths2
    rectangle_extents2 = rectangle_half_lengths2[:, np.newaxis] * rectangle_axes2
    for i1 in range(2):
        for i0 in range(2):
            segment_end, segment_start = convert_rectangle_to_segment(
                rectangle_center2, rectangle_extents2, i0, i1)

            dist, closest_point_rectangle2, closest_point_rectangle1 = line_segment_to_rectangle(
                segment_start, segment_end, rectangle_center1, rectangle_axes1, rectangle_lengths1)

            if dist < best_dist:
                best_closest_point_rectangle1 = closest_point_rectangle1
                best_closest_point_rectangle2 = closest_point_rectangle2
                best_dist = dist
            if dist <= epsilon:
                break

    return best_dist, best_closest_point_rectangle1, best_closest_point_rectangle2
