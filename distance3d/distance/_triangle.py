import numpy as np
import numba
from ..utils import norm_vector, MAX_FLOAT, plane_basis_from_normal
from ..geometry import convert_segment_to_line, convert_rectangle_to_segment
from ._line import _line_to_line_segment
from ._rectangle import line_segment_to_rectangle


@numba.njit(numba.types.Tuple(
    (numba.float64, numba.float64[:]))(numba.float64[:], numba.float64[:, :]),
    cache=True)
def point_to_triangle(point, triangle_points):
    """Compute the shortest distance between point and triangle.

    Implementation adapted from Real-Time Collision Detection by Christer
    Ericson published by Morgan Kaufmann Publishers, Copyright 2005 Elsevier
    Inc.

    Parameters
    ----------
    point : array, shape (3,)
        3D point.

    triangle_points : array, shape (3, 3)
        Each row contains a point of the triangle (A, B, C).

    Returns
    -------
    distance : float
        The shortest distance between point and triangle.

    closest_point : array, shape (3,)
        Closest point on triangle.
    """
    ab = triangle_points[1] - triangle_points[0]
    ac = triangle_points[2] - triangle_points[0]

    # Check if point in vertex region outside A
    ap = point - triangle_points[0]
    d1 = np.dot(ab, ap)
    d2 = np.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        closest_point = triangle_points[0]
        return np.linalg.norm(point - closest_point), closest_point

    # Check if point in vertex region outside B
    bp = point - triangle_points[1]
    d3 = np.dot(ab, bp)
    d4 = np.dot(ac, bp)
    if d3 >= 0 and d4 <= d3:
        closest_point = triangle_points[1]
        return np.linalg.norm(point - closest_point), closest_point

    # Check if point in edge region of AB
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 <= d1 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        closest_point = triangle_points[0] + v * ab
        return np.linalg.norm(point - closest_point), closest_point

    # Check if point in vertex region outside C
    cp = point - triangle_points[2]
    d5 = np.dot(ab, cp)
    d6 = np.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        closest_point = triangle_points[2]
        return np.linalg.norm(point - closest_point), closest_point

    # Check if point in edge region of AC
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 <= d2 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        closest_point = triangle_points[0] + w * ac
        return np.linalg.norm(point - closest_point), closest_point

    # Check if point in edge region of BC
    va = d3 * d6 - d5 * d4
    if va <= 0.0 <= d4 - d3 and d5 - d6 >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        closest_point = triangle_points[1] + w * (triangle_points[2] - triangle_points[1])
        return np.linalg.norm(point - closest_point), closest_point

    # Point inside face region
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    closest_point = triangle_points[0] + ab * v + ac * w

    return np.linalg.norm(point - closest_point), closest_point


def line_to_triangle(line_point, line_direction, triangle_points, epsilon=1e-6):
    """Compute the shortest distance between point and triangle.

    Implementation adapted from Real-Time Collision Detection by Christer
    Ericson published by Morgan Kaufmann Publishers, Copyright 2005 Elsevier
    Inc.

    Parameters
    ----------
    line_point : array, shape (3,)
        Point on line.

    line_direction : array, shape (3,)
        Direction of the line. This is assumed to be of unit length.

    triangle_points : array, shape (3, 3)
        Each row contains a point of the triangle (A, B, C).

    epsilon : float, optional (default: 1e-6)
        Values smaller than epsilon are considered to be 0.

    Returns
    -------
    distance : float
        The shortest distance between line and triangle.

    closest_point_line : array, shape (3,)
        Closest point on line.

    closest_point_triangle : array, shape (3,)
        Closest point on triangle.
    """
    return _line_to_triangle(line_point, line_direction, triangle_points, epsilon)[:3]


@numba.njit(numba.types.Tuple(
    (numba.float64, numba.float64[:], numba.float64[:], numba.float64))
    (numba.float64[::1], numba.float64[::1], numba.float64[:, ::1], numba.float64),
    cache=True)
def _line_to_triangle(line_point, line_direction, triangle_points, epsilon):
    # Test if line intersects triangle. If so, the squared distance is zero.
    edge = np.row_stack((triangle_points[1] - triangle_points[0],
                         triangle_points[2] - triangle_points[0]))
    normal = norm_vector(np.cross(edge[0], edge[1]))
    if abs(normal.dot(line_direction)) > epsilon:
        # The line and triangle are not parallel, so the line intersects
        # the plane of the triangle.
        diff = line_point - triangle_points[0]
        u, v = plane_basis_from_normal(line_direction)
        ude = edge.dot(u)
        vde = edge.dot(v)
        uddiff = u.dot(diff)
        vddiff = v.dot(diff)
        det = ude[0] * vde[1] - ude[1] * vde[0]

        # Barycentric coordinates for the point of intersection.
        b = np.array([vde[1] * uddiff - ude[1] * vddiff,
                      ude[0] * vddiff - vde[0] * uddiff])
        if det != 0.0:
            b /= det
        b2 = 1.0 - b[0] - b[1]

        if b2 >= 0.0 and b[0] >= 0.0 and b[1] >= 0.0:
            # Line parameter for the point of intersection.
            dde = edge.dot(line_direction)
            dddiff = line_direction.dot(diff)
            line_parameter = b.dot(dde) - dddiff

            # The intersection point is inside or on the triangle.
            best_closest_point_line = line_point + line_parameter * line_direction
            best_closest_point_triangle = triangle_points[0] + b.dot(edge)
            return 0.0, best_closest_point_line, best_closest_point_triangle, line_parameter

    # Either (1) the line is not parallel to the triangle and the point of
    # intersection of the line and the plane of the triangle is outside the
    # triangle or (2) the line and triangle are parallel. Regardless, the
    # closest point on the triangle is on an edge of the triangle. Compare
    # the line to all three edges of the triangle.
    best_distance = MAX_FLOAT
    i0 = 2
    i1 = 0
    while i1 < 3:
        distance, closest_point_line, closest_point_segment, t, _ = _line_to_line_segment(
            line_point, line_direction, triangle_points[i0], triangle_points[i1],
            epsilon)

        if distance < best_distance:
            best_closest_point_line = closest_point_line
            best_closest_point_triangle = closest_point_segment
            best_distance = distance
            best_line_parameter = t

        i0 = i1
        i1 += 1
    return best_distance, best_closest_point_line, best_closest_point_triangle, best_line_parameter


def line_segment_to_triangle(
        segment_start, segment_end, triangle_points, epsilon=1e-6):
    """Compute the shortest distance from line segment to triangle.

    Implementation adapted from Real-Time Collision Detection by Christer
    Ericson published by Morgan Kaufmann Publishers, Copyright 2005 Elsevier
    Inc.

    Parameters
    ----------
    segment_start : array, shape (3,)
        Start point of segment.

    segment_end : array, shape (3,)
        End point of segment.

    triangle_points : array, shape (3, 3)
        Each row contains a point of the triangle (A, B, C).

    epsilon : float, optional (default: 1e-6)
        Values smaller than epsilon are considered to be 0.

    Returns
    -------
    distance : float
        The shortest distance between line segment and triangle.

    closest_point_line_segment : array, shape (3,)
        Closest point on line segment.

    closest_point_triangle : array, shape (3,)
        Closest point on triangle.
    """
    segment_direction, segment_length = convert_segment_to_line(
        segment_start, segment_end)

    distance, closest_point_segment, closest_point_triangle, t_closest = _line_to_triangle(
        segment_start, segment_direction, triangle_points, epsilon)

    if t_closest < 0:
        distance, closest_point_triangle = point_to_triangle(
            segment_start, triangle_points)
        closest_point_segment = segment_start
    elif t_closest > segment_length:
        distance, closest_point_triangle = point_to_triangle(
            segment_end, triangle_points)
        closest_point_segment = segment_end

    return distance, closest_point_segment, closest_point_triangle


def triangle_to_triangle(triangle_points1, triangle_points2, epsilon=1e-6):
    """Compute the shortest distance between two triangles.

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
    triangle_points1 : array, shape (3, 3)
        Each row contains a point of the first triangle.

    triangle_points2 : array, shape (3, 3)
        Each row contains a point of the second triangle.

    epsilon : float, optional (default: 1e-6)
        Values smaller than epsilon are considered to be 0.

    Returns
    -------
    distance : float
        Shortest distance.

    closest_point_triangle1 : array, shape (3,)
        Closest point on triangle 1.

    closest_point_triangle2 : array, shape (3,)
        Closest point on triangle 2.
    """
    # compare edges of triangle1 to the interior of triangle2
    best_dist = MAX_FLOAT
    i0 = 2
    i1 = 0
    while i1 < 3:
        segment_start = triangle_points1[i0]
        segment_end = triangle_points1[i1]
        dist, closest_point_segment, closest_point_triangle = line_segment_to_triangle(
            segment_start, segment_end, triangle_points2, epsilon)
        if dist < best_dist:
            best_closest_point_triangle1 = closest_point_segment
            best_closest_point_triangle2 = closest_point_triangle
            best_dist = dist

            if best_dist <= epsilon:
                return 0.0, best_closest_point_triangle1, best_closest_point_triangle2
        i0 = i1
        i1 += 1

    # compare edges of triangle2 to the interior of triangle1
    i0 = 2
    i1 = 0
    while i1 < 3:
        segment_start = triangle_points2[i0]
        segment_end = triangle_points2[i1]
        dist, closest_point_segment, closest_point_triangle = line_segment_to_triangle(
            segment_start, segment_end, triangle_points1, epsilon)
        if dist < best_dist:
            best_closest_point_triangle1 = closest_point_triangle
            best_closest_point_triangle2 = closest_point_segment
            best_dist = dist

            if best_dist <= epsilon:
                return 0.0, best_closest_point_triangle1, best_closest_point_triangle2
        i0 = i1
        i1 += 1

    return best_dist, best_closest_point_triangle1, best_closest_point_triangle2


def triangle_to_rectangle(
        triangle_points, rectangle_center, rectangle_axes, rectangle_lengths):
    """Compute the shortest distance between triangle and rectangle.

    Parameters
    ----------
    triangle_points : array, shape (3, 3)
        Each row contains a point of the triangle.

    rectangle_center : array, shape (3,)
        Center point of the rectangle.

    rectangle_axes : array, shape (2, 3)
        Each row is a vector of unit length, indicating the direction of one
        axis of the rectangle. Both vectors are orthogonal.

    rectangle_lengths : array, shape (2,)
        Lengths of the two sides of the rectangle.

    Returns
    -------
    distance : float
        Shortest distance.

    closest_point_triangle : array, shape (3,)
        Closest point on triangle.

    closest_point_rectangle : array, shape (3,)
        Closest point on rectangle.
    """
    # compare edges of triangle to the interior of rectangle
    best_dist = MAX_FLOAT
    i0 = 2
    i1 = 0
    while i1 < 3:
        segment_start = triangle_points[i0]
        segment_end = triangle_points[i1]
        dist, closest_point_triangle, closest_point_rectangle = line_segment_to_rectangle(
            segment_start, segment_end, rectangle_center, rectangle_axes, rectangle_lengths)
        if dist < best_dist:
            best_closest_point_triangle = closest_point_triangle
            best_closest_point_rectangle = closest_point_rectangle
            best_dist = dist
        i0 = i1
        i1 += 1

    # compare edges of rectangle to the interior of triangle
    rectangle_extents = 0.5 * rectangle_lengths[:, np.newaxis] * rectangle_axes
    for i1 in range(2):
        for i0 in range(2):
            segment_end, segment_start = convert_rectangle_to_segment(
                rectangle_center, rectangle_extents, i0, i1)
            dist, closest_point_rectangle, closest_point_triangle = line_segment_to_triangle(
                segment_start, segment_end, triangle_points)
            if dist < best_dist:
                best_closest_point_triangle = closest_point_triangle
                best_closest_point_rectangle = closest_point_rectangle
                best_dist = dist

    return best_dist, best_closest_point_triangle, best_closest_point_rectangle
