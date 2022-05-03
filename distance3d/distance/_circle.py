import math
from collections import namedtuple
import numpy as np
import pytransform3d.rotations as pr
from ..utils import norm_vector
from ..geometry import convert_segment_to_line


def point_to_circle(point, center, radius, normal, epsilon=1e-6):
    """Compute the shortest distance between point and circle (only line).

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
    point : array, shape (3,)
        3D point.

    center : array, shape (3,)
        Center of the circle.

    radius : float
        Radius of the circle.

    normal : array, shape (3,)
        Normal to the plane in which the circle lies.

    epsilon : float, optional (default: 1e-6)
        Values smaller than epsilon are considered to be 0.

    Returns
    -------
    dist : float
        The shortest distance between point and circle.

    closest_point_circle : array, shape (3,)
        Closest point on the circle.
    """
    # signed distance from point to plane of circle
    diff = point - center
    dist_to_plane = diff.dot(normal)

    # projection of P - C onto plane is Q - C = P - C - dist_to_plane * normal
    diff_in_plane = diff - dist_to_plane * normal
    sqr_len = diff_in_plane.dot(diff_in_plane)

    if sqr_len >= epsilon:
        closest_point_circle = (
            center + (radius / math.sqrt(sqr_len)) * diff_in_plane)
        dist = np.linalg.norm(point - closest_point_circle)
    else:  # on the line defined by center and normal of the circle
        plane_direction = norm_vector(pr.perpendicular_to_vector(normal))
        closest_point_circle = center + radius * plane_direction
        dist = math.sqrt(radius * radius + dist_to_plane * dist_to_plane)

    return dist, closest_point_circle


def line_to_circle(line_point, line_direction, center, radius, normal):
    """Compute the shortest distance between line and circle.

    This is the nonpolynomial-based algorithm that uses bisection.
    More details here:

    * https://www.geometrictools.com/Documentation/DistanceToCircle3.pdf
    * https://www.geometrictools.com/GTE/Mathematics/DistLine3Circle3.h

    Adapted from C++ implementation of
    David Eberly, Geometric Tools, Redmond WA 98052 (Copyright (c) 1998-2022)
    Distributed under the Boost Software License, Version 1.0.

    * https://www.boost.org/LICENSE_1_0.txt
    * https://www.geometrictools.com/License/Boost/LICENSE_1_0.txt

    Parameters
    ----------
    line_point : array, shape (3,)
        Point on line.

    line_direction : array, shape (3,)
        Direction of the line. This is assumed to be of unit length.

    center : array, shape (3,)
        Center of the circle.

    radius : float
        Radius of the circle.

    normal : array, shape (3,)
        Normal to the plane in which the circle lies.

    Returns
    -------
    dist : float
        The shortest distance between line and circle.

    closest_point_line : array, shape (3,)
        Closest point on line.

    closest_point_circle : array, shape (3,)
        Closest point on circle.
    """
    # The line is P(t) == line_point + t * line_direction.
    # The circle is |x - center| == radius with <normal, x - center> == 0.

    # We transform the line to the circle coordinate frame, in which the circle
    # center is at the origin (0, 0, 0) (and the normal is (0, 0, 1)).
    line_point = line_point - center

    line_direction_cross_normal = np.cross(line_direction, normal)
    line_point_cross_normal = np.cross(line_point, normal)

    m0_squared = np.dot(
        line_direction_cross_normal, line_direction_cross_normal)
    if m0_squared > 0.0:
        closest_point_line, closest_point_circle = _case_line_and_normal_not_parallel(
            line_point, line_direction, center, radius, normal,
            m0_squared, line_direction_cross_normal,
            line_point_cross_normal)
    else:  # Line direction and the plane normal are parallel.
        closest_point_line, closest_point_circle = _case_line_and_normal_parallel(
            line_point, line_direction, center, radius, normal,
            line_point_cross_normal)

    dist = np.linalg.norm(closest_point_line - closest_point_circle)
    return dist, closest_point_line, closest_point_circle


def _case_line_and_normal_not_parallel(
        line_point, line_direction, center, radius, normal, m0_squared,
        line_direction_cross_normal, line_point_cross_normal):
    # Compute the critical points s for F'(s) = 0.
    # The line direction M and the plane normal N are not parallel.
    # Move the line_point = (b0, b1, b2) to
    # line_point' = line_point + lmbda * line_direction = (0, b1', b2').
    sin_directions = math.sqrt(m0_squared)
    lmbda = -np.dot(line_direction_cross_normal,
                    line_point_cross_normal) / m0_squared
    old_line_point = line_point
    line_point = old_line_point + lmbda * line_direction
    line_point_cross_normal += lmbda * line_direction_cross_normal
    m2b2 = np.dot(line_direction, line_point)
    b1_squared = np.dot(line_point_cross_normal, line_point_cross_normal)

    if b1_squared > 0.0:
        roots = _case_general(
            b1_squared, m0_squared, m2b2, radius, sin_directions)
    else:
        roots = _case_b1_is_zero(m2b2, radius * sin_directions)

    candidates = [
        _convert_root_to_candidate(
            root, lmbda, old_line_point, line_direction, center, radius, normal)
        for root in roots]
    candidate = candidates[np.argmin([c.dist_squared for c in candidates])]
    return candidate.closest_point_line, candidate.closest_point_circle


def _case_general(b1_squared, m0_squared, m2b2, radius, sin_directions):
    # line_point' = (0, b1', b2') where b1' != 0. See Sections 1.1.2
    # and 1.2.2 of the PDF documentation.
    roots = []
    b1 = math.sqrt(b1_squared)
    radius_m0_squared = radius * m0_squared
    radius_sin_directions = radius * sin_directions
    if radius_m0_squared > b1:
        s_hat2 = abs(m0_squared * b1_squared ** (2.0 / 3.0) - b1_squared)  # TODO why can this be negative?
        s_hat = math.sqrt(s_hat2) / sin_directions
        g_hat = radius_m0_squared * s_hat / math.sqrt(
            m0_squared * s_hat * s_hat + b1_squared)
        cutoff = g_hat - s_hat
        if m2b2 <= -cutoff:
            roots.append(_bisect_line_circle(
                m2b2, radius_m0_squared, m0_squared, b1_squared, -m2b2,
                -m2b2 + radius_sin_directions))
            if m2b2 == -cutoff:
                roots.append(-s_hat)
        elif m2b2 >= cutoff:
            roots.append(_bisect_line_circle(
                m2b2, radius_m0_squared, m0_squared, b1_squared,
                -m2b2 - radius_sin_directions, -m2b2))
            if m2b2 == cutoff:
                roots.append(s_hat)
        elif m2b2 <= 0.0:
            roots.append(_bisect_line_circle(
                m2b2, radius_m0_squared, m0_squared, b1_squared, -m2b2,
                -m2b2 + radius_sin_directions))
            roots.append(_bisect_line_circle(
                m2b2, radius_m0_squared, m0_squared, b1_squared,
                -m2b2 - radius_sin_directions, -s_hat))
        else:
            roots.append(_bisect_line_circle(
                m2b2, radius_m0_squared, m0_squared, b1_squared,
                -m2b2 - radius_sin_directions, -m2b2))
            roots.append(_bisect_line_circle(
                m2b2, radius_m0_squared, m0_squared, b1_squared, s_hat,
                -m2b2 + radius_sin_directions))
    elif m2b2 < 0.0:
        roots.append(_bisect_line_circle(
            m2b2, radius_m0_squared, m0_squared, b1_squared, -m2b2,
            -m2b2 + radius_sin_directions))
    elif m2b2 > 0.0:
        roots.append(_bisect_line_circle(
            m2b2, radius_m0_squared, m0_squared, b1_squared,
            -m2b2 - radius_sin_directions, -m2b2))
    else:
        roots.append(0.0)

    return roots


def _bisect_line_circle(
        m2b2, rm0_squared, m0_squared, b1_squared,
        s_min, s_max):
    # Bisect the function
    #   f(s) = s + m2b2 - radius * m0_squared * s / sqrt(
    #       m0_squared * s * s + b1sqr)
    # on the specified interval [s_min, s_max].
    for i in range(2, 10):
        s = 0.5 * (s_min + s_max)
        if s + m2b2 - rm0_squared * s / math.sqrt(
                m0_squared * s * s + b1_squared) > 0.0:
            s_max = s
        else:
            s_min = s
    return s


def _case_b1_is_zero(m2b2, radius_sin_directions):
    # The new line origin is B' = (0, 0, b2').
    roots = []
    if m2b2 < 0.0:
        roots.append(-m2b2 + radius_sin_directions)
    elif m2b2 > 0.0:
        roots.append(-m2b2 - radius_sin_directions)
    else:
        roots.append(-m2b2 + radius_sin_directions)
        roots.append(-m2b2 - radius_sin_directions)
    return roots


def _convert_root_to_candidate(
        root, lmbda, old_line_point, line_direction, center, radius, normal):
    t = root + lmbda
    normal_cross_delta = np.cross(
        normal, old_line_point + t * line_direction)
    if any(normal_cross_delta != 0.0):
        closest_point_line, closest_point_circle = _line_circle_closest_points(
            old_line_point, line_direction, center, radius, normal, t)
    else:
        u = pr.perpendicular_to_vector(normal)
        closest_point_line = center
        closest_point_circle = center + radius * u
    diff = closest_point_line - closest_point_circle
    dist_squared = np.dot(diff, diff)
    candidate = LineToCircleCandidate(
        closest_point_line, closest_point_circle, dist_squared)
    return candidate


def _line_circle_closest_points(
        line_point, line_direction, center, radius, normal, t):
    delta = line_point + t * line_direction
    line_closest = center + delta
    delta -= np.dot(normal, delta) * normal
    delta = norm_vector(delta)
    circle_closest = center + radius * delta
    return line_closest, circle_closest


LineToCircleCandidate = namedtuple(
    "LineToCircleCandidate",
    "closest_point_line closest_point_circle dist_squared")


def _case_line_and_normal_parallel(
        line_point, line_direction, center, radius, normal,
        line_point_cross_normal):
    if any(line_point_cross_normal != 0.0):
        # The line is A + t * normal but with A != center.
        closest_point_line, closest_point_circle = _line_circle_closest_points(
            line_point, line_direction, center, radius, normal,
            -np.dot(line_direction, line_point))
    else:
        # The line is center + t * normal, so the circle center is the
        # closest point for the line and all circle points are equidistant
        # from it.
        u = pr.perpendicular_to_vector(normal)
        closest_point_line = center
        closest_point_circle = center + radius * u
    return closest_point_line, closest_point_circle


def line_segment_to_circle(segment_start, segment_end, center, radius, normal):
    """Compute the shortest distance between line segment and circle.

    Parameters
    ----------
    segment_start : array, shape (3,)
        Start point of segment.

    segment_end : array, shape (3,)
        End point of segment.

    center : array, shape (3,)
        Center of the circle.

    radius : float
        Radius of the circle.

    normal : array, shape (3,)
        Normal to the plane in which the circle lies.

    Returns
    -------
    dist : float
        The shortest distance between line segment and circle.

    closest_point_line_segment : array, shape (3,)
        Closest point on line segment.

    closest_point_circle : array, shape (3,)
        Closest point on circle.
    """
    return _line_segment_to_circle(
        segment_start, segment_end, center, radius, normal)[:3]


def _line_segment_to_circle(segment_start, segment_end, center, radius, normal):
    segment_direction, segment_length = convert_segment_to_line(
        segment_start, segment_end)

    dist, closest_point_segment, closest_point_circle = line_to_circle(
        segment_start, segment_direction, center, radius, normal)

    comparison_dimensions = np.where(segment_direction != 0.0)[0]
    assert len(comparison_dimensions) > 0
    comparison_dimension = comparison_dimensions[0]
    t = ((closest_point_segment[comparison_dimension]
          - segment_start[comparison_dimension])
         / segment_direction[comparison_dimension])

    if t < 0.0:
        dist, closest_point_circle = point_to_circle(
            segment_start, center, radius, normal)
        closest_point_segment = segment_start
        on_line = False
    elif t > segment_length:
        dist, closest_point_circle = point_to_circle(
            segment_end, center, radius, normal)
        closest_point_segment = segment_end
        on_line = False
    else:
        on_line = True

    return dist, closest_point_segment, closest_point_circle, on_line
