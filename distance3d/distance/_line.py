import math
import numba
import numpy as np


def point_to_line(point, line_point, line_direction):
    """Compute the shortest distance between point and line.

    Parameters
    ----------
    point : array, shape (3,)
        3D point.

    line_point : array, shape (3,)
        Point on line.

    line_direction : array, shape (3,)
        Direction of the line. This is assumed to be of unit length.

    Returns
    -------
    distance : float
        The shortest distance between point and line.

    closest_point_line : array, shape (3,)
        Closest point on line.
    """
    return _point_to_line(point, line_point, line_direction)[:2]


@numba.njit(
    numba.types.Tuple((numba.float64, numba.float64[::1], numba.float64))
    (numba.float64[::1], numba.float64[::1], numba.float64[::1]),
    cache=True)
def _point_to_line(point, line_point, line_direction):
    diff = point - line_point
    t = np.dot(line_direction, diff)
    direction_fraction = t * line_direction
    diff -= direction_fraction
    closest_point_line = line_point + direction_fraction
    return np.linalg.norm(diff), closest_point_line, t


@numba.njit(
    numba.types.Tuple((numba.float64, numba.float64[::1]))
    (numba.float64[::1], numba.float64[::1], numba.float64[::1]),
    cache=True)
def point_to_line_segment(point, segment_start, segment_end):
    """Compute the shortest distance between point and line segment.

    Parameters
    ----------
    point : array, shape (3,)
        3D point.

    segment_start : array, shape (3,)
        Start point of segment.

    segment_end : array, shape (3,)
        End point of segment.

    Returns
    -------
    distance : float
        The shortest distance between point and line segment.

    closest_point_line_segment : array, shape (3,)
        Closest point on line segment.
    """
    segment_direction = segment_end - segment_start
    # Project point onto segment, computing parameterized position
    # s(t) = segment_start + t * (segment_end - segment_start)
    t = (np.dot(point - segment_start, segment_direction) /
         np.dot(segment_direction, segment_direction))
    # If outside segment, clamp t to the closest endpoint
    t = min(max(t, 0.0), 1.0)
    # Compute projected position from the clamped t
    closest_point_line_segment = segment_start + t * segment_direction
    return (np.linalg.norm(point - closest_point_line_segment),
            closest_point_line_segment)


def line_to_line(line_point1, line_direction1, line_point2, line_direction2,
                 epsilon=1e-6):
    """Compute the shortest distance between two lines.

    Parameters
    ----------
    line_point1 : array, shape (3,)
        Point on the first line.

    line_direction1 : array, shape (3,)
        Direction of the first line. This is assumed to be of unit length.

    line_point2 : array, shape (3,)
        Point on the second line.

    line_direction2 : array, shape (3,)
        Direction of the second line. This is assumed to be of unit length.

    epsilon : float, optional (default: 1e-6)
        Values smaller than epsilon are considered to be 0.

    Returns
    -------
    distance : float
        The shortest distance between two lines.

    closest_point_line1 : array, shape (3,)
        Closest point on first line.

    closest_point_line2 : array, shape (3,)
        Closest point on second line.
    """
    return _line_to_line(
        line_point1, line_direction1, line_point2, line_direction2,
        epsilon)[:3]


@numba.njit(
    numba.types.Tuple((numba.float64, numba.float64[::1], numba.float64[::1],
                       numba.float64, numba.float64))
    (numba.float64[::1], numba.float64[::1], numba.float64[::1],
     numba.float64[::1], numba.float64),
    cache=True)
def _line_to_line(line_point1, line_direction1, line_point2, line_direction2,
                  epsilon):
    diff = line_point1 - line_point2
    a12 = -np.dot(line_direction1, line_direction2)
    b1 = np.dot(line_direction1, diff)
    c = np.dot(diff, diff)
    det = 1.0 - a12 * a12

    if abs(det) >= epsilon:
        b2 = -np.dot(line_direction2, diff)
        t1 = (a12 * b2 - b1) / det
        t2 = (a12 * b1 - b2) / det
        dist_squared = (
            t1 * (t1 + a12 * t2 + 2.0 * b1)
            + t2 * (a12 * t1 + t2 + 2.0 * b2) + c)
        closest_point_line2 = line_point2 + t2 * line_direction2
    else:  # parallel lines
        t1 = -b1
        t2 = 0.0
        dist_squared = b1 * t1 + c
        closest_point_line2 = line_point2

    closest_point_line1 = line_point1 + t1 * line_direction1

    return (math.sqrt(abs(dist_squared)), closest_point_line1,
            closest_point_line2, t1, t2)


def line_to_line_segment(
        line_point, line_direction, segment_start, segment_end, epsilon=1e-6):
    """Compute the shortest distance between line and line segment.

    Implementation adapted from Real-Time Collision Detection by Christer
    Ericson published by Morgan Kaufmann Publishers, Copyright 2005 Elsevier
    Inc.

    Parameters
    ----------
    line_point : array, shape (3,)
        Point on line.

    line_direction : array, shape (3,)
        Direction of the line. This is assumed to be of unit length.

    segment_start : array, shape (3,)
        Start point of segment.

    segment_end : array, shape (3,)
        End point of segment.

    epsilon : float, optional (default: 1e-6)
        Values smaller than epsilon are considered to be 0.

    Returns
    -------
    dist : float
        The shortest distance between line and line segment.

    closest_point1 : array, shape (3,)
        Closest point on line.

    closest_point2 : array, shape (3,)
        Closest point on line segment.
    """
    return _line_to_line_segment(
        line_point, line_direction, segment_start, segment_end, epsilon)[:3]


# modified version of line segment to line segment
@numba.njit(numba.types.Tuple(
    (numba.float64, numba.float64[::1], numba.float64[::1], numba.float64, numba.float64))
    (numba.float64[::1], numba.float64[::1], numba.float64[::1],
     numba.float64[::1], numba.float64), cache=True)
def _line_to_line_segment(
        line_point, line_direction, segment_start, segment_end, epsilon):
    # Segment direction vectors
    d = segment_end - segment_start

    # Squared segment lengths, always nonnegative
    a = np.dot(d, d)
    e = np.dot(line_direction, line_direction)

    if a < epsilon and e < epsilon:
        # Both segments degenerate into points
        return (np.linalg.norm(line_point - segment_start),
                segment_start, line_point, 0.0, 0.0)

    r = segment_start - line_point
    f = np.dot(line_direction, r)

    if a < epsilon:
        # First segment degenerates into a point
        s = 0.0
        t = f / e
    else:
        c = np.dot(d, r)
        if e <= epsilon:
            # Second segment degenerates into a point
            t = 0.0
            s = min(max(-c / a, 0.0), 1.0)
        else:
            # General nondegenerate case
            b = np.dot(d, line_direction)
            denom = a * e - b * b  # always nonnegative

            if denom != 0.0:
                # If segements not parallel, compute closest point on line 1 to
                # line 2 and clamp to segment 1.
                s = min(max((b * f - c * e) / denom, 0.0), 1.0)
            else:
                # Parallel case: compute arbitrary s.
                s = 0.0

            t = (b * s + f) / e

    closest_point1 = line_point + t * line_direction
    closest_point2 = segment_start + s * d

    return np.linalg.norm(closest_point2 - closest_point1), closest_point1, closest_point2, t, s


def line_segment_to_line_segment(
        segment_start1, segment_end1, segment_start2, segment_end2, epsilon=1e-6):
    """Compute the shortest distance between two line segments.

    Implementation adapted from Real-Time Collision Detection by Christer
    Ericson published by Morgan Kaufmann Publishers, Copyright 2005 Elsevier
    Inc.

    Parameters
    ----------
    segment_start1 : array, shape (3,)
        Start point of segment 1.

    segment_end1 : array, shape (3,)
        End point of segment 1.

    segment_start2 : array, shape (3,)
        Start point of segment 2.

    segment_end2 : array, shape (3,)
        End point of segment 2.

    epsilon : float, optional (default: 1e-6)
        Values smaller than epsilon are considered to be 0.

    Returns
    -------
    distance : float
        The shortest distance between two line segments.

    closest_point_segment1 : array, shape (3,)
        Closest point on first line segment.

    closest_point_segment2 : array, shape (3,)
        Closest point on second line segment.
    """
    return _line_segment_to_line_segment(
        segment_start1, segment_end1, segment_start2, segment_end2, epsilon)


@numba.njit(numba.types.Tuple(
    (numba.float64, numba.float64[:], numba.float64[:]))
    (numba.float64[:], numba.float64[:], numba.float64[:],
     numba.float64[:], numba.float64), cache=True)
def _line_segment_to_line_segment(
        segment_start1, segment_end1, segment_start2, segment_end2, epsilon):
    # Segment direction vectors
    d1 = segment_end1 - segment_start1
    d2 = segment_end2 - segment_start2

    # Squared segment lengths, always nonnegative
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)

    if a < epsilon and e < epsilon:
        # Both segments degenerate into points
        return (np.linalg.norm(segment_start2 - segment_start1),
                segment_start1, segment_start2)

    r = segment_start1 - segment_start2
    f = np.dot(d2, r)

    if a < epsilon:
        # First segment degenerates into a point
        s = 0.0
        t = min(max(f / e, 0.0), 1.0)
    else:
        c = np.dot(d1, r)
        if e <= epsilon:
            # Second segment degenerates into a point
            t = 0.0
            s = min(max(-c / a, 0.0), 1.0)
        else:
            # General nondegenerate case
            b = np.dot(d1, d2)
            denom = a * e - b * b  # always nonnegative

            if denom != 0.0:
                # If segements not parallel, compute closest point on line 1 to
                # line 2 and clamp to segment 1.
                s = min(max((b * f - c * e) / denom, 0.0), 1.0)
            else:
                # Parallel case: compute arbitrary s.
                s = 0.0

            t = (b * s + f) / e

            # If t in [0, 1] done. Else clamp t, recompute s.
            if t < 0.0:
                t = 0.0
                s = min(max(-c / a, 0.0), 1.0)
            elif t > 1.0:
                t = 1.0
                s = min(max((b - c) / a, 0.0), 1.0)

    closest_point1 = segment_start1 + s * d1
    closest_point2 = segment_start2 + t * d2

    return (np.linalg.norm(closest_point2 - closest_point1), closest_point1,
            closest_point2)
