import math
import numpy as np
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt

from ._line_to_box import _line_to_box
from ..geometry import convert_rectangle_to_segment, convert_segment_to_line


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

    contact_point_line : array, shape (3,)
        Closest point on line.
    """
    return _point_to_line(point, line_point, line_direction)[:2]


def _point_to_line(point, line_point, line_direction):
    diff = point - line_point
    t = np.dot(line_direction, diff)
    direction_fraction = t * line_direction
    diff -= direction_fraction
    point_on_line = line_point + direction_fraction
    return np.linalg.norm(diff), point_on_line, t


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

    contact_point_line_segment : array, shape (3,)
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
    contact_point = segment_start + t * segment_direction
    return np.linalg.norm(point - contact_point), contact_point


def line_to_line(line_point1, line_direction1, line_point2, line_direction2,
                 epsilon=1e-6):
    """Compute the shortest distance between two lines.

    Parameters
    ----------
    line_point1 : array, shape (3,)
        Point on the first line.

    line_direction1 : array, shape (3,)
        Direction of the first line. This is assumed to be of unit length.
        Otherwise, it will only be normalized internally when you set
        normalize_directions to True.

    line_point2 : array, shape (3,)
        Point on the second line.

    line_direction2 : array, shape (3,)
        Direction of the second line. This is assumed to be of unit length.
        Otherwise, it will only be normalized internally when you set
        normalize_directions to True.

    epsilon : float, optional (default: 1e-6)
        Values smaller than epsilon are considered to be 0.

    Returns
    -------
    distance : float
        The shortest distance between two lines.

    contact_point_line1 : array, shape (3,)
        Closest point on first line.

    contact_point_line2 : array, shape (3,)
        Closest point on second line.
    """
    return _line_to_line(
        line_point1, line_direction1, line_point2, line_direction2, epsilon)[:3]


def _line_to_line(line_point1, line_direction1, line_point2, line_direction2,
                  epsilon=1e-6):
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
        contact_point2 = line_point2 + t2 * line_direction2
    else:  # parallel lines
        t1 = -b1
        t2 = 0.0
        dist_squared = b1 * t1 + c
        contact_point2 = line_point2

    contact_point1 = line_point1 + t1 * line_direction1

    return math.sqrt(abs(dist_squared)), contact_point1, contact_point2, t1, t2


def line_to_line_segment(
        line_point, line_direction, segment_start, segment_end, epsilon=1e-6):
    """Compute the closest point between line and line segment.

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

    contact_point1 : array, shape (3,)
        Contact point on line.

    contact_point2 : array, shape (3,)
        Contact point on line segment.
    """
    return _line_to_line_segment(
        line_point, line_direction, segment_start, segment_end, epsilon)[:3]


# modified version of line segment to line segment
def _line_to_line_segment(
        line_point, line_direction, segment_start, segment_end,
        epsilon=1e-6):
    # Segment direction vectors
    d = segment_end - segment_start

    # Squared segment lengths, always nonnegative
    a = np.dot(d, d)
    e = np.dot(line_direction, line_direction)

    if a < epsilon and e < epsilon:
        # Both segments degenerate into points
        return (np.linalg.norm(line_point - segment_start),
                segment_start, line_point)

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

    contact_point1 = line_point + t * line_direction
    contact_point2 = segment_start + s * d

    return np.linalg.norm(contact_point2 - contact_point1), contact_point1, contact_point2, t, s


def line_segment_to_line_segment(
        segment_start1, segment_end1, segment_start2, segment_end2, epsilon=1e-6):
    """Compute the shortest distance between two line segments.

    Implementation according to Ericson: Real-Time Collision Detection (2005).

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

    contact_point_segment1 : array, shape (3,)
        Closest point on first line segment.

    contact_point_segment2 : array, shape (3,)
        Closest point on second line segment.
    """
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

    contact_point1 = segment_start1 + s * d1
    contact_point2 = segment_start2 + t * d2

    return np.linalg.norm(contact_point2 - contact_point1), contact_point1, contact_point2


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
        The shortest distance between two line segments. A sign indicates
        the direction along the normal.

    contact_point_plane : array, shape (3,)
        Closest point on plane.
    """
    t = np.dot(plane_normal, point - plane_point)
    contact_point_plane = point - t * plane_normal
    if not signed:
        t = abs(t)
    return t, contact_point_plane


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

    contact_point_rectangle : array, shape (3,)
        Closest point on the rectangle.
    """
    diff = rectangle_center - point
    rectangle_coordinates = -rectangle_axes.dot(diff)

    rectangle_half_lengths = 0.5 * rectangle_lengths
    rectangle_coordinates = np.clip(
        rectangle_coordinates, -rectangle_half_lengths, rectangle_half_lengths)

    contact_point = rectangle_center + rectangle_coordinates.dot(rectangle_axes)

    return np.linalg.norm(point - contact_point), contact_point


def line_to_rectangle(
        line_point, line_direction,
        rectangle_center, rectangle_axes, rectangle_lengths, epsilon=1e-6):
    """Compute the shortest distance between line and rectangle.

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

    contact_point_line : array, shape (3,)
        Closest point on the line.

    contact_point_rectangle : array, shape (3,)
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
            dist, contact_point_line, contact_point_segment, l_closest, s_closest = _line_to_line_segment(
                line_point, line_direction, segment_start, segment_end, epsilon=epsilon)

            if dist < best_dist:
                best_contact_point_line = contact_point_line
                best_contact_point_rectangle = contact_point_segment
                best_dist = dist
                best_line_parameter = l_closest
            if best_dist < epsilon:
                break

    return best_dist, best_contact_point_line, best_contact_point_rectangle, best_line_parameter


def _line_intersects_rectangle(
        line_point, line_direction, rectangle_center, rectangle_axes,
        rectangle_half_lengths, epsilon):
    rectangle_normal = np.cross(rectangle_axes[0], rectangle_axes[1])
    if abs(rectangle_normal.dot(line_direction)) > epsilon:
        # The line and rectangle are not parallel, so the line intersects
        # the plane of the rectangle.
        diff = line_point - rectangle_center
        u, v = pr.plane_basis_from_normal(line_direction)
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
            contact_point_line = line_point + line_parameter * line_direction
            contact_point_rectangle = rectangle_center + s.dot(rectangle_axes)
            return True, (0.0, contact_point_line, contact_point_rectangle, line_parameter)
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

    contact_point_line_segment : array, shape (3,)
        Closest point on the line segment.

    contact_point_rectangle : array, shape (3,)
        Closest point on the rectangle.
    """
    segment_direction, segment_length = convert_segment_to_line(
        segment_start, segment_end)

    distance, contact_point_segment, contact_point_rectangle, t_closest = _line_to_rectangle(
        segment_start, segment_direction,
        rectangle_center, rectangle_axes, rectangle_lengths, epsilon)

    if t_closest < 0:
        distance, contact_point_rectangle = point_to_rectangle(
            segment_start, rectangle_center, rectangle_axes, rectangle_lengths)
        contact_point_segment = segment_start
    elif t_closest > segment_length:
        distance, contact_point_rectangle = point_to_rectangle(
            segment_end, rectangle_center, rectangle_axes, rectangle_lengths)
        contact_point_segment = segment_end

    return distance, contact_point_segment, contact_point_rectangle


def rectangle_to_rectangle(
        rectangle_center1, rectangle_axes1, rectangle_lengths1,
        rectangle_center2, rectangle_axes2, rectangle_lengths2, epsilon=1e-6):
    """Compute the shortest distance between two rectangles.

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

    contact_point_line_segment : array, shape (3,)
        Closest point on the line segment.

    contact_point_rectangle : array, shape (3,)
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

            dist, contact_point_rectangle1, contact_point_rectangle2 = line_segment_to_rectangle(
                segment_start, segment_end, rectangle_center2, rectangle_axes2, rectangle_lengths2)

            if dist < best_dist:
                best_contact_point_rectangle1 = contact_point_rectangle1
                best_contact_point_rectangle2 = contact_point_rectangle2
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

            dist, contact_point_rectangle2, contact_point_rectangle1 = line_segment_to_rectangle(
                segment_start, segment_end, rectangle_center1, rectangle_axes1, rectangle_lengths1)

            if dist < best_dist:
                best_contact_point_rectangle1 = contact_point_rectangle1
                best_contact_point_rectangle2 = contact_point_rectangle2
                best_dist = dist
            if dist <= epsilon:
                break

    return best_dist, best_contact_point_rectangle1, best_contact_point_rectangle2


def point_to_box(point, box2origin, size, origin2box=None):
    """Compute the shortest distance between point and box.

    Parameters
    ----------
    point : array, shape (3,)
        3D point.

    box2origin : array, shape (4, 4)
        Pose of the box.

    size : array, shape (3,)
        Size of the box along its axes.

    origin2box : array, shape (4, 4), optional (default: None)
        Transform from origin to box coordinates.

    Returns
    -------
    dist : float
        The shortest between point and box.

    contact_point_box : array, shape (3,)
        Closest point on box.
    """
    if origin2box is None:
        origin2box = pt.invert_transform(box2origin)
    point_in_box = origin2box[:3, 3] + origin2box[:3, :3].dot(point)
    half_size = 0.5 * size
    contact_point_in_box = np.clip(point_in_box, -half_size, half_size)
    contact_point = pt.transform(
        box2origin, pt.vector_to_point(contact_point_in_box))[:3]
    return np.linalg.norm(point - contact_point), contact_point


def line_to_box(line_point, line_direction, box2origin, size):
    """Compute the shortest distance between line and box.

    Parameters
    ----------
    line_point : array, shape (3,)
        Point on line.

    line_direction : array, shape (3,)
        Direction of the line. This is assumed to be of unit length.

    box2origin : array, shape (4, 4)
        Pose of the box.

    size : array, shape (3,)
        Size of the box along its axes.

    Returns
    -------
    dist : float
        The shortest between line and box.

    contact_point_line : array, shape (3,)
        Closest point on line.

    contact_point_box : array, shape (3,)
        Closest point on box.
    """
    return _line_to_box(line_point, line_direction, box2origin, size)[:3]


def line_segment_to_box(segment_start, segment_end, box2origin, size, origin2box=None):
    """Compute the shortest distance from line segment to box.

    Parameters
    ----------
    segment_start : array, shape (3,)
        Start point of segment.

    segment_end : array, shape (3,)
        End point of segment.

    box2origin : array, shape (4, 4)
        Pose of the box.

    size : array, shape (3,)
        Size of the box along its axes.

    origin2box : array, shape (4, 4), optional (default: None)
        Transform from origin to box coordinates.

    Returns
    -------
    distance : float
        The shortest distance between line segment and box.

    contact_point_line_segment : array, shape (3,)
        Closest point on line segment.

    contact_point_box : array, shape (3,)
        Closest point on box.
    """
    segment_direction, segment_length = convert_segment_to_line(
        segment_start, segment_end)

    distance, contact_point_segment, contact_point_box, t_closest = _line_to_box(
        segment_start, segment_direction, box2origin, size, origin2box=origin2box)

    if t_closest < 0:
        distance, contact_point_box = point_to_box(
            segment_start, box2origin, size, origin2box=origin2box)
        contact_point_segment = segment_start
    elif t_closest > segment_length:
        distance, contact_point_box = point_to_box(
            segment_end, box2origin, size, origin2box=origin2box)
        contact_point_segment = segment_end

    return distance, contact_point_segment, contact_point_box


def point_to_triangle(point, triangle_points):
    """Compute the shortest distance between point and triangle.

    Implementation according to Ericson: Real-Time Collision Detection (2005).

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

    contact_point : array, shape (3,)
        Closest point on triangle.
    """
    ab = triangle_points[1] - triangle_points[0]
    ac = triangle_points[2] - triangle_points[0]

    # Check if point in vertex region outside A
    ap = point - triangle_points[0]
    d1 = np.dot(ab, ap)
    d2 = np.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        contact_point = triangle_points[0]
        return np.linalg.norm(point - contact_point), contact_point

    # Check if point in vertex region outside B
    bp = point - triangle_points[1]
    d3 = np.dot(ab, bp)
    d4 = np.dot(ac, bp)
    if d3 >= 0 and d4 <= d3:
        contact_point = triangle_points[1]
        return np.linalg.norm(point - contact_point), contact_point

    # Check if point in edge region of AB
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 <= d1 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        contact_point = triangle_points[0] + v * ab
        return np.linalg.norm(point - contact_point), contact_point

    # Check if point in vertex region outside C
    cp = point - triangle_points[2]
    d5 = np.dot(ab, cp)
    d6 = np.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        contact_point = triangle_points[2]
        return np.linalg.norm(point - contact_point), contact_point

    # Check if point in edge region of AC
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 <= d2 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        contact_point = triangle_points[0] + w * ac
        return np.linalg.norm(point - contact_point), contact_point

    # Check if point in edge region of BC
    va = d3 * d6 - d5 * d4
    if va <= 0.0 <= d4 - d3 and d5 - d6 >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        contact_point = triangle_points[1] + w * (triangle_points[2] - triangle_points[1])
        return np.linalg.norm(point - contact_point), contact_point

    # Point inside face region
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    contact_point = triangle_points[0] + ab * v + ac * w

    return np.linalg.norm(point - contact_point), contact_point


def line_to_triangle(line_point, line_direction, triangle_points, epsilon=1e-6):
    """Compute the shortest distance between point and triangle.

    Implementation according to Ericson: Real-Time Collision Detection (2005).

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

    contact_point_line : array, shape (3,)
        Closest point on line.

    contact_point_triangle : array, shape (3,)
        Closest point on triangle.
    """
    return _line_to_triangle(line_point, line_direction, triangle_points, epsilon)[:3]


def _line_to_triangle(line_point, line_direction, triangle_points, epsilon=1e-6):
    # Test if line intersects triangle. If so, the squared distance is zero.
    edge = np.array([triangle_points[1] - triangle_points[0],
                     triangle_points[2] - triangle_points[0]])
    normal = pr.norm_vector(np.cross(edge[0], edge[1]))
    if abs(normal.dot(line_direction)) > epsilon:
        # The line and triangle are not parallel, so the line intersects
        # the plane of the triangle.
        diff = line_point - triangle_points[0]
        u, v = pr.plane_basis_from_normal(line_direction)
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
            best_contact_point_line = line_point + line_parameter * line_direction
            best_contact_point_triangle = triangle_points[0] + b.dot(edge)
            return 0.0, best_contact_point_line, best_contact_point_triangle, line_parameter

    # Either (1) the line is not parallel to the triangle and the point of
    # intersection of the line and the plane of the triangle is outside the
    # triangle or (2) the line and triangle are parallel. Regardless, the
    # closest point on the triangle is on an edge of the triangle. Compare
    # the line to all three edges of the triangle.
    best_distance = np.finfo(float).max
    i0 = 2
    i1 = 0
    while i1 < 3:
        distance, contact_point_line, contact_point_segment, t, _ = _line_to_line_segment(
            line_point, line_direction, triangle_points[i0], triangle_points[i1],
            epsilon=epsilon)

        if distance < best_distance:
            best_contact_point_line = contact_point_line
            best_contact_point_triangle = contact_point_segment
            best_distance = distance
            best_line_parameter = t

        i0 = i1
        i1 += 1
    return best_distance, best_contact_point_line, best_contact_point_triangle, best_line_parameter


def line_segment_to_triangle(
        segment_start, segment_end, triangle_points, epsilon=1e-6):
    """Compute the shortest distance from line segment to triangle.

    Implementation according to Ericson: Real-Time Collision Detection (2005).

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

    contact_point_line_segment : array, shape (3,)
        Closest point on line segment.

    contact_point_triangle : array, shape (3,)
        Closest point on triangle.
    """
    segment_direction, segment_length = convert_segment_to_line(
        segment_start, segment_end)

    distance, contact_point_segment, contact_point_triangle, t_closest = _line_to_triangle(
        segment_start, segment_direction, triangle_points, epsilon)

    if t_closest < 0:
        distance, contact_point_triangle = point_to_triangle(
            segment_start, triangle_points)
        contact_point_segment = segment_start
    elif t_closest > segment_length:
        distance, contact_point_triangle = point_to_triangle(
            segment_end, triangle_points)
        contact_point_segment = segment_end

    return distance, contact_point_segment, contact_point_triangle


def triangle_to_triangle(triangle_points1, triangle_points2, epsilon=1e-6):
    """Compute the shortest distance between two triangles.

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

    contact_point_triangle1 : array, shape (3,)
        Closest point on triangle 1.

    contact_point_triangle2 : array, shape (3,)
        Closest point on triangle 2.
    """
    # compare edges of triangle1 to the interior of triangle2
    best_dist = np.finfo(float).max
    i0 = 2
    i1 = 0
    while i1 < 3:
        segment_start = triangle_points1[i0]
        segment_end = triangle_points1[i1]
        dist, contact_point_segment, contact_point_triangle = line_segment_to_triangle(
            segment_start, segment_end, triangle_points2, epsilon)
        if dist < best_dist:
            best_contact_point_triangle1 = contact_point_segment
            best_contact_point_triangle2 = contact_point_triangle
            best_dist = dist

            if best_dist <= epsilon:
                return 0.0, best_contact_point_triangle1, best_contact_point_triangle2
        i0 = i1
        i1 += 1

    # compare edges of triangle2 to the interior of triangle1
    i0 = 2
    i1 = 0
    while i1 < 3:
        segment_start = triangle_points2[i0]
        segment_end = triangle_points2[i1]
        dist, contact_point_segment, contact_point_triangle = line_segment_to_triangle(
            segment_start, segment_end, triangle_points1, epsilon)
        if dist < best_dist:
            best_contact_point_triangle1 = contact_point_triangle
            best_contact_point_triangle2 = contact_point_segment
            best_dist = dist

            if best_dist <= epsilon:
                return 0.0, best_contact_point_triangle1, best_contact_point_triangle2
        i0 = i1
        i1 += 1

    return best_dist, best_contact_point_triangle1, best_contact_point_triangle2


def point_to_cylinder(point, cylinder2origin, radius, length):
    """Compute the shortest distance between point and cylinder.

    Parameters
    ----------
    point : array, shape (3,)
        3D point.

    cylinder2origin : array, shape (4, 4)
        Pose of the cylinder.

    radius : float
        Radius of the cylinder.

    length : float
        Length of the cylinder.

    Returns
    -------
    distance : float
        The shortest distance between point and triangle.

    contact_point : array, shape (3,)
        Closest point on cylinder.
    """
    # signed distance from point to plane of disk
    diff = point - cylinder2origin[:3, 3]
    dist_to_plane = diff.dot(cylinder2origin[:3, 2])

    # projection of P - C onto plane is Q - C = P - C - dist_to_plane * N
    diff_in_plane = diff - dist_to_plane * cylinder2origin[:3, 2]
    sqr_len = diff_in_plane.dot(diff_in_plane)

    contact_point = (
        cylinder2origin[:3, 3]
        + min(1.0, (radius / math.sqrt(sqr_len))) * diff_in_plane
        + np.clip(dist_to_plane, -0.5 * length, 0.5 * length) * cylinder2origin[:3, 2])
    contact_point_to_point = point - contact_point
    sqr_dist = contact_point_to_point.dot(contact_point_to_point)

    return math.sqrt(sqr_dist), contact_point


def point_to_ellipsoid(
        point, ellipsoid2origin, radii, epsilon=1e-16, max_iter=64):
    """Compute the shortest distance between point and ellipsoid.

    Parameters
    ----------
    point : array, shape (3,)
        3D point.

    ellipsoid2origin : array, shape (4, 4)
        Pose of the ellipsoid.

    radii : array, shape (3,)
        Radii of the ellipsoid.

    epsilon : float, optional (default: 1e-6)
        Values smaller than epsilon are considered to be 0.

    max_iter : int, optional (default: 64)
        Maximum number of iterations of the optimization.

    Returns
    -------
    dist : float
        Shortest distance.

    contact_point_ellipsoid : array, shape (3,)
        Closest point on ellipsoid.
    """
    # compute coordinates of point in ellipsoid coordinate system
    point_in_ellipsoid = pt.transform(pt.invert_transform(ellipsoid2origin),
                                      pt.vector_to_point(point))[:3]

    radii2 = radii ** 2
    point2 = point_in_ellipsoid ** 2
    radii2point2 = radii2 * point2

    # initial guess
    if np.linalg.norm(point_in_ellipsoid / radii) < 1.0:
        # TODO in case we want to compute the distance to the surface:
        #t = 0.0  # and don't return here:
        return 0.0, point
    else:
        t = max(radii) * np.linalg.norm(point_in_ellipsoid)

    # Newton's method
    for i in range(max_iter):
        pqr = t + radii2
        pqr2 = pqr ** 2
        fS = pqr2[0] * pqr2[1] * pqr2[2] - radii2point2[0] * pqr2[1] * pqr2[2] - radii2point2[1] * pqr2[0] * pqr2[2] - radii2point2[2] * pqr2[0] * pqr2[1]
        if abs(fS) < epsilon:
            break

        fPQ = pqr[0] * pqr[1]
        fPR = pqr[0] * pqr[2]
        fQR = pqr[1] * pqr[2]
        fPQR = pqr[0] * pqr[1] * pqr[2]
        fDS = 2.0 * (fPQR * (fQR + fPR + fPQ) - radii2point2[0] * fQR * (pqr[1] + pqr[2]) - radii2point2[1] * fPR * (pqr[0] + pqr[2]) - radii2point2[2] * fPQ * (pqr[0] + pqr[1]))
        t -= fS / fDS

    contact_point_in_ellipsoid = radii2 * point_in_ellipsoid / pqr
    diff = contact_point_in_ellipsoid - point_in_ellipsoid

    contact_point = pt.transform(ellipsoid2origin, pt.vector_to_point(contact_point_in_ellipsoid))[:3]

    return np.linalg.norm(diff), contact_point