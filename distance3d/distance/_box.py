import numpy as np
import pytransform3d.transformations as pt
from ..geometry import convert_segment_to_line
from ._line_to_box import _line_to_box


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
