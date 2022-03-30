import numpy as np
from .utils import norm_vector


def point_to_line(point, line_point, line_direction, normalize_direction=False):
    """Compute the shortest distance between point and line.

    Parameters
    ----------
    point : array, shape (3,)
        3D point.

    line_point : array, shape (3,)
        Point on line.

    line_direction : array, shape (3,)
        Direction of the line. This is assumed to be of unit length. Otherwise
        it will only be normalized internally when you set normalize_direction
        to True.

    normalize_direction : bool, optional (default: False)
        Normalize direction internally.

    Returns
    -------
    distance : float
        The shortest distance between point and line.

    contact_point_line : array, shape (3,)
        Closest point on line.
    """
    if normalize_direction:
        line_direction = norm_vector(line_direction)
    return _point_to_line(point, line_point, line_direction)[:2]


def _point_to_line(point, line_point, line_direction):
    diff = point - line_point
    t = np.dot(line_direction, diff)
    direction_fraction = t * line_direction
    diff -= direction_fraction
    point_on_line = line_point + direction_fraction
    return np.linalg.norm(diff), point_on_line, t
