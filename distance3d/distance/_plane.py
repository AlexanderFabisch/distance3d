import numpy as np


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
