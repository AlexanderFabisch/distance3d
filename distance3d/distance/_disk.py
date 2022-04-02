import math
import numpy as np


def point_to_disk(point, center, radius, normal):
    """Compute the shortest distance between point and disk.

    Parameters
    ----------
    point : array, shape (3,)
        3D point.

    center : array, shape (3,)
        Center of the disk.

    radius : float
        Radius of the disk.

    normal : array, shape (3,)
        Normal to the plane in which the disk lies.

    Returns
    -------
    dist : float
        The shortest distance between point and disk.

    contact_point_disk : array, shape (3,)
        Closest point on the disk.
    """
    # signed distance from point to plane of disk
    diff = point - center
    dist_to_plane = diff.dot(normal)

    # projection of P - C onto plane is Q - C = P - C - dist_to_plane * N
    diff_in_plane = diff - dist_to_plane * normal
    sqr_len = diff_in_plane.dot(diff_in_plane)

    len = math.sqrt(sqr_len)
    t = radius
    if len != 0.0:
        t /= len
    contact_point = center + min(1.0, t) * diff_in_plane

    return np.linalg.norm(point - contact_point), contact_point
