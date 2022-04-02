import math
import numpy as np


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
