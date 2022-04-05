import math
import numpy as np


def point_to_circle(point, center, radius, normal, epsilon=1e-6):
    """Compute the shortest distance between point and circle (only line).

    Implementation adapted from 3D Game Engine Design by David H. Eberly.

    Geometric Tools, Inc.
    http://www.geometrictools.com
    Copyright (c) 1998-2006.  All Rights Reserved

    The Wild Magic Version 4 Foundation Library source code is supplied
    under the terms of the license agreement
        http://www.geometrictools.com/License/Wm4FoundationLicense.pdf
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

    contact_point_circle : array, shape (3,)
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
        closest_point_circle = np.array([np.finfo(float).max] * 3)
        dist = math.sqrt(radius * radius + dist_to_plane * dist_to_plane)

    return dist, closest_point_circle
