import numpy as np
import pytransform3d.transformations as pt


def point_to_ellipsoid(
        point, ellipsoid2origin, radii, distance_to_surface=False,
        epsilon=1e-16, max_iter=64, check=False):
    """Compute the shortest distance between point and ellipsoid.

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

    ellipsoid2origin : array, shape (4, 4)
        Pose of the ellipsoid.

    radii : array, shape (3,)
        Radii of the ellipsoid.

    distance_to_surface : bool, optional (default: False)
        Compute distance to surface or volume otherwise.

    epsilon : float, optional (default: 1e-16)
        Values smaller than epsilon are considered to be 0.

    max_iter : int, optional (default: 64)
        Maximum number of iterations of the optimization.

    check : bool, optional (default: True)
        Check if transformation matrix is valid before inversion.

    Returns
    -------
    dist : float
        Shortest distance.

    closest_point_ellipsoid : array, shape (3,)
        Closest point on ellipsoid.
    """
    # compute coordinates of point in ellipsoid coordinate system
    origin2ellipsoid = pt.invert_transform(ellipsoid2origin, check=check)
    point_in_ellipsoid = origin2ellipsoid[:3, 3] + origin2ellipsoid[:3, :3].dot(point)

    radii2 = radii ** 2
    point2 = point_in_ellipsoid ** 2
    radii2point2 = radii2 * point2

    # initial guess
    normalized_point_norm = np.linalg.norm(point_in_ellipsoid / radii)
    if normalized_point_norm < 1.0:
        if distance_to_surface:
            if normalized_point_norm < 100.0 * epsilon:
                # Point is the center of the ellipsoid, we have two possible
                # solutions on opposite sides and must select one.
                min_radius_idx = np.argmin(radii)
                closest_point = (
                    ellipsoid2origin[:3, 3]
                    + radii[min_radius_idx] * ellipsoid2origin[:3, min_radius_idx])
                return radii[min_radius_idx], closest_point
            t = 0.0
        else:
            return 0.0, point
    else:
        t = max(radii) * np.linalg.norm(point_in_ellipsoid)

    # Newton's method
    for i in range(max_iter):
        pqr = t + radii2
        pqr2 = pqr ** 2
        s = (pqr2[0] * pqr2[1] * pqr2[2]
             - radii2point2[0] * pqr2[1] * pqr2[2]
             - radii2point2[1] * pqr2[0] * pqr2[2]
             - radii2point2[2] * pqr2[0] * pqr2[1])
        if abs(s) < epsilon:
            break

        pq = pqr[0] * pqr[1]
        pr = pqr[0] * pqr[2]
        qr = pqr[1] * pqr[2]
        pqr_ = pqr[0] * pqr[1] * pqr[2]
        ds = 2.0 * (pqr_ * (qr + pr + pq)
                    - radii2point2[0] * qr * (pqr[1] + pqr[2])
                    - radii2point2[1] * pr * (pqr[0] + pqr[2])
                    - radii2point2[2] * pq * (pqr[0] + pqr[1]))
        t -= s / ds

    closest_point_in_ellipsoid = radii2 * point_in_ellipsoid / pqr
    diff = closest_point_in_ellipsoid - point_in_ellipsoid

    closest_point_ellipsoid = (
        ellipsoid2origin[:3, 3]
        + ellipsoid2origin[:3, :3].dot(closest_point_in_ellipsoid))

    return np.linalg.norm(diff), closest_point_ellipsoid
