import numpy as np
import pytransform3d.transformations as pt


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