"""Containment methods compute bounding volumes."""
import numpy as np
from .geometry import convert_box_to_vertices


def axis_aligned_bounding_box(P):
    """Compute axis-aligned bounding box (AABB) that contains points.

    Parameters
    ----------
    P : array, shape (n_points, 3)
        3D points.

    Returns
    -------
    mins : array, shape (3,)
        Minimum coordinates.

    maxs : array, shape (3,)
        Maximum coordinates.
    """
    return np.min(P, axis=0), np.max(P, axis=0)


def sphere_aabb(center, radius):
    """Compute axis-aligned bounding box of sphere.

    Parameters
    ----------
    center : array, shape (3,)
        Center of the sphere.

    radius : float
        Radius of the sphere.

    Returns
    -------
    mins : array, shape (3,)
        Minimum coordinates.

    maxs : array, shape (3,)
        Maximum coordinates.
    """
    return center - radius, center + radius


def box_aabb(box2origin, size):
    """Compute axis-aligned bounding box of an oriented box.

    Parameters
    ----------
    box2origin : array, shape (4, 4)
        Pose of the box.

    size : array, shape (3,)
        Size of the box.

    Returns
    -------
    mins : array, shape (3,)
        Minimum coordinates.

    maxs : array, shape (3,)
        Maximum coordinates.
    """
    vertices = convert_box_to_vertices(box2origin, size)
    return axis_aligned_bounding_box(vertices)


def cylinder_aabb(cylinder2origin, radius, length):
    """Compute axis-aligned bounding box of cylinder.

    Parameters
    ----------
    cylinder2origin : array, shape (4, 4)
        Pose of the cylinder.

    radius : float
        Radius of the cylinder.

    length : float
        Length of the cylinder.

    Returns
    -------
    mins : array, shape (3,)
        Minimum coordinates.

    maxs : array, shape (3,)
        Maximum coordinates.
    """
    # AABB of a cylinder is the same as the AABB of its caps,
    # see https://iquilezles.org/articles/diskbbox/
    axis = cylinder2origin[:3, 2]
    extent = 0.5 * length * np.abs(axis) + radius * np.sqrt(1.0 - axis * axis)
    return cylinder2origin[:3, 3] - extent, cylinder2origin[:3, 3] + extent


def capsule_aabb(capsule2origin, radius, height):
    """Compute axis-aligned bounding box of a capsule.

    Parameters
    ----------
    capsule2origin : array, shape (4, 4)
        Pose of the capsule.

    radius : float
        Radius of the capsule.

    height : float
        Height of the capsule.

    Returns
    -------
    mins : array, shape (3,)
        Minimum coordinates.

    maxs : array, shape (3,)
        Maximum coordinates.
    """
    extent = 0.5 * height * np.abs(capsule2origin[:3, 2]) + radius
    return capsule2origin[:3, 3] - extent, capsule2origin[:3, 3] + extent


def ellipsoid_aabb(ellipsoid2origin, radii):
    """Compute axis-aligned bounding box of a capsule.

    Parameters
    ----------
    ellipsoid2origin : array, shape (4, 4)
        Pose of the ellipsoid.

    radii : array, shape (3,)
        Radii of the ellipsoid.

    Returns
    -------
    mins : array, shape (3,)
        Minimum coordinates.

    maxs : array, shape (3,)
        Maximum coordinates.
    """
    extents = ellipsoid2origin[:3, :3] * radii[np.newaxis]
    extents /= np.linalg.norm(extents, axis=0)
    extents *= radii[np.newaxis]
    extent = np.max(np.dot(ellipsoid2origin[:3, :3], extents.T), axis=0)
    return ellipsoid2origin[:3, 3] - extent, ellipsoid2origin[:3, 3] + extent


def disk_aabb(center, radius, normal):
    """Compute axis-aligned bounding box of a disk.

    Parameters
    ----------
    center : array, shape (3,)
        Center of the disk.

    radius : float
        Radius of the disk.

    normal : array, shape (3,)
        Normal to the plane in which the disk lies.

    Returns
    -------
    mins : array, shape (3,)
        Minimum coordinates.

    maxs : array, shape (3,)
        Maximum coordinates.
    """
    e = radius * np.sqrt(1.0 - normal * normal)
    return center - e, center + e


def cone_aabb(cone2origin, radius, height):
    """Compute axis-aligned bounding box of a cone.

    Parameters
    ----------
    cone2origin : array, shape (4, 4)
        Pose of the cone.

    radius : float
        Radius of the cone.

    height : float
        Length of the cone.

    Returns
    -------
    mins : array, shape (3,)
        Minimum coordinates.

    maxs : array, shape (3,)
        Maximum coordinates.
    """
    pa = cone2origin[:3, 3]
    pb = cone2origin[:3, 3] + height * cone2origin[:3, 2]
    a = pb - pa
    e = np.sqrt(1.0 - a * a / (height * height))
    return np.minimum(pa - e * radius, pb), np.maximum(pa + e * radius, pb)
