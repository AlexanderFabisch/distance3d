"""Containment methods compute bounding volumes."""
import numpy as np
from .geometry import (
    convert_box_to_vertices, cylinder_extreme_along_direction,
    capsule_extreme_along_direction)


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


XM = np.array([-1.0, 0.0, 0.0])
YM = np.array([0.0, -1.0, 0.0])
ZM = np.array([0.0, 0.0, -1.0])
XP = np.array([1.0, 0.0, 0.0])
YP = np.array([0.0, 1.0, 0.0])
ZP = np.array([0.0, 0.0, 1.0])


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
    negative_vertices = np.vstack((
        capsule_extreme_along_direction(XM, capsule2origin, radius, height),
        capsule_extreme_along_direction(YM, capsule2origin, radius, height),
        capsule_extreme_along_direction(ZM, capsule2origin, radius, height),
    ))
    mins = np.min(negative_vertices, axis=0)
    positive_vertices = np.vstack((
        capsule_extreme_along_direction(XP, capsule2origin, radius, height),
        capsule_extreme_along_direction(YP, capsule2origin, radius, height),
        capsule_extreme_along_direction(ZP, capsule2origin, radius, height),
    ))
    maxs = np.max(positive_vertices, axis=0)
    return mins, maxs
