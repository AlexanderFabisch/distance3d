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
    return center - radius, center + radius


def box_aabb(box2origin, size):
    vertices = convert_box_to_vertices(box2origin, size)
    return axis_aligned_bounding_box(vertices)


def cylinder_aabb(cylinder2origin, radius, length):
    negative_vertices = np.vstack((
        cylinder_extreme_along_direction(
            np.array([-1.0, 0.0, 0.0]), cylinder2origin, radius, length),
        cylinder_extreme_along_direction(
            np.array([0.0, -1.0, 0.0]), cylinder2origin, radius, length),
        cylinder_extreme_along_direction(
            np.array([0.0, 0.0, -1.0]), cylinder2origin, radius, length),
    ))
    mins = np.min(negative_vertices, axis=0)
    positive_vertices = np.vstack((
        cylinder_extreme_along_direction(
            np.array([1.0, 0.0, 0.0]), cylinder2origin, radius, length),
        cylinder_extreme_along_direction(
            np.array([0.0, 1.0, 0.0]), cylinder2origin, radius, length),
        cylinder_extreme_along_direction(
            np.array([0.0, 0.0, 1.0]), cylinder2origin, radius, length),
    ))
    maxs = np.max(positive_vertices, axis=0)
    return mins, maxs


def capsule_aabb(capsule2origin, radius, height):
    negative_vertices = np.vstack((
        capsule_extreme_along_direction(
            np.array([-1.0, 0.0, 0.0]), capsule2origin, radius, height),
        capsule_extreme_along_direction(
            np.array([0.0, -1.0, 0.0]), capsule2origin, radius, height),
        capsule_extreme_along_direction(
            np.array([0.0, 0.0, -1.0]), capsule2origin, radius, height),
    ))
    mins = np.min(negative_vertices, axis=0)
    positive_vertices = np.vstack((
        capsule_extreme_along_direction(
            np.array([1.0, 0.0, 0.0]), capsule2origin, radius, height),
        capsule_extreme_along_direction(
            np.array([0.0, 1.0, 0.0]), capsule2origin, radius, height),
        capsule_extreme_along_direction(
            np.array([0.0, 0.0, 1.0]), capsule2origin, radius, height),
    ))
    maxs = np.max(positive_vertices, axis=0)
    return mins, maxs
