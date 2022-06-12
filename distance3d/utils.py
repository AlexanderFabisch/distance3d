"""Utility functions."""
import math
import numba
import numpy as np


MAX_FLOAT = np.finfo(float).max


@numba.njit(numba.float64[::1](numba.float64[::1]), cache=True)
def norm_vector(v):
    """Normalize vector.

    Parameters
    ----------
    v : array, shape (n,)
        nd vector

    Returns
    -------
    u : array, shape (n,)
        nd unit vector with norm 1 or the zero vector
    """
    norm = np.linalg.norm(v)
    if norm == 0.0:
        return v

    return v / norm


@numba.njit(numba.types.Tuple(
    (numba.float64[::1], numba.float64[::1]))(numba.float64[::1]), cache=True)
def plane_basis_from_normal(plane_normal):
    """Compute two basis vectors of a plane from the plane's normal vector.

    Note that there are infinitely many solutions because any rotation of the
    basis vectors about the normal is also a solution. This function
    deterministically picks one of the solutions.

    The two basis vectors of the plane together with the normal form an
    orthonormal basis in 3D space and could be used as columns to form a
    rotation matrix.

    Parameters
    ----------
    plane_normal : array-like, shape (3,)
        Plane normal of unit length.

    Returns
    -------
    x_axis : array, shape (3,)
        x-axis of the plane.

    y_axis : array, shape (3,)
        y-axis of the plane.
    """
    if abs(plane_normal[0]) >= abs(plane_normal[1]):
        # x or z is the largest magnitude component, swap them
        length = math.sqrt(
            plane_normal[0] * plane_normal[0]
            + plane_normal[2] * plane_normal[2])
        x_axis = np.array([-plane_normal[2] / length, 0.0,
                           plane_normal[0] / length])
        y_axis = np.array([
            plane_normal[1] * x_axis[2],
            plane_normal[2] * x_axis[0] - plane_normal[0] * x_axis[2],
            -plane_normal[1] * x_axis[0]])
    else:
        # y or z is the largest magnitude component, swap them
        length = math.sqrt(plane_normal[1] * plane_normal[1]
                           + plane_normal[2] * plane_normal[2])
        x_axis = np.array([0.0, plane_normal[2] / length,
                           -plane_normal[1] / length])
        y_axis = np.array([
            plane_normal[1] * x_axis[2] - plane_normal[2] * x_axis[1],
            -plane_normal[0] * x_axis[2], plane_normal[0] * x_axis[1]])
    return x_axis, y_axis


@numba.njit(numba.float64[::1](numba.float64[:, ::1], numba.float64[::1]),
            cache=True)
def transform_point(A2B, point_in_A):
    """Transform a point from frame A to frame B.

    Parameters
    ----------
    A2B : array, shape (4, 4)
        Transform from frame A to frame B as homogeneous matrix.

    point_in_A : array, shape (3,)
        Point in frame A.

    Returns
    -------
    point_in_B : array, shape (3,)
        Point in frame B.
    """
    return A2B[:3, 3] + np.dot(A2B[:3, :3], point_in_A)


def angles_between_vectors(A, B):
    """Compute angle between two vectors.

    Parameters
    ----------
    A : array, shape (n_vectors, 3)
        3d vectors

    B : array-like, shape (n_vectors, 3)
        3d vectors

    Returns
    -------
    angles : array, shape (n_vectors,)
        Angles between pairs of vectors from A and B
    """
    return np.arccos(
        np.clip(np.sum(A * B, axis=1) / (np.linalg.norm(A, axis=1)
                                         * np.linalg.norm(B, axis=1)),
                -1.0, 1.0))


EPSILON = np.finfo(float).eps
