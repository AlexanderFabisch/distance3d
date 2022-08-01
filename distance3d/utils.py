"""Utility functions."""
import math
import numba
import numpy as np


MAX_FLOAT = np.finfo(float).max
EPSILON = np.finfo(float).eps
HALF_PI = 0.5 * np.pi


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


@numba.njit(cache=True)
def scalar_triple_product(a, b, c):
    r"""Scalar triple product: :math:`a \cdot (b \cross c)`.

    Also known as triple scalar product or box product and written as
    :math:`\left[ a b c \right]`.

    The value of the scalar triple product is the signed volume of a
    parallelepiped formed by the three independent vectors a, b, c. It is
    six times the volume of a tetrahedron spanned by these three vectors.

    The cross and dot product can be interchanged without changing the result:
    :math:`a \cdot (b \cross c) = (a \cross b) \cdot c`. The result also
    remains constant under cyclic permutation of the arguments:
    :math:`(a \cross b) \cdot c = (b \cross c) \cdot a = (c \cross a) \cdot c`.

    The scalar triple product is also the determinant of the matrix
    :math:`\left[ a b c \right]^T` (each vector corresponds to a row).

    The vectors a, b, c lie in the same plane if and only if the scalar
    triple product is 0.

    Parameters
    ----------
    a : array, shape (3,)
        Vector a.

    b : array shape (3,)
        Vector b.

    c : array, shape (3,)
        Vector c.

    Returns
    -------
    d : float
        Scalar triple product.
    """
    return np.dot(a, np.cross(b, c))


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


@numba.njit(numba.float64[::1](numba.float64[:, ::1], numba.float64[::1]),
            cache=True)
def inverse_transform_point(A2B, point_in_B):
    """Transform a point from frame B to frame A.

    Parameters
    ----------
    A2B : array, shape (4, 4)
        Transform from frame A to frame B as homogeneous matrix.

    point_in_B : array, shape (3,)
        Point in frame B.

    Returns
    -------
    point_in_A : array, shape (3,)
        Point in frame A.
    """
    RT = A2B[:3, :3].T
    return np.dot(RT, point_in_B) - np.dot(RT, A2B[:3, 3])


@numba.njit(numba.float64[:, :](numba.float64[:, :]), cache=True)
def invert_transform(A2B):
    """Invert transform.

    Parameters
    ----------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B

    Returns
    -------
    B2A : array-like, shape (4, 4)
        Transform from frame B to frame A
    """
    B2A = np.empty((4, 4))
    RT = A2B[:3, :3].T
    B2A[:3, :3] = RT
    B2A[:3, 3] = -np.dot(RT, A2B[:3, 3])
    B2A[3, :3] = 0.0
    B2A[3, 3] = 1.0
    return B2A


@numba.njit(cache=True)
def cross_product_matrix(v):
    r"""Generate the cross-product matrix of a vector.

    The cross-product matrix :math:`\boldsymbol{V}` satisfies the equation

    .. math::

        \boldsymbol{V} \boldsymbol{w} = \boldsymbol{v} \times
        \boldsymbol{w}

    It is a skew-symmetric (antisymmetric) matrix, i.e.
    :math:`-\boldsymbol{V} = \boldsymbol{V}^T`.

    Parameters
    ----------
    v : array-like, shape (3,)
        3d vector

    Returns
    -------
    V : array-like, shape (3, 3)
        Cross-product matrix
    """
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


@numba.njit(cache=True)
def adjoint_from_transform(A2B):
    """Compute adjoint representation of a transformation matrix.

    The adjoint representation of a transformation
    :math:`\\left[Ad_{\\boldsymbol{T}_{BA}}\\right]`
    from frame A to frame B translates a twist from frame A to frame B
    through the adjoint map

    .. math::

        \\mathcal{V}_{B}
        = \\left[Ad_{\\boldsymbol{T}_{BA}}\\right] \\mathcal{V}_A

    The corresponding matrix form is

    .. math::

        \\left[\\mathcal{V}_{B}\\right]
        = \\boldsymbol{T}_{BA} \\left[\\mathcal{V}_A\\right]
        \\boldsymbol{T}_{BA}^{-1}

    We can also use the adjoint representation to transform a wrench from frame
    A to frame B:

    .. math::

        \\mathcal{F}_B
        = \\left[ Ad_{\\boldsymbol{T}_{AB}} \\right]^T \\mathcal{F}_A

    Note that not only the adjoint is transposed but also the transformation is
    inverted.

    Adjoint representations have the following properties:

    .. math::

        \\left[Ad_{\\boldsymbol{T}_1 \\boldsymbol{T}_2}\\right]
        = \\left[Ad_{\\boldsymbol{T}_1}\\right]
        \\left[Ad_{\\boldsymbol{T}_2}\\right]

    .. math::

        \\left[Ad_{\\boldsymbol{T}}\\right]^{-1} =
        \\left[Ad_{\\boldsymbol{T}^{-1}}\\right]

    Parameters
    ----------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B

    Returns
    -------
    adj_A2B : array, shape (6, 6)
        Adjoint representation of transformation matrix
    """
    R = A2B[:3, :3]
    p = A2B[:3, 3]

    adj_A2B = np.zeros((6, 6))
    adj_A2B[:3, :3] = R
    adj_A2B[3:, :3] = np.dot(cross_product_matrix(p), R)
    adj_A2B[3:, 3:] = R
    return adj_A2B


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
