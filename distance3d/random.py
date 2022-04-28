"""Sample geometric shapes."""
import numpy as np
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
from scipy.spatial import ConvexHull
from .utils import norm_vector


def randn_point(random_state, scale=1.0):
    """Sample 3D point from standard normal distribution.

    Parameters
    ----------
    random_state : np.random.RandomState
        Random number generator.

    scale : float, optional (default: 1)
        Scale of point sampled from normal distribution.

    Returns
    -------
    point : array, shape (3,)
        3D Point sampled from standard normal distribution.
    """
    return scale * random_state.randn(3)


def randn_direction(random_state):
    """Sample 3D direction from standard normal distribution and normalize it.

    Parameters
    ----------
    random_state : np.random.RandomState
        Random number generator.

    Returns
    -------
    direction : array, shape (3,)
        3D direction: 3D vector of unit length.
    """
    return norm_vector(random_state.randn(3))


def randn_line(random_state, scale=1.0):
    """Sample 3D line.

    Parameters
    ----------
    random_state : np.random.RandomState
        Random number generator.

    scale : float, optional (default: 1)
        Scale of point sampled from normal distribution.

    Returns
    -------
    line_point : array, shape (3,)
        3D Point sampled from normal distribution.

    line_direction : array, shape (3,)
        3D direction: 3D vector of unit length.
    """
    line_point = randn_point(random_state, scale=scale)
    line_direction = randn_direction(random_state)
    return line_point, line_direction


def randn_line_segment(random_state, scale=1.0):
    """Sample 3D line segment.

    Parameters
    ----------
    random_state : np.random.RandomState
        Random number generator.

    scale : float, optional (default: 1)
        Scale of points sampled from normal distribution.

    Returns
    -------

    segment_start : array, shape (3,)
        Start point of segment sampled from a normal distribution.

    segment_end : array, shape (3,)
        End point of segment sampled from a normal distribution.
    """
    return (randn_point(random_state, scale=scale),
            randn_point(random_state, scale=scale))


def randn_plane(random_state, scale=1.0):
    """Sample plane in 3D.

    Parameters
    ----------
    random_state : np.random.RandomState
        Random number generator.

    scale : float, optional (default: 1)
        Scale of point sampled from normal distribution.

    Returns
    -------
    plane_point : array, shape (3,)
        3D Point sampled from normal distribution.

    plane_normal : array, shape (3,)
        Plane normal: 3D vector of unit length.
    """
    plane_point = randn_point(random_state, scale=scale)
    plane_normal = randn_direction(random_state)
    return plane_point, plane_normal


def randn_triangle(random_state):
    """Sample triangle.

    Parameters
    ----------
    random_state : np.random.RandomState
        Random number generator.

    Returns
    -------
    triangle_points : array, shape (3, 3)
        Each row contains a point of the triangle (A, B, C) sampled from a
        standard normal distribution.
    """
    return random_state.randn(3, 3)


def randn_rectangle(random_state, center_scale=1.0, length_scale=1.0):
    """Sample rectangle.

    Parameters
    ----------
    random_state : np.random.RandomState
        Random number generator.

    center_scale : float, optional (default: 1)
        Scale the center point by this factor.

    length_scale : float, optional (default: 1)
        Scale the lengths by this factor.

    Returns
    -------
    rectangle_center : array, shape (3,)
        Center point of the rectangle sampled from a normal distribution with
        standard deviation 'center_scale'.

    rectangle_axes : array, shape (2, 3)
        Each row is a vector of unit length, indicating the direction of one
        axis of the rectangle. Both vectors are orthogonal. One direction is
        sampled from a normal distribution. The other one is generated from
        it.

    rectangle_lengths : array, shape (2,)
        Lengths of the two sides of the rectangle sampled from a uniform
        distribution on the interval (0, length_scale].
    """
    rectangle_center = center_scale * randn_point(random_state)
    rectangle_axis1 = randn_direction(random_state)
    rectangle_axis2 = norm_vector(pr.perpendicular_to_vector(rectangle_axis1))
    rectangle_lengths = (1.0 - random_state.rand(2)) * length_scale
    rectangle_axes = np.vstack((rectangle_axis1, rectangle_axis2))
    return rectangle_center, rectangle_axes, rectangle_lengths


def rand_circle(random_state, radius_scale=1.0):
    """Sample circle (or disk).

    Parameters
    ----------
    random_state : np.random.RandomState
        Random number generator.

    radius_scale : float, optional (default: 1)
        Scaling factor for radius.

    Returns
    -------
    center : array, shape (3,)
        Center of the circle.

    radius : float
        Radius of the circle within (0, radius_scale].

    normal : array, shape (3,)
        Normal to the plane in which the circle lies.
    """
    center = random_state.randn(3)
    radius = (1.0 - random_state.rand()) * radius_scale
    normal = norm_vector(random_state.randn(3))
    return center, radius, normal


def rand_box(random_state, center_scale=1.0, size_scale=1.0):
    """Sample box.

    Parameters
    ----------
    random_state : np.random.RandomState
        Random number generator.

    center_scale : float, optional (default: 1)
        Scaling factor for center.

    size_scale : float, optional (default: 1)
        Scaling factor for size.

    Returns
    -------
    box2origin : array, shape (4, 4)
        Pose of the box.

    size : array, shape (3,)
        Sizes of the box along its axes within (0, size_scale].
    """
    box2origin = pt.random_transform(random_state)
    box2origin[:3, 3] *= center_scale
    size = (1.0 - random_state.rand(3)) * size_scale
    return box2origin, size


def rand_capsule(random_state, center_scale=1.0, radius_scale=1.0,
                 height_scale=1.0):
    """Sample capsule.

    Parameters
    ----------
    random_state : np.random.RandomState
        Random number generator.

    center_scale : float, optional (default: 1)
        Scaling factor for center.

    radius_scale : float, optional (default: 1)
        Scaling factor for radius.

    height_scale : float, optional (default: 1)
        Scaling factor for height.

    Returns
    -------
    capsule2origin : array, shape (4, 4)
        Pose of the capsule.

    radius : float
        Radius of the capsule within (0, radius_scale].

    height : float
        Height of the capsule within (0, height_scale].
    """
    capsule2origin = pt.random_transform(random_state)
    capsule2origin[:3, 3] *= center_scale
    radius = (1.0 - random_state.rand()) * radius_scale
    height = (1.0 - random_state.rand()) * height_scale
    return capsule2origin, radius, height


def rand_cylinder(random_state, center_scale=1.0, min_radius=0.0,
                  min_length=0.0):
    """Sample cylinder.

    Parameters
    ----------
    random_state : np.random.RandomState
        Random number generator.

    min_radius : float, optional (default: 0)
        Minimum radius of cylinder.

    min_length : float, optional (default: 0)
        Minimum length of cylinder.

    center_scale : float, optional (default: 1)
        Scaling factor for center.

    Returns
    -------
    cylinder2origin : array, shape (4, 4)
        Pose of the cylinder.

    radius : float
        Radius of the cylinder within (min_radius, min_radius + 1].

    length : float
        Length of the cylinder within (min_length, min_length + 1].
    """
    cylinder2origin = pt.random_transform(random_state)
    cylinder2origin[:3, 3] *= center_scale
    radius = min_radius + (1.0 - random_state.rand())
    length = min_length + (1.0 - random_state.rand())
    return cylinder2origin, radius, length


def rand_sphere(random_state, center_scale=1.0, radius_scale=1.0):
    """Sample sphere.

    Parameters
    ----------
    random_state : np.random.RandomState
        Random number generator.

    center_scale : float, optional (default: 1)
        Scaling factor for center.

    radius_scale : float, optional (default: 1)
        Scaling factor for radius.

    Returns
    -------
    center : array, shape (3,)
        Center of the sphere.

    radius : float
        Radius of the sphere within (0, radius_scale].
    """
    center = random_state.randn(3) * center_scale
    radius = (1.0 - random_state.rand()) * radius_scale
    return center, radius


def randn_convex(random_state, n_points=10, center_scale=1.0, std=1.0, return_indices=False):
    """Sample convex mesh.

    Parameters
    ----------
    random_state : np.random.RandomState
        Random number generator.

    n_points : int
        Number of points to sample from normal distribution.

    center_scale : float, optional (default: 1)
        Scaling factor for center.

    std : float, optional (default: 1)
        Standard deviation of normal distribution.

    return_indices : bool, optional (default: False)
        Return indices of vertices in faces.

    Returns
    -------
    vertices : array, shape (n_convex_points, 3)
        Vertices of the convex mesh.

    faces : array, shape (n_triangles, 3, 3)
        Vertices organized as triangles.

    indices : array, shape (n_triangles, 3)
        Indices of points forming the simplical facets of the convex hull.
    """
    points = random_state.randn(n_points, 3) * std
    ch = ConvexHull(points, qhull_options="Qf")
    simplices = ch.simplices
    faces = np.array([points[[i, j, k]] for i, j, k in simplices])

    reordered_indices = np.array([2, 1, 0], dtype=int)
    for face_idx in range(len(faces)):
        if pr.angle_between_vectors(
                np.cross(faces[face_idx, 2] - faces[face_idx, 0],
                         faces[face_idx, 1] - faces[face_idx, 0]),
                np.mean(faces[face_idx], axis=0)) < 0.5 * np.pi:
            faces[face_idx] = faces[face_idx, reordered_indices]
            simplices[face_idx] = simplices[face_idx, reordered_indices]

    center = random_state.randn(1, 3) * center_scale
    points += center
    faces += center
    vertices = points[ch.vertices]
    ret = [vertices, faces]
    if return_indices:
        ret.extend([ch.points, simplices])
    return ret
