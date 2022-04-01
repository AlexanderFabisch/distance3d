import numpy as np
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
from .utils import norm_vector


def randn_point(random_state):
    """Sample 3D point from standard normal distribution.

    Parameters
    ----------
    random_state : np.random.RandomState
        Random number generator.

    Returns
    -------
    point : array, shape (3,)
        3D Point sampled from standard normal distribution.
    """
    return random_state.randn(3)


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


def randn_line(random_state):
    """Sample 3D line.

    Parameters
    ----------
    random_state : np.random.RandomState
        Random number generator.

    Returns
    -------
    line_point : array, shape (3,)
        3D Point sampled from standard normal distribution.

    line_direction : array, shape (3,)
        3D direction: 3D vector of unit length.
    """
    line_point = randn_point(random_state)
    line_direction = randn_direction(random_state)
    return line_point, line_direction


def randn_line_segment(random_state):
    """Sample 3D line segment.

    Parameters
    ----------
    random_state : np.random.RandomState
        Random number generator.

    Returns
    -------

    segment_start : array, shape (3,)
        Start point of segment sampled from a standard normal distribution.

    segment_end : array, shape (3,)
        End point of segment sampled from a standard normal distribution.
    """
    return randn_point(random_state), randn_point(random_state)


def randn_rectangle(random_state, scale_center=1.0, length_scale=1.0):
    """Sample rectangle.

    Parameters
    ----------
    random_state : np.random.RandomState
        Random number generator.

    scale_center : float, optional (default: 1)
        Scale the center point by this factor.

    length_scale : float, optional (default: 1)
        Scale the lengths by this factor.

    Returns
    -------
    rectangle_center : array, shape (3,)
        Center point of the rectangle sampled from a normal distribution with
        standard deviation 'scale_center'.

    rectangle_axes : array, shape (2, 3)
        Each row is a vector of unit length, indicating the direction of one
        axis of the rectangle. Both vectors are orthogonal. One direction is
        sampled from a normal distribution. The other one is generated from
        it.

    rectangle_lengths : array, shape (2,)
        Lengths of the two sides of the rectangle sampled from a uniform
        distribution on the interval [0, length_scale).
    """
    rectangle_center = scale_center * randn_point(random_state)
    rectangle_axis1 = randn_direction(random_state)
    rectangle_axis2 = norm_vector(pr.perpendicular_to_vector(rectangle_axis1))
    rectangle_lengths = random_state.rand(2) * length_scale
    rectangle_axes = np.vstack((rectangle_axis1, rectangle_axis2))
    return rectangle_center, rectangle_axes, rectangle_lengths


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
        Sizes of the box along its axes.
    """
    box2origin = pt.random_transform(random_state)
    box2origin[:3, 3] *= center_scale
    size = random_state.rand(3) * size_scale
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
        Radius of the capsule.

    height : float
        Height of the capsule.
    """
    capsule2origin = pt.random_transform(random_state)
    capsule2origin[:3, 3] *= center_scale
    radius = random_state.rand() * radius_scale
    height = random_state.rand() * height_scale
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
        Radius of the cylinder.

    length : float
        Length of the cylinder.
    """
    cylinder2origin = pt.random_transform(random_state)
    cylinder2origin[:3, 3] *= center_scale
    radius = min_radius + random_state.rand()
    length = min_length + random_state.rand()
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
        Radius of the sphere.
    """
    center = random_state.randn(3) * center_scale
    radius = random_state.rand() * radius_scale
    return center, radius
