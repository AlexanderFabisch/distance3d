"""Tools for geometric computations."""
import math
from itertools import product
import numba
import numpy as np
from .utils import norm_vector, transform_point, plane_basis_from_normal


def convert_rectangle_to_segment(rectangle_center, rectangle_extents, i0, i1):
    """Extract line segment from rectangle.

    Parameters
    ----------
    rectangle_center : array, shape (3,)
        Center point of the rectangle.

    rectangle_extents : array, shape (3, 2)
        Extents along axes of the rectangles:
        0.5 * rectangle_sizes * rectangle_axes.

    i0 : int
        Either 0 or 1, selecting line segment.

    i1 : int
        Either 0 or 1, selecting line segment.

    Returns
    -------
    segment_start : array, shape (3,)
        Start point of segment.

    segment_end : array, shape (3,)
        End point of segment.
    """
    segment_middle = rectangle_center + (2 * i0 - 1) * rectangle_extents[i1]
    segment_start = segment_middle - rectangle_extents[1 - i1]
    segment_end = segment_middle + rectangle_extents[1 - i1]
    return segment_end, segment_start


RECTANGLE_COORDS = np.array([
    [-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]])


def convert_rectangle_to_vertices(
        rectangle_center, rectangle_axes, rectangle_lengths):
    """Convert rectangle to vertices.

    Parameters
    ----------
    rectangle_center : array, shape (3,)
        Center point of the rectangle.

    rectangle_axes : array, shape (2, 3)
        Each row is a vector of unit length, indicating the direction of one
        axis of the rectangle. Both vectors are orthogonal.

    rectangle_lengths : array, shape (2,)
        Lengths of the two sides of the rectangle.

    Returns
    -------
    rectangle_points : array, shape (4, 3)
        Vertices of the rectangle.
    """
    return rectangle_center + (RECTANGLE_COORDS * rectangle_lengths).dot(rectangle_axes)


def convert_box_to_face(box2origin, size, i, sign):
    """Convert box to face.

    Parameters
    ----------
    box2origin : array, shape (4, 4)
        Pose of the box.

    size : array, shape (3,)
        Size of the box along its axes.

    i : int
        Index of the axis along which we select the face.

    sign : int
        Indicate the direction along the axis.

    Returns
    -------
    face_center : array, shape (3,)
        Center point of the rectangle.

    face_axes : array, shape (2, 3)
        Each row is a vector of unit length, indicating the direction of one
        axis of the rectangle. Both vectors are orthogonal.

    face_lengths : array, shape (2,)
        Lengths of the two sides of the rectangle.
    """
    other_indices = [0, 1, 2]
    other_indices.remove(i)
    face_center = box2origin[:3, 3] + sign * 0.5 * size[i] * box2origin[:3, i]
    face_axes = np.array([box2origin[:3, j] for j in other_indices])
    face_lengths = np.array([size[j] for j in other_indices])
    return face_center, face_axes, face_lengths


@numba.njit(cache=True)
def convert_segment_to_line(segment_start, segment_end):
    """Convert line segment to line.

    Parameters
    ----------
    segment_start : array, shape (3,)
        Start point of segment.

    segment_end : array, shape (3,)
        End point of segment.

    Returns
    -------
    segment_direction : array, shape (3,)
        Line direction with unit length (or 0).

    segment_length : float
        Length of the line segment.
    """
    segment_direction = segment_end - segment_start
    segment_length = np.linalg.norm(segment_direction)
    if segment_length > 0:
        segment_direction /= segment_length
    return segment_direction, segment_length


BOX_COORDS = np.array(list(product([-0.5, 0.5], repeat=3)))


@numba.njit(
    numba.float64[:, :](numba.float64[:, ::1], numba.float64[::1]),
    cache=True)
def convert_box_to_vertices(box2origin, size):
    """Convert box to vertices.

    Parameters
    ----------
    box2origin : array, shape (4, 4)
        Pose of the box.

    size : array, shape (3,)
        Size of the box along its axes.

    Returns
    -------
    box_points : array, shape (8, 3)
        Vertices of the box.
    """
    return box2origin[:3, 3] + (BOX_COORDS * size).dot(box2origin[:3, :3].T)


@numba.njit(
    numba.float64[:](numba.float64[::1], numba.float64[:, ::1], numba.float64,
                     numba.float64),
    cache=True)
def support_function_cylinder(
        search_direction, cylinder2origin, radius, length):
    """Compute extreme point of cylinder along a direction.

    You can find similar implementations here:

    * https://github.com/kevinmoran/GJK/blob/b38d923d268629f30b44c3cf6d4f9974bbcdb0d3/Collider.h#L42
      (Copyright (c) 2017 Kevin Moran, MIT License or Unlicense)
    * https://github.com/bulletphysics/bullet3/blob/e306b274f1885f32b7e9d65062aa942b398805c2/src/BulletCollision/CollisionShapes/btConvexShape.cpp#L167
      (Copyright (c) 2003-2009 Erwin Coumans, zlib license)

    Parameters
    ----------
    search_direction : array, shape (3,)
        Search direction.

    cylinder2origin : array, shape (4, 4)
        Pose of the cylinder.

    radius : float
        Radius of the cylinder.

    length : float
        Radius of the cylinder.

    Returns
    -------
    extreme_point : array, shape (3,)
        Extreme point along search direction.
    """
    local_dir = np.dot(cylinder2origin[:3, :3].T, search_direction)

    s = math.sqrt(local_dir[0] * local_dir[0] + local_dir[1] * local_dir[1])
    if local_dir[2] < 0.0:
        z = -0.5 * length
    else:
        z = 0.5 * length
    if s == 0.0:
        local_vertex = np.array([radius, 0.0, z])
    else:
        d = radius / s
        local_vertex = np.array([local_dir[0] * d, local_dir[1] * d, z])
    return transform_point(cylinder2origin, local_vertex)


@numba.njit(
    numba.float64[:](numba.float64[::1], numba.float64[:, ::1], numba.float64,
                     numba.float64),
    cache=True)
def support_function_capsule(
        search_direction, capsule2origin, radius, height):
    """Compute extreme point of cylinder along a direction.

    You can find similar implementations here:

    * https://github.com/kevinmoran/GJK/blob/b38d923d268629f30b44c3cf6d4f9974bbcdb0d3/Collider.h#L57
      (Copyright (c) 2017 Kevin Moran, MIT License or Unlicense)
    * https://github.com/bulletphysics/bullet3/blob/e306b274f1885f32b7e9d65062aa942b398805c2/src/BulletCollision/CollisionShapes/btConvexShape.cpp#L228
      (Copyright (c) 2003-2009 Erwin Coumans, zlib license)

    Parameters
    ----------
    search_direction : array, shape (3,)
        Search direction.

    capsule2origin : array, shape (4, 4)
        Pose of the capsule.

    radius : float
        Radius of the cylinder.

    height : float
        Height of the cylinder.

    Returns
    -------
    extreme_point : array, shape (3,)
        Extreme point along search direction.
    """
    local_dir = np.dot(capsule2origin[:3, :3].T, search_direction)

    s = math.sqrt(local_dir[0] * local_dir[0] + local_dir[1] * local_dir[1]
                  + local_dir[2] * local_dir[2])
    if s == 0.0:
        local_vertex = np.array([radius, 0, 0])
    else:
        local_vertex = local_dir * (radius / s)
    if local_dir[2] > 0.0:
        local_vertex[2] += 0.5 * height
    else:
        local_vertex[2] -= 0.5 * height

    return transform_point(capsule2origin, local_vertex)


@numba.njit(
    numba.float64[:](numba.float64[::1], numba.float64[:, ::1], numba.float64[::1]),
    cache=True)
def support_function_ellipsoid(
        search_direction, ellipsoid2origin, radii):
    """Compute extreme point of ellipsoid along a direction.

    Parameters
    ----------
    search_direction : array, shape (3,)
        Search direction.

    ellipsoid2origin : array, shape (4, 4)
        Pose of the ellipsoid.

    radii : array, shape (3,)
        Radii of the ellipsoid.

    Returns
    -------
    extreme_point : array, shape (3,)
        Extreme point along search direction.
    """
    local_dir = np.dot(ellipsoid2origin[:3, :3].T, search_direction)
    local_vertex = norm_vector(local_dir * radii) * radii
    return transform_point(ellipsoid2origin, local_vertex)


@numba.njit(
    numba.float64[:](numba.float64[::1], numba.float64[:, ::1], numba.float64[::1]),
    cache=True)
def support_function_box(search_direction, box2origin, half_lengths):
    """Compute extreme point of box along a direction.

    Parameters
    ----------
    search_direction : array, shape (3,)
        Search direction.

    box2origin : array, shape (4, 4)
        Pose of the box.

    half_lengths : array, shape (3,)
        Half lengths of the box along its axes.

    Returns
    -------
    extreme_point : array, shape (3,)
        Extreme point along search direction.
    """
    local_dir = np.dot(box2origin[:3, :3].T, search_direction)
    local_vertex = np.sign(local_dir) * half_lengths
    return transform_point(box2origin, local_vertex)


@numba.njit(
    numba.float64[:](numba.float64[::1], numba.float64[::1], numba.float64),
    cache=True)
def support_function_sphere(search_direction, center, radius):
    """Compute extreme point of box along a direction.

    You can find similar implementations here:

    * https://github.com/kevinmoran/GJK/blob/b38d923d268629f30b44c3cf6d4f9974bbcdb0d3/Collider.h#L33
      (Copyright (c) 2017 Kevin Moran, MIT License or Unlicense)

    Parameters
    ----------
    search_direction : array, shape (3,)
        Search direction.

    center : array, shape (3,)
        Center of the sphere.

    radius : float
        Radius of the sphere.

    Returns
    -------
    extreme_point : array, shape (3,)
        Extreme point along search direction.
    """
    s_norm = np.linalg.norm(search_direction)
    if s_norm == 0.0:
        vertex = center + np.array([0, 0, radius])
    else:
        vertex = center + search_direction / s_norm * radius
    return vertex


@numba.njit(
    numba.float64[:](numba.float64[::1], numba.float64[::1], numba.float64,
                     numba.float64[::1]),
    cache=True)
def support_function_disk(search_direction, center, radius, normal):
    """Compute extreme point of disk along a direction.

    Parameters
    ----------
    search_direction : array, shape (3,)
        Search direction.

    center : array, shape (3,)
        Center of the disk.

    radius : float
        Radius of the disk.

    normal : array, shape (3,)
        Normal to the plane in which the disk lies.

    Returns
    -------
    extreme_point : array, shape (3,)
        Extreme point along search direction.
    """
    x, y = plane_basis_from_normal(normal)
    R = np.column_stack((x, y, normal))
    point = np.dot(R.T, search_direction)
    point[2] = 0.0
    norm = np.linalg.norm(point)
    if norm == 0.0:
        return np.copy(center)
    point *= radius / norm
    return center + np.dot(R, point)


@numba.njit(
    numba.float64[:](numba.float64[::1], numba.float64[:, ::1], numba.float64,
                     numba.float64),
    cache=True)
def support_function_cone(search_direction, cone2origin, radius, height):
    """Compute extreme point of cone along a direction.

    Parameters
    ----------
    search_direction : array, shape (3,)
        Search direction.

    cone2origin : array, shape (4, 4)
        Pose of the cone.

    radius : float
        Radius of the cone.

    height : float
        Length of the cone.

    Returns
    -------
    extreme_point : array, shape (3,)
        Extreme point along search direction.
    """
    local_dir = np.dot(cone2origin[:3, :3].T, search_direction)
    disk_point = np.array([local_dir[0], local_dir[1], 0.0])
    norm = np.linalg.norm(disk_point)
    if norm == 0.0:
        disk_point = np.array([0.0, 0.0, 0.0])
    else:
        disk_point *= radius / norm
    if np.dot(local_dir, disk_point) >= local_dir[2] * height:
        point_in_cone = disk_point
    else:
        point_in_cone = np.array([0.0, 0.0, height])
    return transform_point(cone2origin, point_in_cone)


@numba.njit(cache=True)
def hesse_normal_form(plane_point, plane_normal):
    """Computes Hesse normal form of a plane.

    In the Hesse normal form (x * n - d = 0), x is any point on the plane,
    n is the plane's normal, and d ist the distance from the origin to the
    plane along its normal.

    Parameters
    ----------
    plane_point : array, shape (3,)
        Point on the plane.

    plane_normal : array, shape (3,)
        Normal of the plane. We assume unit length.

    Returns
    -------
    plane_normal : array, shape (3,)
        Normal of the plane. We assume unit length.

    d : float, optional (default: None)
        Distance of the plane to origin in Hesse normal form.
    """
    return plane_normal, np.dot(plane_point, plane_normal)


def line_from_pluecker(line_direction, line_moment):
    """Computes line from Plücker coordinates.

    Parameters
    ----------
    line_direction : array, shape (3,)
        Direction of the line. Not necessarily of unit length.

    line_moment : array, shape (3,)
        Moment of the line.

    Returns
    -------
    line_point : array, shape (3,)
        Point on line.

    line_direction : array, shape (3,)
        Direction of the line. This is assumed to be of unit length.
    """
    line_point = np.cross(line_direction, line_moment)
    line_direction = line_direction
    line_dir_norm_squared = np.dot(line_direction, line_direction)
    if line_dir_norm_squared > 0.0:
        line_point /= line_dir_norm_squared
        line_direction /= math.sqrt(line_dir_norm_squared)
    return line_point, line_direction
