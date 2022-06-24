import numpy as np
from .utils import EPSILON, invert_transform


def points_in_sphere(points, center, radius):
    """Test if points are in sphere.

    Parameters
    ----------
    points : array, shape (n_points, 3)
        Points.

    center : array, shape (3,)
        Center of the sphere.

    radius : float
        Radius of the sphere.

    Returns
    -------
    contained : array, shape (n_points,)
        Boolean array indicating whether each point is contained or not.
    """
    diff = points - center
    squared_dist = np.sum(diff * diff, axis=1)
    return squared_dist <= radius * radius


def points_in_capsule(points, capsule2origin, radius, height):
    """Test if points are in capsule.

    Parameters
    ----------
    points : array, shape (n_points, 3)
        Points.

    capsule2origin : array, shape (4, 4)
        Pose of the capsule.

    radius : float
        Radius of the capsule.

    height : float
        Height of the capsule.

    Returns
    -------
    contained : array, shape (n_points,)
        Boolean array indicating whether each point is contained or not.
    """
    segment_start = capsule2origin[:3, 3] - 0.5 * height * capsule2origin[:3, 2]
    segment_end = capsule2origin[:3, 3] + 0.5 * height * capsule2origin[:3, 2]
    segment_direction = segment_end - segment_start
    t = (np.dot(points - segment_start, segment_direction) /
         np.dot(segment_direction, segment_direction))
    t = np.minimum(np.maximum(t, 0.0), 1.0)
    closest_points_line_segment = (
        segment_start[np.newaxis] +
        t[:, np.newaxis] * segment_direction[np.newaxis])
    diff = points - closest_points_line_segment
    squared_dist = np.sum(diff * diff, axis=1)
    return squared_dist <= radius * radius


def points_in_ellipsoid(points, ellipsoid2origin, radii):
    """Test if points are in ellipsoid.

    Parameters
    ----------
    points : array, shape (n_points, 3)
        Points.

    ellipsoid2origin : array, shape (4, 4)
        Pose of the ellipsoid.

    radii : array, shape (3,)
        Radii of the ellipsoid.

    Returns
    -------
    contained : array, shape (n_points,)
        Boolean array indicating whether each point is contained or not.
    """
    origin2ellipsoid = invert_transform(ellipsoid2origin)
    points = origin2ellipsoid[:3, 3] + np.dot(points, origin2ellipsoid[:3, :3].T)
    normalized_points = points / radii
    return np.sum(normalized_points * normalized_points, axis=1) <= 1.0


def points_in_disk(points, center, radius, normal):
    """Test if points are in disk.

    Parameters
    ----------
    points : array, shape (n_points, 3)
        Points.

    center : array, shape (3,)
        Center of the disk.

    radius : float
        Radius of the disk.

    normal : array, shape (3,)
        Normal to the plane in which the disk lies.

    Returns
    -------
    contained : array, shape (n_points,)
        Boolean array indicating whether each point is contained or not.
    """
    contained = np.empty(len(points), dtype=bool)
    contained[:] = True
    # signed distance from point to plane of disk
    diff = points - center
    dist_to_plane = diff.dot(normal)
    contained[np.abs(dist_to_plane) > 10.0 * EPSILON] = False

    # projection of P - C onto plane is Q - C = P - C - dist_to_plane * N
    diff_in_plane = diff - dist_to_plane[:, np.newaxis] * normal[np.newaxis]
    sqr_dist_in_plane = np.sum(diff_in_plane * diff_in_plane, axis=1)
    contained[sqr_dist_in_plane > radius * radius] = False
    return contained


def points_in_cone(points, cone2origin, radius, height):
    """Test if points are in cone.

    Parameters
    ----------
    points : array, shape (n_points, 3)
        Points.

    cone2origin : array, shape (4, 4)
        Pose of the cone.

    radius : float
        Radius of the cone.

    height : float
        Length of the cone.

    Returns
    -------
    contained : array, shape (n_points,)
        Boolean array indicating whether each point is contained or not.
    """
    contained = np.empty(len(points), dtype=bool)
    contained[:] = True
    half_height = 0.5 * height
    # signed distance from point to plane of cone center
    diff = points - (cone2origin[:3, 3] + half_height * cone2origin[:3, 2])
    dist_to_center_plane = diff.dot(cone2origin[:3, 2])
    outside_z = np.abs(dist_to_center_plane) > half_height
    inside_z = np.logical_not(outside_z)
    contained[outside_z] = False

    # projection of P - C onto plane is Q - C = P - C - dist_to_plane * N
    diff_in_plane = diff[inside_z] - dist_to_center_plane[inside_z, np.newaxis] * cone2origin[np.newaxis, :3, 2]
    sqr_dist_in_plane = np.sum(diff_in_plane * diff_in_plane, axis=1)
    dist_to_base_plane = dist_to_center_plane[inside_z] + half_height
    radii = (1.0 - dist_to_base_plane / height) * radius
    not_contained = sqr_dist_in_plane > radii * radii
    inside_z_indices = np.where(inside_z)[0]
    not_contained_indices = inside_z_indices[not_contained]
    contained[not_contained_indices] = False
    return contained


def points_in_cylinder(points, cylinder2origin, radius, length):
    """Test if points are in cylinder.

    Parameters
    ----------
    points : array, shape (n_points, 3)
        Points.

    cylinder2origin : array, shape (4, 4)
        Pose of the cylinder.

    radius : float
        Radius of the cylinder.

    length : float
        Length of the cylinder.

    Returns
    -------
    contained : array, shape (n_points,)
        Boolean array indicating whether each point is contained or not.
    """
    contained = np.empty(len(points), dtype=bool)
    contained[:] = True
    # signed distance from point to plane of disk
    diff = points - cylinder2origin[:3, 3]
    dist_to_plane = diff.dot(cylinder2origin[:3, 2])
    contained[np.abs(dist_to_plane) > 0.5 * length] = False

    # projection of P - C onto plane is Q - C = P - C - dist_to_plane * N
    diff_in_plane = diff - dist_to_plane[:, np.newaxis] * cylinder2origin[np.newaxis, :3, 2]
    sqr_dist_in_plane = np.sum(diff_in_plane * diff_in_plane, axis=1)
    contained[sqr_dist_in_plane > radius * radius] = False
    return contained


def points_in_box(points, box2origin, size):
    """Test if points are in box.

    Parameters
    ----------
    points : array, shape (n_points, 3)
        Points.

    box2origin : array, shape (4, 4)
        Pose of the box.

    size : array, shape (3,)
        Sizes of the box along its axes.

    Returns
    -------
    contained : array, shape (n_points,)
        Boolean array indicating whether each point is contained or not.
    """
    origin2box = invert_transform(box2origin)
    points = origin2box[:3, 3] + np.dot(points, origin2box[:3, :3].T)
    return np.all(np.abs(points) <= 0.5 * size, axis=1)


def points_in_convex_mesh(points, mesh2origin, vertices, triangles):
    """Test if points are in box.

    Parameters
    ----------
    points : array, shape (n_points, 3)
        Points.

    mesh2origin : array, shape (4, 4)
        Pose of the mesh.

    vertices : array, shape (n_vertices, 3)
        Vertices of the mesh.

    triangles : array, shape (n_triangles, 3)
        Indices of vertices that form triangles of the mesh. Face normals
        must point outwards.

    Returns
    -------
    contained : array, shape (n_points,)
        Boolean array indicating whether each point is contained or not.
    """
    origin2mesh = invert_transform(mesh2origin)
    points = origin2mesh[:3, 3] + np.dot(points, origin2mesh[:3, :3].T)

    faces = vertices[triangles]
    A = faces[:, 1] - faces[:, 0]
    B = faces[:, 2] - faces[:, 0]
    face_normals = np.cross(A, B)
    face_centers = np.mean(faces, axis=1)
    contained = np.empty(len(points), dtype=bool)
    contained[:] = True
    for i, point in enumerate(points):
        normal_projected_points = np.sum(
            face_normals * (point[np.newaxis] - face_centers), axis=1)
        if np.any(normal_projected_points > 0.0):
            contained[i] = False
    return contained
