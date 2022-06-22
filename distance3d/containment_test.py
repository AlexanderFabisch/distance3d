import numpy as np
from .distance import point_to_line_segment
from .utils import EPSILON, invert_transform


def points_in_sphere(points, center, radius):
    diff = points - center
    squared_dist = np.sum(diff * diff, axis=1)
    return squared_dist <= radius * radius


def points_in_capsule(points, capsule2origin, radius, height):
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
    origin2ellipsoid = invert_transform(ellipsoid2origin)
    points = origin2ellipsoid[:3, 3] + np.dot(points, origin2ellipsoid[:3, :3].T)
    normalized_points = points / radii
    return np.sum(normalized_points * normalized_points, axis=1) <= 1.0


def points_in_disk(points, center, radius, normal):
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


def points_in_cylinder(points, cylinder2origin, radius, length):
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
    origin2box = invert_transform(box2origin)
    points = origin2box[:3, 3] + np.dot(points, origin2box[:3, :3].T)
    return np.all(np.abs(points) < 0.5 * size, axis=1)
