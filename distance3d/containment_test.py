import numpy as np
from .distance import point_to_line_segment
from .utils import EPSILON


def point_in_sphere(point, center, radius):
    diff = point - center
    return diff.dot(diff) <= radius * radius


def point_in_capsule(point, capsule2origin, radius, height):
    segment_start = capsule2origin[:, 3] - 0.5 * height * capsule2origin[:3, 2]
    segment_end = capsule2origin[:, 3] + 0.5 * height * capsule2origin[:3, 2]
    dist = point_to_line_segment(point, segment_start, segment_end)[0]
    return dist <= radius


def point_in_ellipsoid(point, ellipsoid2origin, radii):
    origin2ellipsoid = np.linalg.inv(ellipsoid2origin)
    point = origin2ellipsoid[:3, 3] + origin2ellipsoid[:3, :3].dot(point)
    return all(np.abs(point) / radii < 1.0)


def point_in_disk(point, center, radius, normal):
    # signed distance from point to plane of disk
    diff = point - center
    dist_to_plane = diff.dot(normal)
    if abs(dist_to_plane) > 10.0 * EPSILON:
        return False

    # projection of P - C onto plane is Q - C = P - C - dist_to_plane * N
    diff_in_plane = diff - dist_to_plane * normal
    sqr_dist_in_plane = diff_in_plane.dot(diff_in_plane)
    return sqr_dist_in_plane <= radius * radius


def point_in_cylinder(point, cylinder2origin, radius, length):
    # signed distance from point to plane of disk
    diff = point - cylinder2origin[:3, 3]
    dist_to_plane = diff.dot(cylinder2origin[:3, 2])
    if abs(dist_to_plane) > 0.5 * length:
        return False

    # projection of P - C onto plane is Q - C = P - C - dist_to_plane * N
    diff_in_plane = diff - dist_to_plane * cylinder2origin[:3, 2]
    sqr_dist_in_plane = diff_in_plane.dot(diff_in_plane)
    return sqr_dist_in_plane <= radius * radius


def point_in_box(point, box2origin, size):
    origin2box = np.linalg.inv(box2origin)
    point = origin2box[:3, 3] + origin2box[:3, :3].dot(point)
    return all(np.abs(point) < 0.5 * size)
