import numpy as np
from .distance import point_to_line_segment


def point_in_sphere(point, center, radius):
    dist = np.linalg.norm(point - center)
    return dist <= radius


def point_in_capsule(point, capsule2origin, radius, height):
    segment_start = capsule2origin[:, 3] - 0.5 * height * capsule2origin[:3, 2]
    segment_end = capsule2origin[:, 3] + 0.5 * height * capsule2origin[:3, 2]
    dist = point_to_line_segment(point, segment_start, segment_end)[0]
    return dist <= radius
