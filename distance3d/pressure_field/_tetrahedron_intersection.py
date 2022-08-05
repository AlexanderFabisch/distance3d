import numba
import numpy as np
from ..utils import plane_basis_from_normal, norm_vector, EPSILON


TRIANGLES = np.array([[2, 1, 0], [2, 3, 1], [2, 0, 3], [1, 3, 0]], dtype=int)
LINE_SEGMENTS = np.array([[0, 1], [1, 2], [2, 0]], dtype=int)
TRIANGLE_LINE_SEGMENTS = np.array([triangle[LINE_SEGMENTS] for triangle in TRIANGLES], dtype=int)


def intersect_tetrahedron_pairs(pairs, rigid_body1, rigid_body2, X1, X2):
    intersection = False
    contact_planes = []
    contact_polygons = []
    contact_polygon_triangles = []
    intersecting_tetrahedra1 = []
    intersecting_tetrahedra2 = []
    epsilon1 = rigid_body1.tetrahedra_potentials
    epsilon2 = rigid_body2.tetrahedra_potentials
    for i, j in pairs:
        intersecting, contact_details = intersect_tetrahedron_pair(
            rigid_body1.tetrahedra_points[i], epsilon1[i], X1[i],
            rigid_body2.tetrahedra_points[j], epsilon2[j], X2[j])
        if intersecting:
            intersection = True
        else:
            continue

        contact_plane_hnf, contact_polygon, triangles = contact_details

        intersecting_tetrahedra1.append(i)
        intersecting_tetrahedra2.append(j)
        contact_planes.append(contact_plane_hnf)
        contact_polygons.append(contact_polygon)
        contact_polygon_triangles.append(triangles)
    contact_planes = np.vstack(contact_planes)
    return (intersection, contact_planes, contact_polygons,
            contact_polygon_triangles, intersecting_tetrahedra1,
            intersecting_tetrahedra2)


@numba.njit(cache=True)
def intersect_tetrahedron_pair(tetrahedron1, epsilon1, X1,
                               tetrahedron2, epsilon2, X2):
    contact_plane_hnf = contact_plane(X1, X2, epsilon1, epsilon2)
    if not check_tetrahedra_intersect_contact_plane(
            tetrahedron1, tetrahedron2, contact_plane_hnf):
        return False, None

    contact_polygon, triangles = compute_contact_polygon(
        tetrahedron1, tetrahedron2, contact_plane_hnf)
    if contact_polygon is None:
        return False, None

    return True, (contact_plane_hnf, contact_polygon, triangles)


@numba.njit(cache=True)
def contact_plane(X1, X2, epsilon1, epsilon2):
    plane_hnf = epsilon1.dot(X1) - epsilon2.dot(X2)  # TODO Young's modulus, see Eq. 16 of paper
    plane_hnf /= np.linalg.norm(plane_hnf[:3])
    # NOTE in order to obtain proper Hesse normal form of the contact plane
    # we have to multiply the scalar by -1, since it appears as -d in the
    # equation np.dot(normal, x) - d = 0. However, it appears as
    # np.dot(normal, x) + d = 0 in the paper (page 7).
    plane_hnf[3] *= -1
    return plane_hnf


@numba.njit(cache=True)
def check_tetrahedra_intersect_contact_plane(tetrahedron1, tetrahedron2, contact_plane_hnf, epsilon=1e-6):
    plane_distances1 = tetrahedron1.dot(contact_plane_hnf[:3]) - contact_plane_hnf[3]
    plane_distances2 = tetrahedron2.dot(contact_plane_hnf[:3]) - contact_plane_hnf[3]
    return (
        min(plane_distances1) < -epsilon
        and max(plane_distances1) > epsilon
        and min(plane_distances2) < -epsilon
        and max(plane_distances2) > epsilon)


@numba.njit(cache=True)
def compute_contact_polygon(tetrahedron1, tetrahedron2, contact_plane_hnf):
    cart2plane = np.row_stack(plane_basis_from_normal(contact_plane_hnf[:3]))
    halfplanes = np.vstack((
        make_halfplanes(tetrahedron1, contact_plane_hnf, cart2plane),
        make_halfplanes(tetrahedron2, contact_plane_hnf, cart2plane)))
    poly = intersect_halfplanes(halfplanes)

    if poly is None:
        return None, None

    points = np.empty((len(poly), 2))
    for i in range(len(poly)):
        points[i] = poly[i]

    center = np.array([np.mean(points[:, 0]), np.mean(points[:, 1])])
    poly_centered = points - center
    angles = np.arctan2(poly_centered[:, 1], poly_centered[:, 0])
    points = points[np.argsort(angles)]
    # this approach sometimes results in duplicate points, remove them
    unique_points = np.empty((len(points), 2))
    n_unique_points = 0
    for j in range(len(points)):
        if j == len(points) - 1 or np.linalg.norm(points[j + 1] - points[j]) > 10.0 * EPSILON:
            unique_points[n_unique_points] = points[j]
            n_unique_points += 1
    if n_unique_points < 3:
        return None, None

    """
    import matplotlib.pyplot as plt
    plt.figure()
    ax = plt.subplot(111, aspect="equal")
    colors = "rb"
    for i, halfplane in enumerate(halfplanes):
        plot_halfplane(halfplane, ax, colors[i // 4], 0.1)
    plt.scatter(
        points[:, 0], points[:, 1],
        c=["r", "g", "b", "orange", "magenta", "brown", "k"][:len(points)],
        s=100)
    plt.show()
    #"""

    triangles = tesselate_ordered_polygon(unique_points[:n_unique_points])

    poly3d = cartesian_intersection_polygon(unique_points[:n_unique_points], cart2plane, contact_plane_hnf)
    return poly3d, triangles


@numba.njit(cache=True)
def make_halfplanes(tetrahedron_points, plane_hnf, cart2plane):
    plane_normal = plane_hnf[:3]
    d = plane_hnf[3]
    plane_point = plane_normal * d

    P, d_signs, directions = _precompute_edge_intersections(
        d, plane_normal, tetrahedron_points)

    halfplanes = np.empty((4, 4))
    hp_idx = 0
    isect_points = np.empty((2, 3))
    for i, triangle in enumerate(TRIANGLES):
        intersection_points = []
        for line_segment in TRIANGLE_LINE_SEGMENTS[i]:
            i = min(line_segment)
            j = max(line_segment)
            if d_signs[i] != d_signs[j]:
                intersection_points.append(P[i, j])

        if len(intersection_points) != 2:  # TODO what if 3 points?
            continue

        isect_points[0] = intersection_points[0]
        isect_points[1] = intersection_points[1]
        halfplanes[hp_idx, :2], halfplanes[hp_idx, 2:] = make_halfplane(
            cart2plane, directions, isect_points, plane_point, triangle)
        hp_idx += 1
    return halfplanes[:hp_idx]


@numba.njit(cache=True)
def _precompute_edge_intersections(d, plane_normal, tetrahedron_points):
    directions = np.empty((4, 4, 3), np.dtype("float"))
    for i in range(4):
        for j in range(4):
            directions[i, j] = tetrahedron_points[j] - tetrahedron_points[i]
    unnormalized_distances = d - np.dot(tetrahedron_points, plane_normal)
    d_signs = np.sign(unnormalized_distances)
    P = np.empty((4, 4, 3), np.dtype("float"))
    for i in range(4):
        for j in range(i + 1, 4):  # only fill upper triangle
            normal_direction = np.dot(directions[i, j], plane_normal)
            if normal_direction != 0.0:
                t = unnormalized_distances[i] / normal_direction
                P[i, j] = tetrahedron_points[i] + t * directions[i, j]
    return P, d_signs, directions


@numba.njit(cache=True)
def make_halfplane(
        cart2plane, directions, intersection_points, plane_point, triangle):
    # normal pointing inwards
    normal = np.cross(directions[triangle[1], triangle[0]],
                      directions[triangle[2], triangle[0]])
    normal2d = cart2plane.dot(normal)
    intersection_points -= plane_point
    intersection_points = intersection_points.dot(cart2plane.T)
    p, q = intersection_points
    pq = q - p
    if cross2d(pq, normal2d) < 0:
        p = q
        pq *= -1.0
    return p, pq


# replaces from numba.np.extensions import cross2d, which seems to have a bug
# when called with NUMBA_DISABLE_JIT=1
@numba.njit(cache=True)
def cross2d(a, b):
    return a[0] * b[1] - a[1] * b[0]


@numba.njit(cache=True)
def intersect_halfplanes(halfplanes):
    points = []
    for i in range(len(halfplanes)):
        for j in range(i + 1, len(halfplanes)):
            try:
                p = intersect_two_halfplanes(
                    halfplanes[i, :2], halfplanes[i, 2:],
                    halfplanes[j, :2], halfplanes[j, 2:])
            except Exception:
                continue  # parallel halfplanes
            valid = True
            for k in range(len(halfplanes)):
                if k != i and k != j and point_outside_of_halfplane(
                        halfplanes[k, :2], halfplanes[k, 2:], p):
                    valid = False
                    break
            if valid:
                points.append(p)
    if len(points) < 3:
        return None
    return points


@numba.njit(cache=True)
def intersect_two_halfplanes(p1, pq1, p2, pq2):
    denom = cross2d(pq1, pq2)
    if np.abs(denom) < EPSILON:
        raise ValueError("Parallel halfplanes")
    alpha = cross2d((p2 - p1), pq2) / denom
    return p1 + pq1 * alpha


@numba.njit(cache=True)
def point_outside_of_halfplane(p, pq, point):
    return cross2d(pq, point - p) < -EPSILON


def plot_halfplane(ppq, ax, c, alpha):
    line = ppq[:2] + np.linspace(-3.0, 3.0, 101)[:, np.newaxis] * norm_vector(ppq[2:])
    ax.plot(line[:, 0], line[:, 1], lw=3, c=c, alpha=alpha)
    normal2d = np.array([-ppq[3], ppq[2]])
    normal = ppq[:2] + np.linspace(0.0, 1.0, 101)[:, np.newaxis] * norm_vector(normal2d)
    ax.plot(normal[:, 0], normal[:, 1], c=c, alpha=alpha)


@numba.njit(cache=True)
def tesselate_ordered_polygon(poly):
    triangles = np.empty((len(poly) - 2, 3), dtype=np.dtype("int"))
    for i in range(len(poly) - 2):
        triangles[i] = (0, i + 1, i + 2)
    return triangles


@numba.njit(cache=True)
def cartesian_intersection_polygon(poly, cart2plane, contact_plane_hnf):
    plane2cart = cart2plane.T
    plane_point = contact_plane_hnf[:3] * contact_plane_hnf[3]
    return poly.dot(plane2cart.T) + plane_point
