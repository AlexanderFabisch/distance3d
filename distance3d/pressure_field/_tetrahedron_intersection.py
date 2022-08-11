import numba
import numpy as np
from ..utils import plane_basis_from_normal, EPSILON
from ._halfplanes import intersect_halfplanes, plot_halfplanes_and_intersections


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
    if len(contact_planes) > 0:
        contact_planes = np.vstack(contact_planes)
    return (intersection, contact_planes, contact_polygons,
            contact_polygon_triangles, intersecting_tetrahedra1,
            intersecting_tetrahedra2)


@numba.njit(cache=True)
def intersect_tetrahedron_pair(tetrahedron1, epsilon1, X1,
                               tetrahedron2, epsilon2, X2):
    contact_plane_hnf = contact_plane(X1, X2, epsilon1, epsilon2)
    plane_normal = contact_plane_hnf[:3]
    d = contact_plane_hnf[3]
    if not check_tetrahedra_intersect_contact_plane(
            tetrahedron1, tetrahedron2, plane_normal, d):
        return False, None

    contact_polygon, triangles = compute_contact_polygon(
        X1, X2, plane_normal, d)
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
def check_tetrahedra_intersect_contact_plane(tetrahedron1, tetrahedron2, plane_normal, d, epsilon=1e-6):
    plane_distances1 = tetrahedron1.dot(plane_normal) - d
    plane_distances2 = tetrahedron2.dot(plane_normal) - d
    return (
        min(plane_distances1) < -epsilon
        and max(plane_distances1) > epsilon
        and min(plane_distances2) < -epsilon
        and max(plane_distances2) > epsilon)


@numba.njit(cache=True)
def compute_contact_polygon(X1, X2, plane_normal, d):
    plane_point = plane_normal * d
    cart2plane = np.vstack(plane_basis_from_normal(plane_normal))
    halfplanes = np.vstack((make_halfplanes(X1, plane_point, cart2plane),
                            make_halfplanes(X2, plane_point, cart2plane)))

    poly = intersect_halfplanes(halfplanes)

    if poly is None:
        return None, None

    points = np.empty((len(poly), 2))
    for i in range(len(poly)):
        points[i] = poly[i]

    points = order_points(points)
    # this approach sometimes results in duplicate points, remove them
    n_unique_points, unique_points = filter_unique_points(points)
    if n_unique_points < 3:
        return None, None
    unique_points = unique_points[:n_unique_points]

    #plot_halfplanes_and_intersections(halfplanes, unique_points)

    triangles = tesselate_ordered_polygon(len(unique_points))
    poly3d = project_polygon_to_3d(unique_points, cart2plane, plane_point)
    return poly3d, triangles


@numba.njit(cache=True)
def make_halfplanes(X, plane_point, cart2plane):
    halfplanes = np.empty((4, 4))
    normals2d = X[:, :3].dot(cart2plane.T)
    ds = -X[:, 3] - X[:, :3].dot(plane_point)
    hp_idx = 0
    for i in range(4):
        norm = np.linalg.norm(normals2d[i])
        if norm > EPSILON:
            p = normals2d[i] * ds[i] / (norm * norm)
            halfplanes[i, :2] = p
            halfplanes[i, 2] = normals2d[i, 1]
            halfplanes[i, 3] = -normals2d[i, 0]
            hp_idx += 1
    return halfplanes[:hp_idx]


@numba.njit(cache=True)
def tesselate_ordered_polygon(n_vertices):
    triangles = np.empty((n_vertices - 2, 3), dtype=np.dtype("int"))
    triangles[:, 0] = 0
    triangles[:, 1] = np.arange(1, n_vertices - 1)
    triangles[:, 2] = np.arange(2, n_vertices)
    return triangles


@numba.njit(cache=True)
def order_points(points):
    center = np.array([np.mean(points[:, 0]), np.mean(points[:, 1])])
    poly_centered = points - center
    angles = np.arctan2(poly_centered[:, 1], poly_centered[:, 0])
    points = points[np.argsort(angles)]
    return points


@numba.njit(cache=True)
def filter_unique_points(points):
    """Remove duplicate points.

    Parameters
    ----------
    points :  array, shape (n_points, 2)
        Points that should be filtered.

    Returns
    -------
    n_unique_points : int
        Number of unique points.

    unique_points : array, shape (n_points, 2)
        Unique points.
    """
    epsilon = 10.0 * EPSILON
    unique_points = np.empty((len(points), 2))
    n_unique_points = 0
    for j in range(len(points)):
        if j == 0 or np.linalg.norm(points[j] - points[j - 1]) > epsilon:
            unique_points[n_unique_points] = points[j]
            n_unique_points += 1
    return n_unique_points, unique_points


@numba.njit(cache=True)
def project_polygon_to_3d(vertices, cart2plane, plane_point):
    """Project polygon from contact plane to 3D space.

    Parameters
    ----------
    vertices : array, shape (n_vertices, 2)
        Vertices of contact polygon in contact plane.

    cart2plane : array, shape (2, 3)
        Projection from 3D space to contact plane.

    plane_point : array, shape (3,)
        Point on plane. Projection offset from contact plane to 3D space.

    Returns
    -------
    vertices3d : array, shape (n_vertices, 3)
        Vertices of contact polygon in 3D space.
    """
    plane2cart = cart2plane.T
    return vertices.dot(plane2cart.T) + plane_point
