import numba
import numpy as np
from ..utils import plane_basis_from_normal, EPSILON
from ._halfplanes import intersect_halfplanes


def intersect_tetrahedron_pairs(
        pairs, tetrahedra_points1, tetrahedra_points2,
        epsilon1, epsilon2, X1, X2):
    """Intersect pairs of tetrahedra.

    Parameters
    ----------
    pairs : list
        List of index pairs.

    tetrahedra_points1 : array, shape (n_tetrahedra1, 4, 3)
        Points that form the tetrahedra of object 1.

    tetrahedra_points2 : array, shape (n_tetrahedra2, 4, 3)
        Points that form the tetrahedra of object 2.

    epsilon1 : array, shape (n_tetrahedra1, 4)
        Potentials of the vertices of the tetrahedra of object 1.

    epsilon2 : array, shape (n_tetrahedra2, 4)
        Potentials of the vertices of the tetrahedra of object 2.

    X1 : dict
        Maps tetrahedron indices of first rigid body to barycentric transform.

    X2 : dict
        Maps tetrahedron indices of second rigid body to barycentric transform.

    Returns
    -------
    intersection : bool
        Do both rigid bodies overlap?

    contact_planes : array, shape (n_intersections, 4)
        Contact planes of intersection pairs in Hesse normal form.

    contact_polygons : list
        Vertices of contact polygons in counter-clockwise order.

    intersecting_tetrahedra1 : list
        Intersecting tetrahedron indices of first mesh.

    intersecting_tetrahedra2 : list
        Intersecting tetrahedron indices of second mesh.
    """
    intersection = False
    contact_planes = []
    contact_polygons = []
    intersecting_tetrahedra1 = []
    intersecting_tetrahedra2 = []
    for i, j in pairs:
        intersecting, contact_details = intersect_tetrahedron_pair(
            tetrahedra_points1[i], epsilon1[i], X1[i],
            tetrahedra_points2[j], epsilon2[j], X2[j])
        if intersecting:
            intersection = True
        else:
            continue

        contact_plane_hnf, contact_polygon = contact_details

        intersecting_tetrahedra1.append(i)
        intersecting_tetrahedra2.append(j)
        contact_planes.append(contact_plane_hnf)
        contact_polygons.append(contact_polygon)
    if len(contact_planes) > 0:
        contact_planes = np.vstack(contact_planes)
    return (intersection, contact_planes, contact_polygons,
            intersecting_tetrahedra1, intersecting_tetrahedra2)


@numba.njit(cache=True)
def intersect_tetrahedron_pair(tetrahedron1, epsilon1, X1,
                               tetrahedron2, epsilon2, X2):
    """Intersect a pair of tetrahedra.

    Parameters
    ----------
    tetrahedron1 : array, shape (4, 3)
        Vertices of tetrahedron 1.

    epsilon1 : array, shape (4,)
        Potentials of the vertices of tetrahedron 1.

    X1 : array, shape (4, 4)
        Each row is a halfspace that defines the original tetrahedron.

    tetrahedron2 : array, shape (4, 3)
        Vertices of tetrahedron 2.

    epsilon2 : array, shape (4,)
        Potentials of the vertices of tetrahedron 2.

    X2 : array, shape (4, 4)
        Each row is a halfspace that defines the original tetrahedron.

    Returns
    -------
    intersection : bool
        Do both tetrahedra intersect?

    contact_info : tuple
        Contact plane in Hesse normal form, contact polygon and triangles of
        contact polygon.
    """
    contact_plane_hnf = contact_plane(X1, X2, epsilon1, epsilon2)
    plane_normal = contact_plane_hnf[:3]
    d = contact_plane_hnf[3]
    if not check_tetrahedra_intersect_contact_plane(
            tetrahedron1, tetrahedron2, plane_normal, d, tolerance=1e-6):
        return False, None

    contact_polygon = compute_contact_polygon(X1, X2, plane_normal, d)
    if len(contact_polygon) < 3:
        return False, None

    return True, (contact_plane_hnf, contact_polygon)


@numba.njit(
    numba.float64[::1](numba.float64[:, ::1], numba.float64[:, ::1],
                       numba.float64[::1], numba.float64[::1]),
    cache=True)
def contact_plane(X1, X2, epsilon1, epsilon2):
    """Compute contact plane.

    Parameters
    ----------
    X1 : array, shape (4, 4)
        Each row is a halfspace that defines the original tetrahedron.

    X2 : array, shape (4, 4)
        Each row is a halfspace that defines the original tetrahedron.

    epsilon1 : array, shape (4,)
        Potentials of the vertices of tetrahedron 1.

    epsilon2 : array, shape (4,)
        Potentials of the vertices of tetrahedron 2.

    Returns
    -------
    plane_hnf : array, shape (4,)
        Contact plane in Hesse normal form: plane normal and distance to origin
        along plane normal.
    """
    # TODO Young's modulus, see Eq. 16 of paper
    plane_hnf = epsilon1.dot(X1) - epsilon2.dot(X2)
    plane_hnf /= np.linalg.norm(plane_hnf[:3])
    # NOTE in order to obtain proper Hesse normal form of the contact plane
    # we have to multiply the scalar by -1, since it appears as -d in the
    # equation np.dot(normal, x) - d = 0. However, it appears as
    # np.dot(normal, x) + d = 0 in the paper (page 7).
    plane_hnf[3] *= -1
    return plane_hnf


@numba.njit(
    numba.bool_(numba.float64[:, ::1], numba.float64[:, ::1],
                numba.float64[::1], numba.float64, numba.float64),
    cache=True)
def check_tetrahedra_intersect_contact_plane(
        tetrahedron1, tetrahedron2, plane_normal, d, tolerance):
    """Check if tetrahedra intersect contact plane.

    Parameters
    ----------
    tetrahedron1 : array, shape (4, 3)
        Vertices of tetrahedron 1.

    tetrahedron2 : array, shape (4, 3)
        Vertices of tetrahedron 2.

    plane_normal : array, shape (3,)
        Normal of the contact plane.

    d : float
        Distance to origin along normal.

    tolerance : float, optional (default: 1e-6)
        Floating point tolerance.

    Returns
    -------
    intersection : bool
        Do both tetrahedra intersect the contact plane.
    """
    plane_distances1 = tetrahedron1.dot(plane_normal) - d
    plane_distances2 = tetrahedron2.dot(plane_normal) - d
    return (
        min(plane_distances1) < -tolerance
        and max(plane_distances1) > tolerance
        and min(plane_distances2) < -tolerance
        and max(plane_distances2) > tolerance)


@numba.njit(
    numba.float64[:, ::1](numba.float64[:, ::1], numba.float64[::1],
                          numba.float64[:, ::1]),
    cache=True)
def make_halfplanes(X, plane_point, cart2plane):
    """Project triangles of a tetrahedron to contact plane.

    Parameters
    ----------
    X : array, shape (8, 4)
        Each row is a halfspace that defines one of the original tetrahedra.

    plane_point : array, shape (3,)
        Point on plane.

    cart2plane : array, shape (2, 3)
        Projection from 3D Cartesian space to contact plane.

    Returns
    -------
    halfplanes : array, shape (n_halfplanes, 4)
        Halfplanes in contact plane. Each halfplane is defined by a point
        p and a direction pq.
    """
    halfplanes = np.empty((8, 4))
    face_normals = np.ascontiguousarray(X[:, :3])
    normals2d = face_normals.dot(cart2plane.T)
    ds = -X[:, 3] - face_normals.dot(plane_point)
    hp_idx = 0
    for i in range(8):
        norm = np.linalg.norm(normals2d[i])
        if norm > EPSILON:
            p = normals2d[i] * ds[i] / (norm * norm)
            halfplanes[i, :2] = p
            halfplanes[i, 2] = normals2d[i, 1]
            halfplanes[i, 3] = -normals2d[i, 0]
            hp_idx += 1
    return halfplanes[:hp_idx]


@numba.njit(
    numba.float64[:, ::1](numba.float64[:, ::1]),
    cache=True)
def order_points(points):
    """Order points by angle around their center.

    Parameters
    ----------
    points : array, shape (n_points, 2)
        Points.

    Returns
    -------
    ordered_points : shape (n_points, 2)
        Ordered points.
    """
    center = np.array([np.mean(points[:, 0]), np.mean(points[:, 1])])
    poly_centered = points - center
    angles = np.arctan2(poly_centered[:, 1], poly_centered[:, 0])
    return points[np.argsort(angles)]


@numba.njit(
    numba.float64[:, ::1](numba.float64[:, ::1]),
    cache=True)
def filter_unique_points(points):
    """Remove duplicate points.

    Parameters
    ----------
    points :  array, shape (n_points, 2)
        Points that should be filtered.

    Returns
    -------
    unique_points : array, shape (n_unique_points, 2)
        Unique points.
    """
    epsilon = 10.0 * EPSILON
    unique_points = np.empty((len(points), 2))
    n_unique_points = 0
    for j in range(len(points)):
        if j == 0 or np.linalg.norm(points[j] - points[j - 1]) > epsilon:
            unique_points[n_unique_points] = points[j]
            n_unique_points += 1
    return unique_points[:n_unique_points]


@numba.njit(
    numba.float64[:, ::1](numba.float64[:, ::1], numba.float64[:, ::1],
                          numba.float64[::1]),
    cache=True)
def project_polygon_to_3d(vertices, cart2plane, plane_point):
    """Project polygon from contact plane to 3D space.

    Parameters
    ----------
    vertices : array, shape (n_vertices, 2)
        Vertices of contact polygon in contact plane.

    cart2plane : array, shape (2, 3)
        Projection from 3D Cartesian space to contact plane.

    plane_point : array, shape (3,)
        Point on plane. Projection offset from contact plane to 3D space.

    Returns
    -------
    vertices3d : array, shape (n_vertices, 3)
        Vertices of contact polygon in 3D space.
    """
    # Note that we actually apply the transform plane2cart to the vertices,
    # however, it is the transpose of cart2plane and to apply it to a batch
    # of vertices, we also have to transpose it, so both transposes cancel out.
    return vertices.dot(cart2plane) + plane_point


@numba.njit(
    numba.float64[:, :](numba.float64[:, ::1], numba.float64[:, ::1],
                        numba.float64[::1], numba.float64),
    cache=True)
def compute_contact_polygon(X1, X2, plane_normal, d):
    """Compute contact polygon.

    Parameters
    ----------
    X1 : array, shape (4, 4)
        Each row is a halfspace that defines the original tetrahedron.

    X2 : array, shape (4, 4)
        Each row is a halfspace that defines the original tetrahedron.

    plane_normal : array, shape (3,)
        Normal of the contact plane.

    d : float
        Distance to origin along normal.

    Returns
    -------
    polygon3d : array, shape (n_vertices, 3)
        Contact polygon between two tetrahedra. Points are ordered
        counter-clockwise around their center. No intersection is indicated
        by 0 vertices.
    """
    plane_point = plane_normal * d
    cart2plane = np.vstack(plane_basis_from_normal(plane_normal))
    X = np.vstack((X1, X2))
    halfplanes = make_halfplanes(X, plane_point, cart2plane)

    vertices2d = intersect_halfplanes(halfplanes)
    if len(vertices2d) < 3:
        return np.empty((0, 3), dtype=np.dtype("float"))

    # this approach sometimes results in duplicate points, remove them
    vertices2d = order_points(np.ascontiguousarray(vertices2d))
    unique_vertices2d = filter_unique_points(vertices2d)
    if len(unique_vertices2d) < 3:
        return np.empty((0, 3), dtype=np.dtype("float"))

    # from ._halfplanes import plot_halfplanes_and_intersections
    # plot_halfplanes_and_intersections(halfplanes, unique_vertices2d)

    vertices3d = project_polygon_to_3d(unique_vertices2d, cart2plane,
                                       plane_point)
    return vertices3d
