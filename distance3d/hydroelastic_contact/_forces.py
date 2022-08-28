import numba
import numpy as np
from ..utils import adjoint_from_transform


def contact_surface_forces(contact_surface, rigid_body1):
    tetrahedra_potentials1 = rigid_body1.tetrahedra_potentials
    n_intersections = len(contact_surface.intersecting_tetrahedra1)
    coms = np.empty((n_intersections, 3), dtype=float)
    forces = np.empty((n_intersections, 3), dtype=float)
    areas = np.empty(n_intersections, dtype=float)
    triangles = []
    for intersection_idx, tetrahedron_idx in enumerate(
            contact_surface.intersecting_tetrahedra1):
        coms[intersection_idx], forces[intersection_idx], \
        areas[intersection_idx], triangle = compute_contact_force(
            rigid_body1.tetrahedra_points[tetrahedron_idx],
            tetrahedra_potentials1[tetrahedron_idx],
            contact_surface.contact_planes[intersection_idx],
            contact_surface.contact_polygons[intersection_idx])

        triangles.append(triangle)
    return areas, coms, forces, triangles


@numba.njit(cache=True)
def tesselate_ordered_polygon(n_vertices):
    """Tesselate a ccw-ordered polygon.

    Parameters
    ----------
    n_vertices : int
        Number of vertices of the polygon.

    Returns
    -------
    triangles : array, shape (n_vertices - 2, 3)
        Triangles forming the polygon.
    """
    triangles = np.empty((n_vertices - 2, 3), dtype=np.dtype("int"))
    triangles[:, 0] = 0
    triangles[:, 1] = np.arange(1, n_vertices - 1)
    triangles[:, 2] = np.arange(2, n_vertices)
    return triangles


# 8 halfplanes cannot define a polygon with more than 8 vertices
TRIANGLES = tesselate_ordered_polygon(8)


@numba.njit(cache=True)
def compute_contact_force(
        tetrahedron, epsilon, contact_plane_hnf, contact_polygon):
    """Compute contact force from contact polygon.

    Parameters
    ----------
    tetrahedron : array, shape (4, 3)
        Vertices of tetrahedron

    epsilon : array, shape (4,)
        Potentials of vertices of tetrahedron.

    contact_plane_hnf : array, shape (4,)
        Contact plane in Hesse normal form.

    contact_polygon : array, shape (n_vertices, 3)
        Contact polygon between two tetrahedra. Points are ordered
        counter-clockwise around their center.

    Returns
    -------
    intersection_com : array, shape (3,)
        Center of the contact polygon.

    force_vector : array, shape (3,)
        Force vector of this contact.

    total_area : float
        Area of contact polygon.

    triangles : array, shape (n_triangles, 3)
        Vertex indices of triangles.
    """
    total_force = 0.0
    total_area = 0.0
    intersection_com = np.zeros(3)

    X = np.vstack((tetrahedron.T, np.ones((1, 4))))
    com = np.empty(4, dtype=np.dtype("float"))
    com[3] = 1.0
    triangles = TRIANGLES[:len(contact_polygon) - 2]
    for triangle in triangles:
        vertices = contact_polygon[triangle]
        com[:3] = (vertices[0] + vertices[1] + vertices[2]) / 3.0
        res = np.linalg.solve(X, com)
        pressure = np.sum(res * epsilon)
        area = 0.5 * np.linalg.norm(np.cross(vertices[1] - vertices[0],
                                             vertices[2] - vertices[0]))
        total_force += pressure * area
        total_area += area
        intersection_com += area * com[:3]

    if total_area > 0.0:
        intersection_com /= total_area
    else:
        intersection_com = contact_polygon[0]

    force_vector = total_force * contact_plane_hnf[:3]
    return intersection_com, force_vector, total_area, triangles


def accumulate_wrenches(contact_surface, rigid_body1, rigid_body2):
    total_force_21 = np.sum(contact_surface.contact_forces, axis=0)
    total_torque_21 = np.sum(
        np.cross(contact_surface.contact_coms - rigid_body1.com,
                 contact_surface.contact_forces), axis=0)
    total_torque_12 = np.sum(
        np.cross(contact_surface.contact_coms - rigid_body2.com,
                 -contact_surface.contact_forces), axis=0)
    wrench12_in_world, wrench21_in_world = _transform_wrenches(
        contact_surface.frame2world, total_force_21, total_torque_12,
        total_torque_21)
    return wrench12_in_world, wrench21_in_world


@numba.njit(cache=True)
def _transform_wrenches(
        mesh22origin, total_force_21, total_torque_12, total_torque_21):
    wrench21 = np.hstack((total_force_21, total_torque_21))
    wrench12 = np.hstack((-total_force_21, total_torque_12))
    mesh22origin_adjoint = adjoint_from_transform(mesh22origin)
    wrench21_in_world = mesh22origin_adjoint.T.dot(wrench21)
    wrench12_in_world = mesh22origin_adjoint.T.dot(wrench12)
    return wrench12_in_world, wrench21_in_world
