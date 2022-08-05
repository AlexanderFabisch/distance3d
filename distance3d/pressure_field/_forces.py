import numba
import numpy as np
from ..utils import adjoint_from_transform


def contact_surface_forces(contact_surface, rigid_body1):
    tetrahedra_potentials1 = rigid_body1.tetrahedra_potentials
    n_contacts = len(contact_surface.intersecting_tetrahedra1)
    contact_coms = np.empty((n_contacts, 3), dtype=float)
    contact_forces = np.empty((n_contacts, 3), dtype=float)
    contact_areas = np.empty(n_contacts, dtype=float)
    for intersection_idx in range(n_contacts):
        i = contact_surface.intersecting_tetrahedra1[intersection_idx]
        contact_plane_hnf = contact_surface.contact_planes[intersection_idx]
        contact_polygon = contact_surface.contact_polygons[intersection_idx]
        triangles = contact_surface.contact_polygon_triangles[intersection_idx]

        com, force, area = compute_contact_force(
            rigid_body1.tetrahedra_points[i], tetrahedra_potentials1[i],
            contact_plane_hnf, contact_polygon, triangles)

        contact_coms[intersection_idx] = com
        contact_forces[intersection_idx] = force
        contact_areas[intersection_idx] = area
    return contact_areas, contact_coms, contact_forces


@numba.njit(cache=True)
def compute_contact_force(
        tetrahedron, epsilon, contact_plane_hnf, contact_polygon, triangles):
    normal = contact_plane_hnf[:3]

    total_force = 0.0
    intersection_com = np.zeros(3)
    total_area = 0.0

    X = np.vstack((tetrahedron.T, np.ones((1, 4))))
    com = np.empty(4, dtype=np.dtype("float"))
    com[3] = 1.0
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

    intersection_com /= total_area
    force_vector = total_force * normal
    return intersection_com, force_vector, total_area


def accumulate_wrenches(contact_surface, rigid_body1, rigid_body2):
    total_force_21 = np.sum(contact_surface.contact_forces, axis=0)
    total_torque_21 = np.sum(np.cross(contact_surface.contact_coms - rigid_body1.com,
                                      contact_surface.contact_forces), axis=0)
    total_torque_12 = np.sum(np.cross(contact_surface.contact_coms - rigid_body2.com,
                                      -contact_surface.contact_forces), axis=0)
    wrench12_in_world, wrench21_in_world = _transform_wrenches(
        contact_surface.frame2world, total_force_21, total_torque_12, total_torque_21)
    return wrench12_in_world, wrench21_in_world


@numba.njit(cache=True)
def _transform_wrenches(mesh22origin, total_force_21, total_torque_12, total_torque_21):
    wrench21 = np.hstack((total_force_21, total_torque_21))
    wrench12 = np.hstack((-total_force_21, total_torque_12))
    mesh22origin_adjoint = adjoint_from_transform(mesh22origin)
    wrench21_in_world = mesh22origin_adjoint.T.dot(wrench21)
    wrench12_in_world = mesh22origin_adjoint.T.dot(wrench12)
    return wrench12_in_world, wrench21_in_world
