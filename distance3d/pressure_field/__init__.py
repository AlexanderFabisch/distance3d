"""Pressure field model for contact wrenches."""
import aabbtree
import numpy as np
from ..utils import invert_transform
from ..benchmark import Timer
from ._tetra_mesh_creation import make_tetrahedral_cube, make_tetrahedral_icosphere
from ._mesh_processing import tetrahedral_mesh_aabbs
from ._rigid_body import RigidBody
from ._contact_surface import ContactSurface
from ._tetrahedron_intersection import intersect_tetrahedron_pairs
from ._forces import contact_surface_forces, accumulate_wrenches


def contact_forces(
        mesh12origin, vertices1_in_mesh1, tetrahedra1, potentials1,
        mesh22origin, vertices2_in_mesh2, tetrahedra2, potentials2,
        return_details=False, timer=None):
    if timer is None:
        timer = Timer()

    rigid_body1 = RigidBody(mesh12origin, vertices1_in_mesh1, tetrahedra1, potentials1)
    rigid_body2 = RigidBody(mesh22origin, vertices2_in_mesh2, tetrahedra2, potentials2)

    contact_surface = find_contact_plane(rigid_body1, rigid_body2, timer)

    timer.start("accumulate_wrenches")
    wrench12_in_world, wrench21_in_world = accumulate_wrenches(
        contact_surface, rigid_body1, rigid_body2)
    timer.stop_and_add_to_total("accumulate_wrenches")

    if return_details:
        timer.start("make_details")
        if contact_surface.intersection:
            details = contact_surface.make_details(
                rigid_body1.tetrahedra_points, rigid_body2.tetrahedra_points)
        else:
            details = {}
        timer.stop_and_add_to_total("make_details")
        return contact_surface.intersection, wrench12_in_world, wrench21_in_world, details
    else:
        return contact_surface.intersection, wrench12_in_world, wrench21_in_world


def find_contact_plane(rigid_body1, rigid_body2, timer=None):
    if timer is None:
        timer = Timer()

    timer.start("transformation")
    # We transform vertices of mesh1 to mesh2 frame to be able to reuse the
    # AABB tree of mesh2.
    mesh12mesh2 = np.dot(invert_transform(rigid_body2.mesh2origin),
                         rigid_body1.mesh2origin)
    rigid_body1.transform(mesh12mesh2)
    timer.stop_and_add_to_total("transformation")

    timer.start("broad_phase_tetrahedra")
    broad_tetrahedra1, broad_tetrahedra2, broad_pairs = broad_phase_tetrahedra(
        rigid_body1, rigid_body2)
    timer.stop_and_add_to_total("broad_phase_tetrahedra")

    timer.start("barycentric_transform")
    unique_indices1 = np.unique(broad_tetrahedra1)
    unique_indices2 = np.unique(broad_tetrahedra2)
    X1 = barycentric_transforms(rigid_body1.tetrahedra_points[unique_indices1])
    X2 = barycentric_transforms(rigid_body2.tetrahedra_points[unique_indices2])
    X1 = {j: X1[i] for i, j in enumerate(unique_indices1)}
    X2 = {j: X2[i] for i, j in enumerate(unique_indices2)}
    timer.stop_and_add_to_total("barycentric_transform")

    timer.start("intersect_pairs")
    intersection_result = intersect_tetrahedron_pairs(
        broad_pairs, rigid_body1, rigid_body2, X1, X2)
    contact_surface = ContactSurface(rigid_body2.mesh2origin, *intersection_result)
    timer.stop_and_add_to_total("intersect_pairs")

    timer.start("contact_surface_forces")
    contact_areas, contact_coms, contact_forces = contact_surface_forces(
        contact_surface, rigid_body1)
    contact_surface.add_polygon_info(
        contact_areas, contact_coms, contact_forces)
    timer.stop_and_add_to_total("contact_surface_forces")

    return contact_surface


def broad_phase_tetrahedra(rigid_body1, rigid_body2):
    """Broad phase collision detection of tetrahedra."""

    # TODO fix broad phase for cube vs. sphere
    # TODO speed up broad phase
    # TODO store result in RigidBody
    """
    aabbs1 = tetrahedral_mesh_aabbs(rigid_body1.tetrahedra_points)
    aabbs2 = tetrahedral_mesh_aabbs(rigid_body2.tetrahedra_points)
    tree2 = aabbtree.AABBTree()
    for j, aabb in enumerate(aabbs2):
        tree2.add(aabbtree.AABB(aabb), j)
    broad_tetrahedra1 = []
    broad_tetrahedra2 = []
    for i, aabb in enumerate(aabbs1):
        new_indices2 = tree2.overlap_values(aabbtree.AABB(aabb))
        broad_tetrahedra2.extend(new_indices2)
        broad_tetrahedra1.extend([i] * len(new_indices2))
    broad_pairs = zip(broad_tetrahedra1, broad_tetrahedra2)
    """

    # FIXME workaround for broad phase bug:
    from itertools import product
    broad_tetrahedra1 = np.array(list(range(len(rigid_body1.tetrahedra))), dtype=int)
    broad_tetrahedra2 = np.array(list(range(len(rigid_body2.tetrahedra))), dtype=int)
    broad_pairs = list(product(broad_tetrahedra1, broad_tetrahedra2))
    return broad_tetrahedra1, broad_tetrahedra2, broad_pairs


def barycentric_transforms(tetrahedra_points):
    """Returns X. X.dot(coords) = (r, 1), where r is a Cartesian vector."""
    # NOTE that in the original paper it is not obvious that we have to take
    # the inverse
    return np.linalg.pinv(np.hstack((tetrahedra_points.transpose((0, 2, 1)),
                                     np.ones((len(tetrahedra_points), 1, 4)))))


# TODO remove or reuse function
def make_halfplanes2(tetrahedron, cart2plane, plane2cart_offset):  # TODO can we fix this?
    halfplanes = []
    X = barycentric_transform(tetrahedron)
    for i in range(4):
        halfspace = X[i]
        normal2d = cart2plane.dot(halfspace[:3])
        norm = np.linalg.norm(normal2d)
        if norm > 1e-9:
            p = normal2d * (-halfspace[3] - halfspace[:3].dot(plane2cart_offset)) / np.dot(normal2d, normal2d)
            halfplanes.append((p, normal2d))
    return halfplanes


# TODO remove or reuse function
def intersect_halfplanes2(halfplanes):  # TODO can we modify this to work with parallel lines?
    from collections import deque
    dq = deque()
    for hp in halfplanes:
        while len(dq) >= 2 and hp.outside_of(dq[-1].intersect(dq[-2])):
            dq.pop()
        while len(dq) >= 2 and hp.outside_of(dq[0].intersect(dq[1])):
            dq.popleft()
        dq.append(hp)

    while len(dq) >= 3 and dq[0].outside_of(dq[-1].intersect(dq[-2])):
        dq.pop()
    while len(dq) >= 3 and dq[-1].outside_of(dq[0].intersect(dq[1])):
        dq.popleft()

    if len(dq) < 3:
        return None, []
    else:
        polygon = np.row_stack([dq[i].intersect(dq[(i + 1) % len(dq)])
                                for i in range(len(dq))])
        return polygon, list(dq)
