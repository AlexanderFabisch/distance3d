import numpy as np
from ..utils import invert_transform
from ..benchmark import Timer
from ._forces import contact_surface_forces, accumulate_wrenches
from ._broad_phase import broad_phase_tetrahedra
from ._barycentric_transform import barycentric_transforms
from ._tetrahedron_intersection import intersect_tetrahedron_pairs
from ._contact_surface import ContactSurface


def contact_forces(rigid_body1, rigid_body2, return_details=False, timer=None):
    """Contact forces between two objects.

    Parameters
    ----------
    rigid_body1 : RigidBody
        First rigid body.

    rigid_body2 : RigidBody
        Second rigid body.

    return_details : bool
        Return additional contact details.

    timer : Timer, optional (default: None)
        Measures time to execute certain parts of this function.

    Returns
    -------
    intersection : bool
        Do both bodies intersect?

    wrench12_in_world : array, shape (6,)
        Forces and torques caused by body 1 acting on body 2 in world frame.

    wrench21_in_world : array, shape (6,)
        Forces and torques caused by body 2 acting on body 1 in world frame.

    details : dict, optional
        Additional contact details.
    """
    if timer is None:
        timer = Timer()

    contact_surface = find_contact_surface(rigid_body1, rigid_body2, timer)

    timer.start("accumulate_wrenches")
    wrench12_in_world, wrench21_in_world = accumulate_wrenches(
        contact_surface, rigid_body1, rigid_body2)
    timer.stop_and_add_to_total("accumulate_wrenches")

    if return_details:
        timer.start("make_details")
        details = contact_surface.make_details(
            rigid_body1.tetrahedra_points, rigid_body2.tetrahedra_points)
        timer.stop_and_add_to_total("make_details")
        return contact_surface.intersection, wrench12_in_world, wrench21_in_world, details
    else:
        return contact_surface.intersection, wrench12_in_world, wrench21_in_world


def find_contact_surface(rigid_body1, rigid_body2, timer=None):
    """Find contact plane of two rigid bodies.

    Note that this function will transform rigid_body1 into the frame of
    rigid_body2.

    Parameters
    ----------
    rigid_body1 : RigidBody
        First rigid body.

    rigid_body2 : RigidBody
        Second rigid body.

    timer : Timer, optional (default: None)
        Measures time to execute certain parts of this function.

    Returns
    -------
    contact_surface : ContactSurface
        Contact information.
    """
    if timer is None:
        timer = Timer()

    # We transform vertices of rigid_body1 to rigid_body2 frame to be able to
    # reuse the AABB tree of rigid_body2.
    timer.start("transformation")
    rigid_body1.express_in(rigid_body2.body2origin_)
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

    timer.start("intersect_tetrahedron_pairs")
    intersection_result = intersect_tetrahedron_pairs(
        broad_pairs, rigid_body1.tetrahedra_points, rigid_body2.tetrahedra_points,
        rigid_body1.tetrahedra_potentials, rigid_body2.tetrahedra_potentials,
        X1, X2)
    contact_surface = ContactSurface(rigid_body2.body2origin_, *intersection_result)
    timer.stop_and_add_to_total("intersect_tetrahedron_pairs")

    timer.start("contact_surface_forces")
    contact_areas, contact_coms, contact_forces, contact_triangles = contact_surface_forces(
        contact_surface, rigid_body1)
    contact_surface.add_polygon_info(
        contact_areas, contact_coms, contact_forces, contact_triangles)
    timer.stop_and_add_to_total("contact_surface_forces")

    return contact_surface
