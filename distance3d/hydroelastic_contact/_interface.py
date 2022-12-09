from ._forces import contact_surface_forces, accumulate_wrenches
from ._barycentric_transform import barycentric_transforms
from ._tetrahedron_intersection import intersect_tetrahedron_pairs
from ._contact_surface import ContactSurface
from ..aabb_tree import all_aabbs_overlap


def contact_forces(rigid_body1, rigid_body2, return_details=False):
    """Contact forces between two objects.

    Parameters
    ----------
    rigid_body1 : RigidBody
        First rigid body.

    rigid_body2 : RigidBody
        Second rigid body.

    return_details : bool
        Return additional contact details.

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
    contact_surface = find_contact_surface(rigid_body1, rigid_body2)

    wrench12_in_world, wrench21_in_world = accumulate_wrenches(
        contact_surface, rigid_body1, rigid_body2)

    if return_details:
        details = contact_surface.make_details(
            rigid_body1.tetrahedra_points, rigid_body2.tetrahedra_points)
        return (
            contact_surface.intersection, wrench12_in_world, wrench21_in_world,
            details)
    else:
        return (
            contact_surface.intersection, wrench12_in_world, wrench21_in_world)


def find_contact_surface(rigid_body1, rigid_body2, use_aabb_trees=False):
    """Find contact plane of two rigid bodies.

    Note that this function will transform rigid_body1 into the frame of
    rigid_body2.

    Parameters
    ----------
    rigid_body1 : RigidBody
        First rigid body.

    rigid_body2 : RigidBody
        Second rigid body.

    use_aabb_trees : bool, optional (default: False)
        Option to specify the usage of aabb_trees.

    Returns
    -------
    contact_surface : ContactSurface
        Contact information.
    """
    # We transform vertices of rigid_body1 to rigid_body2 frame to be able to
    # reuse the AABB tree of rigid_body2.
    rigid_body1.express_in(rigid_body2.body2origin_)

    if use_aabb_trees:
        _, broad_tetrahedra1, broad_tetrahedra2, broad_pairs \
            = rigid_body1.aabbtree_.overlaps_aabb_tree(rigid_body2.aabbtree_)
    else:
        broad_tetrahedra1, broad_tetrahedra2, broad_pairs = all_aabbs_overlap(rigid_body1.aabbs, rigid_body2.aabbs)

    X1 = barycentric_transforms(rigid_body1.tetrahedra_points[broad_tetrahedra1])
    X2 = barycentric_transforms(rigid_body2.tetrahedra_points[broad_tetrahedra2])
    X1 = {j: X1[i] for i, j in enumerate(broad_tetrahedra1)}
    X2 = {j: X2[i] for i, j in enumerate(broad_tetrahedra2)}

    intersection_result = intersect_tetrahedron_pairs(
        broad_pairs,
        rigid_body1.tetrahedra_points, rigid_body2.tetrahedra_points,
        rigid_body1.tetrahedra_potentials, rigid_body2.tetrahedra_potentials,
        X1, X2, rigid_body1.youngs_modulus, rigid_body2.youngs_modulus)
    contact_surface = ContactSurface(
        rigid_body2.body2origin_, *intersection_result)

    areas, coms, forces, triangles = contact_surface_forces(
        contact_surface, rigid_body1)
    contact_surface.add_polygon_info(areas, coms, forces, triangles)

    return contact_surface
