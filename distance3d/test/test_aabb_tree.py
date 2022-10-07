import numpy as np
from distance3d import visualization, hydroelastic_contact


def test_aabb_tree_creation():
    aabbs = np.array([[[0, 2], [0, 2], [0, 1]],
                      [[1, 3], [1, 3], [0, 1]],
                      [[3, 4], [0, 1], [0, 1]]])

    aabbs2 = np.array([[[1.5, 3.5], [0, 0.5], [0, 1]],
                       [[1.5, 2], [2.5, 4], [0, 1]],
                       [[3.5, 4.5], [0, 0.5], [0, 1]]])

    aabb_tree = hydroelastic_contact.AabbTree(aabbs)

    print(aabb_tree)

    _, overlaps = aabb_tree.overlaps_aabb(aabbs2[0])
    overlaps = np.sort(overlaps)
    assert overlaps[0] == 0 and overlaps[1] == 2

    _, overlaps = aabb_tree.overlaps_aabb(aabbs2[1])
    overlaps = np.sort(overlaps)
    assert overlaps[0] == 1

    _, overlaps = aabb_tree.overlaps_aabb(aabbs2[2])
    overlaps = np.sort(overlaps)
    assert overlaps[0] == 2

    aabb_tree2 = hydroelastic_contact.AabbTree(aabbs)

    aabb_tree.overlaps_aabb_tree(aabb_tree2)


def test_aabb_tree_compare_to_brute_force():

    rigid_body1 = hydroelastic_contact.RigidBody.make_sphere(0.13 * np.ones(3), 0.15, 2)
    rigid_body2 = hydroelastic_contact.RigidBody.make_sphere(0.25 * np.ones(3), 0.15, 2)
    rigid_body1.express_in(rigid_body2.body2origin_)

    aabbs1 = hydroelastic_contact.tetrahedral_mesh_aabbs(rigid_body1.tetrahedra_points)
    aabb_tree1 = hydroelastic_contact.AabbTree(aabbs1)

    aabbs2 = hydroelastic_contact.tetrahedral_mesh_aabbs(rigid_body1.tetrahedra_points)
    aabb_tree2 = hydroelastic_contact.AabbTree(aabbs2)

    broad_tetrahedra1, _, _ = hydroelastic_contact._all_aabbs_overlap(aabbs1, aabbs2)

    _, broad_tetrahedra2, _, _ = aabb_tree1.overlaps_aabb_tree(aabb_tree2)

    assert (np.sort(np.unique(broad_tetrahedra1)) == np.sort(np.unique(broad_tetrahedra2))).all()


