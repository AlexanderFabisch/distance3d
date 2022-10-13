import numpy as np
from distance3d import hydroelastic_contact


def create_test_aabbs():
    aabbs = np.array([[[0, 2], [0, 2], [0, 1]],
                      [[1, 3], [1, 3], [0, 1]],
                      [[3, 4], [0, 1], [0, 1]]])

    aabbs2 = np.array([[[1.5, 3.5], [0, 0.5], [0, 1]],
                       [[1.5, 2], [2.5, 4], [0, 1]],
                       [[3.5, 4.5], [0, 0.5], [0, 1]]])
    return aabbs, aabbs2


def test_aabb_tree_creation():
    aabbs, aabbs2 = create_test_aabbs()

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


def test_aabb_tree_overlap_with_other_aabb_tree():
    aabbs, aabbs2 = create_test_aabbs()

    aabb_tree = hydroelastic_contact.AabbTree(aabbs)
    aabb_tree2 = hydroelastic_contact.AabbTree(aabbs2)

    is_overlapping, overlap_tetrahedron1, overlap_tetrahedron2, overlap_pairs = aabb_tree.overlaps_aabb_tree(aabb_tree2)

    assert is_overlapping

    assert (overlap_tetrahedron1 == np.array([0, 1, 2])).all()

    assert (overlap_tetrahedron2 == np.array([0, 1, 2])).all()



def test_compare_aabb_tree_to_brute_force():
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
