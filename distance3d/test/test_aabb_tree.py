import numpy as np
from distance3d import hydroelastic_contact
from distance3d.aabb_tree import AabbTree, all_aabbs_overlap


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

    aabb_tree = AabbTree()

    aabb_tree.insert_aabbs([])
    external_data = ["test"] * len(aabbs)
    aabb_tree.insert_aabbs(aabbs, external_data, pre_insertion_methode="shuffle")

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

    aabb_tree = AabbTree()
    external_data1 = ["1"] * len(aabbs)
    aabb_tree.insert_aabbs(aabbs, external_data1)
    aabb_tree2 = AabbTree()
    external_data2 = ["2"] * len(aabbs)
    aabb_tree2.insert_aabbs(aabbs2, external_data2)

    is_overlapping, overlap_tetrahedron1, overlap_tetrahedron2, overlap_pairs = aabb_tree.overlaps_aabb_tree(aabb_tree2)

    assert is_overlapping

    assert (overlap_tetrahedron1 == np.array([0, 1, 2])).all()

    assert (overlap_tetrahedron2 == np.array([0, 1, 2])).all()


def test_compare_aabb_tree_to_brute_force():
    rigid_body1 = hydroelastic_contact.RigidBody.make_sphere(0.13 * np.ones(3), 0.15, 2)
    rigid_body2 = hydroelastic_contact.RigidBody.make_sphere(0.25 * np.ones(3), 0.15, 2)
    rigid_body1.express_in(rigid_body2.body2origin_)

    aabbs1 = hydroelastic_contact.tetrahedral_mesh_aabbs(rigid_body1.tetrahedra_points)
    aabb_tree1 = AabbTree()
    aabb_tree1.insert_aabbs(aabbs1)

    aabbs2 = hydroelastic_contact.tetrahedral_mesh_aabbs(rigid_body2.tetrahedra_points)
    aabb_tree2 = AabbTree()
    aabb_tree2.insert_aabbs(aabbs2)

    broad_tetrahedra11, broad_tetrahedra12, broad_pairs1 = all_aabbs_overlap(aabbs1, aabbs2)

    _, broad_tetrahedra21, broad_tetrahedra22, broad_pairs2 = aabb_tree1.overlaps_aabb_tree(aabb_tree2)

    broad_tetrahedra11 = np.sort(np.unique(broad_tetrahedra11))
    broad_tetrahedra12 = np.sort(np.unique(broad_tetrahedra12))
    broad_tetrahedra21 = np.sort(np.unique(broad_tetrahedra21))
    broad_tetrahedra22 = np.sort(np.unique(broad_tetrahedra22))

    assert (broad_tetrahedra11 == broad_tetrahedra21).all()

    assert (broad_tetrahedra12 == broad_tetrahedra22).all()

    assert (np.sort(np.unique(broad_pairs1)) == np.sort(np.unique(broad_pairs2))).all()


def test_aabb_tree_overlap():
    aabb1 = np.array([[0, 10], [0, 10], [0, 10]], dtype=np.float64)
    aabb2 = np.array([[8, 12], [3, 4], [4, 5]], dtype=np.float64)
    aabb3 = np.array([[20, 30], [0, 1], [0, 2]], dtype=np.float64)
    aabb4 = np.array([[-50, 50], [-50, 50], [-50, 50]], dtype=np.float64)

    aabb_tree = AabbTree()
    aabb_tree.insert_aabb(aabb1, 1)
    aabb_tree.insert_aabb(aabb2, 2)
    aabb_tree.insert_aabb(aabb3, 3)

    overlap_flag_1, overlaps_1 = aabb_tree.overlaps_aabb(aabb1)
    external_data = []
    insert_data = []
    for overlap in overlaps_1:
        external_data.append(aabb_tree.external_data_list[overlap])
        insert_data.append(aabb_tree.insert_index_list[overlap])
    assert overlap_flag_1
    assert 1 in external_data
    assert 2 in external_data
    assert 0 in insert_data
    assert 1 in insert_data

    overlap_flag_4, overlaps_4 = aabb_tree.overlaps_aabb(aabb4)
    external_data = []
    insert_data = []
    for overlap in overlaps_4:
        external_data.append(aabb_tree.external_data_list[overlap])
        insert_data.append(aabb_tree.insert_index_list[overlap])
    assert overlap_flag_4
    assert 1 in external_data
    assert 2 in external_data
    assert 3 in external_data
    assert 0 in insert_data
    assert 1 in insert_data
    assert 2 in insert_data
