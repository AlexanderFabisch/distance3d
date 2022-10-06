import numpy as np
from distance3d import hydroelastic_contact


def test_aabb_tree_creation():
    aabbs = np.array([[[0, 2], [0, 2], [0, 1]],
                     [[1, 3], [1, 3], [0, 1]],
                     [[3, 4], [0, 1], [0, 1]]])

    aabbs2 = np.array([[[1.5, 3.5], [0, 0.5], [0, 1]],
                      [[1.5, 2], [2.5, 4], [0, 1]],
                      [[3.5, 4.5], [0, 0.5], [0, 1]]])

    root, nodes, aabbs = hydroelastic_contact.new_tree_from_aabbs(aabbs)

    print("")
    hydroelastic_contact.print_aabb_tree(root, nodes)

    overlaps = hydroelastic_contact.query_overlap(np.array(aabbs2[0]), root, nodes, aabbs)
    overlaps = np.sort(overlaps)
    assert overlaps[0] == 0 and overlaps[1] == 2

    overlaps = hydroelastic_contact.query_overlap(np.array(aabbs2[1]), root, nodes, aabbs)
    overlaps = np.sort(overlaps)
    assert overlaps[0] == 1

    overlaps = hydroelastic_contact.query_overlap(np.array(aabbs2[2]), root, nodes, aabbs)
    overlaps = np.sort(overlaps)
    assert overlaps[0] == 2

    root2, nodes2, aabbs2 = hydroelastic_contact.new_tree_from_aabbs(aabbs2)

    print("")
    hydroelastic_contact.print_aabb_tree(root2, nodes2)

    hydroelastic_contact.query_overlap_of_other_tree(root, nodes, aabbs, root2, nodes2, aabbs2)
    







