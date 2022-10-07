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
