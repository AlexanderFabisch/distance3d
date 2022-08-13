import aabbtree
import numpy as np
from ._mesh_processing import tetrahedral_mesh_aabbs


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

    broad_pairs = list(zip(broad_tetrahedra1, broad_tetrahedra2))
    assert len(broad_tetrahedra1) == len(broad_tetrahedra2)
    for i, aabb in enumerate(aabbs1):
        if tree2.does_overlap(aabbtree.AABB(aabb)):
            assert i in broad_tetrahedra1
        else:
            assert i not in broad_tetrahedra1

    return broad_tetrahedra1, broad_tetrahedra2, broad_pairs
    """

    # FIXME workaround for broad phase bug:
    from itertools import product
    broad_tetrahedra1 = np.array(list(range(len(rigid_body1.tetrahedra_))), dtype=int)
    broad_tetrahedra2 = np.array(list(range(len(rigid_body2.tetrahedra_))), dtype=int)
    broad_pairs = list(product(broad_tetrahedra1, broad_tetrahedra2))
    return broad_tetrahedra1, broad_tetrahedra2, broad_pairs
