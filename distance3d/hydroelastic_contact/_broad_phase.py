import numpy as np
import numba
from ._mesh_processing import tetrahedral_mesh_aabbs
from ._aabb_tree import AabbTree


def broad_phase_tetrahedra(rigid_body1, rigid_body2, use_aabb_trees=False):
    """Broad phase collision detection of tetrahedra.

    Parameters
    ----------
    rigid_body1 : RigidBody
        First rigid body.

    rigid_body2 : RigidBody
        Second rigid body.

    use_aabb_trees : bool, optional (default: False)
        Option to specify the usage of a AABB trees.
        Currently slower than a standard brute search.

    Returns
    -------
    broad_tetrahedra1 : array, shape (n_overlaps,)
        Indices of tetrahedra in first rigid body that overlap with
        second rigid body.

    broad_tetrahedra2 : array, shape (n_overlaps,)
        Indices of tetrahedra in second rigid body that overlap with
        first rigid body.

    broad_pairs : list
        of intersecting tetrahedron pairs.
    """

    aabbs1 = tetrahedral_mesh_aabbs(rigid_body1.tetrahedra_points)
    aabbs2 = tetrahedral_mesh_aabbs(rigid_body2.tetrahedra_points)

    # TODO speed up broad phase with numba
    # TODO store result in RigidBody

    if use_aabb_trees:

        aabb_tree1 = AabbTree(aabbs1)
        aabb_tree2 = AabbTree(aabbs2)

        _, broad_tetrahedra1, broad_tetrahedra2, broad_pairs = aabb_tree1.overlaps_aabb_tree(aabb_tree2)

        return broad_tetrahedra1, broad_tetrahedra2, broad_pairs
    else:
        return _all_aabbs_overlap(aabbs1, aabbs2)


@numba.njit(cache=True)
def _all_aabbs_overlap(aabbs1, aabbs2):
    indices1 = []
    indices2 = []
    broad_pairs = []
    for i in range(len(aabbs1)):
        for j in range(len(aabbs2)):
            if _aabbs_overlap_no_loop(aabbs1[i], aabbs2[j]):
                indices1.append(i)
                indices2.append(j)
                broad_pairs.append((i, j))
    broad_tetrahedra1 = np.array(indices1, dtype=np.dtype("int"))
    broad_tetrahedra2 = np.array(indices2, dtype=np.dtype("int"))
    return broad_tetrahedra1, broad_tetrahedra2, broad_pairs


@numba.njit(cache=True)
def _aabbs_overlap_no_loop(aabb1, aabb2):
    return aabb1[0, 0] < aabb2[0, 1] and aabb2[0, 0] < aabb1[0, 1] \
           and aabb1[1, 0] < aabb2[1, 1] and aabb2[1, 0] < aabb1[1, 1] \
           and aabb1[2, 0] < aabb2[2, 1] and aabb2[2, 0] < aabb1[2, 1]
