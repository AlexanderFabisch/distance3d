import numpy as np
import numba


@numba.njit(cache=True)
def _all_aabbs_overlap(aabbs1, aabbs2):
    indices1 = []
    indices2 = []
    broad_pairs = []
    for i in range(len(aabbs1)):
        for j in range(len(aabbs2)):
            if _aabb_overlap(aabbs1[i], aabbs2[j]):
                indices1.append(i)
                indices2.append(j)
                broad_pairs.append((i, j))
    broad_tetrahedra1 = np.unique(np.array(indices1, dtype=np.dtype("int")))
    broad_tetrahedra2 = np.unique(np.array(indices2, dtype=np.dtype("int")))
    return broad_tetrahedra1, broad_tetrahedra2, broad_pairs


@numba.njit(cache=True)
def _aabb_overlap(aabb1, aabb2):
    return aabb1[0, 0] <= aabb2[0, 1] and aabb1[0, 1] >= aabb2[0, 0] \
           and aabb1[1, 0] <= aabb2[1, 1] and aabb1[1, 1] >= aabb2[1, 0] \
           and aabb1[2, 0] <= aabb2[2, 1] and aabb1[2, 1] >= aabb2[2, 0]


@numba.njit(cache=True)
def _sort_aabbs(aabbs):
    return aabbs[:, 0, 0].argsort()


@numba.njit(cache=True)
def _merge_aabb(aabb1, aabb2):
    return np.array(
        [[min(aabb1[0, 0], aabb2[0, 0]), max(aabb1[0, 1], aabb2[0, 1])],
         [min(aabb1[1, 0], aabb2[1, 0]), max(aabb1[1, 1], aabb2[1, 1])],
         [min(aabb1[0, 0], aabb2[2, 0]), max(aabb1[2, 1], aabb2[2, 1])]]
    )


@numba.njit(cache=True)
def _aabb_surface(aabb):
    return _aabb_x_size(aabb) * _aabb_y_size(aabb) * _aabb_z_size(aabb)


@numba.njit(cache=True)
def _aabb_x_size(aabb):
    return aabb[0, 1] - aabb[0, 0]


@numba.njit(cache=True)
def _aabb_y_size(aabb):
    return aabb[1, 1] - aabb[1, 0]


@numba.njit(cache=True)
def _aabb_z_size(aabb):
    return aabb[2, 1] - aabb[2, 0]
