import numpy as np
import numba


@numba.njit(cache=True)
def _all_aabbs_overlap(aabbs1, aabbs2):
    """Creates result lists of all the overlapping aabbs.

    Parameters
    ----------
    aabbs1 : array, shape(n, 3, 2)
        The aabbs of the first object.

    aabbs2 : array, shape(n, 3, 2)
        The aabbs of the second object.

    Returns
    -------
    broad_tetrahedra1 : array, shape (n)
        Array of all the overlapping aabb indices of aabbs1

    broad_tetrahedra2 : array, shape (n)
        Array of all the overlapping aabb indices of aabbs2

    broad_pairs : array, shape(n, 2)
        A list of a index pairs of the all overlaps.
    """

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
    """Returns true if aabb1 and aabb2 overlap."""
    return aabb1[0, 0] <= aabb2[0, 1] and aabb1[0, 1] >= aabb2[0, 0] \
           and aabb1[1, 0] <= aabb2[1, 1] and aabb1[1, 1] >= aabb2[1, 0] \
           and aabb1[2, 0] <= aabb2[2, 1] and aabb1[2, 1] >= aabb2[2, 0]


@numba.njit(cache=True)
def _sort_aabbs(aabbs):
    """Returns a spatially sorted aabb list."""
    return aabbs[:, 0, 0].argsort()


@numba.njit(cache=True)
def _merge_aabb(aabb1, aabb2):
    """Returns the smallest aabb that contains aabb1 and aabb2."""
    return np.array(
        [[min(aabb1[0, 0], aabb2[0, 0]), max(aabb1[0, 1], aabb2[0, 1])],
         [min(aabb1[1, 0], aabb2[1, 0]), max(aabb1[1, 1], aabb2[1, 1])],
         [min(aabb1[2, 0], aabb2[2, 0]), max(aabb1[2, 1], aabb2[2, 1])]]
    )


@numba.njit(cache=True)
def _aabb_volume(aabb):
    """Returns the volume of the aabb."""
    return _aabb_x_size(aabb) * _aabb_y_size(aabb) * _aabb_z_size(aabb)


@numba.njit(cache=True)
def _aabb_x_size(aabb):
    """Returns the size of the aabb along the x-axsis."""
    return aabb[0, 1] - aabb[0, 0]


@numba.njit(cache=True)
def _aabb_y_size(aabb):
    """Returns the size of the aabb along the y-axsis."""
    return aabb[1, 1] - aabb[1, 0]


@numba.njit(cache=True)
def _aabb_z_size(aabb):
    """Returns the size of the aabb along the Z-axsis."""
    return aabb[2, 1] - aabb[2, 0]
