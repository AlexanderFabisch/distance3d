import numpy as np


def tetrahedral_mesh_aabbs(tetrahedra_points):
    """Compute axis-aligned bounding boxes of tetrahedra.

    Parameters
    ----------
    tetrahedra_points : array, shape (n_tetrahedra, 4, 3)
        Points that form the tetrahedra.

    Returns
    -------
    aabbs : array, shape (n_tetrahedra, 3, 2)
        Axis-aligned bounding boxes of tetrahedra.
    """
    mins = np.min(tetrahedra_points, axis=1)
    maxs = np.max(tetrahedra_points, axis=1)
    aabbs = np.dstack((mins, maxs))
    return aabbs


def center_of_mass_tetrahedral_mesh(tetrahedra_points):
    """Compute center of mass of a tetrahedral mesh.

    Assumes uniform density.

    Parameters
    ----------
    tetrahedra_points : array, shape (n_tetrahedra, 4, 3)
        Points that form the tetrahedra.

    Returns
    -------
    com : array, shape (3,)
        Center of mass.
    """
    volumes = _tetrahedral_mesh_volumes(tetrahedra_points)
    centers = tetrahedra_points.mean(axis=1)
    return np.dot(volumes, centers) / np.sum(volumes)


def _tetrahedral_mesh_volumes(tetrahedra_points):
    """Compute volumes of tetrahedra.

    Parameters
    ----------
    tetrahedra_points : array, shape (n_tetrahedra, 4, 3)
        Points that form the tetrahedra.

    Returns
    -------
    volumes : array, shape (n_tetrahedra,)
        Volumes of tetrahedra.
    """
    tetrahedra_edges = tetrahedra_points[:, 1:] - tetrahedra_points[:, np.newaxis, 0]
    return np.abs(np.sum(
        np.cross(tetrahedra_edges[:, 0], tetrahedra_edges[:, 1])
        * tetrahedra_edges[:, 2], axis=1)) / 6.0
