"""Features related to input and output from and to disk."""
import numpy as np


def load_mesh(filename, scale=1.0):
    """Load mesh from file.

    This functions attempts to load the mesh with Open3D first. If this fails,
    either because the library is not available or because the mesh format
    is not supported, it tries to use trimesh to load the file.

    Parameters
    ----------
    filename : str
        Path to mesh file.

    scale : float, optional (default: 1)
        Scale of vertex coordinates.

    Returns
    -------
    vertices : array, shape (n_vertices, 3)
        Vertices of the mesh.

    triangles : array, shape (n_triangles, 3)
        Indices of vertices that form triangles of the mesh.
    """
    try:
        import open3d as o3d
        mesh = o3d.io.read_triangle_mesh(filename)
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        try_trimesh = False
    except OSError:
        try_trimesh = True
    except ImportError:
        try_trimesh = True

    if try_trimesh:
        import trimesh
        mesh = trimesh.load(filename)
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.faces)

    vertices *= scale
    return vertices, triangles
