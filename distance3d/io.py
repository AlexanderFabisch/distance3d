"""Features related to input and output from and to disk."""
import os
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


def load_tetrahedral_mesh(filename, scale=1.0):
    """Load tetrahedral mesh from file.

    Tetrahedral meshes are used mainly for simulation of deformable objects.

    Note that only the VTK format is currently supported
    (https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf). You can
    use TetWild (https://github.com/Yixin-Hu/TetWild) to convert triangular
    meshes to tetrahedral meshes.

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

    tetrahedra : array, shape (n_tetrahedra, 4)
        Indices of vertices that form tetrahedra of the mesh.
    """
    assert filename.endswith(".vtk")

    with open(filename, "r") as f:
        content = f.read()
    lines = content.split(os.linesep)
    lines = [line.strip() for line in lines]

    point_lines = None
    cells_lines = None
    for i, line in enumerate(lines):
        if line.startswith("POINTS"):
            n_points = int(line.split(" ")[1])
            points_start = i + 1
            points_end = points_start + n_points
            point_lines = lines[points_start:points_end]
        elif line.startswith("CELLS"):
            n_cells = int(line.split(" ")[1])
            cells_start = i + 1
            cells_end = cells_start + n_cells
            cells_lines = lines[cells_start:cells_end]
    assert point_lines is not None
    assert cells_lines is not None

    points = np.row_stack([np.fromstring(line, sep=" ", dtype=float)
                           for line in point_lines])
    cells = np.row_stack([np.fromstring(line, sep=" ", dtype=int)
                          for line in cells_lines])
    assert all(cells[:, 0] == 4)

    vertices = points * scale
    tetrahedra = cells[:, 1:]

    return vertices, tetrahedra
