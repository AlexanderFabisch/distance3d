"""Visualization tools."""
import numpy as np
import pytransform3d.visualizer as pv
import open3d as o3d


class Mesh(pv.Artist):
    """A mesh.

    Parameters
    ----------
    mesh2origin : array, shape (4, 4)
        Pose of the mesh.

    vertices : array, shape (n_vertices, 3)
        Vertices of the mesh.

    triangles : array, shape (n_triangles, 3)
        Indices of vertices that form triangles of the mesh.

    c : array-like, shape (3,), optional (default: None)
        Color(s)
    """
    def __init__(self, mesh2origin, vertices, triangles, c=None):
        self.mesh = o3d.geometry.TriangleMesh()
        self.mesh.vertices = o3d.utility.Vector3dVector(vertices)
        self.mesh.triangles = o3d.utility.Vector3iVector(triangles)
        self.mesh.compute_vertex_normals()
        if c is not None:
            self.mesh.paint_uniform_color(c)
        self.mesh2origin = None
        self.set_data(mesh2origin)

    def set_data(self, mesh2origin):
        """Update data.

        Parameters
        ----------
        mesh2origin : array-like, shape (4, 4)
            Pose of the mesh.
        """
        previous_mesh2origin = self.mesh2origin
        if previous_mesh2origin is None:
            previous_mesh2origin = np.eye(4)
        self.mesh2origin = mesh2origin

        self.mesh.transform(np.linalg.inv(previous_mesh2origin))
        self.mesh.transform(self.mesh2origin)

    @property
    def geometries(self):
        """Expose geometries.

        Returns
        -------
        geometries : list
            List of geometries that can be added to the visualizer.
        """
        return [self.mesh]


class Ellipse(pv.Artist):
    """An ellipse.

    Parameters
    ----------
    center : array, shape (3,)
        Center of ellipse.

    axes : array, shape (2, 3)
        Axes of ellipse.

    radii : array, shape (2,)
        Radii of ellipse.

    c : array-like, shape (3,), optional (default: None)
        Color(s)
    """
    def __init__(self, center, axes, radii, c=None):
        self.mesh = o3d.geometry.TriangleMesh()
        ellipse2origin = np.eye(4)
        ellipse2origin[:3, :2] = axes.T
        ellipse2origin[:3, 2] = np.cross(axes[0], axes[1])
        ellipse2origin[:3, 3] = center
        ellipse = np.array([
            np.hstack(((radii * np.array([np.cos(angle), np.sin(angle)])), (0.0,)))
            for angle in np.linspace(0, 2 * np.pi, 20)])
        vertices = np.vstack((np.zeros((1, 3)), ellipse))
        triangles = np.vstack((
            np.array([[i, j, 0] for i, j in zip(range(1, len(vertices) - 1),
                                                range(2, len(vertices)))]),
            np.array([[0, j, i] for i, j in zip(range(1, len(vertices) - 1),
                                                range(2, len(vertices)))]),
        ))
        self.mesh.vertices = o3d.utility.Vector3dVector(vertices)
        self.mesh.triangles = o3d.utility.Vector3iVector(triangles)
        self.mesh.compute_vertex_normals()
        if c is not None:
            self.mesh.paint_uniform_color(c)
        self.ellipse2origin = None
        self.set_data(ellipse2origin)

    def set_data(self, ellipse2origin):
        """Update data.

        Parameters
        ----------
        ellipse2origin : array-like, shape (4, 4)
            Pose of the ellipse.
        """
        previous_ellipse2origin = self.ellipse2origin
        if previous_ellipse2origin is None:
            previous_ellipse2origin = np.eye(4)
        self.ellipse2origin = ellipse2origin

        self.mesh.transform(np.linalg.inv(previous_ellipse2origin))
        self.mesh.transform(self.ellipse2origin)

    @property
    def geometries(self):
        """Expose geometries.

        Returns
        -------
        geometries : list
            List of geometries that can be added to the visualizer.
        """
        return [self.mesh]
