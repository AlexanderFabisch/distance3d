"""Visualization tools."""
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.visualizer as pv
import open3d as o3d


class Mesh(pv.Artist):
    """A triangular mesh.

    Parameters
    ----------
    mesh2origin : array, shape (4, 4)
        Pose of the mesh.

    vertices : array, shape (n_vertices, 3)
        Vertices of the mesh.

    triangles : array, shape (n_triangles, 3)
        Indices of vertices that form triangles of the mesh.

    c : array-like, shape (3,), optional (default: None)
        Color
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


class TetraMesh(pv.Artist):
    """A tetrahedral mesh.

    Parameters
    ----------
    mesh2origin : array, shape (4, 4)
        Pose of the mesh.

    vertices : array, shape (n_vertices, 3)
        Vertices of the mesh.

    tetrahedra : array, shape (n_tetrahedra, 4)
        Indices of vertices that form tetrahedra of the mesh.

    c : array-like, shape (3,), optional (default: None)
        Color
    """
    def __init__(self, mesh2origin, vertices, tetrahedra, c=None):
        self.mesh = o3d.geometry.TetraMesh(
            o3d.utility.Vector3dVector(vertices),
            o3d.utility.Vector4iVector(tetrahedra))
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


class Tetrahedron(Mesh):  # pragma: no cover
    """Tetrahedron.

    Parameters
    ----------
    tetrahedron_points : array, shape (4, 3)
        Points of the tetrahedron.

    c : array-like, shape (3,), optional (default: None)
        Color
    """
    def __init__(self, tetrahedron_points, c=None):
        mesh2origin = np.eye(4)
        triangles = np.array([[0, 1, 2], [1, 3, 2], [3, 0, 2], [0, 3, 1]], dtype=int)
        super(Tetrahedron, self).__init__(mesh2origin, tetrahedron_points, triangles, c)


class Ellipse(pv.Artist):  # pragma: no cover
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


class ContactSurface(pv.Artist):  # pragma: no cover
    """A pressure field.

    Parameters
    ----------
    mesh2origin : array, shape (4, 4)
        Pose of the mesh.

    contact_vertices : list
        Each entry is an array of vertices of a contact polygon.

    contact_triangles : list
        Each entry is an array of trangles of a contact polygon.

    pressures : array, shape (n_contact_polygons)
        Pressure values for contact polygons.
    """
    def __init__(self, mesh2origin, contact_vertices, contact_triangles, pressures):
        self.cmap = plt.get_cmap("plasma")
        self.mesh = o3d.geometry.TriangleMesh()
        self.set_data(mesh2origin, contact_vertices, contact_triangles, pressures)

    def set_data(self, mesh2origin, contact_vertices, contact_triangles, pressures):
        """Update contact surface.

        Parameters
        ----------
        mesh2origin : array, shape (4, 4)
            Pose of the mesh.

        contact_vertices : list
            Each entry is an array of vertices of a contact polygon.

        contact_triangles : list
            Each entry is an array of trangles of a contact polygon.

        pressures : array, shape (n_contact_polygons)
            Pressure values for contact polygons.
        """
        if len(contact_vertices) > 0:
            vertices = np.vstack(contact_vertices)
            triangles = []
            n_vertices = 0
            pressures_per_face = []
            for i in range(len(contact_triangles)):
                triangles.extend(contact_triangles[i] + n_vertices)
                n_vertices_i = len(contact_vertices[i])
                n_vertices += n_vertices_i
                pressures_per_face.extend([pressures[i]] * n_vertices_i)
            triangles = np.vstack(triangles)
            triangles = np.vstack((triangles, triangles[:, ::-1]))
            max_pressure = max(pressures)
            if max_pressure <= 0.0:
                max_pressure = 1.0
            colors = np.vstack([self.cmap(pressure_per_face / max_pressure)[:3]
                                for pressure_per_face in pressures_per_face])
        else:
            vertices = np.empty((0, 3), dtype=float)
            triangles = np.empty((0, 4), dtype=int)
            colors = np.empty((0, 3), dtype=float)
        self.mesh.vertices = o3d.utility.Vector3dVector(vertices)
        self.mesh.triangles = o3d.utility.Vector3iVector(triangles)
        self.mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        self.mesh.transform(mesh2origin)

    @property
    def geometries(self):
        """Expose geometries.

        Returns
        -------
        geometries : list
            List of geometries that can be added to the visualizer.
        """
        return [self.mesh]


class RigidBodyTetrahedralMesh(pv.Artist):  # pragma: no cover
    """Rigid body represented by a tetrahedral mesh.

    Parameters
    ----------
    body2origin : array, shape (4, 4)
        Pose of the mesh.

    vertices : array, shape (n_vertices, 3)
        Vertices of the mesh.

    tetrahedra : array, shape (n_tetrahedra, 4)
        Indices of vertices that form tetrahedra of the mesh.
    """
    def __init__(self, body2origin, vertices, tetrahedra):
        self.mesh = o3d.geometry.TetraMesh()
        self.set_data(body2origin, vertices, tetrahedra)

    def set_data(self, body2origin, vertices, tetrahedra):
        """Update rigid body.

        Parameters
        ----------
        body2origin : array, shape (4, 4)
            Pose of the mesh.

        vertices : array, shape (n_vertices, 3)
            Vertices of the mesh.

        tetrahedra : array, shape (n_tetrahedra, 4)
            Indices of vertices that form tetrahedra of the mesh.
        """
        self.mesh.vertices = o3d.utility.Vector3dVector(vertices)
        self.mesh.tetras = o3d.utility.Vector4iVector(tetrahedra)
        self.mesh.transform(body2origin)

    @property
    def geometries(self):
        """Expose geometries.

        Returns
        -------
        geometries : list
            List of geometries that can be added to the visualizer.
        """
        return [self.mesh]
