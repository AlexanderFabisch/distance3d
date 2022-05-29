import numpy as np
import pytransform3d.visualizer as pv
import pytransform3d.transformations as pt
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
    """
    def __init__(self, mesh2origin, vertices, triangles):
        self.mesh = o3d.geometry.TriangleMesh()
        self.mesh.vertices = o3d.utility.Vector3dVector(vertices)
        self.mesh.triangles = o3d.utility.Vector3iVector(triangles)
        self.mesh.compute_vertex_normals()
        self.mesh2origin = None
        self.set_data(mesh2origin)

    def set_data(self, mesh2origin):
        previous_mesh2origin = self.mesh2origin
        if previous_mesh2origin is None:
            previous_mesh2origin = np.eye(4)
        self.mesh2origin = mesh2origin

        self.mesh.transform(pt.invert_transform(previous_mesh2origin, check=False))
        self.mesh.transform(self.mesh2origin)

    @property
    def geometries(self):
        return [self.mesh]
