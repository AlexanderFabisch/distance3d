"""Tools for convex meshes."""
import numba
import numpy as np
from scipy.spatial import ConvexHull

from .utils import angles_between_vectors


class MeshHillClimbingSupportFunction:
    """Mesh support function with hill climbing.

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
        self.mesh2origin = mesh2origin
        self.vertices = vertices

        self.first_idx = np.min(triangles)

        connections = {}
        for i, j, k in triangles:
            if i not in connections:
                connections[i] = set()
            if j not in connections:
                connections[j] = set()
            if k not in connections:
                connections[k] = set()
            connections[i].update((j, k))
            connections[j].update((i, k))
            connections[k].update((i, j))

        self.shortcut_connections = np.array([
            np.argmax(self.vertices[:, 0]), np.argmax(self.vertices[:, 1]),
            np.argmax(self.vertices[:, 2]), np.argmin(self.vertices[:, 0]),
            np.argmin(self.vertices[:, 1]), np.argmin(self.vertices[:, 2])])

        self.connections = numba.typed.Dict.empty(numba.int64, numba.int64[:])
        for idx, connected_indices in connections.items():
            self.connections[idx] = np.fromiter(
                connected_indices, dtype=int, count=len(connected_indices))

    def update_pose(self, mesh2origin):
        """Update pose.

        Parameters
        ----------
        mesh2origin : array, shape (4, 4)
            New pose of the mesh.
        """
        self.mesh2origin = mesh2origin

    def __call__(self, search_direction):
        """Support function.

        Parameters
        ----------
        search_direction : array, shape (3,)
            Search direction.

        Returns
        -------
        idx : int
            Index of support point.

        support_point : array, shape (3,)
            Support point.
        """
        search_direction_in_mesh = np.dot(
            self.mesh2origin[:3, :3].T, search_direction)
        idx = hill_climb_mesh_extreme(
            search_direction_in_mesh, self.first_idx, self.vertices,
            self.connections, self.shortcut_connections)
        self.first_idx = idx  # vertex caching
        return idx, self.mesh2origin[:3, 3] + np.dot(
            self.mesh2origin[:3, :3], self.vertices[idx])


@numba.njit(
    numba.int64(numba.float64[:], numba.int64, numba.float64[:, :],
                numba.types.DictType(numba.int64, numba.int64[:]),
                numba.optional(numba.int64[:])),
    cache=True)
def hill_climb_mesh_extreme(
        search_direction, start_idx, vertices, connections,
        shortcut_connections):
    """Hill climbing to find support point of mesh.

    Parameters
    ----------
    search_direction : array, shape (3,)
        Search direction in mesh frame.

    start_idx : int
        Index of vertex from which we start hill climbing.

    vertices : array, shape (n_vertices, 3)
        Vertices of the mesh.

    connections : dict
        Maps vertex indices to neighboring vertex indices.

    shortcut_connections : array, shape (n_shortcut_connections,)
        Indices of extreme vertices of the mesh in mesh frame.
    """
    search_direction = np.ascontiguousarray(search_direction)
    best_idx = start_idx

    if shortcut_connections is not None:
        for connected_idx in shortcut_connections:
            vertex_diff = np.ascontiguousarray(
                vertices[connected_idx] - vertices[best_idx])
            projected_length = search_direction.dot(vertex_diff)
            if projected_length > 0.0:
                best_idx = connected_idx

    converged = False
    while not converged:
        converged = True
        for connected_idx in connections[best_idx]:
            vertex_diff = np.ascontiguousarray(
                vertices[connected_idx] - vertices[best_idx])
            projected_length = search_direction.dot(vertex_diff)
            if projected_length > 0.0:
                best_idx = connected_idx
                converged = False

    return best_idx


class MeshSupportFunction:
    """Standard mesh support function.

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
        self.mesh2origin = mesh2origin
        self.vertices = vertices
        self.first_idx = 0

    def update_pose(self, mesh2origin):
        """Update pose.

        Parameters
        ----------
        mesh2origin : array, shape (4, 4)
            New pose of the mesh.
        """
        self.mesh2origin = mesh2origin

    def __call__(self, search_direction):
        """Support function.

        Parameters
        ----------
        search_direction : array, shape (3,)
            Search direction.

        Returns
        -------
        idx : int
            Index of support point.

        support_point : array, shape (3,)
            Support point.
        """
        search_direction_in_mesh = np.dot(
            self.mesh2origin[:3, :3].T, search_direction)
        idx = np.argmax(self.vertices.dot(search_direction_in_mesh))
        return idx, self.mesh2origin[:3, 3] + np.dot(
            self.mesh2origin[:3, :3], self.vertices[idx])


def make_convex_mesh(vertices):
    """Make convex mesh from vertices.

    Parameters
    ----------
    vertices : array, shape (n_vertices, 3)
        Vertices of the mesh (not necessarily convex, but might be).

    Returns
    -------
    triangles : array, shape (n_triangles, 3)
        Indices of vertices forming the faces of a convex mesh. Normals of the
        faces point to the outside of the mesh.
    """
    vertices = vertices - np.mean(vertices, axis=0)
    ch = ConvexHull(vertices)
    triangles = ch.simplices
    faces = vertices[triangles]
    A = faces[:, 2] - faces[:, 0]
    B = faces[:, 1] - faces[:, 0]
    face_normals = np.cross(A, B)
    face_centers = np.mean(faces, axis=1)
    angles = angles_between_vectors(face_normals, face_centers)
    indices = np.where(angles < 0.5 * np.pi)[0]
    triangles[indices] = triangles[indices, ::-1]
    return triangles
