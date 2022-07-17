"""Tools for convex meshes."""
import math
import numba
import numpy as np
from scipy.spatial import ConvexHull

from .utils import EPSILON, HALF_PI, angles_between_vectors, transform_point


PROJECTION_LENGTH_EPSILON = 10.0 * EPSILON


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
            if projected_length > PROJECTION_LENGTH_EPSILON:
                best_idx = connected_idx

    converged = False
    while not converged:
        converged = True
        for connected_idx in connections[best_idx]:
            vertex_diff = np.ascontiguousarray(
                vertices[connected_idx] - vertices[best_idx])
            projected_length = search_direction.dot(vertex_diff)
            if projected_length > PROJECTION_LENGTH_EPSILON:
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
    # test if normal derived from wrong order points from origin to center of
    # triangle, should actually point from origin away from center
    indices = np.where(angles < HALF_PI)[0]
    triangles[indices] = triangles[indices, ::-1]
    return triangles


def make_triangular_icosphere(center, radius, order=4):
    """Creates an triangular icosphere mesh.

    Source: https://observablehq.com/@mourner/fast-icosphere-mesh

    Parameters
    ----------
    center : array, shape (3,)
        Center of the sphere.

    radius : float
        Radius of the sphere.

    order : int, optional (default: 4)
        Number of subdivisions of initial 20 triangles.

    Returns
    -------
    vertices : array, shape (n_vertices, 3)
        Vertices of the mesh.

    triangles : array, shape (n_triangles, 3)
        Indices of vertices that form triangles of the mesh.
    """
    f = (1 + 5 ** 0.5) / 2
    vertices = np.zeros(((10 * 4 ** order + 2), 3))
    vertices[:12] = np.array([
        [-1, f, 0], [1, f, 0], [-1, -f, 0],
        [1, -f, 0], [0, -1, f], [0, 1, f],
        [0, -1, -f], [0, 1, -f], [f, 0, -1],
        [f, 0, 1], [-f, 0, -1], [-f, 0, 1]])
    triangles = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [11, 10, 2],
        [5, 11, 4], [1, 5, 9], [7, 1, 8], [10, 7, 6], [3, 9, 4], [3, 4, 2],
        [3, 2, 6], [3, 6, 8], [3, 8, 9], [9, 8, 1], [4, 9, 5], [2, 4, 11],
        [6, 2, 10], [8, 6, 7]
    ], dtype=int)
    v = 12
    mid_cache = dict()  # midpoint vertices cache to avoid duplicating shared vertices

    def add_mid_point(a, b, mid_cache, v):
        # Cantor's pairing function
        key = math.floor((a + b) * (a + b + 1) / 2) + min(a, b)
        i = mid_cache.get(key, None)
        if i is not None:
            del mid_cache[key]
            return i, v
        mid_cache[key] = v
        vertices[v] = 0.5 * (vertices[a] + vertices[b])
        i = v
        v += 1
        return i, v

    # repeatedly subdivide each triangle into 4 triangles
    triangles_prev = triangles
    for i in range(order):
        triangles = np.empty(
            (4 * triangles.shape[0], triangles.shape[1]),
            dtype=int)
        for k in range(len(triangles_prev)):
            v1, v2, v3 = triangles_prev[k]
            a, v = add_mid_point(v1, v2, mid_cache, v)
            b, v = add_mid_point(v2, v3, mid_cache, v)
            c, v = add_mid_point(v3, v1, mid_cache, v)
            t = k * 4
            triangles[t] = v1, a, c
            triangles[t + 1] = v2, b, a
            triangles[t + 2] = v3, c, b
            triangles[t + 3] = a, b, c
        triangles_prev = triangles

    vertices /= 1.0 / radius * np.linalg.norm(vertices, axis=1)[:, np.newaxis]
    vertices += center[np.newaxis]
    return vertices, triangles


def make_tetrahedral_icosphere(center, radius, order=4):
    """Creates an tetrahedral icosphere mesh.

    Source: https://github.com/ekzhang/hydroelastics

    Parameters
    ----------
    center : array, shape (3,)
        Center of the sphere.

    radius : float
        Radius of the sphere.

    order : int, optional (default: 4)
        Number of subdivisions of initial 20 triangles.

    Returns
    -------
    vertices : array, shape (n_vertices, 3)
        Vertices of the mesh.

    tetrahedra : array, shape (n_tetrahedra, 4)
        Indices of vertices that form tetrahedra of the mesh.
    """
    vertices, triangles = make_triangular_icosphere(center, radius, order)
    center_idx = len(vertices)
    vertices = np.vstack((vertices, center[np.newaxis]))
    tetrahedra = np.hstack(
        (triangles, center_idx * np.ones((len(triangles), 1), dtype=int)))
    return vertices, tetrahedra


def center_of_mass_tetrahedral_mesh(mesh2origin, vertices, tetrahedra):
    """TODO"""
    tetrahedra_vertices = vertices[tetrahedra]
    centers = tetrahedra_vertices.mean(axis=1)
    tetrahedra_edges = tetrahedra_vertices[:, 1:] - tetrahedra_vertices[:, np.newaxis, 0]
    volumes = np.abs(np.sum(
        np.cross(tetrahedra_edges[:, 0], tetrahedra_edges[:, 1])
        * tetrahedra_edges[:, 2], axis=1)) / 6.0
    center = np.dot(volumes, centers) / np.sum(volumes)
    return transform_point(mesh2origin, center)
