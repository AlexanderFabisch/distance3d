import numba
import numpy as np
from .containment import axis_aligned_bounding_box
from aabbtree import AABB


class MeshHillClimbingSupportFunction:
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
        self.mesh2origin = mesh2origin

    def __call__(self, search_direction):
        search_direction_in_mesh = np.dot(
            self.mesh2origin[:3, :3].T, search_direction)
        idx = hill_climb_mesh_extreme(
            search_direction_in_mesh, self.first_idx, self.vertices,
            self.connections, self.shortcut_connections)
        self.first_idx = idx  # vertex caching
        return idx, self.mesh2origin[:3, 3] + np.dot(
            self.mesh2origin[:3, :3], self.vertices[idx])

    def aabb(self):
        """
        mins, maxs = axis_aligned_bounding_box(
            self.mesh2origin[np.newaxis, :3, 3] + np.dot(
                self.vertices, self.mesh2origin[:3, :3].T))
        return AABB(np.array([mins, maxs]).T)

        #import time
        #start = time.time()
        mins, maxs = axis_aligned_bounding_box(
            self.mesh2origin[np.newaxis, :3, 3] + np.dot(
                self.vertices, self.mesh2origin[:3, :3].T))
        #stop = time.time()
        #print(stop - start)
        #print(mins, maxs)
        """

        #import time
        #start = time.time()
        idx_xm = hill_climb_mesh_extreme(
            -self.mesh2origin[0, :3], self.first_idx, self.vertices,
            self.connections, self.shortcut_connections)
        idx_xp = hill_climb_mesh_extreme(
            self.mesh2origin[0, :3], self.first_idx, self.vertices,
            self.connections, self.shortcut_connections)
        idx_ym = hill_climb_mesh_extreme(
            -self.mesh2origin[1, :3], self.first_idx, self.vertices,
            self.connections, self.shortcut_connections)
        idx_yp = hill_climb_mesh_extreme(
            self.mesh2origin[1, :3], self.first_idx, self.vertices,
            self.connections, self.shortcut_connections)
        idx_zm = hill_climb_mesh_extreme(
            -self.mesh2origin[2, :3], self.first_idx, self.vertices,
            self.connections, self.shortcut_connections)
        idx_zp = hill_climb_mesh_extreme(
            self.mesh2origin[2, :3], self.first_idx, self.vertices,
            self.connections, self.shortcut_connections)
        min_vertices = self.vertices[[idx_xm, idx_ym, idx_zm]]
        max_vertices = self.vertices[[idx_xp, idx_yp, idx_zp]]
        min_vertices = self.mesh2origin[np.newaxis, :3, 3] + min_vertices.dot(self.mesh2origin[:3, :3].T)
        max_vertices = self.mesh2origin[np.newaxis, :3, 3] + max_vertices.dot(self.mesh2origin[:3, :3].T)
        mins = np.diag(min_vertices)
        maxs = np.diag(max_vertices)
        #stop = time.time()
        #print(stop - start)
        #print(mins, maxs)
        return AABB(np.array([mins, maxs]).T)


@numba.njit(
    numba.int64(numba.float64[:], numba.int64, numba.float64[:, :],
                numba.types.DictType(numba.int64, numba.int64[:]),
                numba.optional(numba.int64[:])),
    cache=True)
def hill_climb_mesh_extreme(
        search_direction, start_idx, vertices, connections,
        shortcut_connections):
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
    def __init__(self, mesh2origin, vertices, triangles):
        self.mesh2origin = mesh2origin
        self.vertices = vertices
        self.first_idx = 0

    def update_pose(self, mesh2origin):
        self.mesh2origin = mesh2origin

    def __call__(self, search_direction):
        search_direction_in_mesh = np.dot(
            self.mesh2origin[:3, :3].T, search_direction)
        idx = np.argmax(self.vertices.dot(search_direction_in_mesh))
        return idx, self.mesh2origin[:3, 3] + np.dot(
            self.mesh2origin[:3, :3], self.vertices[idx])

    def aabb(self):
        mins, maxs = axis_aligned_bounding_box(
            self.mesh2origin[np.newaxis, :3, 3] + np.dot(
                self.vertices, self.mesh2origin[:3, :3].T))
        return AABB(np.array([mins, maxs]).T)
