import numba
import numpy as np


class MeshHillClimber:
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
