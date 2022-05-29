import numba
import numpy as np


class MeshHillClimber:
    def __init__(self, mesh2origin, vertices, triangles):
        self.mesh2origin = mesh2origin
        self.vertices = vertices

        connections = {}
        for i, j, k in triangles:
            if i not in connections:
                connections[i] = set()
            if j not in connections:
                connections[j] = set()
            if k not in connections:
                connections[k] = set()
            connections[i].add(j)
            connections[i].add(k)
            connections[j].add(i)
            connections[j].add(k)
            connections[k].add(i)
            connections[k].add(j)

        self.connections = numba.typed.Dict.empty(numba.int64, numba.int64[:])
        for idx, connected_indices in connections.items():
            self.connections[idx] = np.asarray(list(connected_indices), dtype=int)

        self.first_idx = np.min(triangles)

    def update_pose(self, mesh2origin):
        self.mesh2origin = mesh2origin

    def __call__(self, search_direction):
        search_direction_in_mesh = np.dot(
            self.mesh2origin[:3, :3].T, search_direction)
        idx = hill_climb_mesh_extreme(
            search_direction_in_mesh, self.first_idx, self.vertices,
            self.connections)
        return idx, self.mesh2origin[:3, 3] + np.dot(self.mesh2origin[:3, :3], self.vertices[idx])


@numba.njit(
    numba.int64(numba.float64[:], numba.int64, numba.float64[:, :],
                numba.types.DictType(numba.int64, numba.int64[:])),
    cache=True)
def hill_climb_mesh_extreme(search_direction, start_idx, vertices, connections):
    search_direction = np.ascontiguousarray(search_direction)
    best_idx = start_idx
    converged = False
    while not converged:
        updated = False
        for connected_idx in connections[best_idx]:
            vertex_diff = np.ascontiguousarray(
                vertices[connected_idx] - vertices[best_idx])
            projected_length = search_direction.dot(vertex_diff)
            if projected_length > 0.0:
                best_idx = connected_idx
                updated = True
        converged = not updated
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
        return idx, self.mesh2origin[:3, 3] + np.dot(self.mesh2origin[:3, :3], self.vertices[idx])
