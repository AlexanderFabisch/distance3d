import numpy as np
from .geometry import (
    capsule_extreme_along_direction, cylinder_extreme_along_direction,
    convert_box_to_vertices)


class Convex:
    """Wraps convex hull of a set of vertices for GJK algorithm.

    Parameters
    ----------
    vertices : array, shape (n_vertices, 3)
        Vertices of the convex shape.
    """
    def __init__(self, vertices):
        self.vertices = vertices

    @staticmethod
    def from_box(box2origin, size):
        """TODO"""
        vertices = convert_box_to_vertices(box2origin, size)
        return Convex(vertices)

    @staticmethod
    def from_mesh(filename, A2B, scale=1.0):
        """TODO"""
        import open3d as o3d
        mesh = o3d.io.read_triangle_mesh(filename)
        mesh.transform(A2B)
        vertices = o3d.utility.Vector3dVector(
            np.asarray(mesh.vertices) * scale)
        return Convex(vertices)

    def first_vertex(self):
        return self.vertices[0]

    def support_function(self, search_direction):
        idx = np.argmax(self.vertices.dot(search_direction))
        return idx, self.vertices[idx]

    def compute_point(self, barycentric_coordinates, indices):
        return np.dot(barycentric_coordinates, self.vertices[indices])


class Cylinder:
    """Wraps cylinder for GJK algorithm."""
    def __init__(self, cylinder2origin, radius, length):
        self.cylinder2origin = cylinder2origin
        self.radius = radius
        self.length = length
        self.vertices = []

    def first_vertex(self):
        vertex = self.cylinder2origin[:3, 3] + 0.5 * self.length * self.cylinder2origin[:3, 2]
        self.vertices.append(vertex)
        return vertex

    def support_function(self, search_direction):
        vertex = cylinder_extreme_along_direction(
            search_direction, self.cylinder2origin, self.radius, self.length)
        vertex_idx = len(self.vertices)
        self.vertices.append(vertex)
        return vertex_idx, vertex

    def compute_point(self, barycentric_coordinates, indices):
        return np.dot(barycentric_coordinates, np.array([self.vertices[i] for i in indices]))


class Capsule:
    """Wraps capsule for GJK algorithm."""
    def __init__(self, capsule2origin, radius, height):
        self.capsule2origin = capsule2origin
        self.radius = radius
        self.height = height
        self.vertices = []

    def first_vertex(self):
        vertex = self.capsule2origin[:3, 3] - (self.radius + 0.5 * self.height) * self.capsule2origin[:3, 2]
        self.vertices.append(vertex)
        return vertex

    def support_function(self, search_direction):
        vertex = capsule_extreme_along_direction(
            search_direction, self.capsule2origin, self.radius, self.height)
        vertex_idx = len(self.vertices)
        self.vertices.append(vertex)
        return vertex_idx, vertex

    def compute_point(self, barycentric_coordinates, indices):
        return np.dot(barycentric_coordinates, np.array([self.vertices[i] for i in indices]))


class Sphere:
    """Wraps sphere for GJK algorithm."""
    # TODO https://github.com/kevinmoran/GJK/blob/master/Collider.h#L33
    def __init__(self, center, radius):
        self.c = center
        self.radius = radius
        self.vertices = []

    def first_vertex(self):
        vertex = self.c + np.array([0, 0, self.radius])
        self.vertices.append(vertex)
        return vertex

    def support_function(self, search_direction):
        s_norm = np.linalg.norm(search_direction)
        if s_norm == 0.0:
            vertex = self.c + np.array([0, 0, self.radius])
        else:
            vertex = self.c + search_direction / s_norm * self.radius
        vertex_idx = len(self.vertices)
        self.vertices.append(vertex)
        return vertex_idx, vertex

    def compute_point(self, barycentric_coordinates, indices):
        return np.dot(barycentric_coordinates, np.array([self.vertices[i] for i in indices]))
