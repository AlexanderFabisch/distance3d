import numpy as np
from .geometry import (
    capsule_extreme_along_direction, cylinder_extreme_along_direction,
    convert_box_to_vertices)


class ColliderTree:
    """TODO document"""
    def __init__(self, tm, base_frame):
        self.tm = tm
        self.base_frame = base_frame
        self.colliders = {}

    def add_collider(self, frame, collider):
        self.colliders[frame] = collider

    def update_collider_poses(self):
        for frame in self.colliders:
            A2B = self.tm.get_transform(frame, self.base_frame)
            self.colliders[frame].update_pose(A2B)

    def get_colliders(self):
        return self.colliders.values()

    def get_artists(self):
        return [collider.artist for collider in self.colliders.values()
                if collider.artist is not None]


class Convex:
    """Wraps convex hull of a set of vertices for GJK algorithm.

    Parameters
    ----------
    vertices : array, shape (n_vertices, 3)
        Vertices of the convex shape.

    artist : pytransform3d.visualizer.Artist, optional (default: None)
        Artist for visualizer.
    """
    def __init__(self, vertices, artist):
        self.vertices = vertices
        self.artist = artist

    def first_vertex(self):
        return self.vertices[0]

    def support_function(self, search_direction):
        idx = np.argmax(self.vertices.dot(search_direction))
        return idx, self.vertices[idx]

    def compute_point(self, barycentric_coordinates, indices):
        return np.dot(barycentric_coordinates, self.vertices[indices])

    def update_pose(self, vertices):
        self.vertices = vertices
        # TODO how to update artist?


class Box(Convex):
    """Wraps box for GJK algorithm."""
    def __init__(self, box2origin, size, artist=None):
        super(Box, self).__init__(
            convert_box_to_vertices(box2origin, size), artist)
        self.box2origin = box2origin
        self.size = size

    def update_pose(self, pose):
        self.box2origin = pose
        if self.artist is not None:
            self.artist.set_data(pose)


class Mesh(Convex):
    """Wraps mesh for GJK algorithm (we assume a convex mesh)."""
    def __init__(self, filename, A2B, scale=1.0, artist=None):
        import pytransform3d.visualizer as pv
        if artist is None:
            artist = pv.Mesh(filename=filename, A2B=A2B, s=scale)
        vertices = np.asarray(artist.mesh.vertices)
        super(Mesh, self).__init__(vertices, artist)

    def update_pose(self, pose):
        self.artist.set_data(pose)
        self.vertices = np.asarray(self.artist.mesh.vertices)


class Cylinder:
    """Wraps cylinder for GJK algorithm."""
    def __init__(self, cylinder2origin, radius, length, artist=None):
        self.cylinder2origin = cylinder2origin
        self.radius = radius
        self.length = length
        self.artist = artist
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

    def update_pose(self, pose):
        self.cylinder2origin = pose
        self.vertices = []
        if self.artist is not None:
            self.artist.set_data(pose)


class Capsule:
    """Wraps capsule for GJK algorithm."""
    def __init__(self, capsule2origin, radius, height, artist=None):
        self.capsule2origin = capsule2origin
        self.radius = radius
        self.height = height
        self.artist = artist
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

    def update_pose(self, pose):
        self.capsule2origin = pose
        self.vertices = []
        if self.artist is not None:
            self.artist.set_data(pose)


class Sphere:
    """Wraps sphere for GJK algorithm."""
    # TODO https://github.com/kevinmoran/GJK/blob/master/Collider.h#L33
    def __init__(self, center, radius, artist=None):
        self.c = center
        self.radius = radius
        self.artist = artist
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

    def update_pose(self, pose):
        self.c = pose[:3, 3]
        self.vertices = []
        if self.artist is not None:
            self.artist.set_data(pose)
