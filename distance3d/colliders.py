import abc
import numpy as np
from .geometry import (
    capsule_extreme_along_direction, cylinder_extreme_along_direction,
    convert_box_to_vertices)


class ColliderTree:
    """Wraps multiple colliders that are connected through transformations.

    Parameters
    ----------
    tm : pytransform3d.transform_manager.TransformManager
        Transform manager that stores the transformations.

    base_frame : str
        Name of the base frame in which colliders are represented.
    """
    def __init__(self, tm, base_frame):
        self.tm = tm
        self.base_frame = base_frame
        self.colliders = {}

    def add_collider(self, frame, collider):
        """Add collider.

        Parameters
        ----------
        frame : str
            Frame in which the collider is located.

        collider : ConvexCollider
            Collider.
        """
        self.colliders[frame] = collider

    def update_collider_poses(self):
        """Update poses of all colliders from transform manager."""
        for frame in self.colliders:
            A2B = self.tm.get_transform(frame, self.base_frame)
            self.colliders[frame].update_pose(A2B)

    def get_colliders(self):
        """Get all colliders.

        Returns
        -------
        colliders : list
            List of colliders.
        """
        return self.colliders.values()

    def get_artists(self):
        """Get all artists.

        Returns
        -------
        artists : list
            List of artists.
        """
        return [collider.artist for collider in self.colliders.values()
                if collider.artist is not None]


class ConvexCollider(abc.ABC):
    """Convex collider base class.

    Parameters
    ----------
    vertices : iterable
        Vertices of the convex collider.

    Attributes
    ----------
    vertices_ : iterable
        Vertices of the convex collider.
    """
    def __init__(self, vertices):
        self.vertices_ = vertices

    @abc.abstractmethod
    def first_vertex(self):
        """Get any vertex from collider to initialize GJK algorithm.

        Returns
        -------
        vertex : array, shape (3,)
            Vertex from collider.
        """

    @abc.abstractmethod
    def support_function(self, search_direction):
        """Support function for collider.

        Parameters
        ----------
        search_direction : array, shape (3,)
            Direction in which we search for extreme point of the collider.

        Returns
        -------
        extreme_point : array, shape (3,)
            Extreme point along search direction.
        """

    def compute_point(self, barycentric_coordinates, indices):
        """Compute point from barycentric coordinates.

        Parameters
        ----------
        barycentric_coordinates : array, shape (n_vertices,)
            Barycentric coordinates of the point that we compute.

        indices : array, shape (n_vertices,)
            Vertex indices to which the barycentric coordinates apply.

        Returns
        -------
        point : array, shape (3,)
            Point that we compute from barycentric coordinates.
        """
        return np.dot(barycentric_coordinates,
                      np.array([self.vertices_[i] for i in indices]))

    @abc.abstractmethod
    def update_pose(self, pose):
        """Update pose of collider.

        Parameters
        ----------
        pose : array, shape (4, 4)
            New pose of the collider.
        """


class Convex(ConvexCollider):
    """Wraps convex hull of a set of vertices for GJK algorithm.

    Parameters
    ----------
    vertices : array, shape (n_vertices, 3)
        Vertices of the convex shape.

    artist : pytransform3d.visualizer.Artist, optional (default: None)
        Artist for visualizer.
    """
    def __init__(self, vertices, artist):
        super(Convex, self).__init__(vertices)
        self.artist = artist

    def first_vertex(self):
        return self.vertices_[0]

    def support_function(self, search_direction):
        idx = np.argmax(self.vertices_.dot(search_direction))
        return idx, self.vertices_[idx]

    def compute_point(self, barycentric_coordinates, indices):
        return np.dot(barycentric_coordinates, self.vertices_[indices])

    def update_pose(self, vertices):
        self.vertices_ = vertices
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
        self.vertices_ = convert_box_to_vertices(pose, self.size)
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
        self.vertices_ = np.asarray(self.artist.mesh.vertices)


class Cylinder(ConvexCollider):
    """Wraps cylinder for GJK algorithm."""
    def __init__(self, cylinder2origin, radius, length, artist=None):
        super(Cylinder, self).__init__([])
        self.cylinder2origin = cylinder2origin
        self.radius = radius
        self.length = length
        self.artist = artist

    def first_vertex(self):
        vertex = self.cylinder2origin[:3, 3] + 0.5 * self.length * self.cylinder2origin[:3, 2]
        self.vertices_.append(vertex)
        return vertex

    def support_function(self, search_direction):
        vertex = cylinder_extreme_along_direction(
            search_direction, self.cylinder2origin, self.radius, self.length)
        vertex_idx = len(self.vertices_)
        self.vertices_.append(vertex)
        return vertex_idx, vertex

    def update_pose(self, pose):
        self.cylinder2origin = pose
        self.vertices_ = []
        if self.artist is not None:
            self.artist.set_data(pose)


class Capsule(ConvexCollider):
    """Wraps capsule for GJK algorithm."""
    def __init__(self, capsule2origin, radius, height, artist=None):
        super(Capsule, self).__init__([])
        self.capsule2origin = capsule2origin
        self.radius = radius
        self.height = height
        self.artist = artist
        self.vertices_ = []

    def first_vertex(self):
        vertex = self.capsule2origin[:3, 3] - (self.radius + 0.5 * self.height) * self.capsule2origin[:3, 2]
        self.vertices_.append(vertex)
        return vertex

    def support_function(self, search_direction):
        vertex = capsule_extreme_along_direction(
            search_direction, self.capsule2origin, self.radius, self.height)
        vertex_idx = len(self.vertices_)
        self.vertices_.append(vertex)
        return vertex_idx, vertex

    def update_pose(self, pose):
        self.capsule2origin = pose
        self.vertices_ = []
        if self.artist is not None:
            self.artist.set_data(pose)


class Sphere(ConvexCollider):
    """Wraps sphere for GJK algorithm."""
    # TODO https://github.com/kevinmoran/GJK/blob/master/Collider.h#L33
    def __init__(self, center, radius, artist=None):
        super(Sphere, self).__init__([])
        self.c = center
        self.radius = radius
        self.artist = artist
        self.vertices_ = []

    def first_vertex(self):
        vertex = self.c + np.array([0, 0, self.radius])
        self.vertices_.append(vertex)
        return vertex

    def support_function(self, search_direction):
        s_norm = np.linalg.norm(search_direction)
        if s_norm == 0.0:
            vertex = self.c + np.array([0, 0, self.radius])
        else:
            vertex = self.c + search_direction / s_norm * self.radius
        vertex_idx = len(self.vertices_)
        self.vertices_.append(vertex)
        return vertex_idx, vertex

    def update_pose(self, pose):
        self.c = pose[:3, 3]
        self.vertices_ = []
        if self.artist is not None:
            self.artist.set_data(pose)
