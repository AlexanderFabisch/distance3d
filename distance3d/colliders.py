"""Colliders used for collision detection with GJK algorithm."""
import abc
import warnings
import numpy as np
from pytransform3d import urdf
import pytransform3d.visualizer as pv
from .geometry import (
    capsule_extreme_along_direction, cylinder_extreme_along_direction,
    convert_box_to_vertices)
from .containment import (
    axis_aligned_bounding_box, sphere_aabb, box_aabb, cylinder_aabb,
    capsule_aabb)
from aabbtree import AABB, AABBTree


class ColliderTree:
    """Wraps multiple colliders that are connected through transformations.

    In addition, these colliders are stored in an AABB tree for broad phase
    collision detection.

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
        self.collider_frames = set()
        self.aabbtree = AABBTree()
        self.colliders = {}

    def fill_tree_with_colliders(self, tm):
        """Fill tree with colliders from URDF transform manager.

        Parameters
        ----------
        tm : pytransform3d.urdf.UrdfTransformManager
            Transform manager that has colliders.
        """
        for obj in tm.collision_objects:
            A2B = tm.get_transform(obj.frame, self.base_frame)
            try:
                if isinstance(obj, urdf.Sphere):
                    collider = Sphere(center=A2B[:3, 3], radius=obj.radius)
                elif isinstance(obj, urdf.Box):
                    collider = Box(A2B, obj.size)
                elif isinstance(obj, urdf.Cylinder):
                    collider = Cylinder(
                        cylinder2origin=A2B, radius=obj.radius,
                        length=obj.length)
                else:
                    assert isinstance(obj, urdf.Mesh)
                    collider = Mesh(obj.filename, A2B, obj.scale)
                collider.make_artist()
                self.add_collider(obj.frame, collider)
            except RuntimeError as e:
                warnings.warn(str(e))

    def add_collider(self, frame, collider):
        """Add collider.

        Parameters
        ----------
        frame : str
            Frame in which the collider is located.

        collider : ConvexCollider
            Collider.
        """
        self.collider_frames.add(frame)
        self.colliders[frame] = collider
        self.aabbtree.add(collider.aabb(), (frame, collider))

    def update_collider_poses(self):
        """Update poses of all colliders from transform manager."""
        self.aabbtree = AABBTree()
        for frame in self.colliders:
            A2B = self.tm.get_transform(frame, self.base_frame)
            collider = self.colliders[frame]
            collider.update_pose(A2B)
            self.aabbtree.add(collider.aabb(), (frame, collider))

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
        return [collider.artist_ for collider in self.colliders.values()
                if collider.artist_ is not None]

    def aabb_overlapping_colliders(self, collider):
        """Get colliders with an overlapping AABB.

        This function performs broad phase collision detection with a bounding
        volume hierarchy, where the bounding volumes are axies-aligned bounding
        boxes.

        Parameters
        ----------
        collider : ConvexCollider
            Collider.

        Returns
        -------
        colliders : dict
            Colliders with overlapping AABB.
        """
        aabb = collider.aabb()
        return dict(self.aabbtree.overlap_values(aabb))

    def get_collider_frames(self):
        """Get collider frames.

        Returns
        -------
        collider_frames : set
            Collider frames.
        """
        return self.collider_frames


class ConvexCollider(abc.ABC):
    """Convex collider base class.

    Parameters
    ----------
    vertices : iterable
        Vertices of the convex collider.

    artist : pytransform3d.visualizer.Artist, optional (default: None)
        Corresponding artist for visualizer.

    Attributes
    ----------
    vertices_ : iterable
        Vertices of the convex collider.

    artist_ : pytransform3d.visualizer.Artist, optional (default: None)
        Corresponding artist for visualizer.
    """
    def __init__(self, vertices, artist=None):
        self.vertices_ = vertices
        self.artist_ = artist

    @abc.abstractmethod
    def make_artist(self, c=None):
        """Make artist that represents this collider.

        Parameters
        ----------
        c : array-like, shape (3,), optional (default: None)
            Color of artist.
        """

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

    @abc.abstractmethod
    def aabb(self):
        """Get axis-aligned bounding box.

        Returns
        -------
        aabb : AABB
            Axis-aligned bounding box.
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
    def __init__(self, vertices, artist=None):
        super(Convex, self).__init__(vertices, artist)

    def make_artist(self, c=None):
        self.artist_ = pv.PointCollection3D(self.vertices_, s=0.005, c=c)

    def first_vertex(self):
        return self.vertices_[0]

    def support_function(self, search_direction):
        idx = np.argmax(self.vertices_.dot(search_direction))
        return idx, self.vertices_[idx]

    def compute_point(self, barycentric_coordinates, indices):
        return np.dot(barycentric_coordinates, self.vertices_[indices])

    def update_pose(self, vertices):
        self.vertices_ = vertices
        if self.artist_ is not None:
            self.artist_.set_data(self.vertices_)

    def aabb(self):
        mins, maxs = axis_aligned_bounding_box(self.vertices_)
        return AABB(np.array([mins, maxs]).T)


class Box(Convex):
    """Wraps box for GJK algorithm.

    Parameters
    ----------
    box2origin : array, shape (4, 4)
        Pose of the box.

    size : array, shape (3,)
        Sizes of the box along its axes.

    artist : pytransform3d.visualizer.Artist, optional (default: None)
        Corresponding artist for visualizer.
    """
    def __init__(self, box2origin, size, artist=None):
        super(Box, self).__init__(
            convert_box_to_vertices(box2origin, size), artist)
        self.box2origin = box2origin
        self.size = size

    def make_artist(self, c=None):
        self.artist_ = pv.Box(size=self.size, A2B=self.box2origin, c=c)

    def update_pose(self, pose):
        self.box2origin = pose
        self.vertices_ = convert_box_to_vertices(pose, self.size)
        if self.artist_ is not None:
            self.artist_.set_data(pose)

    def aabb(self):
        mins, maxs = box_aabb(self.box2origin, self.size)
        return AABB(np.array([mins, maxs]).T)


class Mesh(Convex):
    """Wraps mesh for GJK algorithm (we assume a convex mesh).

    Parameters
    ----------
    filename : str
        Path to mesh file.

    A2B : array, shape (4, 4)
        Center of the mesh.

    scale : float, optional (default: 1)
        Scaling of the mesh.

    artist : pytransform3d.visualizer.Artist, optional (default: None)
        Corresponding artist for visualizer.
    """
    def __init__(self, filename, A2B, scale=1.0, artist=None):
        import pytransform3d.visualizer as pv
        if artist is None:
            artist = pv.Mesh(filename=filename, A2B=A2B, s=scale)
        vertices = np.asarray(artist.mesh.vertices)
        super(Mesh, self).__init__(vertices, artist)

    def make_artist(self, c=None):
        assert self.artist_ is not None

    def update_pose(self, pose):
        self.artist_.set_data(pose)
        self.vertices_ = np.asarray(self.artist_.mesh.vertices)

    def aabb(self):
        mins, maxs = axis_aligned_bounding_box(self.vertices_)
        return AABB(np.array([mins, maxs]).T)


class Cylinder(ConvexCollider):
    """Wraps cylinder for GJK algorithm.

    Parameters
    ----------
    cylinder2origin : array, shape (4, 4)
        Pose of the cylinder.

    radius : float
        Radius of the cylinder.

    length : float
        Length of the cylinder.

    artist : pytransform3d.visualizer.Artist, optional (default: None)
        Corresponding artist for visualizer.
    """
    def __init__(self, cylinder2origin, radius, length, artist=None):
        super(Cylinder, self).__init__([], artist)
        self.cylinder2origin = cylinder2origin
        self.radius = radius
        self.length = length

    def make_artist(self, c=None):
        self.artist_ = pv.Cylinder(
            length=self.length, radius=self.radius, A2B=self.cylinder2origin,
            c=c)

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
        if self.artist_ is not None:
            self.artist_.set_data(pose)

    def aabb(self):
        mins, maxs = cylinder_aabb(
            self.cylinder2origin, self.radius, self.length)
        return AABB(np.array([mins, maxs]).T)


class Capsule(ConvexCollider):
    """Wraps capsule for GJK algorithm.

    Parameters
    ----------
    capsule2origin : array, shape (4, 4)
        Pose of the capsule.

    radius : float
        Radius of the capsule.

    height : float
        Height of the capsule.

    artist : pytransform3d.visualizer.Artist, optional (default: None)
        Corresponding artist for visualizer.
    """
    def __init__(self, capsule2origin, radius, height, artist=None):
        super(Capsule, self).__init__([], artist)
        self.capsule2origin = capsule2origin
        self.radius = radius
        self.height = height
        self.vertices_ = []

    def make_artist(self, c=None):
        self.artist_ = pv.Capsule(
            height=self.height, radius=self.radius, A2B=self.capsule2origin,
            c=c)

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
        if self.artist_ is not None:
            self.artist_.set_data(pose)

    def aabb(self):
        mins, maxs = capsule_aabb(
            self.capsule2origin, self.radius, self.height)
        return AABB(np.array([mins, maxs]).T)


class Sphere(ConvexCollider):
    """Wraps sphere for GJK algorithm.

    Parameters
    ----------
    center : array, shape (3,)
        Center of the sphere.

    radius : float
        Radius of the sphere.

    artist : pytransform3d.visualizer.Artist, optional (default: None)
        Corresponding artist for visualizer.
    """
    def __init__(self, center, radius, artist=None):
        super(Sphere, self).__init__([], artist)
        self.c = center
        self.radius = radius
        self.vertices_ = []

    def make_artist(self, c=None):
        sphere2origin = np.eye(4)
        sphere2origin[:3, 3] = self.c
        self.artist_ = pv.Sphere(radius=self.radius, A2B=sphere2origin, c=c)

    def first_vertex(self):
        vertex = self.c + np.array([0, 0, self.radius])
        self.vertices_.append(vertex)
        return vertex

    def support_function(self, search_direction):
        # Similar implementation:
        # https://github.com/kevinmoran/GJK/blob/b38d923d268629f30b44c3cf6d4f9974bbcdb0d3/Collider.h#L33
        # (Copyright (c) 2017 Kevin Moran, MIT License or Unlicense)
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
        if self.artist_ is not None:
            self.artist_.set_data(pose)

    def aabb(self):
        mins, maxs = sphere_aabb(self.c, self.radius)
        return AABB(np.array([mins, maxs]).T)
