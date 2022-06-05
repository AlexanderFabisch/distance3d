"""Colliders used for collision detection with GJK algorithm."""
import abc
import warnings
import numpy as np
from pytransform3d import urdf
from .geometry import (
    support_function_capsule, support_function_cylinder,
    convert_box_to_vertices, support_function_ellipsoid,
    support_function_sphere)
from .containment import (
    axis_aligned_bounding_box, sphere_aabb, box_aabb, cylinder_aabb,
    capsule_aabb, ellipsoid_aabb)
from .urdf_utils import self_collision_whitelists
from .mesh import MeshHillClimbingSupportFunction, MeshSupportFunction
from aabbtree import AABB, AABBTree


class BoundingVolumeHierarchy:
    """Bounding volume hierarchy (BVH) for broad phase collision detection.

    Wraps multiple colliders that are connected through transformations.
    In addition, these colliders are stored in an AABB tree for broad phase
    collision detection.

    Parameters
    ----------
    tm : pytransform3d.transform_manager.TransformManager
        Transform manager that stores the transformations.

    base_frame : str
        Name of the base frame in which colliders are represented.

    Attributes
    ----------
    aabbtree_ : AABBTree
        Tree of axis-aligned bounding boxes.

    colliders_ : dict
        Maps frames of collision objects to colliders.

    self_collision_whitelists_ : dict
        Whitelists for self-collision detection in case this BVH represents
        a robot.
    """
    def __init__(self, tm, base_frame):
        self.tm = tm
        self.base_frame = base_frame
        self.collider_frames = set()
        self.aabbtree_ = AABBTree()
        self.colliders_ = {}
        self.self_collision_whitelists_ = {}

    def fill_tree_with_colliders(
            self, tm, make_artists=False,
            fill_self_collision_whitelists=False):
        """Fill tree with colliders from URDF transform manager.

        Parameters
        ----------
        tm : pytransform3d.urdf.UrdfTransformManager
            Transform manager that has colliders.

        make_artists : bool, optional (default: False)
            Create artist for visualization for each collision object.

        fill_self_collision_whitelists : bool, optional (default: False)
            Fill whitelists for self collision detection. All collision
            objects connected to the current link, child, and parent links
            will be ignored.
        """
        for obj in tm.collision_objects:
            try:
                collider = self._make_collider(tm, obj, make_artists)
                self.add_collider(obj.frame, collider)
            except RuntimeError as e:
                warnings.warn(str(e))

        if fill_self_collision_whitelists:
            self.self_collision_whitelists_.update(
                self_collision_whitelists(tm))

    def _make_collider(self, tm, obj, make_artists):
        A2B = tm.get_transform(obj.frame, self.base_frame)
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
            import open3d as o3d
            mesh = o3d.io.read_triangle_mesh(obj.filename)
            vertices = np.asarray(mesh.vertices) * obj.scale
            triangles = np.asarray(mesh.triangles)
            collider = MeshGraph(A2B, vertices, triangles)
        if make_artists:
            collider.make_artist()
        return collider

    def add_collider(self, frame, collider):
        """Add collider.

        Parameters
        ----------
        frame : Hashable
            Frame in which the collider is located.

        collider : ConvexCollider
            Collider.
        """
        self.collider_frames.add(frame)
        self.colliders_[frame] = collider
        self.aabbtree_.add(collider.aabb(), (frame, collider))

    def update_collider_poses(self):
        """Update poses of all colliders from transform manager."""
        self.aabbtree_ = AABBTree()
        for frame in self.colliders_:
            A2B = self.tm.get_transform(frame, self.base_frame)
            collider = self.colliders_[frame]
            collider.update_pose(A2B)
            self.aabbtree_.add(collider.aabb(), (frame, collider))

    def get_colliders(self):
        """Get all colliders.

        Returns
        -------
        colliders : list
            List of colliders.
        """
        return self.colliders_.values()

    def get_artists(self):
        """Get all artists.

        Returns
        -------
        artists : list
            List of artists.
        """
        return [collider.artist_ for collider in self.colliders_.values()
                if collider.artist_ is not None]

    def aabb_overlapping_colliders(self, collider, whitelist=()):
        """Get colliders with an overlapping AABB.

        This function performs broad phase collision detection with a bounding
        volume hierarchy, where the bounding volumes are axis-aligned bounding
        boxes.

        Parameters
        ----------
        collider : ConvexCollider
            Collider.

        whitelist : sequence
            Names of frames to which collisions are allowed.

        Returns
        -------
        colliders : dict
            Maps frame names to colliders with overlapping AABB.
        """
        aabb = collider.aabb()
        colliders = dict(self.aabbtree_.overlap_values(aabb))
        for frame in whitelist:
            colliders.pop(frame, None)
        return colliders

    def get_collider_frames(self):
        """Get collider frames.

        Returns
        -------
        collider_frames : set
            Collider frames.
        """
        return self.collider_frames


# for backwards compatibility:
ColliderTree = BoundingVolumeHierarchy


class ConvexCollider(abc.ABC):
    """Convex collider base class.

    Parameters
    ----------
    artist : pytransform3d.visualizer.Artist, optional (default: None)
        Corresponding artist for visualizer.

    Attributes
    ----------
    artist_ : pytransform3d.visualizer.Artist, optional (default: None)
        Corresponding artist for visualizer.
    """
    def __init__(self, artist=None):
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
        """Get vertex with index 0 from collider to initialize GJK algorithm.

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
        support_point : array, shape (3,)
            Extreme point along search direction.
        """

    @abc.abstractmethod
    def center(self):
        """Returns the (approximate) center of the collider.

        Returns
        -------
        center : array, shape (3,)
            Center of the collider.
        """

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
        super(Convex, self).__init__(artist)
        self.vertices = vertices

    def make_artist(self, c=None):
        import pytransform3d.visualizer as pv
        self.artist_ = pv.PointCollection3D(self.vertices, s=0.005, c=c)

    def first_vertex(self):
        return self.vertices[0]

    def support_function(self, search_direction):
        return self.vertices[np.argmax(self.vertices.dot(search_direction))]

    def center(self):
        return np.mean(self.vertices, axis=0)

    def update_pose(self, pose):
        raise NotImplementedError("update_pose is not implemented!")

    def aabb(self):
        mins, maxs = axis_aligned_bounding_box(self.vertices)
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
        import pytransform3d.visualizer as pv
        self.artist_ = pv.Box(size=self.size, A2B=self.box2origin, c=c)

    def center(self):
        return self.box2origin[:3, 3]

    def update_pose(self, pose):
        self.box2origin = pose
        self.vertices = convert_box_to_vertices(pose, self.size)
        if self.artist_ is not None:
            self.artist_.set_data(pose)

    def aabb(self):
        mins, maxs = box_aabb(self.box2origin, self.size)
        return AABB(np.array([mins, maxs]).T)


class MeshGraph(ConvexCollider):
    """Wraps mesh for GJK and use triangles for hill climbing.

    Parameters
    ----------
    mesh2origin : array, shape (4, 4)
        Pose of the mesh.

    vertices : array, shape (n_vertices, 3)
        Vertices of the mesh.

    triangles : array, shape (n_triangles, 3)
        Indices of vertices that form triangles of the mesh.

    artist : pytransform3d.visualizer.Artist, optional (default: None)
        Corresponding artist for visualizer.
    """
    def __init__(self, mesh2origin, vertices, triangles, artist=None):
        super(MeshGraph, self).__init__(artist)
        self.mesh2origin = mesh2origin
        self.vertices = vertices
        self.triangles = triangles
        self._support_function = MeshHillClimbingSupportFunction(
            mesh2origin, vertices, triangles)

    def make_artist(self, c=None):
        from .visualization import Mesh
        self.artist_ = Mesh(
            self.mesh2origin, self.vertices, self.triangles, c=c)

    def first_vertex(self):
        return self.mesh2origin[:3, 3] + np.dot(
            self.mesh2origin[:3, :3], self.vertices[0])

    def support_function(self, search_direction):
        return self._support_function(search_direction)[1]

    def center(self):
        return self.mesh2origin[:3, 3] + np.dot(
            self.mesh2origin[:3, :3], np.mean(self.vertices, axis=0))

    def update_pose(self, mesh2origin):
        self.mesh2origin = mesh2origin
        self._support_function.update_pose(mesh2origin)
        if self.artist_ is not None:
            self.artist_.set_data(mesh2origin)

    def aabb(self):
        mins, maxs = axis_aligned_bounding_box(
            self.mesh2origin[np.newaxis, :3, 3] + np.dot(
                self.vertices, self.mesh2origin[:3, :3].T))
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
        super(Cylinder, self).__init__(artist)
        self.cylinder2origin = cylinder2origin
        self.radius = radius
        self.length = length

    def make_artist(self, c=None):
        import pytransform3d.visualizer as pv
        self.artist_ = pv.Cylinder(
            length=self.length, radius=self.radius, A2B=self.cylinder2origin,
            c=c)

    def center(self):
        return self.cylinder2origin[:3, 3]

    def first_vertex(self):
        return self.cylinder2origin[:3, 3] + 0.5 * self.length * self.cylinder2origin[:3, 2]

    def support_function(self, search_direction):
        return support_function_cylinder(
            search_direction, self.cylinder2origin, self.radius, self.length)

    def update_pose(self, pose):
        self.cylinder2origin = pose
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
        super(Capsule, self).__init__(artist)
        self.capsule2origin = capsule2origin
        self.radius = radius
        self.height = height

    def make_artist(self, c=None):
        import pytransform3d.visualizer as pv
        self.artist_ = pv.Capsule(
            height=self.height, radius=self.radius, A2B=self.capsule2origin,
            c=c)

    def center(self):
        return self.capsule2origin[:3, 3]

    def first_vertex(self):
        return self.capsule2origin[:3, 3] - (self.radius + 0.5 * self.height) * self.capsule2origin[:3, 2]

    def support_function(self, search_direction):
        return support_function_capsule(
            search_direction, self.capsule2origin, self.radius, self.height)

    def update_pose(self, pose):
        self.capsule2origin = pose
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
        super(Sphere, self).__init__(artist)
        self.c = center
        self.radius = radius

    def make_artist(self, c=None):
        import pytransform3d.visualizer as pv
        sphere2origin = np.eye(4)
        sphere2origin[:3, 3] = self.c
        self.artist_ = pv.Sphere(radius=self.radius, A2B=sphere2origin, c=c)

    def center(self):
        return self.c

    def first_vertex(self):
        return self.c + np.array([0, 0, self.radius], dtype=float)

    def support_function(self, search_direction):
        return support_function_sphere(
            search_direction, np.ascontiguousarray(self.c), self.radius)

    def update_pose(self, pose):
        self.c = pose[:3, 3]
        if self.artist_ is not None:
            self.artist_.set_data(pose)

    def aabb(self):
        mins, maxs = sphere_aabb(self.c, self.radius)
        return AABB(np.array([mins, maxs]).T)


class Ellipsoid(ConvexCollider):
    """Wraps ellipsoid for GJK algorithm.

    Parameters
    ----------
    ellipsoid2origin : array, shape (4, 4)
        Pose of the ellipsoid.

    radii : array, shape (3,)
        Radii of the ellipsoid.

    artist : pytransform3d.visualizer.Artist, optional (default: None)
        Corresponding artist for visualizer.
    """
    def __init__(self, ellipsoid2origin, radii, artist=None):
        super(Ellipsoid, self).__init__(artist)
        self.ellipsoid2origin = ellipsoid2origin
        self.radii = radii

    def make_artist(self, c=None):
        import pytransform3d.visualizer as pv
        self.artist_ = pv.Ellipsoid(
            radii=self.radii, A2B=self.ellipsoid2origin, c=c)

    def center(self):
        return self.ellipsoid2origin[:3, 3]

    def first_vertex(self):
        return self.ellipsoid2origin[:3, 3] + self.radii[2] * self.ellipsoid2origin[:3, 2]

    def support_function(self, search_direction):
        return support_function_ellipsoid(
            search_direction, self.ellipsoid2origin, self.radii)

    def update_pose(self, pose):
        self.ellipsoid2origin = pose
        if self.artist_ is not None:
            self.artist_.set_data(pose)

    def aabb(self):
        mins, maxs = ellipsoid_aabb(self.ellipsoid2origin, self.radii)
        return AABB(np.array([mins, maxs]).T)
