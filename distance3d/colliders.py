"""Colliders used for collision detection with GJK and MPR algorithms."""
import abc
import numpy as np
from .geometry import (
    support_function_capsule, support_function_cylinder,
    convert_box_to_vertices, support_function_ellipsoid,
    support_function_sphere, support_function_cone, support_function_disk)
from .containment import (
    axis_aligned_bounding_box, sphere_aabb, box_aabb, cylinder_aabb,
    capsule_aabb, ellipsoid_aabb, cone_aabb, disk_aabb)
from .mesh import MeshHillClimbingSupportFunction
from .utils import plane_basis_from_normal
from aabbtree import AABB


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


class ConvexHullVertices(ConvexCollider):
    """Convex hull of a set of vertices.

    Parameters
    ----------
    vertices : array, shape (n_vertices, 3)
        Vertices of the convex shape.

    artist : pytransform3d.visualizer.Artist, optional (default: None)
        Artist for visualizer.
    """
    def __init__(self, vertices, artist=None):
        super(ConvexHullVertices, self).__init__(artist)
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


class Box(ConvexHullVertices):
    """Box collider.

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
    """Mesh collider that uses triangles for hill climbing.

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


class Sphere(ConvexCollider):
    """Sphere collider.

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


class Capsule(ConvexCollider):
    """Capsule collider.

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
        return self.capsule2origin[:3, 3] - (
            self.radius + 0.5 * self.height) * self.capsule2origin[:3, 2]

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


class Ellipsoid(ConvexCollider):
    """Ellipsoid collider.

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
        return (self.ellipsoid2origin[:3, 3]
                + self.radii[2] * self.ellipsoid2origin[:3, 2])

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


class Cylinder(ConvexCollider):
    """Cylinder collider.

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
        return (self.cylinder2origin[:3, 3]
                + 0.5 * self.length * self.cylinder2origin[:3, 2])

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


class Disk(ConvexCollider):
    """Disk collider.

    Parameters
    ----------
    center : array, shape (3,)
        Center of the disk.

    radius : float
        Radius of the disk.

    normal : array, shape (3,)
        Normal to the plane in which the disk lies.

    artist : pytransform3d.visualizer.Artist, optional (default: None)
        Corresponding artist for visualizer.
    """
    def __init__(self, center, radius, normal, artist=None):
        super(Disk, self).__init__(artist)
        self.c = center
        self.radius = radius
        self.normal = normal

    def make_artist(self, c=None):
        import pytransform3d.visualizer as pv
        x, y = plane_basis_from_normal(self.normal)
        disk2origin = np.eye(4)
        disk2origin[:3, :3] = np.column_stack((x, y, self.normal))
        disk2origin[:3, 3] = self.c
        self.artist_ = pv.Cylinder(
            A2B=disk2origin, radius=self.radius, length=0.01 * self.radius,
            c=c)

    def center(self):
        return self.c

    def first_vertex(self):
        x, _ = plane_basis_from_normal(self.normal)
        return self.c + self.radius * x

    def support_function(self, search_direction):
        return support_function_disk(
            search_direction, self.c, self.radius, self.normal)

    def update_pose(self, pose):
        self.c = pose[:3, 3]
        self.normal = pose[:3, 2]
        if self.artist_ is not None:
            self.artist_.set_data(pose)

    def aabb(self):
        mins, maxs = disk_aabb(self.c, self.radius, self.normal)
        return AABB(np.array([mins, maxs]).T)


class Cone(ConvexCollider):
    """Cone collider.

    Parameters
    ----------
    cone2origin : array, shape (4, 4)
        Pose of the cone.

    radius : float
        Radius of the cone.

    height : float
        Length of the cone.

    artist : pytransform3d.visualizer.Artist, optional (default: None)
        Corresponding artist for visualizer.
    """
    def __init__(self, cone2origin, radius, height, artist=None):
        super(Cone, self).__init__(artist)
        self.cone2origin = cone2origin
        self.radius = radius
        self.height = height

    def make_artist(self, c=None):
        import pytransform3d.visualizer as pv
        self.artist_ = pv.Cone(
            A2B=self.cone2origin, radius=self.radius, height=self.height, c=c)

    def center(self):
        return (self.cone2origin[:3, 3]
                + 0.5 * self.height * self.cone2origin[:3, 2])

    def first_vertex(self):
        return (self.cone2origin[:3, 3]
                + self.height * self.cone2origin[:3, 2])

    def support_function(self, search_direction):
        return support_function_cone(
            search_direction, self.cone2origin, self.radius, self.height)

    def update_pose(self, pose):
        self.cone2origin = pose
        if self.artist_ is not None:
            self.artist_.set_data(pose)

    def aabb(self):
        mins, maxs = cone_aabb(
            self.cone2origin, self.radius, self.height)
        return AABB(np.array([mins, maxs]).T)
