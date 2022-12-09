import numpy as np
from ..utils import transform_points, invert_transform
from ._tetra_mesh_creation import (
    make_tetrahedral_sphere, make_tetrahedral_ellipsoid,
    make_tetrahedral_cube, make_tetrahedral_box, make_tetrahedral_cylinder,
    make_tetrahedral_capsule)
from ._mesh_processing import center_of_mass_tetrahedral_mesh, tetrahedral_mesh_aabbs
from ..aabb_tree import AabbTree
from ..visualization import RigidBodyTetrahedralMesh


class RigidBody:
    """Rigid body represented by tetrahedral mesh.

    Parameters
    ----------
    body2origin : array, shape (4, 4)
        Pose of the mesh.

    vertices : array, shape (n_vertices, 3)
        Vertices of the mesh.

    tetrahedra : array, shape (n_tetrahedra, 4)
        Indices of vertices that form tetrahedra of the mesh.

    potentials : array, shape (n_vertices,)
        Potential of each vertex. Shortest distance to surface.

    youngs_modulus : float, optional (default: 1.0)
        The stiffness of the material the rigidbody is made out of.
    """
    def __init__(self, body2origin, vertices, tetrahedra, potentials, youngs_modulus=1.0):
        self.body2origin_ = body2origin
        self.vertices_ = vertices
        self.tetrahedra_ = tetrahedra
        self.potentials_ = potentials

        self._tetrahedra_points = None
        self._com = None

        self._aabbs = None
        self._aabb_tree = None

        self._youngs_modulus = youngs_modulus
        self._artist = None

    @staticmethod
    def make_sphere(center, radius, order=4):
        """Create sphere.

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
        rigid_body : RigidBody
            Sphere.
        """
        mesh2origin = np.eye(4)
        mesh2origin[:3, 3] = center
        vertices, tetrahedra, potentials = make_tetrahedral_sphere(
            radius, order)
        return RigidBody(mesh2origin, vertices, tetrahedra, potentials)

    @staticmethod
    def make_ellipsoid(ellipsoid2origin, radii, order=4):
        """Create ellipsoid.

        Parameters
        ----------
        ellipsoid2origin : array, shape (4, 4)
            Pose of the ellipsoid.

        radii : array, shape (3,)
            Radii of the ellipsoid.

        order : int, optional (default: 4)
            Number of subdivisions of initial 20 triangles.

        Returns
        -------
        rigid_body : RigidBody
            Ellipsoid.
        """
        vertices, tetrahedra, potentials = make_tetrahedral_ellipsoid(
            radii, order)
        return RigidBody(ellipsoid2origin, vertices, tetrahedra, potentials)

    @staticmethod
    def make_cube(cube2origin, size):
        """Create cube.

        Parameters
        ----------
        cube2origin : array, shape (4, 4)
            Pose of the cube.

        size : float
            Length of the edges in each dimension.

        Returns
        -------
        rigid_body : RigidBody
            Cube.
        """
        vertices, tetrahedra, potentials = make_tetrahedral_cube(size)
        return RigidBody(cube2origin, vertices, tetrahedra, potentials)

    @staticmethod
    def make_box(box2origin, size):
        """Create box.

        Parameters
        ----------
        box2origin : array, shape (4, 4)
            Pose of the box.

        size : array, shape (3,)
            Lengths of the edges in each dimension.

        Returns
        -------
        rigid_body : RigidBody
            Box.
        """
        vertices, tetrahedra, potentials = make_tetrahedral_box(size)
        return RigidBody(box2origin, vertices, tetrahedra, potentials)

    @staticmethod
    def make_cylinder(cylinder2origin, radius, length, resolution_hint=0.1):
        """Create cylinder.

        Parameters
        ----------
        cylinder2origin : array, shape (4, 4)
            Pose of the cylinder.

        radius : float
            Radius of the cylinder.

        length : float
            Length of the cylinder.

        resolution_hint : float, optional (default: 0.1)
            Controls the fineness of the tetrahedral mesh. The coarsest mesh
            that produces desirable results will allow simulation to run as
            efficiently as possible. The circles of the cylinder will have
            2 * pi * radius / resolution_hint edges.

        Returns
        -------
        rigid_body : RigidBody
            Cylinder.
        """
        vertices, tetrahedra, potentials = make_tetrahedral_cylinder(
            radius, length, resolution_hint)
        return RigidBody(cylinder2origin, vertices, tetrahedra, potentials)

    @staticmethod
    def make_capsule(capsule2origin, radius, height, resolution_hint=0.1):
        """Create capsule.

        Parameters
        ----------
        capsule2origin : array, shape (4, 4)
            Pose of the capsule.

        radius : float
            Radius of the capsule.

        height : float
            Height of the capsule.

        resolution_hint : float, optional (default: 0.1)
            Controls the fineness of the tetrahedral mesh. The coarsest mesh
            that produces desirable results will allow simulation to run as
            efficiently as possible. The circles of the cylinder and great
            circles of each hemisphere will have 2 * pi * radius /
            resolution_hint edges.

        Returns
        -------
        rigid_body : RigidBody
            Capsule.
        """
        vertices, tetrahedra, potentials = make_tetrahedral_capsule(
            radius, height, resolution_hint)
        return RigidBody(capsule2origin, vertices, tetrahedra, potentials)

    @property
    def tetrahedra_points(self):
        """All points of all tetrahedra in body frame."""
        if self._tetrahedra_points is None:
            self._tetrahedra_points = self.vertices_[self.tetrahedra_]
        return self._tetrahedra_points

    @property
    def tetrahedra_potentials(self):
        """Potentials of tetrahedra."""
        return self.potentials_[self.tetrahedra_]

    @property
    def com(self):
        """Center of mass."""
        if self._com is None:
            self._com = center_of_mass_tetrahedral_mesh(self.tetrahedra_points)
        return self._com

    def express_in(self, new_body2origin):
        """Express tetrahedral meshes in another frame.

        Vertices will be transformed so that they are expressed in the new
        frame.

        Parameters
        ----------
        new_body2origin : array, shape (4, 4)
            New frame.
        """
        origin2new_body = invert_transform(new_body2origin)
        body2new_body = np.dot(origin2new_body, self.body2origin_)
        self.vertices_ = transform_points(body2new_body, self.vertices_)
        self.body2origin_ = np.copy(new_body2origin)

        self._tetrahedra_points = None
        self._com = None
        self._aabbs = None
        self._aabb_tree = None

    def aabb(self):
        return self.aabb_tree.get_root_aabb()

    @property
    def aabbs(self):
        """The aabbs for broad phase collision detection"""
        if self._aabbs is None:
            self._aabbs = tetrahedral_mesh_aabbs(self.tetrahedra_points)
        return self._aabbs

    @property
    def aabb_tree(self):
        """The aabb_tree for broad phase collision detection"""
        if self._aabb_tree is None:
            self._aabb_tree = AabbTree()
            self._aabb_tree.insert_aabbs(self.aabbs, pre_insertion_methode="sort")
        return self._aabb_tree

    @property
    def youngs_modulus(self):
        """Get the young's modulus of the Object. (stiffness)"""
        return self._youngs_modulus

    @youngs_modulus.setter
    def youngs_modulus(self, value):
        """Set the young's modulus of the Object. (stiffness)"""
        self._youngs_modulus = value

    @property
    def artist_(self):
        if self._artist is None:
            self.make_artist()
        return self._artist

    def make_artist(self, c=None):
        self._artist = RigidBodyTetrahedralMesh(self.body2origin_, self.vertices_, self.tetrahedra_, c)

