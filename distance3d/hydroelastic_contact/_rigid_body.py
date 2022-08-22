import numpy as np
from ..utils import transform_points, invert_transform
from ._tetra_mesh_creation import (
    make_tetrahedral_icosphere, make_tetrahedral_cube, make_tetrahedral_box)
from ._mesh_processing import center_of_mass_tetrahedral_mesh


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
    """
    def __init__(self, body2origin, vertices, tetrahedra, potentials):
        self.body2origin_ = body2origin
        self.vertices_ = vertices
        self.tetrahedra_ = tetrahedra
        self.potentials_ = potentials

        self._tetrahedra_points = None
        self._com = None

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
        vertices, tetrahedra, potentials = make_tetrahedral_icosphere(
            np.zeros(3), radius, order)
        return RigidBody(mesh2origin, vertices, tetrahedra, potentials)

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
            Pose of the cube.

        size : float
            Lengths of the edges in each dimension.

        Returns
        -------
        rigid_body : RigidBody
            Box.
        """
        vertices, tetrahedra, potentials = make_tetrahedral_box(size)
        return RigidBody(box2origin, vertices, tetrahedra, potentials)

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
