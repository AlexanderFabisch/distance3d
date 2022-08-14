import numpy as np
from ..utils import transform_points, invert_transform
from ._tetra_mesh_creation import (
    make_tetrahedral_icosphere, make_tetrahedral_cube)
from ._mesh_processing import center_of_mass_tetrahedral_mesh


class RigidBody:
    def __init__(self, mesh2origin, vertices, tetrahedra, potentials):
        self.body2origin_ = mesh2origin
        self.vertices_ = vertices
        self.tetrahedra_ = tetrahedra
        self.potentials_ = potentials

        self._tetrahedra_points = None
        self._com = None

    @staticmethod
    def make_sphere(center, radius, order=4):
        mesh2origin = np.eye(4)
        mesh2origin[:3, 3] = center
        vertices, tetrahedra, potentials = make_tetrahedral_icosphere(
            np.zeros(3), radius, order)
        return RigidBody(mesh2origin, vertices, tetrahedra, potentials)

    @staticmethod
    def make_cube(cube2origin, size):
        vertices, tetrahedra, potentials = make_tetrahedral_cube(size)
        return RigidBody(cube2origin, vertices, tetrahedra, potentials)

    @property
    def tetrahedra_points(self):
        if self._tetrahedra_points is None:
            self._tetrahedra_points = self.vertices_[self.tetrahedra_]
        return self._tetrahedra_points

    @property
    def tetrahedra_potentials(self):
        return self.potentials_[self.tetrahedra_]

    @property
    def com(self):
        if self._com is None:
            self._com = center_of_mass_tetrahedral_mesh(self.tetrahedra_points)
        return self._com

    def express_in(self, new_body2origin):
        origin2new_body = invert_transform(new_body2origin)
        body2new_body = np.dot(origin2new_body, self.body2origin_)
        self.vertices_ = transform_points(body2new_body, self.vertices_)
        self.body2origin_ = np.copy(new_body2origin)

        self._tetrahedra_points = None
        self._com = None
