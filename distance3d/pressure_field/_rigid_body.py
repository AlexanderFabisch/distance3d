import numpy as np
from ..utils import transform_points, invert_transform
from ._mesh_processing import center_of_mass_tetrahedral_mesh


class RigidBody:
    def __init__(self, mesh2origin, vertices, tetrahedra, potentials):
        self.body2origin_ = mesh2origin
        self.vertices_ = vertices
        self.tetrahedra_ = tetrahedra
        self.potentials_ = potentials

        self._tetrahedra_points = None
        self._com = None

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

    def transform(self, old2new):
        self.body2origin_ = np.dot(self.body2origin_, invert_transform(old2new))
        self.vertices_ = transform_points(old2new, self.vertices_)
        self._tetrahedra_points = None
        self._com = None
