from ..utils import transform_points
from ._mesh_processing import center_of_mass_tetrahedral_mesh


class RigidBody:
    def __init__(self, mesh2origin, vertices_in_mesh, tetrahedra, potentials):
        self.mesh2origin = mesh2origin
        self.vertices_in_mesh = vertices_in_mesh
        self.tetrahedra = tetrahedra
        self.potentials = potentials

        self._tetrahedra_points = None
        self._com = None

    @property
    def tetrahedra_points(self):
        if self._tetrahedra_points is None:
            self._tetrahedra_points = self.vertices_in_mesh[self.tetrahedra]
        return self._tetrahedra_points

    @property
    def tetrahedra_potentials(self):
        return self.potentials[self.tetrahedra]

    @property
    def com(self):
        if self._com is None:
            self._com = center_of_mass_tetrahedral_mesh(self.tetrahedra_points)
        return self._com

    def transform(self, old2new):
        self.vertices_in_mesh = transform_points(old2new, self.vertices_in_mesh)
        self._tetrahedra_points = None
        self._com = None
