"""Hydroelastic contact model for contact wrenches of rigid bodies.

The original name for this model is pressure field model.
"""
from ._interface import contact_forces, find_contact_surface
from ._rigid_body import RigidBody
from ._contact_surface import ContactSurface

from ._halfplanes import intersect_halfplanes, plot_halfplanes_and_intersections
from ._tetra_mesh_creation import make_tetrahedral_icosphere
from ._mesh_processing import center_of_mass_tetrahedral_mesh, tetrahedral_mesh_aabbs
from ._barycentric_transform import barycentric_transforms
from ._tetrahedron_intersection import intersect_tetrahedron_pair
from ._forces import compute_contact_force
from ._broad_phase import broad_phase_tetrahedra


__all__ = [
    "contact_forces", "find_contact_surface", "RigidBody", "ContactSurface",
    # exported for unit tests and specific examples:
    "intersect_halfplanes", "plot_halfplanes_and_intersections",
    "make_tetrahedral_icosphere", "center_of_mass_tetrahedral_mesh",
    "tetrahedral_mesh_aabbs", "barycentric_transforms",
    "intersect_tetrahedron_pair", "compute_contact_force",
    "broad_phase_tetrahedra"
]
