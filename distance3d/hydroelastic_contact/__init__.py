"""Hydroelastic contact model for contact wrenches of rigid bodies.

The original name for this model is pressure field model. It has been
presented in:

R. Elandt, E. Drumwright, M. Sherman and A. Ruina: A pressure field model for
fast, robust approximation of net contact force and moment between nominally
rigid objects, IEEE/RSJ International Conference on Intelligent Robots and
Systems (IROS), 2019, pp. 8238-8245, doi: 10.1109/IROS40897.2019.8968548.

The implementation is different, but is inspired by:

V. Huang, F. Wang, E. Zhang: An Efficient Implementation of Pressure Field
Models, https://www.ekzhang.com/assets/pdf/Hydroelastics.pdf.
"""
from ._interface import contact_forces, find_contact_surface
from ._rigid_body import RigidBody
from ._contact_surface import ContactSurface

from ._halfplanes import (
    intersect_halfplanes, plot_halfplanes_and_intersections)
from ._mesh_processing import (
    center_of_mass_tetrahedral_mesh, tetrahedral_mesh_aabbs,
    tetrahedral_mesh_volumes)
from ._barycentric_transform import barycentric_transforms
from ._tetrahedron_intersection import intersect_tetrahedron_pair
from ._forces import compute_contact_force
from ._broad_phase import HydroelasticBoundingVolumeHierarchy


__all__ = [
    "contact_forces", "find_contact_surface", "RigidBody", "ContactSurface","HydroelasticBoundingVolumeHierarchy",
    # exported for unit tests and specific examples:
    "intersect_halfplanes", "plot_halfplanes_and_intersections",
    "center_of_mass_tetrahedral_mesh", "tetrahedral_mesh_aabbs",
    "tetrahedral_mesh_volumes", "barycentric_transforms",
    "intersect_tetrahedron_pair", "compute_contact_force",
]
