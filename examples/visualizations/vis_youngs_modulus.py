"""
=======================================================
Visualize pressure fields with different young's moduli
=======================================================
"""
print(__doc__)

import numpy as np
import pytransform3d.visualizer as pv
from distance3d import visualization, hydroelastic_contact, benchmark

rigid_body1 = hydroelastic_contact.RigidBody.make_sphere(np.array([0.5, 0, 0]), 0.15, 2)
rigid_body2 = hydroelastic_contact.RigidBody.make_sphere(np.array([0.5, 0.12, 0]), 0.15, 2)

rigid_body1.youngs_modulus = 10
rigid_body2.youngs_modulus = 0.1

rigid_body3 = hydroelastic_contact.RigidBody.make_sphere(np.array([0, 0, 0]), 0.15, 2)
rigid_body4 = hydroelastic_contact.RigidBody.make_sphere(np.array([0, 0.12, 0]), 0.15, 2)

rigid_body3.youngs_modulus = 10
rigid_body4.youngs_modulus = 15

rigid_body5 = hydroelastic_contact.RigidBody.make_sphere(np.array([-0.5, 0, 0]), 0.15, 2)
rigid_body6 = hydroelastic_contact.RigidBody.make_sphere(np.array([-0.5, 0.12, 0]), 0.15, 2)

rigid_body5.youngs_modulus = 0.1
rigid_body6.youngs_modulus = 0.2

timer = benchmark.Timer()
timer.start("contact_forces")

intersection1, _, _, details1 = hydroelastic_contact.contact_forces(rigid_body1, rigid_body2, return_details=True)
intersection2, _, _, details2 = hydroelastic_contact.contact_forces(rigid_body3, rigid_body4, return_details=True)
intersection3, _, _, details3 = hydroelastic_contact.contact_forces(rigid_body5, rigid_body6, return_details=True)

print(f"time: {timer.stop('contact_forces')}")

fig = pv.figure()
fig.plot_transform(np.eye(4), s=0.1)


def vis_mesh(rigid_body):
    visualization.RigidBodyTetrahedralMesh(
        rigid_body.body2origin_, rigid_body.vertices_, rigid_body.tetrahedra_).add_artist(fig)


vis_mesh(rigid_body1)
vis_mesh(rigid_body2)
vis_mesh(rigid_body3)
vis_mesh(rigid_body4)
vis_mesh(rigid_body5)
vis_mesh(rigid_body6)


def vis_contact_surface(details):
    contact_surface = visualization.ContactSurface(
        np.eye(4), details["contact_polygons"],
        details["contact_polygon_triangles"], details["pressures"])
    contact_surface.add_artist(fig)


vis_contact_surface(details1)
vis_contact_surface(details2)
vis_contact_surface(details3)

fig.view_init()

if "__file__" in globals():
    fig.show()
else:
    fig.save_image("__open3d_rendered_image.jpg")
