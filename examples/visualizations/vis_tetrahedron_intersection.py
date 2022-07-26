"""
====================================
Visualize Intersection of Tetrahedra
====================================
"""
print(__doc__)
import numpy as np
import pytransform3d.visualizer as pv
from distance3d import mesh, pressure_field


vertices1, tetrahedra1 = mesh.make_tetrahedral_icosphere(np.array([0.1, 0.2, 0.1]), 1.0, order=2)
vertices2, tetrahedra2 = mesh.make_tetrahedral_icosphere(np.array([0.05, 0.15, 1.6]), 1.0, order=2)

tetrahedron1 = vertices1[tetrahedra1[257]]
tetrahedron2 = vertices2[tetrahedra2[310]]

epsilon1 = np.array([0.0, 0.0, 0.0, 1.0])
epsilon2 = np.array([0.0, 0.0, 0.0, 1.0])

contact_plane_hnf = pressure_field.contact_plane(
    tetrahedron1, tetrahedron2, epsilon1, epsilon2)
assert pressure_field.check_tetrahedra_intersect_contact_plane(
    tetrahedron1, tetrahedron2, contact_plane_hnf)
contact_polygon = pressure_field.compute_contact_polygon(
    tetrahedron1, tetrahedron2, contact_plane_hnf)

intersection_com, force_vector, _ = pressure_field.contact_force(
    tetrahedron1, epsilon1, contact_plane_hnf, contact_polygon)

fig = pv.figure()
fig.scatter(tetrahedron1, s=0.01, c=(1, 0, 0))
fig.scatter(tetrahedron2, s=0.01, c=(0, 0, 1))
fig.plot_transform(np.eye(4), s=0.05)
fig.plot_plane(normal=contact_plane_hnf[:3], d=contact_plane_hnf[3])
fig.scatter(contact_polygon, s=0.01, c=(1, 0, 1))
fig.plot_vector(intersection_com, 100.0 * force_vector, c=(1, 0, 0))
fig.view_init()

if "__file__" in globals():
    fig.show()
else:
    fig.save_image("__open3d_rendered_image.jpg")
