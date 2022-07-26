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

contact_plane_hnf = pressure_field.contact_plane(tetrahedron1, tetrahedron2, epsilon1, epsilon2)
assert pressure_field.check_tetrahedra_intersect_contact_plane(tetrahedron1, tetrahedron2, contact_plane_hnf)
poly3d = pressure_field.contact_polygon(tetrahedron1, tetrahedron2, contact_plane_hnf)
normal = contact_plane_hnf[:3]
#assert np.dot(normal, mesh12world[:3, :3].dot(com1) + mesh12world[:3, 3]) - np.dot(normal, mesh22world[:3, :3].dot(com2) + mesh22world[:3, 3]) >= 0.0  # Otherwise normal *= -1

total_force = 0.0
intersection_com = np.zeros(3)
total_area = 0.0

# TODO transform to world
t1 = np.vstack((tetrahedron1.T, np.ones((1, 4))))
for i in range(2, len(poly3d)):
    vertices = poly3d[np.array([0, i - 1, i], dtype=int)]
    com = np.hstack((np.mean(vertices, axis=0), (1,)))
    res = np.linalg.solve(t1, com)
    pressure = sum(res * epsilon1)
    area = 0.5 * np.linalg.norm(np.cross(vertices[0] - vertices[1], vertices[0] - vertices[2]))
    total_force += pressure * area
    total_area += area
    intersection_com += area * com[:3]

intersection_com /= total_area
force_vector = total_force * normal

fig = pv.figure()
fig.scatter(tetrahedron1, s=0.01, c=(1, 0, 0))
fig.scatter(tetrahedron2, s=0.01, c=(0, 0, 1))
fig.plot_transform(np.eye(4), s=0.05)
fig.plot_plane(normal=contact_plane_hnf[:3], d=contact_plane_hnf[3])
fig.scatter(poly3d, s=0.01, c=(1, 0, 1))
fig.plot_vector(intersection_com, 100.0 * force_vector, c=(1, 0, 0))
fig.view_init()

if "__file__" in globals():
    fig.show()
else:
    fig.save_image("__open3d_rendered_image.jpg")
