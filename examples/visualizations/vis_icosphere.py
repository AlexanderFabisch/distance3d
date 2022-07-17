"""
===================
Visualize Icosphere
===================
"""
print(__doc__)
import numpy as np
import pytransform3d.visualizer as pv
from distance3d import mesh, visualization, benchmark, mpr, colliders, geometry, io


timer = benchmark.Timer()
timer.start("make_tetrahedral_icosphere")
vertices1, tetrahedra1 = mesh.make_tetrahedral_icosphere(0.1 * np.ones(3), 0.15, 2)
vertices2, tetrahedra2 = mesh.make_tetrahedral_icosphere(0.25 * np.ones(3), 0.15, 2)
print(timer.stop("make_tetrahedral_icosphere"))
#vertices2, tetrahedra2 = io.load_tetrahedral_mesh("test/data/insole.vtk")

c1 = colliders.ConvexHullVertices(vertices1)
c2 = colliders.ConvexHullVertices(vertices2)
timer.start("mpr_penetration")
intersection, depth, normal, contact_point = mpr.mpr_penetration(c1, c2)
assert intersection
print(timer.stop("mpr_penetration"))

# TODO mesh2origin

# TODO refactor

def points_to_plane_signed(points, plane_point, plane_normal):
    return np.dot(points - plane_point.reshape(1, -1), plane_normal)

def intersecting_tetrahedra(vertices, tetrahedra):
    d = points_to_plane_signed(vertices, contact_point, normal)[tetrahedra]
    mins = np.min(d, axis=1)
    maxs = np.max(d, axis=1)
    return np.where(np.sign(mins) != np.sign(maxs))[0]

from distance3d.distance import point_to_plane
def point_in_plane(plane_point, plane_normal, tetrahedron_points):  # TODO triangle projection?
    return np.mean(np.array([point_to_plane(p, plane_point, plane_normal)[1]
                             for p in tetrahedron_points]), axis=0)

potentials1 = np.zeros(len(vertices1))
potentials1[-1] = 0.15

potentials2 = np.zeros(len(vertices1))
potentials2[-1] = 0.15

indices1 = intersecting_tetrahedra(vertices1, tetrahedra1)
tetras1 = vertices1[tetrahedra1[indices1]]
tetras1_on_plane = np.array([point_in_plane(contact_point, normal, t) for t in tetras1])
coords1 = np.array([geometry.barycentric_coordinates_tetrahedron(p, t) for p, t in zip(tetras1_on_plane, tetras1)])
pressure1 = np.array([np.dot(c, potentials1[t]) for c, t in zip(coords1, tetrahedra1)])
force1 = np.sum(pressure1)
print(force1)
indices2 = intersecting_tetrahedra(vertices2, tetrahedra2)
tetras2 = vertices2[tetrahedra2[indices2]]
tetras2_on_plane = np.array([point_in_plane(contact_point, normal, t) for t in tetras2])
coords2 = np.array([geometry.barycentric_coordinates_tetrahedron(p, t) for p, t in zip(tetras2_on_plane, tetras2)])
pressure2 = np.array([np.dot(c, potentials2[t]) for c, t in zip(coords2, tetrahedra2)])
force2 = np.sum(pressure2)
print(force2)

# TODO something with halfplanes?

timer.start("center_of_mass_tetrahedral_mesh")
com = mesh.center_of_mass_tetrahedral_mesh(np.eye(4), vertices1, tetrahedra1)
print(timer.stop("center_of_mass_tetrahedral_mesh"))

timer.start("tetrahedral_mesh_aabbs")
aabbs = mesh.tetrahedral_mesh_aabbs(np.eye(4), vertices1, tetrahedra1)
print(timer.stop("tetrahedral_mesh_aabbs"))

fig = pv.figure()
fig.plot_transform(np.eye(4), s=0.1)
tetra_mesh1 = visualization.TetraMesh(np.eye(4), vertices1, tetrahedra1)
tetra_mesh1.add_artist(fig)
tetra_mesh2 = visualization.TetraMesh(np.eye(4), vertices2, tetrahedra2)
tetra_mesh2.add_artist(fig)
#fig.plot_plane(normal=normal, point_in_plane=contact_point)
fig.scatter(tetras1_on_plane, s=0.003, c=(1, 0, 0))
fig.scatter(tetras2_on_plane, s=0.003, c=(0, 1, 0))
fig.view_init()

if "__file__" in globals():
    fig.show()
else:
    fig.save_image("__open3d_rendered_image.jpg")
