"""
===================
Visualize Icosphere
===================
"""
print(__doc__)

import numpy as np
import open3d as o3d
import pytransform3d.visualizer as pv
from distance3d import mesh, visualization, pressure_field, benchmark, io


mesh12origin = np.eye(4)
mesh22origin = np.eye(4)
vertices1, tetrahedra1 = mesh.make_tetrahedral_icosphere(0.13 * np.ones(3), 0.15, 2)
vertices2, tetrahedra2 = mesh.make_tetrahedral_icosphere(0.25 * np.ones(3), 0.15, 2)
#vertices2_in_mesh2, tetrahedra2 = io.load_tetrahedral_mesh("test/data/insole.vtk")

# The pressure function assigns to each point in the interior of the object
# a nonnegative real number representing the pressure at that point, which
# is an intuitive notion of how much resistance a foreign body protruding
# into the object would experience at that point.
# Source: https://www.ekzhang.com/assets/pdf/Hydroelastics.pdf
# TODO general distance to surface
potentials1 = np.zeros(len(vertices1))
potentials1[-1] = 0.15
potentials2 = np.zeros(len(vertices2))
potentials2[-1] = 0.15

timer = benchmark.Timer()
timer.start("contact_forces")
intersection, wrench12, wrench21, details = pressure_field.contact_forces(
    mesh12origin, vertices1, tetrahedra1, potentials1,
    mesh22origin, vertices2, tetrahedra2, potentials2,
    return_details=True)
print(f"time: {timer.stop('contact_forces')}")

print(f"force 12: {wrench12}")
print(f"force 21: {wrench21}")

fig = pv.figure()
fig.plot_transform(np.eye(4), s=0.1)
visualization.TetraMesh(mesh12origin, vertices1, tetrahedra1).add_artist(fig)
visualization.TetraMesh(mesh22origin, vertices2, tetrahedra2).add_artist(fig)

isect_idx = 3
fig.scatter(details["contact_polygons"][isect_idx], s=0.001, c=(1, 0, 1))
fig.scatter(details["intersecting_tetrahedra1"][isect_idx], s=0.001, c=(1, 0, 0))
fig.scatter(details["intersecting_tetrahedra2"][isect_idx], s=0.001, c=(0, 0, 1))
#visualization.Tetrahedron(details["intersecting_tetrahedra1"][isect_idx]).add_artist(fig)
#visualization.Tetrahedron(details["intersecting_tetrahedra2"][isect_idx]).add_artist(fig)

pressures = np.linalg.norm(details["contact_forces"], axis=1) / np.asarray(details["contact_areas"])
max_pressure = max(pressures)
c = [(pressure / max_pressure, 0, 0) for pressure in pressures]
#fig.scatter(details["contact_coms"], s=0.003, c=c)

if np.linalg.norm(wrench21) > 0:
    fig.plot_vector(details["contact_point"], 1000 * wrench21[:3], (1, 0, 0))
if np.linalg.norm(wrench12) > 0:
    fig.plot_vector(details["contact_point"], 1000 * wrench12[:3], (0, 1, 0))

contact_polygons = zip(c, details["contact_polygons"])
contact_polygons = list(contact_polygons)[isect_idx:isect_idx + 1]
for color, points in contact_polygons:
    triangles = np.array([[0, i - 1, i] for i in range(2, len(points))], dtype=int)
    contact_surface_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(points),
        o3d.utility.Vector3iVector(triangles))
    contact_surface_mesh.paint_uniform_color(color)
    fig.add_geometry(contact_surface_mesh)

fig.view_init()

if "__file__" in globals():
    fig.show()
else:
    fig.save_image("__open3d_rendered_image.jpg")
