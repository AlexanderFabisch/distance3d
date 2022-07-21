"""
===================
Visualize Icosphere
===================
"""
print(__doc__)

import numpy as np
import open3d as o3d
import pytransform3d.visualizer as pv
from distance3d import mesh, visualization, pressure_field, benchmark


mesh12origin = np.eye(4)
mesh22origin = np.eye(4)
vertices1_in_mesh1, tetrahedra1 = mesh.make_tetrahedral_icosphere(0.13 * np.ones(3), 0.15, 2)
vertices2_in_mesh2, tetrahedra2 = mesh.make_tetrahedral_icosphere(0.25 * np.ones(3), 0.15, 2)
#vertices2_in_mesh2, tetrahedra2 = io.load_tetrahedral_mesh("test/data/insole.vtk")

# The pressure function assigns to each point in the interior of the object
# a nonnegative real number representing the pressure at that point, which
# is an intuitive notion of how much resistance a foreign body protruding
# into the object would experience at that point.
# Source: https://www.ekzhang.com/assets/pdf/Hydroelastics.pdf
# TODO general distance to surface
potentials1 = np.zeros(len(vertices1_in_mesh1))
potentials1[-1] = 0.15
potentials2 = np.zeros(len(vertices2_in_mesh2))
potentials2[-1] = 0.15

timer = benchmark.Timer()
timer.start("contact_forces")
intersection, wrench12, wrench21, details = pressure_field.contact_forces(
    mesh12origin, vertices1_in_mesh1, tetrahedra1, potentials1,
    mesh22origin, vertices2_in_mesh2, tetrahedra2, potentials2,
    return_details=True)
print(f"time: {timer.stop('contact_forces')}")

print(f"force 1: {wrench12}")
print(f"force 2: {wrench21}")

fig = pv.figure()
fig.plot_transform(np.eye(4), s=0.1)
tetra_mesh1 = visualization.TetraMesh(mesh12origin, vertices1_in_mesh1, tetrahedra1)
tetra_mesh1.add_artist(fig)
tetra_mesh2 = visualization.TetraMesh(mesh22origin, vertices2_in_mesh2, tetrahedra2)
tetra_mesh2.add_artist(fig)

max_pressure1 = max(details["object1_pressures"])
max_pressure2 = max(details["object2_pressures"])
c1 = [(pressure / max_pressure1, 0, 0) for pressure in details["object1_pressures"]]
fig.scatter(details["object1_coms"], s=0.003, c=c1)
c2 = [(0, pressure / max_pressure2, 0) for pressure in details["object2_pressures"]]
fig.scatter(details["object2_coms"], s=0.003, c=c2)

for color, points in zip(c1, details["object1_polys"]):
    if len(points) == 3:
        triangles = np.array([[0, 1, 2], [2, 1, 0]], dtype=int)
    else:
        assert len(points) == 4
        triangles = np.array([[0, 1, 2], [2, 1, 0], [3, 1, 2], [2, 1, 3]], dtype=int)
    contact_surface_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(points),
        o3d.utility.Vector3iVector(triangles))
    contact_surface_mesh.paint_uniform_color(color)
    fig.add_geometry(contact_surface_mesh)

"""
for color, points in zip(c2, details["object2_polys"]):
    if len(points) == 3:
        triangles = np.array([[0, 1, 2], [2, 1, 0]], dtype=int)
    else:
        assert len(points) == 4
        triangles = np.array([[0, 1, 2], [2, 1, 0], [3, 1, 2], [2, 1, 3]], dtype=int)
    contact_surface_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(points),
        o3d.utility.Vector3iVector(triangles))
    contact_surface_mesh.paint_uniform_color(color)
    fig.add_geometry(contact_surface_mesh)
"""

fig.view_init()

if "__file__" in globals():
    fig.show()
else:
    fig.save_image("__open3d_rendered_image.jpg")
