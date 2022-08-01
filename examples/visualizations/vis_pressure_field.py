"""
=================================================
Visualize Pressure Field of Two Colliding Objects
=================================================
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import pytransform3d.rotations as pr
import pytransform3d.visualizer as pv
from distance3d import mesh, visualization, pressure_field, benchmark


highlight_isect_idx = None

# The pressure function assigns to each point in the interior of the object
# a nonnegative real number representing the pressure at that point, which
# is an intuitive notion of how much resistance a foreign body protruding
# into the object would experience at that point.
# Source: https://www.ekzhang.com/assets/pdf/Hydroelastics.pdf
mesh12origin = np.eye(4)
vertices1, tetrahedra1 = mesh.make_tetrahedral_icosphere(0.13 * np.ones(3), 0.15, 2)
# TODO general distance to surface
potentials1 = np.zeros(len(vertices1))
potentials1[-1] = 0.15
mesh22origin = np.eye(4)
mesh22origin[:3, :3] = pr.active_matrix_from_extrinsic_euler_zyx([0.1, 0.3, 0.5])
mesh22origin[:3, 3] = 0.25 * np.ones(3)
vertices2, tetrahedra2, potentials2 = mesh.make_tetrahedral_cube(0.15)

timer = benchmark.Timer()
timer.start("contact_forces")
intersection, wrench12, wrench21, details = pressure_field.contact_forces(
    mesh12origin, vertices1, tetrahedra1, potentials1,
    mesh22origin, vertices2, tetrahedra2, potentials2,
    return_details=True)
print(f"time: {timer.stop('contact_forces')}")

print(f"force 12: {np.round(wrench12, 8)}")
print(f"force 21: {np.round(wrench21, 8)}")

fig = pv.figure()
fig.plot_transform(np.eye(4), s=0.1)
visualization.TetraMesh(mesh12origin, vertices1, tetrahedra1).add_artist(fig)
visualization.TetraMesh(mesh22origin, vertices2, tetrahedra2).add_artist(fig)

if highlight_isect_idx is not None:
    fig.plot_plane(normal=details["plane_normals"][highlight_isect_idx],
                   point_in_plane=details["contact_coms"][highlight_isect_idx], s=0.05)
    fig.scatter(details["contact_polygons"][highlight_isect_idx], s=0.001, c=(1, 0, 1))
    fig.scatter(details["intersecting_tetrahedra1"][highlight_isect_idx], s=0.001, c=(1, 0, 0))
    fig.scatter(details["intersecting_tetrahedra2"][highlight_isect_idx], s=0.001, c=(0, 0, 1))
    visualization.Tetrahedron(details["intersecting_tetrahedra1"][highlight_isect_idx]).add_artist(fig)
    visualization.Tetrahedron(details["intersecting_tetrahedra2"][highlight_isect_idx]).add_artist(fig)

fig.plot_vector(details["contact_point"], 100 * wrench21[:3], (1, 0, 0))
fig.plot_vector(details["contact_point"], 100 * wrench12[:3], (0, 1, 0))

pressures = np.linalg.norm(details["contact_forces"], axis=1) / np.asarray(details["contact_areas"])
max_pressure = max(pressures)
cmap = plt.get_cmap("plasma")
colors = [cmap(pressure / max_pressure)[:3] for pressure in pressures]
contact_polygons = zip(colors, details["contact_polygons"], details["contact_polygon_triangles"])
for color, points, triangles in contact_polygons:
    triangles = np.vstack((triangles, triangles[:, ::-1]))
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
