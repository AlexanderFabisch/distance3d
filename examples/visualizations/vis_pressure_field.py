"""
=================================================
Visualize Pressure Field of Two Colliding Objects
=================================================
"""
print(__doc__)

import pprint
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import pytransform3d.rotations as pr
import pytransform3d.visualizer as pv
from distance3d import visualization, pressure_field, benchmark, utils


highlight_isect_idx = None
show_broad_phase = False

# The pressure function assigns to each point in the interior of the object
# a nonnegative real number representing the pressure at that point, which
# is an intuitive notion of how much resistance a foreign body protruding
# into the object would experience at that point.
# Source: https://www.ekzhang.com/assets/pdf/Hydroelastics.pdf
rigid_body1 = pressure_field.RigidBody.make_sphere(0.13 * np.ones(3), 0.15, 2)
rigid_body2 = pressure_field.RigidBody.make_sphere(0.25 * np.ones(3), 0.15, 2)
#cube2origin = np.eye(4)
#cube2origin[:3, :3] = pr.active_matrix_from_extrinsic_euler_zyx([0.1, 0.3, 0.5])
#cube2origin[:3, 3] = 0.25 * np.ones(3)
#rigid_body2 = pressure_field.RigidBody.make_cube(cube2origin, 0.15)

timer = benchmark.Timer()
timer.start("contact_forces")
intersection, wrench12, wrench21, details = pressure_field.contact_forces(
    rigid_body1, rigid_body2, return_details=True, timer=timer)
pprint.pprint(timer.total_time_)
print(f"time: {timer.stop('contact_forces')}")

assert intersection

print(f"force 12: {np.round(wrench12, 8)}")
print(f"force 21: {np.round(wrench21, 8)}")

fig = pv.figure()
fig.plot_transform(np.eye(4), s=0.1)
visualization.TetraMesh(rigid_body1.body2origin_, rigid_body1.vertices_, rigid_body1.tetrahedra_).add_artist(fig)
visualization.TetraMesh(rigid_body2.body2origin_, rigid_body2.vertices_, rigid_body2.tetrahedra_).add_artist(fig)

if show_broad_phase:
    broad_overlapping_indices1, broad_overlapping_indices2, broad_pairs = pressure_field.broad_phase_tetrahedra(
        rigid_body1, rigid_body2)
    for i in np.unique(broad_overlapping_indices1):
        tetrahedron_points1 = rigid_body1.tetrahedra_points[i].dot(
            rigid_body1.body2origin_[:3, :3].T) + rigid_body1.body2origin_[:3, 3]
        visualization.Tetrahedron(tetrahedron_points1, c=(1, 0, 0)).add_artist(fig)
    for j in np.unique(broad_overlapping_indices2):
        tetrahedron_points2 = rigid_body2.tetrahedra_points[j].dot(
            rigid_body2.body2origin_[:3, :3].T) + rigid_body2.body2origin_[:3, 3]
        visualization.Tetrahedron(tetrahedron_points2, c=(1, 0, 0)).add_artist(fig)

if highlight_isect_idx is not None:
    fig.plot_plane(normal=details["contact_planes"][highlight_isect_idx, :3],
                   d=details["contact_planes"][highlight_isect_idx, -1], s=0.15)
    fig.scatter(details["contact_polygons"][highlight_isect_idx], s=0.001, c=(1, 0, 1))
    fig.scatter(details["intersecting_tetrahedra1"][highlight_isect_idx], s=0.001, c=(1, 0, 0))
    fig.scatter(details["intersecting_tetrahedra2"][highlight_isect_idx], s=0.001, c=(0, 0, 1))
    fig.scatter([details["contact_coms"][highlight_isect_idx]], s=0.002, c=(1, 0, 1))
    #visualization.Tetrahedron(details["intersecting_tetrahedra1"][highlight_isect_idx]).add_artist(fig)
    #visualization.Tetrahedron(details["intersecting_tetrahedra2"][highlight_isect_idx]).add_artist(fig)

fig.plot_vector(details["contact_point"], 100 * wrench21[:3], (1, 0, 0))
fig.plot_vector(details["contact_point"], 100 * wrench12[:3], (0, 1, 0))

max_pressure = max(details["pressures"])
cmap = plt.get_cmap("plasma")
colors = [cmap(pressure / max_pressure)[:3] for pressure in details["pressures"]]
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
