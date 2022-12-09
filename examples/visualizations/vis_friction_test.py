"""
=================================================
Visualize Pressure Field of Two Colliding Objects
=================================================
"""
from distance3d.hydroelastic_contact._broad_phase import HydroelasticBoundingVolumeHierarchy

print(__doc__)

import os
from pytransform3d.urdf import UrdfTransformManager
import distance3d
import numpy as np
import pytransform3d.visualizer as pv
from distance3d import visualization, hydroelastic_contact, benchmark, broad_phase


show_normals = True

g = np.array([0, 9.81, 0])
ball_mass = 1
my_static = 0.2
GPa = 100000000
finger_angle = [1, 1, 1, 1, 0.2]

joint_names = ["j_index_fle", "j_mrl_fle", "j_ring_fle", "j_little_fle", "j_thumb_fle"]

F_g = g * ball_mass

fig = pv.figure()


BASE_DIR = "test/data/"
data_dir = BASE_DIR
search_path = ".."
while (not os.path.exists(data_dir) and
       os.path.dirname(search_path) != "distance3d"):
    search_path = os.path.join(search_path, "..")
    data_dir = os.path.join(search_path, BASE_DIR)

tm = UrdfTransformManager()
filename = os.path.join(data_dir, "mia_hand_description/urdf/mia_hand.urdf")
with open(filename, "r") as f:
    robot_urdf = f.read()
    tm.load_urdf(robot_urdf, mesh_path=data_dir)

for i, joint_name in enumerate(joint_names):
    tm.set_joint(joint_name, finger_angle[i])

robot_bvh = HydroelasticBoundingVolumeHierarchy(tm, "mia_hand")
robot_bvh.fill_tree_with_colliders(tm, make_artists=True)

sphere_rb = hydroelastic_contact.RigidBody.make_sphere(np.array([-0.03, 0.05, -0.01]), 0.03, 1)
sphere_rb.youngs_modulus = 100 * GPa
artist = visualization.RigidBodyTetrahedralMesh(
        sphere_rb.body2origin_, sphere_rb.vertices_, sphere_rb.tetrahedra_)
artist.add_artist(fig)

contact_forces = []
contact_norms = []
contact_force_sum = np.array([0,0,0], dtype=float)
for hand_rb in robot_bvh.get_colliders():
    hand_rb.youngs_modulus = 100 * GPa
    hand_rb.artist_.add_artist(fig)

    intersection, wrench12, wrench21, details = hydroelastic_contact.contact_forces(
        hand_rb, sphere_rb, return_details=True)

    if not intersection:
        continue

    for i, force in enumerate(details["contact_forces"]):
        polygon = details["contact_polygons"][i]

        pos = (polygon[0] + polygon[1] + polygon[2]) / 3
        force_len = np.linalg.norm(force)
        normal = (-force / force_len)

        contact_forces.append(-force)
        contact_norms.append(normal)
        contact_force_sum += -force

        if show_normals:
            fig.plot_vector(pos, 0.01* normal, (1, 0, 0))

    contact_surface = visualization.ContactSurface(
        np.eye(4), details["contact_polygons"],
        details["contact_polygon_triangles"], details["pressures"])
    contact_surface.add_artist(fig)

contact_force_sum_len = np.linalg.norm(contact_force_sum)

# contact_norms.append(F_g)
fig.plot_vector(sphere_rb.body2origin_[:3, 3], 0.000001 * contact_force_sum, (0, 1, 0))

F_f_sum = 0
for i in range(len(contact_norms)):
    normal = contact_norms[i]
    angle = np.arccos(np.dot(normal, contact_force_sum) /
                      (np.linalg.norm(normal) * np.linalg.norm(contact_force_sum)))

    if np.degrees(angle) > 90:
        continue

    F_f = contact_forces[i] * my_static / np.sin(angle)
    fig.plot_vector(sphere_rb.body2origin_[:3, 3], 0.000001 * normal*F_f, (0, 0, 1))

    F_f_sum += np.linalg.norm(F_f)

print(f"Sliding: {F_f_sum < contact_force_sum_len}")

fig.view_init()

fig.show()


