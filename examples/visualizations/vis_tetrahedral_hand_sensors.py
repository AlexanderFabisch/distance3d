import os
from pytransform3d.urdf import UrdfTransformManager

import numpy as np
import pytransform3d.visualizer as pv
from distance3d import visualization, hydroelastic_contact, benchmark, broad_phase

from distance3d.hydroelastic_contact._broad_phase import HydroelasticBoundingVolumeHierarchy

GPa = 100000000

fig = pv.figure()

# Hand
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

finger_angle = [0, 0, 0, 0.1, 0.9]
joint_names = ["j_index_fle", "j_mrl_fle", "j_ring_fle", "j_little_fle", "j_thumb_fle"]

for i, joint_name in enumerate(joint_names):
    tm.set_joint(joint_name, finger_angle[i])

robot_bvh = HydroelasticBoundingVolumeHierarchy(tm, "mia_hand")
robot_bvh.fill_tree_with_colliders(tm, make_artists=True)

for rb in robot_bvh.get_colliders():
    rb.youngs_modulus = 1 * GPa
    rb.artist_.add_artist(fig)

# Box
box2origin = np.eye(4)
box2origin[:3, 3] = np.array([-0.04, 0.07, -0.04])
box_rb = hydroelastic_contact.RigidBody.make_box(box2origin, np.array([0.03, 0.1, 0.15]))
box_rb.youngs_modulus = 1 * GPa
artist = visualization.RigidBodyTetrahedralMesh(
        box_rb.body2origin_, box_rb.vertices_, box_rb.tetrahedra_)
artist.add_artist(fig)

finger_force = [np.zeros(3),np.zeros(3),np.zeros(3)]

# Friction force calculations
for frame in robot_bvh.colliders_:
    hand_rb = robot_bvh.colliders_[frame]

    intersection, wrench12, wrench21, details = hydroelastic_contact.contact_forces(
        hand_rb, box_rb, return_details=True)

    if not intersection:
        continue

    contact_surface = visualization.ContactSurface(
        np.eye(4), details["contact_polygons"],
        details["contact_polygon_triangles"], details["pressures"])
    contact_surface.add_artist(fig)

    if "index" in frame:
        finger_force[0] += wrench12[:3]

    if "middle" in frame:
        finger_force[1] += wrench12[:3]

    if "thumb" in frame:
        finger_force[2] += wrench12[:3]

MM_PER_M = 1000.0

F_index_norm = (finger_force[0][0] * MM_PER_M) / 2.854
F_index_tang = (finger_force[0][1] * MM_PER_M) / 2.006

F_mrl_norm = (finger_force[1][0] * MM_PER_M) / 2.854
F_mrl_tang = (finger_force[1][1] * MM_PER_M) / 2.006

F_thumb_norm = (finger_force[2][0]) / 0.056
F_thumb_tang = (finger_force[2][1] * MM_PER_M) / 3.673

print(f"F_index_norm: {F_index_norm}")
print(f"F_index_tang: {F_index_tang}")

print(f"F_mrl_norm: {F_mrl_norm}")
print(f"F_mrl_tang: {F_mrl_tang}")

print(f"F_thumb_norm: {F_thumb_norm}")
print(f"F_thumb_tang: {F_thumb_tang}")

fig.view_init()
fig.show()
