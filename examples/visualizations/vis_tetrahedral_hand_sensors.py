import os
from pytransform3d.urdf import UrdfTransformManager

import numpy as np
import pytransform3d.visualizer as pv
from distance3d import visualization, hydroelastic_contact, benchmark, broad_phase

from distance3d.hydroelastic_contact._broad_phase import HydroelasticBoundingVolumeHierarchy

GPa = 100000000
MPa = 1000000

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

fig = pv.figure()

# ------ Load Hand ------
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

finger_angle = [0, 0, 0, 0, 1]
joint_names = ["j_index_fle", "j_mrl_fle", "j_ring_fle", "j_little_fle", "j_thumb_fle"]

for i, joint_name in enumerate(joint_names):
    tm.set_joint(joint_name, finger_angle[i])

robot_bvh = HydroelasticBoundingVolumeHierarchy(tm, "mia_hand")
robot_bvh.fill_tree_with_colliders(tm, make_artists=True)

for rb in robot_bvh.get_colliders():
    rb.youngs_modulus = 1 * GPa
    rb.artist_.add_artist(fig)


# ------ Load Box ------
box2origin = np.eye(4)
box2origin[:3, 3] = np.array([-0.037, 0.07, -0.04])
box_rb = hydroelastic_contact.RigidBody.make_box(box2origin, np.array([0.03, 0.1, 0.15]))
box_rb.youngs_modulus = 0.04 * MPa
artist = visualization.RigidBodyTetrahedralMesh(box_rb.body2origin_, box_rb.vertices_, box_rb.tetrahedra_)
artist.add_artist(fig)



# ------ Extract Finger directions ------
index_pre_sensor = robot_bvh.colliders_['collision:index_fle/collision_index_pre_sensor']
middle_pre_sensor = robot_bvh.colliders_['collision:middle_fle/collision_middle_pre_sensor']
thumb_proximal = robot_bvh.colliders_['collision:thumb_fle/collision_thumb_proximal']

finger_pos = np.array([index_pre_sensor.body2origin_[:3, 3],
                       middle_pre_sensor.body2origin_[:3, 3],
                       thumb_proximal.body2origin_[:3, 3]])

finger_dir = np.array([-index_pre_sensor.body2origin_[:3, 2],
                       -middle_pre_sensor.body2origin_[:3, 2],
                       -thumb_proximal.body2origin_[:3, 2]])

finger_norm_dir = np.array([index_pre_sensor.body2origin_[:3, 1],
                            middle_pre_sensor.body2origin_[:3, 1],
                            -thumb_proximal.body2origin_[:3, 0]])

finger_tang_dir = np.array([index_pre_sensor.body2origin_[:3, 0],
                            middle_pre_sensor.body2origin_[:3, 0],
                            thumb_proximal.body2origin_[:3, 1]])

# ------ Plot Finger directions ------
fig.plot_vector(finger_pos[0], 0.01 * finger_dir[0], (1, 0, 0))
fig.plot_vector(finger_pos[1], 0.01 * finger_dir[1], (1, 0, 0))
fig.plot_vector(finger_pos[2], 0.01 * finger_dir[2], (1, 0, 0))

fig.plot_vector(finger_pos[0], 0.01 * finger_norm_dir[0], (0, 1, 0))
fig.plot_vector(finger_pos[1], 0.01 * finger_norm_dir[1], (0, 1, 0))
fig.plot_vector(finger_pos[2], 0.01 * finger_norm_dir[2], (0, 1, 0))

fig.plot_vector(finger_pos[0], 0.01 * finger_tang_dir[0], (0, 0, 1))
fig.plot_vector(finger_pos[1], 0.01 * finger_tang_dir[1], (0, 0, 1))
fig.plot_vector(finger_pos[2], 0.01 * finger_tang_dir[2], (0, 0, 1))



# ------ Get Finger Forces ------
finger_force = np.array([np.zeros(3), np.zeros(3), np.zeros(3)])
finger_force_pos =  np.array([np.zeros(3), np.zeros(3), np.zeros(3)])
finger_counter =  np.array([0, 0, 0])

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
        finger_force_pos[0] += details['contact_point']
        finger_counter[0] += 1

    if "middle" in frame:
        finger_force[1] += wrench12[:3]
        finger_force_pos[1] += details['contact_point']
        finger_counter[1] += 1

    if "thumb" in frame:
        finger_force[2] += wrench12[:3]
        finger_force_pos[2] += details['contact_point']
        finger_counter[2] += 1

if finger_counter[0] > 0:
    finger_force_pos[0] /= finger_counter[0]
if finger_counter[1] > 0:
    finger_force_pos[1] /= finger_counter[1]
if finger_counter[2] > 0:
    finger_force_pos[2] /= finger_counter[2]


# ------ Plot Finger directions ------
fig.plot_vector(finger_force_pos[0], finger_force[0], (0, 1, 0))
fig.plot_vector(finger_force_pos[1], finger_force[1], (0, 1, 0))
fig.plot_vector(finger_force_pos[2], finger_force[2], (0, 1, 0))


# ------ Project Finger Force on to finger line ------
def point_on_line(start_vector, direction_vetor, point):
    n = direction_vetor / np.linalg.norm(direction_vetor)
    v = point - start_vector
    t = np.dot(v, n)
    d = start_vector + t * n
    return d


point_on_line =  np.array([point_on_line(finger_pos[0], finger_dir[0], finger_force_pos[0]),
                           point_on_line(finger_pos[1], finger_dir[1], finger_force_pos[1]),
                           point_on_line(finger_pos[2], finger_dir[2], finger_force_pos[2])])

# ------ Plot projected Finger Force ------
fig.plot_vector(point_on_line[0], finger_force[0], (0, 0, 1))
fig.plot_vector(point_on_line[1], finger_force[1], (0, 0, 1))
fig.plot_vector(point_on_line[2], finger_force[2], (0, 0, 1))

# ------ Calculate distance of contact point to finger sensor joint. ------
finger_force_dist = np.array([np.linalg.norm(point_on_line[0] - finger_pos[0]),
                              np.linalg.norm(point_on_line[1] - finger_pos[1]),
                              np.linalg.norm(point_on_line[2] - finger_pos[2])])

print(f"Force Dist: {finger_force_dist}")


# ------ Get sub force vector with same direction as the norm and tang vector. ------
finger_norm_force = finger_force * finger_norm_dir
finger_tang_force = finger_force * finger_tang_dir


# ------ Convert force into mia sensor units. ------
MM_PER_M = 1000.0
F_index_norm = (np.linalg.norm(finger_norm_force[0]) * MM_PER_M) / 2.854
F_index_tang = (np.linalg.norm(finger_tang_force[0]) * MM_PER_M) / 2.006

F_mrl_norm = (np.linalg.norm(finger_norm_force[1]) * MM_PER_M) / 2.854
F_mrl_tang = (np.linalg.norm(finger_tang_force[1]) * MM_PER_M) / 2.006

F_thumb_norm = (np.linalg.norm(finger_norm_force[2])) / 0.056
F_thumb_tang = (np.linalg.norm(finger_tang_force[2]) * MM_PER_M) / 3.673


# ------ Print sensor units. ------
print(f"F_index_norm: {F_index_norm}")
print(f"F_index_tang: {F_index_tang}")

print(f"F_mrl_norm: {F_mrl_norm}")
print(f"F_mrl_tang: {F_mrl_tang}")

print(f"F_thumb_norm: {F_thumb_norm}")
print(f"F_thumb_tang: {F_thumb_tang}")

fig.view_init()
fig.show()
