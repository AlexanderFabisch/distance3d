import os
from pytransform3d.urdf import UrdfTransformManager

import numpy as np
import pytransform3d.visualizer as pv
from distance3d import hydroelastic_contact

from distance3d.hydroelastic_contact._broad_phase import HydroelasticBoundingVolumeHierarchy



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

finger_angle = [1, 0.8, 0.5, 0.3, 0.2]
joint_names = ["j_index_fle", "j_mrl_fle", "j_ring_fle", "j_little_fle", "j_thumb_fle"]

for i, joint_name in enumerate(joint_names):
    tm.set_joint(joint_name, finger_angle[i])

robot_bvh = HydroelasticBoundingVolumeHierarchy(tm, "mia_hand")
robot_bvh.fill_tree_with_colliders(tm, make_artists=True)

for artist in robot_bvh.get_artists():
    artist.add_artist(fig)

fig.view_init()
fig.show()
