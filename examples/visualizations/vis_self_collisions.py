"""
==========================
Self-collisions of a robot
==========================
"""
print(__doc__)
import os
import numpy as np
from pytransform3d.urdf import UrdfTransformManager
import pytransform3d.visualizer as pv
from distance3d import colliders, self_collision


BASE_DIR = "test/data/"
data_dir = BASE_DIR
search_path = ".."
while (not os.path.exists(data_dir) and
       os.path.dirname(search_path) != "distance3d"):
    search_path = os.path.join(search_path, "..")
    data_dir = os.path.join(search_path, BASE_DIR)

tm = UrdfTransformManager()
filename = os.path.join(data_dir, "robot.urdf")
with open(filename, "r") as f:
    robot_urdf = f.read()
    tm.load_urdf(robot_urdf, mesh_path=data_dir)
tm.set_joint("joint2", 1.57)
tm.set_joint("joint3", 1.57)
tm.set_joint("joint5", 2.05)

bvh = colliders.BoundingVolumeHierarchy(tm, "robot_arm")
bvh.fill_tree_with_colliders(tm, make_artists=True, fill_self_collision_whitelists=True)

contacts = self_collision.detect(bvh)

fig = pv.figure()
fig.plot_transform(np.eye(4), s=0.2)

for frame, collider in bvh.colliders_.items():
    if contacts[frame]:
        collider.artist_.geometries[0].paint_uniform_color((1, 0, 0))
    collider.artist_.add_artist(fig)
    fig.plot_transform(collider.artist_.A2B, s=0.1)
fig.view_init()
fig.set_zoom(1.5)
if "__file__" in globals():
    fig.show()
else:
    fig.save_image("__open3d_rendered_image.jpg")
