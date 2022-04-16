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
from distance3d import colliders, gjk


BASE_DIR = "test/data/"
data_dir = BASE_DIR
search_path = ".."
while (not os.path.exists(data_dir) and
       os.path.dirname(search_path) != "distance3d"):
    search_path = os.path.join(search_path, "../..")
    data_dir = os.path.join(search_path, BASE_DIR)

tm = UrdfTransformManager()
filename = os.path.join(data_dir, "robot.urdf")
with open(filename, "r") as f:
    robot_urdf = f.read()
    tm.load_urdf(robot_urdf, mesh_path=data_dir)
tm.set_joint("joint2", 2.1)
tm.set_joint("joint3", 2.1)
tm.set_joint("joint5", 2.1)

colls = colliders.BoundingVolumeHierarchy(tm, "robot_arm")
colls.fill_tree_with_colliders(tm, make_artists=True, fill_self_collision_whitelists=True)

collision_margin = 1e-3
for frame, collider in colls.colliders_.items():
    candidates = colls.aabb_overlapping_colliders(collider, whitelist=colls.self_collision_whitelists_[frame])
    in_contact = []
    for frame2, collider2 in candidates.items():
        if collider is collider2:
            continue
        dist, _, _, _ = gjk.gjk_with_simplex(collider, collider2)
        if dist < collision_margin:
            in_contact.append(collider2)
    collider.artist_.geometries[0].paint_uniform_color((1, 0, 0))

fig = pv.figure()

for artist in colls.get_artists():
    artist.add_artist(fig)
fig.view_init()
fig.set_zoom(1.5)
if "__file__" in globals():
    fig.show()
else:
    fig.save_image("__open3d_rendered_image.jpg")
