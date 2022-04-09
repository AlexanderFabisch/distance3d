"""
========================================
Collisions between robot and environment
========================================
"""
print(__doc__)
import os
import time
import numpy as np
import open3d as o3d
from pytransform3d.urdf import UrdfTransformManager
import pytransform3d.visualizer as pv
from distance3d import random, colliders, gjk


class AnimationCallback:
    def __init__(self, with_aabb_tree=True, n_frames=100, verbose=0):
        self.with_aabb_tree = with_aabb_tree
        self.n_frames = n_frames
        self.verbose = verbose
        self.total_time = 0.0

    def __call__(self, step, n_frames, tm, colls, boxes, joint_names):
        if step == 0:
            self.total_time = 0.0

        angle = 0.5 * np.cos(2.0 * np.pi * (step / n_frames))
        for joint_name in joint_names:
            tm.set_joint(joint_name, angle)
        colls.update_collider_poses()

        in_contact = {frame: False for frame in colls.get_collider_frames()}
        in_aabb = {frame: False for frame in colls.get_collider_frames()}

        total_time = 0.0
        if self.with_aabb_tree:
            for box in boxes:
                start = time.time()
                overlapping_colls = colls.aabb_overlapping_colliders(
                    box).items()
                for frame, collider in overlapping_colls:
                    dist = gjk.gjk_with_simplex(collider, box)[0]
                    in_aabb[frame] |= True
                    in_contact[frame] |= dist < 1e-6
                stop = time.time()
                total_time += stop - start
            if self.verbose:
                print(f"With AABBTree: {total_time}")
        else:
            for frame, collider in colls.colliders.items():
                start = time.time()
                for box in boxes:
                    dist = gjk.gjk_with_simplex(collider, box)[0]
                    in_contact[frame] |= dist < 1e-6
                stop = time.time()
                total_time += stop - start
            if self.verbose:
                print(f"Without AABBTree: {total_time}")

        self.total_time += total_time

        for frame in in_contact:
            geometry = colls.colliders[frame].artist_.geometries[0]
            if in_contact[frame]:
                geometry.paint_uniform_color((1, 0, 0))
            elif in_aabb[frame]:
                geometry.paint_uniform_color((1, 0.5, 0))
            else:
                geometry.paint_uniform_color((0.5, 0.5, 0.5))

        if step == self.n_frames - 1:
            print(f"Total time: {self.total_time}")

        return colls.get_artists()


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
joint_names = ["joint%d" % i for i in range(1, 7)]
for joint_name in joint_names:
    tm.set_joint(joint_name, 0.7)

colls = colliders.ColliderTree(tm, "robot_arm")
colls.fill_tree_with_colliders(tm, make_artists=True)

random_state = np.random.RandomState(5)

fig = pv.figure()

boxes = []
for _ in range(15):
    box2origin, size = random.rand_box(
        random_state, center_scale=0.3, size_scale=0.3)
    box2origin[:3, 3] += 0.2
    color = random_state.rand(3)
    box_artist = pv.Box(size=size, A2B=box2origin, c=color)
    box_artist.add_artist(fig)
    box = colliders.Box(box2origin, size, artist=box_artist)
    boxes.append(box)

    aabb = box.aabb()
    aabb = o3d.geometry.AxisAlignedBoundingBox(aabb[:, 0], aabb[:, 1])
    aabb.color = (1, 0, 0)
    fig.add_geometry(aabb)

for artist in colls.get_artists():
    artist.add_artist(fig)
fig.view_init()
fig.set_zoom(1.5)
n_frames = 100
animation_callback = AnimationCallback(
    with_aabb_tree=True, n_frames=n_frames, verbose=0)
if "__file__" in globals():
    fig.animate(animation_callback, n_frames, loop=True,
                fargs=(n_frames, tm, colls, boxes, joint_names))
    fig.show()
else:
    fig.save_image("__open3d_rendered_image.jpg")
