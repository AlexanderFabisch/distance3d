import os
import time
import numpy as np
from pytransform3d.urdf import UrdfTransformManager
import pytransform3d.visualizer as pv
from distance3d import robot, random, colliders, gjk


def animation_callback(
        step, n_frames, tm, colls, boxes, joint_names):
    angle = 0.5 * np.cos(2.0 * np.pi * (step / n_frames))
    for joint_name in joint_names:
        tm.set_joint(joint_name, angle)
    colls.update_collider_poses()

    total_time = 0.0
    for box in boxes:
        start = time.time()
        for collider in colls.overlapping_colliders(box):
            dist = gjk.gjk_with_simplex(collider, box)[0]
        stop = time.time()
        total_time += stop - start
    print(f"With AABBTree: {total_time}")

    total_time = 0.0
    for idx, collider in enumerate(colls.get_colliders()):
        start = time.time()
        had_contact = False
        for box in boxes:
            dist = gjk.gjk_with_simplex(collider, box)[0]
            if dist < 1e-6:
                had_contact = True
        stop = time.time()
        total_time += stop - start

        geometry = collider.artist_.geometries[0]
        if had_contact:
            geometry.paint_uniform_color((1, 0, 0))
        else:
            geometry.paint_uniform_color((0.5, 0.5, 0.5))
    print(f"Without AABBTree: {total_time}")

    return colls.get_artists()


BASE_DIR = "test/data/"
data_dir = BASE_DIR
search_path = "."
while (not os.path.exists(data_dir) and
       os.path.dirname(search_path) != "distance3d"):
    search_path = os.path.join(search_path, "..")
    data_dir = os.path.join(search_path, BASE_DIR)

tm = UrdfTransformManager()
filename = os.path.join(data_dir, "robot.urdf")
with open(filename, "r") as f:
    robot_urdf = f.read()
    tm.load_urdf(robot_urdf, mesh_path=data_dir)
joint_names = ["joint%d" % i for i in range(1, 7)]
for joint_name in joint_names:
    tm.set_joint(joint_name, 0.7)

colls = robot.get_colliders(tm, "robot_arm")

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

for artist in colls.get_artists():
    artist.add_artist(fig)
fig.view_init()
fig.set_zoom(1.5)
n_frames = 100
if "__file__" in globals():
    fig.animate(animation_callback, n_frames, loop=True,
                fargs=(n_frames, tm, colls, boxes, joint_names))
    fig.show()
else:
    fig.save_image("__open3d_rendered_image.jpg")
