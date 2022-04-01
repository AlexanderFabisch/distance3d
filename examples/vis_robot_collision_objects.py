import os
import time
import numpy as np
from pytransform3d.urdf import UrdfTransformManager
import pytransform3d.visualizer as pv
from distance3d import robot, random, colliders, gjk


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

geometries = robot.get_geometries(tm, "robot_arm")
colls = robot.get_colliders(tm, "robot_arm")

random_state = np.random.RandomState(5)

fig = pv.figure()

for _ in range(15):
    box2origin, size = random.rand_box(
        random_state, center_scale=0.3, size_scale=0.3)
    box2origin[:3, 3] += 0.2
    box = colliders.Convex.from_box(box2origin, size)
    color = random_state.rand(3)

    start = time.time()
    for collider, geometry in zip(colls, geometries):
        dist = gjk.gjk_with_simplex(collider, box)[0]
        if dist < 1e-3:
            geometry.paint_uniform_color(color)
    stop = time.time()
    print(stop - start)
    fig.plot_box(size, box2origin, c=color)

for g in geometries:
    fig.add_geometry(g)
fig.view_init()
fig.set_zoom(1.5)
if "__file__" in globals():
    fig.show()
else:
    fig.save_image("__open3d_rendered_image.jpg")
