import os
from pytransform3d.urdf import UrdfTransformManager
import pytransform3d.visualizer as pv
from distance3d import robot


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
    tm.set_joint(joint_name, 0.5)
geometries = robot.get_geometries(tm, "robot_arm")

fig = pv.figure()
for g in geometries:
    fig.add_geometry(g)
fig.view_init()
fig.set_zoom(1.5)
if "__file__" in globals():
    fig.show()
else:
    fig.save_image("__open3d_rendered_image.jpg")
