import timeit
import numpy as np
import distance3d.broad_phase
import os
from pytransform3d.urdf import UrdfTransformManager

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
joint_names = ["joint%d" % i for i in range(1, 7)]
for joint_name in joint_names:
    tm.set_joint(joint_name, 0.7)

bvh = distance3d.broad_phase.BoundingVolumeHierarchy(tm, "robot_arm")


def bench():
    bvh.fill_tree_with_colliders(tm, make_artists=False)


times = timeit.repeat(bench, repeat=10, number=10)
print(f"Mean: {np.mean(times):.5f}; Std. dev.: {np.std(times):.5f}")
