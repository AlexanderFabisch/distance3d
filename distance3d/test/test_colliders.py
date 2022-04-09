import os
import numpy as np
from pytransform3d.urdf import UrdfTransformManager
from distance3d.colliders import BoundingVolumeHierarchy, Box


def test_bvh():
    tm = UrdfTransformManager()
    data_dir = "test/data/"
    filename = os.path.join(data_dir, "robot.urdf")
    with open(filename, "r") as f:
        robot_urdf = f.read()
        tm.load_urdf(robot_urdf, mesh_path=data_dir)

    bvh = BoundingVolumeHierarchy(tm, "robot_arm")
    bvh.fill_tree_with_colliders(tm, make_artists=False)
    assert len(bvh.get_artists()) == 0
    assert len(bvh.get_collider_frames()) == 8
    assert len(bvh.get_colliders()) == 8

    tm.set_joint("joint1", 3.1415926535)
    bvh.update_collider_poses()
    box = Box(np.eye(4), np.array([1, 1, 1]))
    colliders = bvh.aabb_overlapping_colliders(box)
    assert len(colliders) == 5
