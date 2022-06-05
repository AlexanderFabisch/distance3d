import os
import numpy as np
from pytransform3d.urdf import UrdfTransformManager

import distance3d.broad_phase
from distance3d import colliders, self_collision


def test_self_collision_detection():
    tm = UrdfTransformManager()
    data_dir = "test/data/"
    filename = os.path.join(data_dir, "robot.urdf")
    with open(filename, "r") as f:
        robot_urdf = f.read()
        tm.load_urdf(robot_urdf, mesh_path=data_dir)

    bvh = distance3d.broad_phase.BoundingVolumeHierarchy(tm, "robot_arm")
    bvh.fill_tree_with_colliders(
        tm, make_artists=True, fill_self_collision_whitelists=True)

    contacts = list(self_collision.detect(bvh).values())
    assert np.count_nonzero(contacts) == 0
    assert not self_collision.detect_any(bvh)

    tm.set_joint("joint2", 1.57)
    tm.set_joint("joint3", 1.57)
    tm.set_joint("joint5", 1.93)
    bvh.update_collider_poses()

    contacts = list(self_collision.detect(bvh).values())
    assert np.count_nonzero(contacts) == 0
    assert not self_collision.detect_any(bvh)

    tm.set_joint("joint2", 1.57)
    tm.set_joint("joint3", 1.57)
    tm.set_joint("joint5", 2.05)
    bvh.update_collider_poses()

    contacts = list(self_collision.detect(bvh).values())
    assert np.count_nonzero(contacts) == 3
    assert self_collision.detect_any(bvh)


def test_self_collision_detection_mesh():
    tm = UrdfTransformManager()
    data_dir = "test/data/"
    filename = os.path.join(data_dir, "simple_mechanism.urdf")
    with open(filename, "r") as f:
        robot_urdf = f.read()
        tm.load_urdf(robot_urdf, mesh_path=data_dir)

    bvh = distance3d.broad_phase.BoundingVolumeHierarchy(tm, "simple_mechanism")
    bvh.fill_tree_with_colliders(
        tm, make_artists=True, fill_self_collision_whitelists=True)

    contacts = list(self_collision.detect(bvh).values())
    assert np.count_nonzero(contacts) == 0
    assert not self_collision.detect_any(bvh)

    tm.set_joint("joint1", 1.5)
    tm.set_joint("joint2", 1.5)
    tm.set_joint("joint3", 1.5)
    bvh.update_collider_poses()

    contacts = list(self_collision.detect(bvh).values())
    assert np.count_nonzero(contacts) == 2
    assert self_collision.detect_any(bvh)

    tm.set_joint("joint1", 2.7)
    tm.set_joint("joint2", 2.7)
    tm.set_joint("joint3", 2.7)
    bvh.update_collider_poses()

    contacts = list(self_collision.detect(bvh).values())
    assert np.count_nonzero(contacts) == 4
    assert self_collision.detect_any(bvh)
