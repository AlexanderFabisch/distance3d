import os
import numpy as np
from pytransform3d.transform_manager import TransformManager
from pytransform3d.urdf import UrdfTransformManager
import pytransform3d.transformations as pt
from distance3d import colliders, random


def test_bvh():
    tm = UrdfTransformManager()
    data_dir = "test/data/"
    filename = os.path.join(data_dir, "robot.urdf")
    with open(filename, "r") as f:
        robot_urdf = f.read()
        tm.load_urdf(robot_urdf, mesh_path=data_dir)

    bvh = colliders.BoundingVolumeHierarchy(tm, "robot_arm")
    bvh.fill_tree_with_colliders(tm, make_artists=False)
    assert len(bvh.get_artists()) == 0
    assert len(bvh.get_collider_frames()) == 8
    assert len(bvh.get_colliders()) == 8

    tm.set_joint("joint1", 3.1415926535)
    bvh.update_collider_poses()
    box = colliders.Box(np.eye(4), np.array([1, 1, 1], dtype=float))
    colls = bvh.aabb_overlapping_colliders(box)
    assert len(colls) == 5


def test_bvh_from_colliders():
    tm = TransformManager()
    bvh = colliders.BoundingVolumeHierarchy(tm, "origin")
    random_state = np.random.RandomState(232)

    box2origin, size = random.rand_box(random_state, center_scale=1.0)
    bvh.add_collider("box", colliders.Box(box2origin, size))
    tm.add_transform("box", "origin", box2origin)

    center, radius = random.rand_sphere(random_state, center_scale=1.0)
    sphere2origin = pt.transform_from(R=np.eye(3), p=center)
    bvh.add_collider("sphere", colliders.Sphere(center, radius))
    tm.add_transform("sphere", "origin", sphere2origin)

    capsule2origin, radius, height = random.rand_capsule(
        random_state, center_scale=1.0)
    bvh.add_collider("capsule", colliders.Capsule(capsule2origin, radius, height))
    tm.add_transform("capsule", "origin", capsule2origin)

    cylinder2origin, radius, length = random.rand_cylinder(
        random_state, center_scale=1.0)
    bvh.add_collider("cylinder", colliders.Cylinder(cylinder2origin, radius, length))
    tm.add_transform("cylinder", "origin", cylinder2origin)

    ellipsoid2origin, radii = random.rand_ellipsoid(
        random_state, center_scale=1.0)
    bvh.add_collider("ellipsoid", colliders.Ellipsoid(ellipsoid2origin, radii))
    tm.add_transform("ellipsoid", "origin", ellipsoid2origin)

    overlapping_aabbs = dict()
    for frame1, coll1 in bvh.colliders_.items():
        overlapping_aabbs[frame1] = []
        colls = bvh.aabb_overlapping_colliders(coll1, whitelist=(frame1,))
        for frame2, coll2 in colls.items():
            overlapping_aabbs[frame1].append(frame2)

    box2origin = np.copy(box2origin)
    box2origin[2, 3] -= 1.0
    tm.add_transform("box", "origin", box2origin)
    sphere2origin = np.copy(sphere2origin)
    sphere2origin[2, 3] -= 1.0
    tm.add_transform("sphere", "origin", sphere2origin)
    capsule2origin = np.copy(capsule2origin)
    capsule2origin[2, 3] -= 1.0
    tm.add_transform("capsule", "origin", capsule2origin)
    cylinder2origin = np.copy(cylinder2origin)
    cylinder2origin[2, 3] -= 1.0
    tm.add_transform("cylinder", "origin", cylinder2origin)
    ellipsoid2origin = np.copy(ellipsoid2origin)
    ellipsoid2origin[2, 3] -= 1.0
    tm.add_transform("ellipsoid", "origin", ellipsoid2origin)
    bvh.update_collider_poses()

    overlapping_aabbs2 = dict()
    for frame1, coll1 in bvh.colliders_.items():
        overlapping_aabbs2[frame1] = []
        colls = bvh.aabb_overlapping_colliders(coll1, whitelist=(frame1,))
        for frame2, coll2 in colls.items():
            overlapping_aabbs2[frame1].append(frame2)

    assert overlapping_aabbs == overlapping_aabbs2
