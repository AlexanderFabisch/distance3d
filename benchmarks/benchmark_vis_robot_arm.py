import timeit
from functools import partial
from distance3d.aabb_tree import all_aabbs_overlap
import os
import numpy as np
from pytransform3d.urdf import UrdfTransformManager
import pytransform3d.visualizer as pv
from distance3d import random, colliders, gjk, broad_phase
import matplotlib.pyplot as plt


def run(with_aabb_tree, robotBVH, worldBVH):
    if with_aabb_tree:

        pairs = robotBVH.aabb_overlapping_with_other_BVH(worldBVH)

        for pair in pairs:
            frame, collider = pair[0]
            _, box = pair[1]

            gjk.gjk_intersection(collider, box)
    else:
        aabbs1 = []
        robot_coll_list = list(robotBVH.colliders_.items())
        for frame, collider in robot_coll_list:
            aabbs1.append(collider.aabb())

        aabbs2 = []
        world_coll_list = list(worldBVH.colliders_.items())
        for _, box in world_coll_list:
            aabbs2.append(box.aabb())

        _, _, pairs = all_aabbs_overlap(aabbs1, aabbs2)

        for pair in pairs:
            frame, collider = robot_coll_list[pair[0]]
            _, box = world_coll_list[pair[1]]

            gjk.gjk_intersection(collider, box)


def start(amount, space_size):
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

    robot_bvh = broad_phase.BoundingVolumeHierarchy(tm, "robot_arm")
    robot_bvh.fill_tree_with_colliders(tm, make_artists=True)

    world_bvh = broad_phase.BoundingVolumeHierarchy(tm, "world")

    random_state = np.random.RandomState(5)

    for i in range(amount):
        box2origin, size = random.rand_box(
            random_state, center_scale=space_size, size_scale=0.3)
        box2origin[:3, 3] += 0.2
        color = random_state.rand(3)
        box_artist = pv.Box(size=size, A2B=box2origin, c=color)
        box = colliders.Margin(
            colliders.Box(box2origin, size, artist=box_artist), 0.03)
        world_bvh.add_collider("Box %s" % i, box)

    return robot_bvh, world_bvh


def bench(robotBVH, worldBVH, use_tree):
    times = timeit.repeat(partial(run, robotBVH=robotBVH, worldBVH=worldBVH, with_aabb_tree=use_tree),
                          repeat=10, number=10)
    print(f"Mean: {np.mean(times):.5f}; Std. dev.: {np.std(times):.5f}")

    return np.mean(times)


plt.xlabel("aabb amount")
plt.ylabel("time in sec")

size = 0.3
times_brute = []
times_tree = []
y_steps = []
for amount in range(1, 100, 1):
    robot_bvh, world_bvh = start(amount, size)

    print("Amount: %d Size %f" % (amount, size))

    tree_time = bench(robot_bvh, world_bvh, True)
    brute_time = bench(robot_bvh, world_bvh, False)
    times_brute.append(brute_time)
    times_tree.append(tree_time)
    y_steps.append(amount)

plt.plot(y_steps, times_tree, markersize=20, label="Tree size: %0.1f" % size)
plt.plot(y_steps, times_brute, markersize=20, label="Brute size: %0.1f" % size)

plt.legend()
plt.show()
