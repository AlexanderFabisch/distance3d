"""
========================================
Benchmark collision detection for meshes
========================================
"""
print(__doc__)
import numpy as np
from pytransform3d.transform_manager import TransformManager
import pytransform3d.visualizer as pv
from distance3d.random import randn_convex
from distance3d.colliders import BoundingVolumeHierarchy, MeshGraph
from distance3d.gjk import gjk_with_simplex
from distance3d.urdf_utils import fast_transform_manager_initialization
from distance3d.benchmark import Timer


collision_margin = 1e-3
tm = TransformManager(check=False)
bvh = BoundingVolumeHierarchy(tm, "base")
random_state = np.random.RandomState(32)

timer = Timer()
box_frames = list(range(200))
fast_transform_manager_initialization(tm, box_frames, "base")
for i in box_frames:
    _, _, vertices, triangles = randn_convex(
        random_state, n_points=10000, center_scale=2.0, std=0.1)
    collider = MeshGraph(vertices, triangles)
    collider.make_artist(c=(0, 1, 0))

    timer.start("aabbtree")
    tm.add_transform(i, "base", np.eye(4))
    bvh.add_collider(i, collider)
    timer.stop_and_add_to_total("aabbtree")

timer.start("collision")
collisions = []
for frame1, collider1 in bvh.colliders_.items():
    timer.start("broad phase")
    colliders = bvh.aabb_overlapping_colliders(collider1, whitelist=(frame1,))
    timer.stop_and_add_to_total("broad phase")
    for frame2, collider2 in colliders.items():
        timer.start("gjk")
        dist, point1, point2, _ = gjk_with_simplex(collider1, collider2)
        timer.stop_and_add_to_total("gjk")
        if dist < collision_margin:
            collisions.append((frame1, frame2))
total_collision_detection = timer.stop("collision")
print(f"Insertion in AABB tree: {timer.total_time_['aabbtree']} s")
print(f"Collision detection (broad phase + GJK): "
      f"{total_collision_detection} s")
print(f"Broad phase: {timer.total_time_['broad phase']}")
print(f"GJK: {timer.total_time_['gjk']}")
print(f"{len(collisions)} collisions")

for frame1, frame2 in collisions:
    bvh.colliders_[frame1].artist_.geometries[0].paint_uniform_color((1, 0, 0))

fig = pv.figure()
for artist in bvh.get_artists():
    artist.add_artist(fig)

if "__file__" in globals():
    fig.show()
else:
    fig.save_image("__open3d_rendered_image.jpg")
