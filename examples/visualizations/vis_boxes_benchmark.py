"""
=======================================
Benchmark collision detection for boxes
=======================================
"""
print(__doc__)
import time
import numpy as np
from pytransform3d.transform_manager import TransformManager
import pytransform3d.visualizer as pv
from distance3d.random import rand_box
from distance3d.colliders import BoundingVolumeHierarchy, Box
from distance3d.gjk import gjk_with_simplex


collision_margin = 1e-3
tm = TransformManager(check=False)
bvh = BoundingVolumeHierarchy(tm, "base")
random_state = np.random.RandomState(32)

total_aabbtree = 0.0
for i in range(300):
    box2origin, size = rand_box(
        random_state, center_scale=2.0, size_scale=0.5)
    collider = Box(box2origin, size)
    collider.make_artist(c=(0, 1, 0))
    start_aabbtree = time.time()
    tm.add_transform(i, "base", box2origin)
    stop_aabbtree = time.time()
    total_aabbtree += stop_aabbtree - start_aabbtree
    bvh.add_collider(i, collider)
print("Insertion in AABB tree:")
print(total_aabbtree)

# TODO simplify timing
start = time.time()
total_broad_phase = 0.0
total_gjk = 0.0
collisions = []
for frame1, collider1 in bvh.colliders_.items():
    start_broad_phase = time.time()
    colliders = bvh.aabb_overlapping_colliders(collider1, whitelist=(frame1,))
    stop_broad_phase = time.time()
    total_broad_phase += stop_broad_phase - start_broad_phase
    for frame2, collider2 in colliders.items():
        start_gjk = time.time()
        dist, point1, point2, _ = gjk_with_simplex(collider1, collider2)
        stop_gjk = time.time()
        total_gjk += stop_gjk - start_gjk
        if dist < collision_margin:
            collisions.append((frame1, frame2))
stop = time.time()
total_collision_detection = stop - start
print("Collision detection (broad phase + GJK):")
print(total_collision_detection)
print("Broad phase:")
print(total_broad_phase)
print("GJK:")
print(total_gjk)
print("Collisions between boxes:")
print(collisions)

for frame1, frame2 in collisions:
    bvh.colliders_[frame1].artist_.geometries[0].paint_uniform_color((1, 0, 0))

fig = pv.figure()
for artist in bvh.get_artists():
    artist.add_artist(fig)

if "__file__" in globals():
    fig.show()
else:
    fig.save_image("__open3d_rendered_image.jpg")
