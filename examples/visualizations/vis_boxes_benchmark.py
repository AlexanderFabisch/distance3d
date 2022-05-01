"""
=======================================
Benchmark collision detection for boxes
=======================================
"""
print(__doc__)
import numpy as np
from pytransform3d.transform_manager import TransformManager
import pytransform3d.visualizer as pv
from distance3d.random import rand_box
from distance3d.colliders import BoundingVolumeHierarchy, Box
from distance3d.gjk import gjk_with_simplex
from distance3d.benchmark import Timer


collision_margin = 1e-3
tm = TransformManager(check=False)
bvh = BoundingVolumeHierarchy(tm, "base")
random_state = np.random.RandomState(32)

timer = Timer()
for i in range(300):
    box2origin, size = rand_box(
        random_state, center_scale=2.0, size_scale=0.5)
    collider = Box(box2origin, size)
    collider.make_artist(c=(0, 1, 0))
    timer.start("aabbtree")
    tm.add_transform(i, "base", box2origin)
    timer.stop_and_add_to_total("aabbtree")
    bvh.add_collider(i, collider)

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
