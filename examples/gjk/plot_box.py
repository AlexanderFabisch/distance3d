"""
===============================
Distance between boxes with GJK
===============================
"""
print(__doc__)
import time
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.plot_utils as ppu
from distance3d import gjk, random, geometry, plotting, colliders


random_state = np.random.RandomState(0)
box2origin, size = random.rand_box(random_state, 0.1, 3)
vertices = geometry.convert_box_to_vertices(box2origin, size)
box_collider = colliders.Box(box2origin, size)

ax = ppu.make_3d_axis(ax_s=3)

accumulated_time = 0.0
for i in range(3000):
    box2origin2, size2 = random.rand_box(random_state, 2, 1)
    box_collider2 = colliders.Box(box2origin2, size2)
    start = time.time()
    dist, closest_point_box, closest_point_box2, _ = gjk.gjk(
        box_collider, box_collider2)
    end = time.time()
    accumulated_time += end - start
    print(dist)
    if i > 10:
        continue
    plotting.plot_segment(
        ax, closest_point_box, closest_point_box2, c="k", lw=1)
    ppu.plot_box(ax=ax, A2B=box2origin2, size=size2, wireframe=False, alpha=0.2)
print(f"{accumulated_time=}")

ppu.plot_box(
    ax=ax, A2B=box2origin, size=size, wireframe=False, alpha=0.5,
    color="yellow")
plt.show()
