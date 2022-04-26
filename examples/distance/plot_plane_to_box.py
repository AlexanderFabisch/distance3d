"""
==========================
Distance from point to box
==========================
"""
print(__doc__)
import time
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.transformations as pt
import pytransform3d.plot_utils as ppu
from distance3d.distance import plane_to_box
from distance3d import random, plotting


random_state = np.random.RandomState(3)
plane_point, plane_normal = random.randn_plane(random_state)

ax = ppu.make_3d_axis(ax_s=3)

accumulated_time = 0.0
for i in range(30000):
    box2origin, size = random.rand_box(random_state)
    start = time.time()
    dist, closest_point_plane, closest_point_box = plane_to_box(
        plane_point, plane_normal, box2origin, size)
    end = time.time()
    accumulated_time += end - start
    print(dist)
    if i > 5:
        continue
    plotting.plot_segment(ax, closest_point_plane, closest_point_box,
                          c="k", lw=1)
    ppu.plot_box(ax=ax, A2B=box2origin, size=size, wireframe=False, alpha=0.5)
    pt.plot_transform(ax=ax, A2B=box2origin, s=0.1)
print(f"{accumulated_time=}")

plotting.plot_plane(
    ax=ax, plane_point=plane_point, plane_normal=plane_normal, s=2, color="b")
ax.view_init(azim=150, elev=30)
plt.show()
