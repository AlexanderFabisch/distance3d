"""
============================
Distance from plane to plane
============================
"""
print(__doc__)
import time
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.plot_utils as ppu
from distance3d.distance import plane_to_plane
from distance3d import random, plotting


random_state = np.random.RandomState(3)
plane_point, plane_normal = random.randn_plane(random_state)

ax = ppu.make_3d_axis(ax_s=2)

accumulated_time = 0.0
for i in range(15000):
    plane_point2, plane_normal2 = random.randn_plane(random_state)
    if random_state.rand() >= 0.5:
        plane_normal2 = plane_normal
    start = time.time()
    dist, closest_point1, closest_point2 = plane_to_plane(
        plane_point2, plane_normal2, plane_point, plane_normal)
    end = time.time()
    accumulated_time += end - start
    print(dist)
    if i > 2:
        continue
    plotting.plot_segment(ax, closest_point1, closest_point2, c="k", lw=1)
    plotting.plot_plane(
        ax=ax, plane_point=plane_point2, plane_normal=plane_normal2, s=2)
print(f"{accumulated_time=}")

plotting.plot_plane(
    ax=ax, plane_point=plane_point, plane_normal=plane_normal, s=2, color="r")
ax.view_init(azim=150, elev=30)
plt.show()
