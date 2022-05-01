"""
================================
Distance from plane to rectangle
================================
"""
print(__doc__)
import time
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.plot_utils as ppu
from distance3d.distance import plane_to_rectangle
from distance3d import random, plotting


random_state = np.random.RandomState(3)
plane_point, plane_normal = random.randn_plane(random_state)

ax = ppu.make_3d_axis(ax_s=2)

accumulated_time = 0.0
for i in range(30000):
    rectangle_center, rectangle_axes, rectangle_lengths = random.randn_rectangle(
        random_state, length_scale=3.0)
    start = time.time()
    dist, closest_point_plane, closest_point_rectangle = plane_to_rectangle(
        plane_point, plane_normal, rectangle_center, rectangle_axes,
        rectangle_lengths)
    end = time.time()
    accumulated_time += end - start
    print(dist)
    if i > 5:
        continue
    plotting.plot_segment(
        ax, closest_point_plane, closest_point_rectangle, c="k", lw=1)
    plotting.plot_rectangle(
        ax, rectangle_center, rectangle_axes, rectangle_lengths)
print(f"{accumulated_time=}")

plotting.plot_plane(
    ax=ax, plane_point=plane_point, plane_normal=plane_normal, s=2)
ax.view_init(azim=150, elev=30)
plt.show()
