"""
===========================
Distance from line to plane
===========================
"""
print(__doc__)
import time
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.plot_utils as ppu
import pytransform3d.rotations as pr
from distance3d.distance import line_to_plane
from distance3d import random, plotting


random_state = np.random.RandomState(3)
plane_point, plane_normal = random.randn_plane(random_state)

ax = ppu.make_3d_axis(ax_s=2)

accumulated_time = 0.0
for i in range(120000):
    line_point, line_direction = random.randn_line(random_state)
    if random_state.rand() < 0.5:
        line_direction = pr.perpendicular_to_vector(plane_normal)
    start = time.time()
    dist, closest_point_line, closest_point_plane = line_to_plane(
        line_point, line_direction, plane_point, plane_normal)
    end = time.time()
    accumulated_time += end - start
    print(dist)
    if i > 10:
        continue
    plotting.plot_segment(ax, closest_point_line, closest_point_plane, c="k", lw=1)
    plotting.plot_line(ax, line_point, line_direction)
print(f"{accumulated_time=}")

plotting.plot_plane(
    ax=ax, plane_point=plane_point, plane_normal=plane_normal, s=2)
ax.view_init(azim=150, elev=30)
plt.show()
