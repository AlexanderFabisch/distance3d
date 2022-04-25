"""
===================================
Distance from line segment to plane
===================================
"""
print(__doc__)
import time
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.plot_utils as ppu
import pytransform3d.rotations as pr
from distance3d.distance import line_segment_to_plane
from distance3d import random, plotting


random_state = np.random.RandomState(3)
plane_point, plane_normal = random.randn_plane(random_state)

ax = ppu.make_3d_axis(ax_s=2)

accumulated_time = 0.0
for i in range(50000):
    segment_start, segment_end = random.randn_line_segment(random_state, scale=2)
    if random_state.rand() < 0.5:
        segment_direction = pr.perpendicular_to_vector(plane_normal)
        segment_end = segment_start + segment_direction * random_state.rand()
    start = time.time()
    dist, closest_point_line, closest_point_plane = line_segment_to_plane(
        segment_start, segment_end, plane_point, plane_normal)
    end = time.time()
    accumulated_time += end - start
    print(dist)
    if i > 10:
        continue
    plotting.plot_segment(ax, closest_point_line, closest_point_plane, c="k", lw=1)
    plotting.plot_segment(ax, segment_start, segment_end)
print(f"{accumulated_time=}")

plotting.plot_plane(
    ax=ax, plane_point=plane_point, plane_normal=plane_normal, s=2)
ax.view_init(azim=150, elev=30)
plt.show()
