"""
============================
Distance from point to plane
============================
"""
print(__doc__)
import time
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.plot_utils as ppu
from distance3d.distance import point_to_plane
from distance3d import random, plotting


random_state = np.random.RandomState(3)
plane_point, plane_normal = random.randn_plane(random_state)

ax = ppu.make_3d_axis(ax_s=2)

accumulated_time = 0.0
for i in range(150000):
    point = random.randn_point(random_state)
    start = time.time()
    dist, closest_point = point_to_plane(point, plane_point, plane_normal)
    end = time.time()
    accumulated_time += end - start
    print(dist)
    if i > 10:
        continue
    plotting.plot_segment(ax, point, closest_point, lw=1)
print(f"{accumulated_time=}")

plotting.plot_plane(
    ax=ax, plane_point=plane_point, plane_normal=plane_normal, s=2)
ax.view_init(azim=150, elev=30)
plt.show()
