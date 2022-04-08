"""
============================
Distance from point to plane
============================
"""
print(__doc__)
import time
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.rotations as pr
import pytransform3d.plot_utils as ppu
from distance3d.distance import point_to_plane
from distance3d import random, plotting


random_state = np.random.RandomState(0)
# TODO move to random
plane_point = random_state.randn(3)
plane_normal = pr.norm_vector(random_state.randn(3))

ax = ppu.make_3d_axis(ax_s=2)

accumulated_time = 0.0
for i in range(150000):
    point = random.randn_point(random_state)
    start = time.time()
    dist, contact_point = point_to_plane(point, plane_point, plane_normal)
    end = time.time()
    accumulated_time += end - start
    print(dist)
    if i > 10:
        continue
    plotting.plot_segment(ax, point, contact_point, c="k", lw=1)
print(f"{accumulated_time=}")

# TODO how to plot a plane? part of pytransform3d?
ppu.plot_vector(ax=ax, start=plane_point, direction=plane_normal)
plt.show()
