"""
================================
Distance from point to rectangle
================================
"""
print(__doc__)
import time
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.plot_utils as ppu
from distance3d.distance import point_to_rectangle
from distance3d import random, plotting


random_state = np.random.RandomState(9)
rectangle_center, rectangle_axes, rectangle_lengths = random.randn_rectangle(
    random_state, length_scale=5.0)

ax = ppu.make_3d_axis(ax_s=2)

accumulated_time = 0.0
for i in range(45000):
    point = random.randn_point(random_state)
    start = time.time()
    dist, contact_point = point_to_rectangle(point, rectangle_center, rectangle_axes, rectangle_lengths)
    end = time.time()
    accumulated_time += end - start
    print(dist)
    if i > 15:
        continue
    plotting.plot_segment(ax, point, contact_point, c="k", lw=1)
print(f"{accumulated_time=}")

plotting.plot_rectangle(ax, rectangle_center, rectangle_axes, rectangle_lengths)
plt.show()
