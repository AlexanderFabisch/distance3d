"""
===============================
Distance from line to rectangle
===============================
"""
print(__doc__)
import time
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.plot_utils as ppu
from distance3d.distance import line_to_rectangle
from distance3d import random, plotting


random_state = np.random.RandomState(0)
rectangle_center, rectangle_axes, rectangle_lengths = random.randn_rectangle(
    random_state, center_scale=0.3, length_scale=3.0)

ax = ppu.make_3d_axis(ax_s=2)

accumulated_time = 0.0
for i in range(7000):
    line_point, line_direction = random.randn_line(random_state)
    start = time.time()
    dist, closest_point_line, closest_point_rectangle = line_to_rectangle(
        line_point, line_direction, rectangle_center, rectangle_axes, rectangle_lengths)
    end = time.time()
    accumulated_time += end - start
    print(dist)
    if i > 10:
        continue
    plotting.plot_segment(
        ax, closest_point_line, closest_point_rectangle, c="k", lw=1)
    plotting.plot_line(ax, line_point, line_direction)
print(f"{accumulated_time=}")

plotting.plot_rectangle(ax, rectangle_center, rectangle_axes, rectangle_lengths)
plt.show()
