"""
==========================
Distance from line to line
==========================
"""
print(__doc__)
import time
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.plot_utils as ppu
from distance3d.distance import line_to_line
from distance3d import random, plotting


random_state = np.random.RandomState(0)
line_point1, line_direction1 = random.randn_line(random_state)

ax = ppu.make_3d_axis(ax_s=3)

accumulated_time = 0.0
for i in range(85000):
    line_point2, line_direction2 = random.randn_line(random_state)
    if random_state.rand() < 0.5:
        line_direction2 = line_direction1
    start = time.time()
    dist, closest_point1, closest_point2 = line_to_line(
        line_point1, line_direction1, line_point2, line_direction2)
    end = time.time()
    accumulated_time += end - start
    print(dist)
    if i > 10:
        continue
    plotting.plot_segment(ax, closest_point1, closest_point2, c="k", lw=1)
    plotting.plot_line(ax, line_point2, line_direction2)
print(f"{accumulated_time=}")

plotting.plot_line(ax, line_point1, line_direction1)
plt.show()
