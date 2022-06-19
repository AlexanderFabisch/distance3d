"""
===========================
Distance from point to line
===========================
"""
print(__doc__)
import time
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.plot_utils as ppu
from distance3d.distance import point_to_line
from distance3d import random, plotting


random_state = np.random.RandomState(0)
line_point, line_direction = random.randn_line(random_state)

ax = ppu.make_3d_axis(ax_s=2)

accumulated_time = 0.0
for i in range(75000):
    point = random.randn_point(random_state)
    start = time.time()
    dist, closest_point_line = point_to_line(point, line_point, line_direction)
    end = time.time()
    accumulated_time += end - start
    print(dist)
    if i > 10:
        continue
    plotting.plot_segment(ax, point, closest_point_line, lw=1)
print(f"{accumulated_time=}")

plotting.plot_line(ax, line_point, line_direction)
plt.show()
