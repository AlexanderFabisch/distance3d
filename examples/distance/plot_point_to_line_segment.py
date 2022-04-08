"""
===================================
Distance from point to line segment
===================================
"""
print(__doc__)
import time
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.plot_utils as ppu
from distance3d.distance import point_to_line_segment
from distance3d import random, plotting


random_state = np.random.RandomState(3)
segment_start, segment_end = random.randn_line_segment(random_state, scale=2)

ax = ppu.make_3d_axis(ax_s=5)

accumulated_time = 0.0
for i in range(50000):
    point = random.randn_point(random_state)
    start = time.time()
    dist, contact_point_line = point_to_line_segment(point, segment_start, segment_end)
    end = time.time()
    accumulated_time += end - start
    print(dist)
    if i > 25:
        continue
    plotting.plot_segment(ax, point, contact_point_line, c="k", lw=1)
print(f"{accumulated_time=}")

plotting.plot_segment(ax, segment_start, segment_end)
plt.show()
