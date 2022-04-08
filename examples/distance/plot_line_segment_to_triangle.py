"""
======================================
Distance from line segment to triangle
======================================
"""
print(__doc__)
import time
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.plot_utils as ppu
from distance3d.distance import line_segment_to_triangle
from distance3d import random, plotting


random_state = np.random.RandomState(4)
triangle_points = random.randn_triangle(random_state)

ax = ppu.make_3d_axis(ax_s=2)

accumulated_time = 0.0
for i in range(6500):
    segment_start, segment_end = random.randn_line_segment(random_state)
    start = time.time()
    dist, contact_point_segment, contact_point_triangle = line_segment_to_triangle(
        segment_start, segment_end, triangle_points)
    end = time.time()
    accumulated_time += end - start
    print(dist)
    if i > 10:
        continue
    plotting.plot_segment(ax, contact_point_segment, contact_point_triangle)
    plotting.plot_segment(ax, segment_start, segment_end)
print(f"{accumulated_time=}")

plotting.plot_triangle(ax, triangle_points)
plt.show()
