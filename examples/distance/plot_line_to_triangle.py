"""
==============================
Distance from line to triangle
==============================
"""
print(__doc__)
import time
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.plot_utils as ppu
from distance3d.distance import line_to_triangle
from distance3d import random
from distance3d import plotting


random_state = np.random.RandomState(4)
triangle_points = random.randn_triangle(random_state)

ax = ppu.make_3d_axis(ax_s=2)

accumulated_time = 0.0
for i in range(7500):
    line_point, line_direction = random.randn_line(random_state)
    start = time.time()
    dist, contact_point_line, contact_point_triangle = line_to_triangle(
        line_point, line_direction, triangle_points)
    end = time.time()
    accumulated_time += end - start
    print(dist)
    if i > 10:
        continue
    plotting.plot_segment(ax, contact_point_line, contact_point_triangle)
    plotting.plot_line(ax, line_point, line_direction)
print(f"{accumulated_time=}")

plotting.plot_triangle(ax, triangle_points)
plt.show()
