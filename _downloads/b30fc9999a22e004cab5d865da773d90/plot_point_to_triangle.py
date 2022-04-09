"""
===============================
Distance from point to triangle
===============================
"""
print(__doc__)
import time
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.plot_utils as ppu
from distance3d.distance import point_to_triangle
from distance3d import random, plotting


random_state = np.random.RandomState(8)
triangle_points = random.randn_triangle(random_state) * 0.5

ax = ppu.make_3d_axis(ax_s=2)

accumulated_time = 0.0
for i in range(40000):
    point = random.randn_point(random_state) * 2
    start = time.time()
    dist, contact_point = point_to_triangle(point, triangle_points)
    end = time.time()
    accumulated_time += end - start
    print(dist)
    if i > 30:
        continue
    plotting.plot_segment(ax, point, contact_point, lw=1)
print(f"{accumulated_time=}")

plotting.plot_triangle(ax, triangle_points)
plt.show()
