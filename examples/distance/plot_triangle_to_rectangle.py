"""
===================================
Distance from triangle to rectangle
===================================
"""
print(__doc__)
import time
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.plot_utils as ppu
from distance3d.distance import triangle_to_rectangle
from distance3d import random, plotting


random_state = np.random.RandomState(0)
rectangle_center, rectangle_axes, rectangle_lengths = random.randn_rectangle(
    random_state, center_scale=0.1, length_scale=5)

ax = ppu.make_3d_axis(ax_s=5)

accumulated_time = 0.0
for i in range(750):
    triangle_points = random.randn_triangle(random_state) + 4 * np.sign(np.random.randn(1, 3))
    start = time.time()
    dist, closest_point_triangle, closest_point_rectangle = triangle_to_rectangle(
        triangle_points, rectangle_center, rectangle_axes, rectangle_lengths)
    end = time.time()
    accumulated_time += end - start
    print(dist)
    if i > 10:
        continue
    plotting.plot_segment(
        ax, closest_point_triangle, closest_point_rectangle, c="k", lw=1)
    plotting.plot_triangle(ax, triangle_points, surface_alpha=0.8)
print(f"{accumulated_time=}")

plotting.plot_rectangle(
    ax, rectangle_center, rectangle_axes, rectangle_lengths, show_axes=True,
    surface_alpha=0.1)

plt.show()
