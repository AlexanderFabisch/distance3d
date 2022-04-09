"""
==============================
Distance between two triangles
==============================
"""
print(__doc__)
import time
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.plot_utils as ppu
from distance3d.distance import triangle_to_triangle
from distance3d import random, plotting


random_state = np.random.RandomState(0)
triangle_points1 = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=float)

ax = ppu.make_3d_axis(ax_s=1)

accumulated_time = 0.0
for i in range(1300):
    triangle_points2 = random.randn_triangle(random_state) * 0.3 + 0.7 * np.sign(random_state.randn(3))
    start = time.time()
    dist, contact_point_triangle1, contact_point_triangle2 = triangle_to_triangle(
        triangle_points1, triangle_points2)
    end = time.time()
    accumulated_time += end - start
    print(dist)
    if i > 10:
        continue
    plotting.plot_segment(
        ax, contact_point_triangle1, contact_point_triangle2, c="k", lw=1)
    plotting.plot_triangle(ax, triangle_points2)
print(f"{accumulated_time=}")

plotting.plot_triangle(ax, triangle_points1, surface_alpha=0.8)
plt.show()
