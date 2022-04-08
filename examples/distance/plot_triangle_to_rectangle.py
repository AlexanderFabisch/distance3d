"""
===================================
Distance from triangle to rectangle
===================================
"""
print(__doc__)
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.plot_utils as ppu
from distance3d.distance import triangle_to_rectangle
from distance3d import plotting, random


random_state = np.random.RandomState(0)
rectangle_center, rectangle_axes, rectangle_lengths = random.randn_rectangle(
    random_state, length_scale=5)

ax = ppu.make_3d_axis(ax_s=3)

for i in range(10):
    triangle_points = random.randn_triangle(random_state) * 2
    dist, contact_point_triangle, contact_point_rectangle = triangle_to_rectangle(
        triangle_points, rectangle_center, rectangle_axes, rectangle_lengths)
    print(dist)
    points = np.vstack((contact_point_triangle, contact_point_rectangle))
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.plot(points[:, 0], points[:, 1], points[:, 2])
    plotting.plot_triangle(ax, triangle_points)

plotting.plot_rectangle(
    ax, rectangle_center, rectangle_axes, rectangle_lengths, show_axes=True)

plt.show()
