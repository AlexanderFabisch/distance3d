"""
==============================
Distance from rectangle to box
==============================
"""
print(__doc__)
import time
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.transformations as pt
import pytransform3d.plot_utils as ppu
from distance3d.distance import rectangle_to_box
from distance3d import plotting, random


random_state = np.random.RandomState(0)
box2origin, size = random.rand_box(
    random_state, center_scale=0.1, size_scale=2)

ax = ppu.make_3d_axis(ax_s=5)

accumulated_time = 0.0
for i in range(90):
    rectangle_center, rectangle_axes, rectangle_lengths = random.randn_rectangle(
        random_state, center_scale=2.5, length_scale=5)

    start = time.time()
    dist, contact_point_rectangle, contact_point_box = rectangle_to_box(
        rectangle_center, rectangle_axes, rectangle_lengths, box2origin, size)
    end = time.time()
    accumulated_time += end - start
    print(dist)
    if i > 10:
        continue
    plotting.plot_segment(
        ax, contact_point_rectangle, contact_point_box, c="k", lw=1)
    plotting.plot_rectangle(
        ax, rectangle_center, rectangle_axes, rectangle_lengths,
        show_axes=False)
print(f"{accumulated_time=}")

ppu.plot_box(ax=ax, A2B=box2origin, size=size, wireframe=True, alpha=0.5)
ppu.plot_box(ax=ax, A2B=box2origin, size=size, wireframe=False, alpha=0.5)
pt.plot_transform(ax=ax, A2B=box2origin, s=0.1)
plt.show()
