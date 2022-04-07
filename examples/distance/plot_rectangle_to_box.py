"""TODO"""
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.transformations as pt
import pytransform3d.plot_utils as ppu
from distance3d.distance import rectangle_to_box
from distance3d import plotting, random


random_state = np.random.RandomState(0)
box2origin, size = random.rand_box(
    random_state, center_scale=0.1, size_scale=2)

ax = ppu.make_3d_axis(ax_s=3)

for i in range(10):
    rectangle_center, rectangle_axes, rectangle_lengths = random.randn_rectangle(
        random_state, length_scale=5)

    dist, contact_point_rectangle, contact_point_box = rectangle_to_box(
        rectangle_center, rectangle_axes, rectangle_lengths, box2origin, size)
    print(dist)
    plotting.plot_segment(ax, contact_point_rectangle, contact_point_box)
    plotting.plot_rectangle(
        ax, rectangle_center, rectangle_axes, rectangle_lengths,
        show_axes=False)

ppu.plot_box(ax=ax, A2B=box2origin, size=size, wireframe=True, alpha=0.5)
ppu.plot_box(ax=ax, A2B=box2origin, size=size, wireframe=False, alpha=0.5)
pt.plot_transform(ax=ax, A2B=box2origin, s=0.1)

plt.show()
