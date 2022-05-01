"""
=================================
Distance from line segment to box
=================================
"""
print(__doc__)
import time
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.plot_utils as ppu
from distance3d.distance import line_segment_to_box
from distance3d import random, plotting


random_state = np.random.RandomState(2)
box2origin = np.eye(4)
size = np.ones(3)

ax = ppu.make_3d_axis(ax_s=2)

accumulated_time = 0.0
for i in range(8000):
    segment_start, segment_end = random.randn_line_segment(random_state)
    start = time.time()
    dist, closest_point_segment, closest_point_box = line_segment_to_box(
        segment_start, segment_end, box2origin, size)
    end = time.time()
    accumulated_time += end - start
    print(dist)
    if i > 15:
        continue
    plotting.plot_segment(
        ax, closest_point_segment, closest_point_box, c="k", lw=1)
    plotting.plot_segment(ax, segment_start, segment_end)
print(f"{accumulated_time=}")

ppu.plot_box(ax=ax, A2B=box2origin, size=size, wireframe=False, alpha=0.5)
plt.show()
