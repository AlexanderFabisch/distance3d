"""
==========================================
Distance from line segment to line segment
==========================================
"""
print(__doc__)
import time
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.plot_utils as ppu
from distance3d.distance import line_segment_to_line_segment
from distance3d import random, plotting


random_state = np.random.RandomState(0)
segment_start1, segment_end1 = random.randn_line_segment(random_state)
segment_direction1 = segment_end1 - segment_start1
segment_direction1 /= np.linalg.norm(segment_direction1)

ax = ppu.make_3d_axis(ax_s=3)

accumulated_time = 0.0
for i in range(50000):
    segment_start2, segment_end2 = random.randn_line_segment(random_state)
    if random_state.rand() < 0.5:
        segment_end2 = (segment_start2 + segment_direction1
                        * np.linalg.norm(segment_end2 - segment_start2))
    start = time.time()
    dist, closest_point_segment1, closest_point_segment2 = line_segment_to_line_segment(
        segment_start1, segment_end1, segment_start2, segment_end2)
    end = time.time()
    accumulated_time += end - start
    print(dist)
    if i > 15:
        continue
    plotting.plot_segment(
        ax, closest_point_segment1, closest_point_segment2, c="k", lw=1)
    plotting.plot_segment(ax, segment_start2, segment_end2)
print(f"{accumulated_time=}")

plotting.plot_segment(ax, segment_start1, segment_end1)
plt.show()
