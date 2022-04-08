"""
===========================
Distance from point to disk
===========================
"""
print(__doc__)
import time
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.rotations as pr
import pytransform3d.plot_utils as ppu
from distance3d.distance import point_to_disk
from distance3d import random, plotting


random_state = np.random.RandomState(0)
point = np.array([-0.2, -0.6, -0.4])
center = np.array([0.0, 0.0, 0.3])
radius = 0.5
normal = pr.norm_vector(np.array([0.5, -0.3, 0.4]))

ax = ppu.make_3d_axis(ax_s=1)

accumulated_time = 0.0
for i in range(62000):
    point = random.randn_point(random_state)
    start = time.time()
    distance, contact_point = point_to_disk(point, center, radius, normal)
    end = time.time()
    accumulated_time += end - start
    print(f"{distance}")
    if i > 25:
        continue
    plotting.plot_segment(ax, point, contact_point, c="k", lw=1)
print(f"{accumulated_time=}")

ax.scatter(center[0], center[1], center[2])
plotting.plot_circle(ax, center, radius, normal)
plt.show()
