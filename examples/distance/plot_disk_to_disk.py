"""
==========================
Distance between two disks
==========================
"""
print(__doc__)
import time
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.plot_utils as ppu
from distance3d.distance import disk_to_disk
from distance3d import plotting, random


random_state = np.random.RandomState(0)
center = np.zeros(3)
radius = 1.0
normal = np.array([0.0, 0.0, 1.0])

ax = ppu.make_3d_axis(ax_s=1.5)

accumulated_time = 0.0
for i in range(3000):
    center2, radius2, normal2 = random.rand_circle(random_state)
    start = time.time()
    dist, closest_point1, closest_point2 = disk_to_disk(
        center2, radius2, normal2, center, radius, normal)
    end = time.time()
    accumulated_time += end - start
    print(dist)
    if i > 10:
        continue
    plotting.plot_segment(ax, closest_point1, closest_point2)
    plotting.plot_circle(ax, center2, radius2, normal2, show_normal=True)
print(f"{accumulated_time=}")

plotting.plot_circle(ax, center, radius, normal, show_normal=True)
plt.show()
