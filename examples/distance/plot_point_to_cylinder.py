"""
===============================
Distance from point to cylinder
===============================
"""
print(__doc__)
import time
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.transformations as pt
import pytransform3d.plot_utils as ppu
from distance3d.distance import point_to_cylinder
from distance3d import random, plotting


random_state = np.random.RandomState(3)
cylinder2origin, radius, length = random.rand_cylinder(random_state, min_radius=0.5, min_length=1.0)

ax = ppu.make_3d_axis(ax_s=2)

accumulated_time = 0.0
for i in range(25000):
    point = random.randn_point(random_state)
    start = time.time()
    dist, contact_point = point_to_cylinder(point, cylinder2origin, radius, length)
    end = time.time()
    accumulated_time += end - start
    print(dist)
    if i > 50:
        continue
    plotting.plot_segment(ax, point, contact_point, c="k", lw=1)
print(f"{accumulated_time=}")

ppu.plot_cylinder(ax=ax, A2B=cylinder2origin, radius=radius, length=length, wireframe=False, alpha=0.5)
pt.plot_transform(ax=ax, A2B=cylinder2origin, s=0.1)
plt.show()
