"""
===============================
Distance from plane to cylinder
===============================
"""
print(__doc__)
import time
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.transformations as pt
import pytransform3d.plot_utils as ppu
from distance3d.distance import plane_to_cylinder
from distance3d import random, plotting


random_state = np.random.RandomState(2)
cylinder2origin, radius, length = random.rand_cylinder(
    random_state, min_radius=0.1, min_length=2.0)

ax = ppu.make_3d_axis(ax_s=2)

accumulated_time = 0.0
for i in range(15000):
    plane_point, plane_normal = random.randn_plane(random_state, scale=5.0)
    start = time.time()
    dist, closest_point_plane, closest_point_cylinder = plane_to_cylinder(
        plane_point, plane_normal, cylinder2origin, radius, length)
    end = time.time()
    accumulated_time += end - start
    print(dist)
    if i > 3:
        continue
    plotting.plot_segment(ax, closest_point_plane, closest_point_cylinder,
                          c="k", lw=1)
    plotting.plot_plane(
        ax=ax, plane_point=plane_point, plane_normal=plane_normal, s=10,
        color="b")
print(f"{accumulated_time=}")

ppu.plot_cylinder(ax=ax, A2B=cylinder2origin, radius=radius, length=length,
                  wireframe=False, alpha=0.5)
pt.plot_transform(ax=ax, A2B=cylinder2origin, s=0.1)

plt.show()
