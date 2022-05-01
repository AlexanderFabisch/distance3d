"""
================================
Distance from plane to ellipsoid
================================
"""
print(__doc__)
import time
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.transformations as pt
import pytransform3d.plot_utils as ppu
from distance3d.distance import plane_to_ellipsoid
from distance3d import random, plotting


random_state = np.random.RandomState(2)
ellipsoid2origin, radii = random.rand_ellipsoid(random_state)

ax = ppu.make_3d_axis(ax_s=2)

accumulated_time = 0.0
for i in range(15000):
    plane_point, plane_normal = random.randn_plane(random_state)
    start = time.time()
    dist, closest_point_plane, closest_point_ellipsoid = plane_to_ellipsoid(
        plane_point, plane_normal, ellipsoid2origin, radii)
    end = time.time()
    accumulated_time += end - start
    print(dist)
    if i > 1:
        continue
    plotting.plot_segment(ax, closest_point_plane, closest_point_ellipsoid,
                          c="k", lw=1)
    plotting.plot_plane(
        ax=ax, plane_point=plane_point, plane_normal=plane_normal, s=2,
        color="b")
print(f"{accumulated_time=}")

ppu.plot_ellipsoid(ax=ax, A2B=ellipsoid2origin, radii=radii, wireframe=False, alpha=0.5)
pt.plot_transform(ax=ax, A2B=ellipsoid2origin, s=0.1)

plt.show()
